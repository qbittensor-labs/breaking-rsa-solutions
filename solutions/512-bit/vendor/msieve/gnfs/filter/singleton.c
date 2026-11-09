/*--------------------------------------------------------------------
This source distribution is placed in the public domain by its author,
Jason Papadopoulos. You may use it for any purpose, free of charge,
without having to notify anyone. I disclaim any responsibility for any
errors.

Optionally, please be nice and tell me if you find this source to be
useful. Again optionally, if you add to the functionality present here
please consider making those additions public too, so that others may 
benefit from your work.	

$Id$
--------------------------------------------------------------------*/

#include "filter.h"
#include <omp.h>

/*--------------------------------------------------------------------*/
void nfs_write_lp_file(msieve_obj *obj, factor_base_t *fb,
			filter_t *filter, uint32 max_relations,
			uint32 pass) {

	/* read through the relation file and form a packed 
	   array of relation_ideal_t structures. This is the
	   first step to get the NFS relations into the 
	   algorithm-independent form that the rest of the
	   filtering will use */

	uint32 i;
	savefile_t *savefile = &obj->savefile;
	FILE *relation_fp;
	FILE *final_fp;
	char buf[LINE_BUF_SIZE];
	size_t header_words;
	uint32 next_relation;
	uint32 curr_relation;
	uint32 num_relations;
	hashtable_t unique_ideals;
	relation_ideal_t packed_ideal;
	uint32 have_skip_list = (pass == 0);

	logprintf(obj, "commencing singleton removal, initial pass\n");

	savefile_open(savefile, SAVEFILE_READ);
	sprintf(buf, "%s.d", savefile->name);
	relation_fp = fopen(buf, "rb");
	if (relation_fp == NULL) {
		logprintf(obj, "error: can't open dup file\n");
		exit(-1);
	}
	sprintf(buf, "%s.lp", savefile->name);
	final_fp = fopen(buf, "wb");
	if (final_fp == NULL) {
		logprintf(obj, "error: can't open output LP file\n");
		exit(-1);
	}

	hashtable_init(&unique_ideals, (uint32)WORDS_IN(ideal_t), 0);
	header_words = (sizeof(relation_ideal_t) - 
			sizeof(packed_ideal.ideal_list)) / sizeof(uint32);

	/* for each relation that survived the duplicate removal.
	   The per-relation cost is nfs_read_relation() (GMP norm
	   eval + factor division) plus find_large_ideals(), both pure
	   functions of the relation line and the read-only polynomials
	   -- essentially all of this pass's time. So parse a batch of
	   surviving relations in PARALLEL (each thread with its own
	   deep-copied poly scratch), then map ideals to unique integers
	   and dump to disk SEQUENTIALLY in the original relation order.
	   The ideal->integer assignment (hashtable_find) is insertion-
	   order dependent, so keeping the merge in order makes the .lp
	   file byte-for-byte identical to the serial version; only the
	   GMP-heavy parse is threaded. */

	curr_relation = (uint32)(-1);
	next_relation = (uint32)(-1);
	num_relations = 0;

	{
	#define LP_PASS_BATCH 65536
		uint32 nthreads = obj->num_threads ? obj->num_threads : 1;
		uint32 t, k, cnt;
		int batch_done = 0;
		char (*lines)[LINE_BUF_SIZE] = (char (*)[LINE_BUF_SIZE])
			xmalloc((size_t)LP_PASS_BATCH * LINE_BUF_SIZE);
		uint32 *lrel = (uint32 *)xmalloc(LP_PASS_BATCH * sizeof(uint32));
		int32 *rstat = (int32 *)xmalloc(LP_PASS_BATCH * sizeof(int32));
		relation_lp_t *rlp = (relation_lp_t *)xmalloc(
				(size_t)LP_PASS_BATCH * sizeof(relation_lp_t));
		factor_base_t *tfb;
		mpz_t *tpv;

		if (nthreads < 1) nthreads = 1;
		if (nthreads > 64) nthreads = 64;

		/* per-thread factor base: shallow copy of fb but with private
		   deep-copied polynomials, so the tmp1/2/3 scratch that
		   nfs_read_relation writes is not shared across threads */
		tfb = (factor_base_t *)xmalloc(nthreads * sizeof(factor_base_t));
		tpv = (mpz_t *)xmalloc(nthreads * sizeof(mpz_t));
		for (t = 0; t < nthreads; t++) {
			uint32 j;
			tfb[t] = *fb;
			mpz_poly_init(&tfb[t].rfb.poly);
			mpz_poly_init(&tfb[t].afb.poly);
			tfb[t].rfb.poly.degree = fb->rfb.poly.degree;
			tfb[t].afb.poly.degree = fb->afb.poly.degree;
			for (j = 0; j <= MAX_POLY_DEGREE; j++) {
				mpz_set(tfb[t].rfb.poly.coeff[j],
						fb->rfb.poly.coeff[j]);
				mpz_set(tfb[t].afb.poly.coeff[j],
						fb->afb.poly.coeff[j]);
			}
			mpz_init(tpv[t]);
		}

		fread(&next_relation, (size_t)1,
				sizeof(uint32), relation_fp);
		savefile_read_line(buf, sizeof(buf), savefile);

		while (!batch_done && !savefile_eof(savefile)) {

			/* (1) fill a batch of surviving relation lines
			   (sequential read + skip-list logic, identical
			   to the serial control flow) */
			cnt = 0;
			while (cnt < LP_PASS_BATCH && !savefile_eof(savefile)) {
				if (buf[0] != '-' && !isdigit(buf[0])) {
					savefile_read_line(buf, sizeof(buf),
							savefile);
					continue;
				}
				curr_relation++;
				if (max_relations &&
						curr_relation >= max_relations) {
					batch_done = 1;
					break;
				}
				if (have_skip_list) {
					if (curr_relation == next_relation) {
						fread(&next_relation,
							sizeof(uint32),
							(size_t)1, relation_fp);
						savefile_read_line(buf,
							sizeof(buf), savefile);
						continue;
					}
				}
				else {
					if (curr_relation < next_relation) {
						savefile_read_line(buf,
							sizeof(buf), savefile);
						continue;
					}
					fread(&next_relation, sizeof(uint32),
							(size_t)1, relation_fp);
				}
				memcpy(lines[cnt], buf, LINE_BUF_SIZE);
				lrel[cnt] = curr_relation;
				cnt++;
				savefile_read_line(buf, sizeof(buf), savefile);
			}

			/* (2) parse + find large ideals in parallel */
			#pragma omp parallel for schedule(dynamic, 256) \
				num_threads(nthreads)
			for (k = 0; k < cnt; k++) {
				int tid = omp_get_thread_num();
				uint8 tf[COMPRESSED_P_MAX_SIZE];
				uint32 asz;
				relation_t rel;
				rel.factors = tf;
				rstat[k] = nfs_read_relation(lines[k],
						&tfb[tid], &rel, &asz, 1,
						tpv[tid], 0);
				if (rstat[k] == 0)
					find_large_ideals(&rel, &rlp[k],
						filter->filtmin_r,
						filter->filtmin_a);
			}

			/* (3) map ideals to integers + dump, in original
			   order -> byte-identical to serial */
			for (k = 0; k < cnt; k++) {
				if (rstat[k] != 0)
					continue;

				num_relations++;
				packed_ideal.rel_index = lrel[k];
				packed_ideal.gf2_factors = rlp[k].gf2_factors;
				packed_ideal.ideal_count = rlp[k].ideal_count;

				for (i = 0; i < rlp[k].ideal_count; i++) {
					hashtable_find(&unique_ideals,
						rlp[k].ideal_list + i,
						packed_ideal.ideal_list + i,
						NULL);
				}

				fwrite(&packed_ideal, sizeof(uint32),
					header_words + rlp[k].ideal_count,
					final_fp);
			}
		}

		for (t = 0; t < nthreads; t++) {
			mpz_poly_free(&tfb[t].rfb.poly);
			mpz_poly_free(&tfb[t].afb.poly);
			mpz_clear(tpv[t]);
		}
		free(tfb); free(tpv);
		free(lines); free(lrel); free(rstat); free(rlp);
	#undef LP_PASS_BATCH
	}

	filter->num_relations = num_relations;
	filter->num_ideals = hashtable_get_num(&unique_ideals);
	filter->relation_array = NULL;
	logprintf(obj, "memory use: %.1f MB\n",
			(double)hashtable_sizeof(&unique_ideals) / 1048576);
	hashtable_free(&unique_ideals);
	savefile_close(savefile);
	fclose(relation_fp);
	fclose(final_fp);

	sprintf(buf, "%s.lp", savefile->name);
	filter->lp_file_size = get_file_size(buf);

	sprintf(buf, "%s.d", savefile->name);
	if (remove(buf) != 0) {
		logprintf(obj, "error: can't delete dup file\n");
		exit(-1);
	}
}
