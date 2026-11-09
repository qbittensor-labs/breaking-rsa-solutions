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

/* produce <savefile_name>.d, a binary file containing the
   line numbers of relations in the savefile that should *not*
   graduate to the singleton removal (or are just plain invalid).

   This code has to touch all of the relations, and when the
   dataset is large avoiding excessive memory use is tricky.
   Cavallar's paper suggests dropping relations into a hashtable
   and ignoring relations that map to the same hash bin. This keeps
   memory use down but causes good relations to be thrown away.
   Another option is to put the (a,b) values of relations into the
   hashtable, so that hash collisions can be resolved rigorously.
   Unfortunately this means we have to budget 12 bytes for each
   unique relation, and there could be tens (hundreds!) of millions 
   of them.

   The implementation here is a compromise: we do duplicate removal
   in two passes. The first pass maps relations into a hashtable
   of bits, and we save (on disk) the list of hash bins where two 
   or more relations collide. The second pass refills the hashtable
   of bits with just these entries, then reads through the complete
   dataset again and saves the (a,b) values of any relation that
   maps to one of the filled-in hash bins. The memory use in the
   first pass is constant, and the memory use of the second pass
   is 12 bytes per duplicate relation. Assuming unique relations
   greatly outnumber duplicates, this solution finds all the duplicates
   with no false positives, and the memory use is low enough so
   that singleton filtering is a larger memory bottleneck 
   
   One useful optimization for really big problems would turn the
   first-pass hashtable into a Bloom filter using several hash
   functions. This would make it much more effective at avoiding
   false positives as the hashtable gets more congested */

static const uint8 hashmask[] = {0x01, 0x02, 0x04, 0x08,
				 0x10, 0x20, 0x40, 0x80};

static uint32 purge_duplicates_pass2(msieve_obj *obj,
				uint32 log2_hashtable1_size,
				uint32 max_relations) {

	savefile_t *savefile = &obj->savefile;
	FILE *bad_relation_fp;
	FILE *collision_fp;
	FILE *out_fp;
	uint32 i;
	char buf[LINE_BUF_SIZE];
	uint32 num_duplicates;
	uint32 num_relations;
	uint32 next_bad_relation;
	uint32 curr_relation;
	uint8 *bit_table;
	hashtable_t duplicates;
	uint32 key[2];

	logprintf(obj, "commencing duplicate removal, pass 2\n");

	/* fill in the list of hash collisions */

	sprintf(buf, "%s.hc", savefile->name);
	collision_fp = fopen(buf, "rb");
	if (collision_fp == NULL) {
		logprintf(obj, "error: dup2 can't open collision file\n");
		exit(-1);
	}
	bit_table = (uint8 *)xcalloc(
			(size_t)1 << (log2_hashtable1_size - 3), 
			sizeof(uint8));

	while (fread(&i, (size_t)1, sizeof(uint32), collision_fp) != 0) {
		if (i < ((uint32)1 << log2_hashtable1_size)) {
			bit_table[i / 8] |= 1 << (i % 8);
		}
	}
	fclose(collision_fp);

	/* set up for reading the list of relations */

	savefile_open(savefile, SAVEFILE_READ);
	sprintf(buf, "%s.br", savefile->name);
	bad_relation_fp = fopen(buf, "rb");
	if (bad_relation_fp == NULL) {
		logprintf(obj, "error: dup2 can't open rel file\n");
		exit(-1);
	}
	sprintf(buf, "%s.d", savefile->name);
	out_fp = fopen(buf, "wb");
	if (out_fp == NULL) {
		logprintf(obj, "error: dup2 can't open output file\n");
		exit(-1);
	}
	hashtable_init(&duplicates, (uint32)WORDS_IN(key), 0);

	num_duplicates = 0;
	num_relations = 0;
	curr_relation = (uint32)(-1);
	next_bad_relation = (uint32)(-1);
	fread(&next_bad_relation, (size_t)1, 
			sizeof(uint32), bad_relation_fp);
	savefile_read_line(buf, sizeof(buf), savefile);

	while (!savefile_eof(savefile)) {
		
		uint32 hashval;
		int64 a;
		uint32 b;
		char *next_field;

		if (buf[0] != '-' && !isdigit(buf[0])) {

			/* no relation on this line */

			savefile_read_line(buf, sizeof(buf), savefile);
			continue;
		}
		if (++curr_relation == next_bad_relation) {

			/* this relation isn't valid; save it and
			   read in the next invalid relation line number */

			fwrite(&curr_relation, (size_t)1, 
					sizeof(uint32), out_fp);
			fread(&next_bad_relation, (size_t)1, 
					sizeof(uint32), bad_relation_fp);
			savefile_read_line(buf, sizeof(buf), savefile);
			continue;
		}

		if (max_relations && curr_relation >= max_relations)
			break;

		/* determine if the (a,b) coordinates of the
		   relation collide in the table of bits */

		a = strtoll(buf, &next_field, 10);
		b = strtoul(next_field + 1, NULL, 10);
		key[0] = (uint32)a;
		key[1] = ((a >> 32) & 0x1f) | (b << 5);

		hashval = (HASH1(key[0]) ^ HASH2(key[1])) >>
				(32 - log2_hashtable1_size);

		if (bit_table[hashval/8] & hashmask[hashval % 8]) {

			/* relation collides in the first hashtable;
			   use the second hashtable to determine 
			   rigorously if the relation was previously seen */

			uint32 is_dup;
			hashtable_find(&duplicates, key, NULL, &is_dup);

			if (!is_dup) {

				/* relation was seen for the first time;
				   doesn't count as a duplicate */

				num_relations++;
			}
			else {
				/* relation was previously seen; this
				   time it's a duplicate */

				fwrite(&curr_relation, (size_t)1, 
						sizeof(uint32), out_fp);
				num_duplicates++;
			}
		}
		else {
			/* no collision; relation is unique */

			num_relations++;
		}

		savefile_read_line(buf, sizeof(buf), savefile);
	}

	logprintf(obj, "found %u duplicates and %u unique relations\n", 
				num_duplicates, num_relations);
	logprintf(obj, "memory use: %.1f MB\n", 
			(double)((1 << (log2_hashtable1_size-3)) +
			hashtable_sizeof(&duplicates)) / 1048576);

	/* clean up and finish */

	savefile_close(savefile);
	fclose(bad_relation_fp);
	fclose(out_fp);
	sprintf(buf, "%s.hc", savefile->name);
	remove(buf);
	sprintf(buf, "%s.br", savefile->name);
	remove(buf);

	free(bit_table);
	hashtable_free(&duplicates);
	return num_relations;
}

/*--------------------------------------------------------------------*/
static double estimate_rel_size(savefile_t *savefile) {

	uint32 i;
	char buf[LINE_BUF_SIZE];
	uint32 num_relations = 0;
	uint32 totlen = 0;

	savefile_open(savefile, SAVEFILE_READ);
	savefile_read_line(buf, sizeof(buf), savefile);
	for (i = 0; i < 100 && !savefile_eof(savefile); i++) {

		if (buf[0] != '-' && !isdigit(buf[0])) {
			/* no relation on this line */
			savefile_read_line(buf, sizeof(buf), savefile);
			continue;
		}

		num_relations++;
		totlen += strlen(buf);
	}

	savefile_close(savefile);
	if (num_relations == 0)
		return 0;
	return (double)totlen / num_relations;
}

/*--------------------------------------------------------------------*/
#define LOG2_BIN_SIZE 17
#define BIN_SIZE (1 << (LOG2_BIN_SIZE))
#define TARGET_HITS_PER_PRIME 40.0

uint32 nfs_purge_duplicates(msieve_obj *obj, factor_base_t *fb,
				uint32 max_relations, 
				uint32 *num_relations_out) {

	uint32 i;
	savefile_t *savefile = &obj->savefile;
	FILE *bad_relation_fp;
	FILE *collision_fp;
	uint32 curr_relation;
	char buf[LINE_BUF_SIZE];
	uint32 num_relations;
	uint32 num_collisions;
	uint32 num_skipped_b;
	uint32 num_composite;
	uint8 *hashtable;
	uint32 blob[2];
	uint32 log2_hashtable1_size;
	double rel_size = estimate_rel_size(savefile);
	mpz_t scratch;

	uint8 *free_relation_bits;
	uint32 *free_relations;
	uint32 num_free_relations;
	uint32 num_free_relations_alloc;

	uint32 *prime_bins;
	double bin_max;

	uint8 tmp_factors[COMPRESSED_P_MAX_SIZE];
	uint32 array_size;
	relation_t tmp_rel;

	tmp_rel.factors = tmp_factors;

	logprintf(obj, "commencing duplicate removal, pass 1\n");

	savefile_open(savefile, SAVEFILE_READ);
	sprintf(buf, "%s.br", savefile->name);
	bad_relation_fp = fopen(buf, "wb");
	if (bad_relation_fp == NULL) {
		logprintf(obj, "error: dup1 can't open relation file\n");
		exit(-1);
	}
	sprintf(buf, "%s.hc", savefile->name);
	collision_fp = fopen(buf, "wb");
	if (collision_fp == NULL) {
		logprintf(obj, "error: dup1 can't open collision file\n");
		exit(-1);
	}

	/* figure out how large the stage 1 hashtable should be.
	   We want there to be many more bins in the hashtable than
	   relations in the savefile, but it takes too long to
	   actually count the relations. So we estimate the average
	   relation size and then the number of relations */

	log2_hashtable1_size = 28;
	if (rel_size > 0.0) {
		double num_rels; /* estimated */
#if 0 /* WAS: !defined(WIN32) && !defined(_WIN64) */
		if (savefile->isCompressed) {
			char name_gz[256];
			sprintf(name_gz, "%s.gz", savefile->name);
			num_rels = get_file_size(name_gz) / 0.55 / rel_size;
		} else 
#endif /* get_file_size( ) will now do the same internally */
			num_rels = get_file_size(savefile->name) / rel_size;
		log2_hashtable1_size = log(num_rels * 10.0) / M_LN2 + 0.5;
	}
	if (log2_hashtable1_size < 25)
		log2_hashtable1_size = 25;
	if (log2_hashtable1_size > 31)
		log2_hashtable1_size = 31;

	hashtable = (uint8 *)xcalloc((size_t)1 << 
				(log2_hashtable1_size - 3), sizeof(uint8));
	prime_bins = (uint32 *)xcalloc((size_t)1 << (32 - LOG2_BIN_SIZE),
					sizeof(uint32));

	/* set up the structures for tracking free relations */

	free_relation_bits = (uint8 *)xcalloc(
				((size_t)(FREE_RELATION_LIMIT/2) + 7) / 8,
				(size_t)1);
	num_free_relations = 0;
	num_free_relations_alloc = 5000;
	free_relations = (uint32 *)xmalloc(num_free_relations_alloc *
						sizeof(uint32));

	curr_relation = (uint32)(-1);
	num_relations = 0;
	num_collisions = 0;
	num_skipped_b = 0;
	num_composite = 0;
	mpz_init(scratch);

	/* ---- parallel pass 1 ------------------------------------------------
	   The per-relation cost here is nfs_read_relation(): it re-derives the
	   norm and divides out every factor with GMP -- essentially all of
	   pass-1's time, and completely independent between relations. So parse
	   a batch of relation lines in PARALLEL (each thread with its own poly
	   scratch), then apply every shared-state update (hashtable toggle,
	   collision file, prime_bins, free relations, bad-relation list,
	   counters) SEQUENTIALLY in the original relation order. The parse of
	   relation k is a pure function of its line + the read-only polynomials,
	   so the batched result is identical to the serial one and the output
	   (collision file, prime_bins, free-rel bits, counts) is byte-for-byte
	   the same; only the GMP-heavy parse is threaded. */
	{
	#define DUP_PASS1_BATCH 65536
		uint32 nthreads = obj->num_threads ? obj->num_threads : 1;
		uint32 t, k, cnt;
		int batch_done = 0;
		char (*lines)[LINE_BUF_SIZE] = (char (*)[LINE_BUF_SIZE])
			xmalloc((size_t)DUP_PASS1_BATCH * LINE_BUF_SIZE);
		uint32 *lrel = (uint32 *)xmalloc(DUP_PASS1_BATCH * sizeof(uint32));
		int32 *rstat = (int32 *)xmalloc(DUP_PASS1_BATCH * sizeof(int32));
		int64 *ra = (int64 *)xmalloc(DUP_PASS1_BATCH * sizeof(int64));
		uint32 *rb = (uint32 *)xmalloc(DUP_PASS1_BATCH * sizeof(uint32));
		uint8 *rnfr = (uint8 *)xmalloc(DUP_PASS1_BATCH);
		uint8 *rnfa = (uint8 *)xmalloc(DUP_PASS1_BATCH);
		uint8 *rfac = (uint8 *)xmalloc((size_t)DUP_PASS1_BATCH *
						COMPRESSED_P_MAX_SIZE);
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

		savefile_read_line(buf, sizeof(buf), savefile);
		while (!batch_done && !savefile_eof(savefile)) {

			/* (1) fill a batch of relation lines (sequential read) */
			cnt = 0;
			while (cnt < DUP_PASS1_BATCH && !savefile_eof(savefile)) {
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
				memcpy(lines[cnt], buf, LINE_BUF_SIZE);
				lrel[cnt] = curr_relation;
				cnt++;
				savefile_read_line(buf, sizeof(buf), savefile);
			}

			/* (2) parse the batch in parallel (GMP-heavy) */
			#pragma omp parallel for schedule(dynamic, 256) \
				num_threads(nthreads)
			for (k = 0; k < cnt; k++) {
				int tid = omp_get_thread_num();
				uint32 asz;
				relation_t rel;
				rel.factors = rfac +
					(size_t)k * COMPRESSED_P_MAX_SIZE;
				rstat[k] = nfs_read_relation(lines[k], &tfb[tid],
						&rel, &asz, 1, tpv[tid], 1);
				if (rstat[k] == 0) {
					ra[k] = rel.a;
					rb[k] = rel.b;
					rnfr[k] = rel.num_factors_r;
					rnfa[k] = rel.num_factors_a;
				}
			}

			/* (3) merge in original order -> identical to serial */
			for (k = 0; k < cnt; k++) {
				uint32 hashval;

				if (rstat[k] != 0) {
					fwrite(&lrel[k], (size_t)1,
						sizeof(uint32), bad_relation_fp);
					if (rstat[k] == -99)
						num_skipped_b++;
					else if (rstat[k] == -98)
						num_composite++;
					else
						logprintf(obj, "error %d reading "
							"relation %u\n",
							rstat[k], lrel[k]);
					continue;
				}

				if (lrel[k] > 0 && (lrel[k] % 10000000 == 0))
					printf("read %uM relations\n",
							lrel[k] / 1000000);

				num_relations++;
				blob[0] = (uint32)ra[k];
				blob[1] = ((ra[k] >> 32) & 0x1f) | (rb[k] << 5);

				hashval = (HASH1(blob[0]) ^ HASH2(blob[1])) >>
						(32 - log2_hashtable1_size);

				if (hashtable[hashval / 8] & hashmask[hashval % 8]) {
					fwrite(&hashval, (size_t)1,
						sizeof(uint32), collision_fp);
					num_collisions++;
					hashtable[hashval / 8] &=
						~hashmask[hashval % 8];
				}
				else {
					hashtable[hashval / 8] |=
						hashmask[hashval % 8];
				}

				if (rb[k] == 0) {
					if (num_free_relations ==
						num_free_relations_alloc) {
						num_free_relations_alloc *= 2;
						free_relations = (uint32 *)xrealloc(
							free_relations,
							num_free_relations_alloc *
							sizeof(uint32));
					}
					free_relations[num_free_relations++] =
							(uint32)(ra[k]);
				}
				else {
					uint32 num_r = rnfr[k];
					uint32 num_a = rnfa[k];
					uint8 *fptr = rfac +
						(size_t)k * COMPRESSED_P_MAX_SIZE;

					for (i = array_size = 0;
							i < num_r + num_a; i++) {
						uint64 p = decompress_p(fptr,
								&array_size);

						if (p >= ((uint64)1 << 32))
							continue;

						prime_bins[p / BIN_SIZE]++;

						if (i >= num_r &&
						    p > MAX_PACKED_PRIME &&
						    p < FREE_RELATION_LIMIT) {
							p = p / 2;
							free_relation_bits[p / 8] |=
								hashmask[p % 8];
						}
					}
				}
			}
		}

		for (t = 0; t < nthreads; t++) {
			mpz_poly_free(&tfb[t].rfb.poly);
			mpz_poly_free(&tfb[t].afb.poly);
			mpz_clear(tpv[t]);
		}
		free(tfb); free(tpv);
		free(lines); free(lrel); free(rstat); free(ra); free(rb);
		free(rnfr); free(rnfa); free(rfac);
	#undef DUP_PASS1_BATCH
	}

	mpz_clear(scratch);
	free(hashtable);
	savefile_close(savefile);
	fclose(bad_relation_fp);
	fclose(collision_fp);

	if (num_skipped_b > 0)
		logprintf(obj, "skipped %d relations with b > 2^32\n",
				num_skipped_b);
	if (num_composite > 0)
		logprintf(obj, "skipped %d relations with composite factors\n",
				num_composite);
	logprintf(obj, "found %u hash collisions in %u relations\n", 
				num_collisions, num_relations);

	if (max_relations == 0 || max_relations > curr_relation + 1) {

		/* cancel out any free relations that are 
		   already present in the dataset, then add
		   free relations that remain */

		for (i = 0; i < num_free_relations; i++) {
			uint32 p = free_relations[i];

			if (p < FREE_RELATION_LIMIT) {
				p = p / 2;
				free_relation_bits[p / 8] &= ~hashmask[p % 8];
			}
		}
		num_relations += add_free_relations(obj, fb,
					free_relation_bits);
	}
	free(free_relations);
	free(free_relation_bits);

	if (num_collisions == 0) {

		/* no duplicates; no second pass is necessary */

		char buf2[256];
		sprintf(buf, "%s.hc", savefile->name);
		remove(buf);
		sprintf(buf, "%s.br", savefile->name);
		sprintf(buf2, "%s.d", savefile->name);
		if (rename(buf, buf2) != 0) {
			logprintf(obj, "error: dup1 can't rename outfile\n");
			exit(-1);
		}
	}
	else {
		num_relations = purge_duplicates_pass2(obj,
					log2_hashtable1_size,
					max_relations);
	}

	/* the large prime cutoff for the rest of the filtering
	   process should be chosen here. We don't want the bound
	   to depend on an arbitrarily chosen factor base, since
	   that bound may be too large or much too small. The former
	   would make filtering take too long, and the latter 
	   could make filtering impossible.

	   Conceptually, we want the bound to be the point below
	   which large primes appear too often in the dataset. */

	i = 1 << (32 - LOG2_BIN_SIZE);
	bin_max = (double)BIN_SIZE * i /
			log((double)BIN_SIZE * i);
	for (i--; i > 2; i--) {
		double bin_min = (double)BIN_SIZE * i /
				log((double)BIN_SIZE * i);
		double hits_per_prime = (double)prime_bins[i] /
						(bin_max - bin_min);
		if (hits_per_prime > TARGET_HITS_PER_PRIME)
			break;
		bin_max = bin_min;
	}

	free(prime_bins);
	*num_relations_out = num_relations;
	return BIN_SIZE * (i + 0.5);
}
