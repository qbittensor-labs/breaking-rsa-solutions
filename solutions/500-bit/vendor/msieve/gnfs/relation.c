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

#include <common.h>
#include "gnfs.h"
#include <omp.h>

/*--------------------------------------------------------------------*/
static
#ifdef __GNUC__
__attribute__((always_inline))
#endif
uint32 divide_factor_out(mpz_t polyval, uint64 p, 
			uint8 *factors, uint32 *array_size_in,
			uint32 *num_factors, uint32 compress,
			mpz_t tmp1, mpz_t tmp2, mpz_t tmp3) {

	/* read the rational factors. Note that the following
	   will work whether a given factor appears only once
	   or whether its full multiplicity is in the relation */

	uint32 i = *num_factors;
	uint32 array_size = *array_size_in;
	uint32 multiplicity = 0;

	if (p < ((uint64)1 << 32)) {
		while (1) {
			uint32 rem = mpz_tdiv_q_ui(tmp1, polyval, p);
			if (rem != 0 || mpz_cmp_ui(tmp1, 0) == 0)
				break;

			multiplicity++;
			mpz_swap(tmp1, polyval);
		}
	}
	else {
		uint64_2gmp(p, tmp1);
		while (1) {
			mpz_tdiv_qr(tmp2, tmp3, polyval, tmp1);
			if (mpz_cmp_ui(tmp3, 0) != 0 || 
			    mpz_cmp_ui(tmp2, 0) == 0)
				break;

			multiplicity++;
			mpz_swap(tmp2, polyval);
		}
	}
	if (i + multiplicity >= TEMP_FACTOR_LIST_SIZE)
		return 1;

	if (compress) {
		if (multiplicity & 1) {
			array_size = compress_p(factors, p, array_size);
			i++;
		}
	}
	else if (multiplicity) {
		i += multiplicity;
		while (multiplicity--)
			array_size = compress_p(factors, p, array_size);
	}
	*num_factors = i;
	*array_size_in = array_size;
	return 0;
}

/*--------------------------------------------------------------------*/
#define RELATION_TF_BOUND 1000

int32 nfs_read_relation(char *buf, factor_base_t *fb, 
			relation_t *r, uint32 *array_size_out,
			uint32 compress, mpz_t polyval,
			uint32 test_primality) {

	/* note that only the polynomials within the factor
	   base need to be initialized */

	uint32 i; 
	uint64 p, btmp;
	int64 a, atmp;
	uint32 b;
	char *tmp, *next_field;
	mpz_poly_t *rpoly = &fb->rfb.poly;
	mpz_poly_t *apoly = &fb->afb.poly;
	uint32 num_factors_r;
	uint32 num_factors_a;
	uint32 array_size = 0;
	uint8 *factors = r->factors;

	/* read the relation coordinates */

	a = strtoll(buf, &next_field, 10);
	tmp = next_field;
	if (tmp[0] != ',' || !isdigit(tmp[1]))
		return -1;

	btmp = strtoull(tmp+1, &next_field, 10);
	tmp = next_field;
	b = (uint32)btmp;
	if (btmp != (uint64)b)
		return -99; /* cannot use large b */

	num_factors_r = 0;
	num_factors_a = 0;
	r->a = a;
	r->b = b;

	/* for free relations, store the roots and not
	   the prime factors */

	if (b == 0) {
		uint32 i;
		uint32 roots[MAX_POLY_DEGREE];
		uint32 high_coeff, num_roots;

		/* note that the finite field rootfinder must
		   be given p < 2^32 */

		p = (uint64)a;
		if (p == 0 || p >= ((uint64)1 << 32))
			return -2;

		if (test_primality && !mp_is_prime_1((uint32)p))
			return -98;

		array_size = compress_p(factors, p, array_size);

		num_roots = poly_get_zeros(roots, apoly,
						(uint32)p, &high_coeff, 0);
		if (num_roots != apoly->degree || high_coeff == 0)
			return -4;
		for (i = 0; i < num_roots; i++) {
			array_size = compress_p(factors, (uint64)roots[i], 
						array_size);
		}

		r->num_factors_r = 1;
		r->num_factors_a = num_roots;
		*array_size_out = array_size;
		return 0;
	}

	if (tmp[0] != ':')
		return -5;
	
	atmp = a % (int64)b;
	if (atmp < 0)
		atmp += b;

	if (mp_gcd_1((uint32)atmp, b) != 1)
		return -6;

	/* handle a rational factor of -1 */

	eval_poly(polyval, a, b, rpoly);
	if (mpz_cmp_ui(polyval, 0) == 0)
		return -6;
	if (mpz_cmp_ui(polyval, 0) < 0) {
		array_size = compress_p(factors, 0, array_size);
		num_factors_r++;
		mpz_abs(polyval, polyval);
	}

	/* read the rational factors (possibly an empty list) */

	if (isxdigit(tmp[1])) {
		do {
			p = strtoull(tmp + 1, &next_field, 16);

			if (test_primality && 
			    p > RELATION_TF_BOUND && 
			    p < ((uint64)1 << 32) &&
	    		    !mp_is_prime_1((uint32)p))
				return -98;

			if (p > 1 && divide_factor_out(polyval, p, 
						factors, &array_size,
						&num_factors_r, compress,
						rpoly->tmp1, rpoly->tmp2,
						rpoly->tmp3)) {
				return -8;
			}
			tmp = next_field;
		} while (tmp[0] == ',' && isxdigit(tmp[1]));
	}
	else {
		tmp++;
	}

	if (tmp[0] != ':')
		return -9;

	/* if there are rational factors still to be accounted
	   for, assume they are small and find them by trial division */

	for (i = p = 0; mpz_cmp_ui(polyval, 1) != 0 && 
				p < RELATION_TF_BOUND; i++) {

		p += prime_delta[i];
		if (divide_factor_out(polyval, p, factors, 
				&array_size, &num_factors_r, 
				compress, rpoly->tmp1, 
				rpoly->tmp2, rpoly->tmp3)) {
			return -10;
		}
	}

	if (mpz_cmp_ui(polyval, 1) != 0)
		return -11;

	/* read the algebraic factors */

	eval_poly(polyval, a, b, apoly);
	if (mpz_cmp_ui(polyval, 0) == 0)
		return -12;
	mpz_abs(polyval, polyval);

	if (isxdigit(tmp[1])) {
		do {
			p = strtoull(tmp + 1, &next_field, 16);

			if (test_primality &&
			    p > RELATION_TF_BOUND && 
			    p < ((uint64)1 << 32) &&
	    		    !mp_is_prime_1((uint32)p))
				return -98;

			if (p > 1 && divide_factor_out(polyval, p, 
						factors, &array_size,
						&num_factors_a, compress,
						apoly->tmp1, apoly->tmp2,
						apoly->tmp3)) {
				return -13;
			}
			tmp = next_field;
		} while (tmp[0] == ',' && isxdigit(tmp[1]));
	}

	/* if there are algebraic factors still to be accounted
	   for, assume they are small and find them by trial division */

	for (i = p = 0; mpz_cmp_ui(polyval, 1) != 0 && 
					p < RELATION_TF_BOUND; i++) {

		p += prime_delta[i];
		if (divide_factor_out(polyval, p, factors, 
				&array_size, &num_factors_a, 
				compress, apoly->tmp1,
				apoly->tmp2, apoly->tmp3)) {
			return -14;
		}
	}

	if (mpz_cmp_ui(polyval, 1) != 0)
		return -15;
	
	r->num_factors_r = num_factors_r;
	r->num_factors_a = num_factors_a;
	*array_size_out = array_size;
	return 0;
}

/*--------------------------------------------------------------------*/
uint32 find_large_ideals(relation_t *rel, 
			relation_lp_t *out, 
			uint32 filtmin_r, uint32 filtmin_a) {
	uint32 i;
	uint32 num_ideals = 0;
	uint32 array_size = 0;
	uint32 num_factors_r;
	int64 a = rel->a;
	uint32 b = rel->b;

	out->gf2_factors = 0;

	/* handle free relations */

	if (b == 0) {
		uint64 p = decompress_p(rel->factors, &array_size);
		uint64 compressed_p = (p - 1) / 2;

		if (p > filtmin_r) {
			ideal_t *ideal = out->ideal_list + num_ideals;

			ideal->p_lo = (uint32)compressed_p;
			ideal->p_hi = (uint16)(compressed_p >> 32);
			ideal->rat_or_alg = RATIONAL_IDEAL;
			ideal->r_lo = (uint32)p;
			ideal->r_hi = (uint16)(p >> 32);
			num_ideals++;
		}
		else if (p > MAX_PACKED_PRIME) {
			out->gf2_factors++;
		}

		if (p > filtmin_a) {
			for (i = 0; i < rel->num_factors_a; i++) {
				ideal_t *ideal = out->ideal_list + 
							num_ideals + i;
				uint64 root = decompress_p(rel->factors,
							&array_size);

				ideal->p_lo = (uint32)compressed_p;
				ideal->p_hi = (uint16)(compressed_p >> 32);
				ideal->rat_or_alg = ALGEBRAIC_IDEAL;
				ideal->r_lo = (uint32)root;
				ideal->r_hi = (uint16)(root >> 32);
			}
			num_ideals += rel->num_factors_a;
		}
		else if (p > MAX_PACKED_PRIME) {
			out->gf2_factors += rel->num_factors_a;
		}

		out->ideal_count = num_ideals;
		return num_ideals;
	}

	/* find the large rational ideals */

	num_factors_r = rel->num_factors_r;

	for (i = 0; i < num_factors_r; i++) {
		uint64 p = decompress_p(rel->factors, &array_size);
		uint64 compressed_p = (p - 1) / 2;

		/* if processing all the ideals, make up a
		   separate unique entry for rational factors of -1 */

		if (p == 0 && filtmin_r == 0) {
			ideal_t *ideal = out->ideal_list + num_ideals;
			ideal->p_lo = (uint32)(-1);
			ideal->p_hi = 0x7fff;
			ideal->rat_or_alg = RATIONAL_IDEAL;
			ideal->r_lo = (uint32)(-1);
			ideal->r_hi = 0xffff;
			num_ideals++;
			continue;
		}

		if (p > filtmin_r) {

			/* make a single unique entry for p, instead
			   of finding the exact number r for which
			   rational_poly(r) mod p is zero */

			ideal_t *ideal = out->ideal_list + num_ideals;

			if (num_ideals >= TEMP_FACTOR_LIST_SIZE)
				return TEMP_FACTOR_LIST_SIZE + 1;

			ideal->p_lo = (uint32)compressed_p;
			ideal->p_hi = (uint16)(compressed_p >> 32);
			ideal->rat_or_alg = RATIONAL_IDEAL;
			ideal->r_lo = (uint32)p;
			ideal->r_hi = (uint16)(p >> 32);
			num_ideals++;
		}
		else if (p > MAX_PACKED_PRIME) {

			/* we only keep a count of the ideals that are
			   too small to list explicitly. NFS filtering
			   will work a little better if we completely
			   ignore the smallest ideals */

			out->gf2_factors++;
		}
	}

	/* repeat for the large algebraic ideals */

	for (i = 0; i < (uint32)rel->num_factors_a; i++) {
		uint64 p = decompress_p(rel->factors, &array_size);
		uint64 compressed_p = (p - 1) / 2;

		if (p > filtmin_a) {
			ideal_t *ideal = out->ideal_list + num_ideals;
			uint32 bmodp;

			if (num_ideals >= TEMP_FACTOR_LIST_SIZE)
				return TEMP_FACTOR_LIST_SIZE + 1;

			/* this time we have to find the exact r */

			bmodp = b % p;
			if (bmodp == 0) {
				ideal->r_lo = (uint32)p;
				ideal->r_hi = (uint16)(p >> 32);
			}
			else {
				uint64 root;
				int64 mapped_a = a % (int64)p;
				if (mapped_a < 0)
					mapped_a += p;

				root = (uint64)mapped_a;
				if (p < ((uint64)1 << 32)) {
					root = mp_modmul_1((uint32)root, 
						    mp_modinv_1(bmodp, 
						    	(uint32)p), (uint32)p);
				}
				else {
					root = mp_modmul_2(root, 
						    mp_modinv_2(bmodp, p), p);
				}
				ideal->r_lo = (uint32)root;
				ideal->r_hi = (uint16)(root >> 32);

			}
			ideal->p_lo = (uint32)compressed_p;
			ideal->p_hi = (uint16)(compressed_p >> 32);
			ideal->rat_or_alg = ALGEBRAIC_IDEAL;
			num_ideals++;
		}
		else if (p > MAX_PACKED_PRIME) {
			out->gf2_factors++;
		}
	}

	out->ideal_count = num_ideals;
	return num_ideals;
}

/*--------------------------------------------------------------------*/
static int bsearch_relation(const void *key, const void *rel) {
	relation_t *r = (relation_t *)rel;
	uint32 *k = (uint32 *)key;

	if ((*k) < r->rel_index)
		return -1;
	if ((*k) > r->rel_index)
		return 1;
	return 0;
}

static void remap_relation_numbers(msieve_obj *obj, 
				uint32 num_cycles, 
				la_col_t *cycle_list, 
				uint32 num_relations,
				relation_t *rlist) {
	uint32 i, j;

	/* walk through the list of cycles and convert
	   each occurence of a line number in the savefile
	   to an offset in the relation array */

	for (i = 0; i < num_cycles; i++) {
		la_col_t *c = cycle_list + i;

		for (j = 0; j < c->cycle.num_relations; j++) {

			/* since relations were read in order of increasing
			   relation index (= savefile line number), use 
			   binary search to locate relation j for this
			   cycle, then save a pointer to it */

			relation_t *rptr = (relation_t *)bsearch(
						c->cycle.list + j,
						rlist,
						(size_t)num_relations,
						sizeof(relation_t),
						bsearch_relation);
			if (rptr == NULL) {
				/* this cycle is corrupt somehow */
				logprintf(obj, "error: cannot locate "
						"relation %u\n", 
						c->cycle.list[j]);
				exit(-1);
			}
			else {
				c->cycle.list[j] = rptr - rlist;
			}
		}
	}
}

/*--------------------------------------------------------------------*/
static int compare_uint32(const void *x, const void *y) {
	uint32 *xx = (uint32 *)x;
	uint32 *yy = (uint32 *)y;
	if (*xx > *yy)
		return 1;
	if (*xx < *yy)
		return -1;
	return 0;
}

typedef struct {
	uint32 relidx;
	uint32 count;
} relcount_t;

/*--------------------------------------------------------------------*/
/* ---- BLOCK LINE READER (2026-08-21) --------------------------------------
   savefile_read_line() is one gzgets() per line, and nfs_get_cycle_relations()
   below calls it once for EVERY line of m.dat -- ~16.3 M times over 2.1 GB on
   c151 -- to find the ones it actually wants. Worth-Try.md §2a measures the
   whole read+parse phase at 44 s, the largest serial item left inside -nc2,
   and -nc2 is the largest CPU-exposed stage of a run that is otherwise 94%
   GPU-bound. A slower-single-core validator host is charged that 44 s in full.

   This pulls the file in 8 MB gzread() blocks and splits lines in-process.
   gzread() is used (NOT read()) so a gzipped savefile still works exactly as
   before -- zlib decompresses transparently either way, which is the whole
   reason msieve went through gzgets in the first place.

   Semantics deliberately match gzgets(): copy at most max_len-1 bytes, stop
   after a '\n', NUL-terminate. It returns the number of bytes written, so 0
   means end of file -- and that return is what replaces the savefile_eof()
   test in the caller's loop. */
typedef struct {
	gzFile *fp;   /* same (sloppy) type savefile_t uses */
	char *blk;
	size_t blk_len;
	size_t blk_off;
	int eof;
} blkreader_t;

#define BLKREAD_SIZE (8u << 20)

static void blk_init(blkreader_t *br, gzFile *fp) {
	br->fp = fp;
	br->blk = (char *)xmalloc(BLKREAD_SIZE);
	br->blk_len = br->blk_off = 0;
	br->eof = 0;
}

static void blk_free(blkreader_t *br) {
	free(br->blk);
	br->blk = NULL;
}

static size_t blk_getline(blkreader_t *br, char *buf, size_t max_len) {

	size_t j = 0;

	if (max_len == 0)
		return 0;

	while (j + 1 < max_len) {
		char c;
		if (br->blk_off == br->blk_len) {
			int n;
			if (br->eof)
				break;
			n = gzread((gzFile)br->fp, br->blk,
					(unsigned)BLKREAD_SIZE);
			if (n <= 0) {
				br->eof = 1;
				break;
			}
			br->blk_len = (size_t)n;
			br->blk_off = 0;
		}
		c = br->blk[br->blk_off++];
		buf[j++] = c;
		if (c == '\n')
			break;
	}
	buf[j] = 0;
	return j;
}

/*--------------------------------------------------------------------*/
static void nfs_get_cycle_relations(msieve_obj *obj, 
				factor_base_t *fb, uint32 num_cycles, 
				la_col_t *cycle_list, 
				uint32 *num_relations_out,
				relation_t **rlist_out,
				uint32 compress,
				uint32 dependency) {
	uint32 i, j;
	char buf[LINE_BUF_SIZE];
	relation_t *rlist;
	savefile_t *savefile = &obj->savefile;

	hashtable_t unique_relidx;
	uint32 num_unique_relidx;
	uint32 *relidx_list;
	relcount_t *entry;

	hashtable_init(&unique_relidx, 
			(uint32)WORDS_IN(relcount_t), 
			(uint32)1);

	/* fill the hashtable */

	for (i = 0; i < num_cycles; i++) {
		la_col_t *c = cycle_list + i;
		uint32 num_relations = c->cycle.num_relations;
		uint32 *list = c->cycle.list;

		for (j = 0; j < num_relations; j++) {
			uint32 already_seen;
			entry = (relcount_t *)hashtable_find(
						&unique_relidx, list + j, 
						NULL, &already_seen);
			if (!already_seen)
				entry->count = 1;
			else
				entry->count++;
		}
	}

	/* convert the internal list of hashtable entries into
	   a list of 32-bit relation numbers. If reading in just
	   the relations in one dependency, squeeze out relations
	   that appear an even number of times */

	hashtable_close(&unique_relidx);
	num_unique_relidx = hashtable_get_num(&unique_relidx);
	entry = (relcount_t *)hashtable_get_first(&unique_relidx);
	relidx_list = unique_relidx.match_array;

	for (i = j = 0; i < num_unique_relidx; i++) {
		if (dependency == 0 || entry->count % 2 != 0)
			relidx_list[j++] = entry->relidx;

		entry = (relcount_t *)hashtable_get_next(
					&unique_relidx, entry);
	}
	num_unique_relidx = j;

	/* sort the list in order of increasing relation number */

	qsort(relidx_list, (size_t)num_unique_relidx, 
		sizeof(uint32), compare_uint32);

	logprintf(obj, "cycles contain %u unique relations\n", 
				num_unique_relidx);

	savefile_open(savefile, SAVEFILE_READ);

	/* read the list of relations */

	rlist = (relation_t *)xmalloc(num_unique_relidx * sizeof(relation_t));

	/* ---- PARALLEL RELATION PARSE (2026-08-21) -------------------------------
	   The loop this replaces did two things per line, both serially: pull the
	   line (one gzgets, now blk_getline above), and -- for the ~16.3 M lines
	   the cycles actually need -- run nfs_read_relation on it. The parse is by
	   far the larger half and it is PURE per line: nfs_read_relation reads the
	   factor base read-only and writes only into caller-provided buffers, with
	   its mpz_t scratch passed in. relation.c holds no statics (the same
	   property build_matrix_core relies on in gf2.c).

	   So the reader stays serial -- it must, the line INDEX i defines which
	   relations are wanted -- and only the parse fans out. Lines are collected
	   into a fixed-stride chunk buffer and parsed by OpenMP once the chunk is
	   full. Slot k of the chunk always lands in rlist[jbase + k], so rlist ends
	   up in EXACTLY the order the serial loop produced it, independent of the
	   thread count. That is what makes the result invariant, and it is checked
	   the same way -nc2's other parallel stage is: an md5 of m.dat.mat.

	   Chunk memory is GCR_CHUNK x LINE_BUF_SIZE = 19.6 MB, sized off
	   LINE_BUF_SIZE so a line can never overflow its slot; against a measured
	   34.4 GB peak on an 85 GB budget that is noise.

	   Threads come from obj->num_threads (msieve's own -t), so -t 1 walks this
	   code with one thread and one chunk, and a 1-CPU host is unaffected. */
	{
	uint32 nthreads = obj->num_threads ? obj->num_threads : 1;
	uint32 nbuf = 0, t;
	int32 bad_line = -1;
	blkreader_t br;
	char *chunk;
	uint32 *cidx;
	uint8 **tfactors;
	mpz_t *tscratch;
	factor_base_t *tfb;

	if (nthreads > 64)
		nthreads = 64;

#define GCR_CHUNK 65536

	chunk = (char *)xmalloc((size_t)GCR_CHUNK * LINE_BUF_SIZE);
	cidx = (uint32 *)xmalloc(GCR_CHUNK * sizeof(uint32));
	tfactors = (uint8 **)xmalloc(nthreads * sizeof(uint8 *));
	tscratch = (mpz_t *)xmalloc(nthreads * sizeof(mpz_t));
	/* ---- PER-THREAD FACTOR BASE ---------------------------------------
	   nfs_read_relation() is NOT reentrant against a shared factor_base_t.
	   It reads the coefficients, which is fine, but it also uses
	   rpoly->tmp1/tmp2/tmp3 and apoly->tmp1/tmp2/tmp3 -- the "scratch
	   quantities for evaluating the homogeneous form of poly" declared in
	   mpz_poly_t -- as WORKING mpz_t's, for eval_poly() and for every
	   divide_factor_out() call. Two threads sharing those scratch values
	   realloc the same GMP limb array concurrently, which corrupts the heap:
	   measured as `realloc(): invalid next size` a few seconds into the
	   parse, on the very first attempt at this change.

	   The fix is a shallow struct copy per thread plus private scratch. The
	   copy shares coeff[] with the caller's factor base, which is correct
	   precisely because those are read-only on this path -- only tmp1/2/3
	   are written. Nothing else in factor_base_t is touched by
	   nfs_read_relation: it takes &fb->rfb.poly and &fb->afb.poly and
	   nothing more. Only the mpz_t's initialised here are cleared below, so
	   the caller's factor base is never freed through a copy. */
	tfb = (factor_base_t *)xmalloc(nthreads * sizeof(factor_base_t));
	for (t = 0; t < nthreads; t++) {
		tfactors[t] = (uint8 *)xmalloc(COMPRESSED_P_MAX_SIZE *
						sizeof(uint8));
		mpz_init(tscratch[t]);
		tfb[t] = *fb;
		mpz_init(tfb[t].rfb.poly.tmp1);
		mpz_init(tfb[t].rfb.poly.tmp2);
		mpz_init(tfb[t].rfb.poly.tmp3);
		mpz_init(tfb[t].afb.poly.tmp1);
		mpz_init(tfb[t].afb.poly.tmp2);
		mpz_init(tfb[t].afb.poly.tmp3);
	}

/* parse the nbuf buffered lines into rlist[j ... j+nbuf-1] and advance j */
#define GCR_FLUSH()							\
	do {								\
		int32 _k;						\
		_Pragma("omp parallel for schedule(static) num_threads(nthreads)") \
		for (_k = 0; _k < (int32)nbuf; _k++) {			\
			uint32 _tid = (uint32)omp_get_thread_num();	\
			relation_t _tmp;				\
			uint32 _fsize;					\
			relation_t *_r = rlist + j + _k;		\
			_tmp.factors = tfactors[_tid];			\
			if (nfs_read_relation(chunk + (size_t)_k *	\
					LINE_BUF_SIZE, tfb + _tid, &_tmp,\
						&_fsize, compress,	\
						tscratch[_tid], 0)) {	\
				/* the filtering stage should already have	\
				   dropped an unreadable relation */	\
				_Pragma("omp critical (gcr_bad)")	\
				if (bad_line < 0)			\
					bad_line = (int32)cidx[_k];	\
			}						\
			else {						\
				*_r = _tmp;				\
				_r->rel_index = cidx[_k];		\
				_r->factors = (uint8 *)xmalloc(_fsize *	\
							sizeof(uint8));	\
				memcpy(_r->factors, _tmp.factors,	\
						_fsize * sizeof(uint8)); \
			}						\
		}							\
		if (bad_line >= 0) {					\
			logprintf(obj, "error: relation %u corrupt\n",	\
					(uint32)bad_line);		\
			exit(-1);					\
		}							\
		j += nbuf;						\
		nbuf = 0;						\
	} while (0)

	i = (uint32)(-1);
	j = 0;
	blk_init(&br, savefile->fp);
	blk_getline(&br, buf, sizeof(buf));
	while (buf[0] != 0 && j + nbuf < num_unique_relidx) {

		if (buf[0] != '-' && !isdigit(buf[0])) {

			/* no relation at this line */

			blk_getline(&br, buf, sizeof(buf));
			continue;
		}
		if (++i < relidx_list[j + nbuf]) {

			/* relation not needed */

			blk_getline(&br, buf, sizeof(buf));
			continue;
		}

		/* wanted: stash the raw line and parse it with the chunk */

		memcpy(chunk + (size_t)nbuf * LINE_BUF_SIZE, buf,
				LINE_BUF_SIZE);
		cidx[nbuf] = i;
		if (++nbuf == GCR_CHUNK)
			GCR_FLUSH();

		blk_getline(&br, buf, sizeof(buf));
	}
	if (nbuf > 0)
		GCR_FLUSH();

#undef GCR_FLUSH
#undef GCR_CHUNK

	blk_free(&br);
	for (t = 0; t < nthreads; t++) {
		mpz_clear(tfb[t].rfb.poly.tmp1);
		mpz_clear(tfb[t].rfb.poly.tmp2);
		mpz_clear(tfb[t].rfb.poly.tmp3);
		mpz_clear(tfb[t].afb.poly.tmp1);
		mpz_clear(tfb[t].afb.poly.tmp2);
		mpz_clear(tfb[t].afb.poly.tmp3);
		mpz_clear(tscratch[t]);
		free(tfactors[t]);
	}
	free(tfb);
	free(tscratch);
	free(tfactors);
	free(cidx);
	free(chunk);
	}

	num_unique_relidx = *num_relations_out = j;
	logprintf(obj, "read %u relations\n", j);
	savefile_close(savefile);
	hashtable_free(&unique_relidx);
	*rlist_out = rlist;
}

/*--------------------------------------------------------------------*/
void nfs_read_cycles(msieve_obj *obj, 
			factor_base_t *fb,
			uint32 *num_cycles_out, 
			la_col_t **cycle_list_out, 
			uint32 *num_relations_out,
			relation_t **rlist_out,
			uint32 compress,
			uint32 dependency) {

	uint32 num_cycles;
	uint32 num_relations;
	la_col_t *cycle_list = NULL;
	relation_t *rlist;

	/* read the raw list of relation numbers for each cycle */

	read_cycles(obj, &num_cycles, &cycle_list, dependency, NULL);

	if (num_cycles == 0) {
		free(cycle_list);
		if (num_cycles_out != NULL)
			*num_cycles_out = 0;

		if (cycle_list_out != NULL)
			*cycle_list_out = NULL;

		if (num_relations_out != NULL)
			*num_relations_out = 0;

		if (rlist_out != NULL)
			*rlist_out = NULL;
		return;
	}

	/* finish if caller doesn't want the relations as well */

	if (fb == NULL || num_relations_out == NULL || rlist_out == NULL) {
		*num_cycles_out = num_cycles;
		*cycle_list_out = cycle_list;
		return;
	}

	/* now read the list of relations needed by the
	   list of cycles */

	nfs_get_cycle_relations(obj, fb, num_cycles, cycle_list, 
				&num_relations, &rlist, compress,
				dependency);

	*num_relations_out = num_relations;
	*rlist_out = rlist;

	/* if both the cycles and relations are wanted by 
	   callers, then modify the cycles to point to the
	   relations in memory and not on disk */

	if (num_cycles_out != NULL && cycle_list_out != NULL) {

		remap_relation_numbers(obj, num_cycles, cycle_list,
					num_relations, rlist);

		*num_cycles_out = num_cycles;
		*cycle_list_out = cycle_list;
	}
	else {
		free_cycle_list(cycle_list, num_cycles);
	}
}

/*--------------------------------------------------------------------*/
void nfs_free_relation_list(relation_t *rlist, uint32 num_relations) {

	uint32 i;

	for (i = 0; i < num_relations; i++)
		free(rlist[i].factors);
	free(rlist);
}

/*--------------------------------------------------------------------*/
typedef struct {
	uint32 purge_idx;
	uint32 rel_idx;
} relconvert_t;

static int bsearch_relconvert(const void *key, const void *t) {
	relconvert_t *c = (relconvert_t *)t;
	uint32 *k = (uint32 *)key;

	if ((*k) < c->purge_idx)
		return -1;
	if ((*k) > c->purge_idx)
		return 1;
	return 0;
}

void nfs_convert_cado_cycles(msieve_obj *obj) {

	uint32 i, j;

	char buf[LINE_BUF_SIZE];
	char purgefile[LINE_BUF_SIZE];
	savefile_t s;
	savefile_t *savefile = &s;

	uint32 num_cycles;
	la_col_t *cycle_list = NULL;
	uint32 num_unique_relidx;
	relconvert_t *convert;

	/* read the raw list of relation numbers for each cycle */

	read_cycles(obj, &num_cycles, &cycle_list, 0, NULL);

	/* for CADO filtering results, the cycle file contains
	   line numbers in a purge file, not the line numbers of
	   relations like we want. This is arguably better, since
	   the purge file results have already assigned unique
	   numbers to ideals so we could use the purge file to
	   directly build the matrix. Unfortunately, if we did that
	   then both the linear algebra and the square root would
	   need modifications to understand the purge file format.
	   The alternative is to use the purge file lines to convert 
	   the purge file line numbers to relation numbers */

	sprintf(purgefile, "%s.purged", obj->savefile.name);
	savefile_init(savefile, purgefile);
	savefile_open(savefile, SAVEFILE_READ);
	savefile_read_line(buf, sizeof(buf), savefile);

	num_unique_relidx = strtoul(buf, NULL, 10);

	logprintf(obj, "cycles contain %u unique purge entries\n", 
				num_unique_relidx);

	convert = (relconvert_t *)xmalloc(num_unique_relidx *
					sizeof(relconvert_t));

	for (i = 0; i < num_unique_relidx; i++) {

		relconvert_t *curr = convert + i;

		savefile_read_line(buf, sizeof(buf), savefile);

		/* the relation number is the first entry in the
		   line from the purge file */

		curr->rel_idx = strtoul(buf, NULL, 10);
		curr->purge_idx = i;
	}

	savefile_close(savefile);
	savefile_free(savefile);

	/* walk through the list of cycles and convert
	   each purge file number to a relation file number */

	for (i = 0; i < num_cycles; i++) {
		la_col_t *c = cycle_list + i;

		for (j = 0; j < c->cycle.num_relations; j++) {

			relconvert_t *t = (relconvert_t *)bsearch(
						c->cycle.list + j,
						convert,
						(size_t)num_unique_relidx,
						sizeof(relconvert_t),
						bsearch_relconvert);
			if (t == NULL) {
				/* this cycle is corrupt somehow */
				logprintf(obj, "error: cannot locate "
						"relation %u\n", 
						c->cycle.list[j]);
				exit(-1);
			}
			else {
				c->cycle.list[j] = t->rel_idx;
			}
		}
	}

	dump_cycles(obj, cycle_list, num_cycles);
	free_cycle_list(cycle_list, num_cycles);
	free(convert);
}

