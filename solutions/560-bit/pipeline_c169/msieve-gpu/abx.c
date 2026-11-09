// Copyright (C) 2026 qBitTensor Labs.
// Original author: an anonymous competition participant (Enigma / Breaking RSA competition).
// IP in custom components assigned to qBitTensor Labs under the Enigma rules.
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
// details. You should have received a copy of the license with this program;
// if not, see <https://www.gnu.org/licenses/>.

/* abx.c -- the (a,b) extraction of CADO `sqrt -ab`, split so that its pass over purged.gz runs DURING the LA.
 *
 * CADO `sqrt -ab -purged P -index I -ker K -prefix X` (sqrt.cpp, task_prepare_dependencies) does, after
 * `characters`: read the index (merged row -> purged relation numbers), XOR each row's kernel word into
 * abs[relation], then stream the whole purged file and write "a b\n" (decimal, a signed) of every relation
 * whose abs word has bit j set into X.<j:03d>. Everything except the XOR and the writes depends only on
 * purged + index, which exist as soon as replay has run -- ~3,000 s before characters ends. So:
 *
 *   abx prep <purged[.gz]> <index[.gz]> <out.abx>    (ds_cado.sh SQRT_ABX=1: in the background during the LA)
 *       parses purged (a,b only, hex, '#' and empty lines skipped -- the numbering of filter_rels() and of the
 *       parallel characters) and the index ("%lu" rows, then per row "%d" and that many "%x"), checks the
 *       relation count against the purged header "# nrows ncols", writes one binary file.
 *   abx deps <in.abx> <ker> <prefix> [maxdep]         (after characters + the MAXDEP mixing)
 *       abs[] from the kernel words (first uint64 per row, like CADO with ker_stride), then one file per
 *       non-zero bit j in increasing order, named exactly like get_depname(): <prefix>.<k:03d>, or with a prefix
 *       ending in .gz <base>.<k:03d>.gz written through `gzip -1` (same content as sqrt -ab's gzip -6 files; for a
 *       small /tmp: 10 plain c169 dependencies are ~5.2 GB), lines "%lld %llu\n" in purged order. Prints
 *       "<file> : <n> (a,b) pairs" like CADO. Output is byte-identical to CADO's with the same plain prefix
 *       (checked by gpu/s23_cp.sh step abx: md5 of every file).
 *
 * Binary format (little endian): "ABX1" u32 0, u64 nrel, u64 nrows_idx, u64 ncols_idx,
 *   int64 a[nrel], uint64 b[nrel], uint64 off[nrows_idx+1], uint32 col[ncols_idx].
 * Build: gcc -O2 -pthread -o abx abx.c   (no other dependency; .gz inputs are read through `gzip -dc`)
 * Exit: 0 ok, 1 usage/IO/format error (named on stderr) -- ds_cado.sh then falls back to `sqrt -ab`.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <pthread.h>
#include <sys/stat.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
#define DIE(...) do { fprintf(stderr, "abx: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)

static int ends_with(const char *s, const char *t) { size_t a = strlen(s), b = strlen(t); return a >= b && !strcmp(s + a - b, t); }
static int is_compressed(const char *s) { return ends_with(s, ".gz") || ends_with(s, ".bz2") || ends_with(s, ".lzma") || ends_with(s, ".xz"); }

/* ---- buffered reader over a file or a `gzip -dc` pipe ---- */
typedef struct { FILE *f; int is_pipe; char *buf; size_t len, pos, cap; int eof; const char *name; } rd_t;
static void rd_open(rd_t *r, const char *name) {
  memset(r, 0, sizeof *r); r->name = name; r->cap = 64u << 20; r->buf = malloc(r->cap); if (!r->buf) DIE("out of memory");
  if (ends_with(name, ".gz")) {
    char *cmd; if (asprintf(&cmd, "gzip -dc '%s'", name) < 0) DIE("asprintf");
    r->f = popen(cmd, "r"); r->is_pipe = 1; free(cmd);
  } else if (is_compressed(name)) DIE("%s: only plain or .gz input", name);
  else r->f = fopen(name, "rb");
  if (!r->f) DIE("cannot open %s: %s", name, strerror(errno));
}
static int rd_fill(rd_t *r) {   /* keeps the unread tail; returns bytes available */
  if (r->pos > 0) { memmove(r->buf, r->buf + r->pos, r->len - r->pos); r->len -= r->pos; r->pos = 0; }
  while (!r->eof && r->len < r->cap) {
    size_t g = fread(r->buf + r->len, 1, r->cap - r->len, r->f);
    if (g == 0) { if (ferror(r->f)) DIE("read error on %s", r->name); r->eof = 1; break; }
    r->len += g;
    if (r->len >= r->cap / 2) break;
  }
  return (int) (r->len - r->pos > 0);
}
static void rd_close(rd_t *r) {
  if (r->is_pipe) { int rc = pclose(r->f); if (rc != 0) DIE("gzip -dc %s failed (status %d): truncated/corrupt input?", r->name, rc); }
  else fclose(r->f);
  free(r->buf);
}
/* next line [*s, *e) without the newline; 0 at EOF. A last line without '\n' is an error (truncated input). */
static int rd_line(rd_t *r, char **s, char **e) {
  for (;;) {
    char *p = r->buf + r->pos, *end = r->buf + r->len;
    char *nl = memchr(p, '\n', (size_t) (end - p));
    if (nl) { *s = p; *e = nl; r->pos = (size_t) (nl + 1 - r->buf); return 1; }
    if (r->eof) { if (p != end) DIE("%s does not end with a newline (truncated input?)", r->name); return 0; }
    if (r->pos == 0 && r->len == r->cap) { r->cap *= 2; r->buf = realloc(r->buf, r->cap); if (!r->buf) DIE("out of memory"); }
    rd_fill(r);
  }
}

static const signed char *hexv(void) {
  static signed char t[256]; static int init = 0;
  if (!init) { memset(t, -1, sizeof t); for (int c = 0; c < 10; c++) t['0' + c] = (signed char) c;
               for (int c = 0; c < 6; c++) { t['a' + c] = (signed char) (10 + c); t['A' + c] = (signed char) (10 + c); } init = 1; }
  return t;
}

static void put_all(FILE *f, const void *p, size_t sz, size_t n, const char *what) { if (fwrite(p, sz, n, f) != n) DIE("write error (%s): %s", what, strerror(errno)); }
static void get_all(FILE *f, void *p, size_t sz, size_t n, const char *what) { if (fread(p, sz, n, f) != n) DIE("short read (%s)", what); }

/* ---- prep ---- */
static int cmd_prep(const char *purged, const char *index, const char *out) {
  double t0 = now(); const signed char *H = hexv();
  /* purged */
  rd_t r; rd_open(&r, purged); rd_fill(&r);
  uint64_t hdr_n = 0, hdr_c = 0; int have_hdr = 0;
  size_t cap = 1u << 24, n = 0; int64_t *A = malloc(cap * 8); uint64_t *B = malloc(cap * 8);
  if (!A || !B) DIE("out of memory");
  char *s, *e;
  while (rd_line(&r, &s, &e)) {
    if (s == e) continue;                                  /* empty line: not a relation */
    if (*s == '#') {
      if (!have_hdr) { char tmp[256]; size_t L = (size_t) (e - s) < 255 ? (size_t) (e - s) : 255; memcpy(tmp, s, L); tmp[L] = 0;
        if (sscanf(tmp, "# %" SCNu64 " %" SCNu64, &hdr_n, &hdr_c) == 2) have_hdr = 1; }
      continue;
    }
    if (n == cap) { cap *= 2; A = realloc(A, cap * 8); B = realloc(B, cap * 8); if (!A || !B) DIE("out of memory"); }
    const char *p = s; int neg = 0; uint64_t w = 0; int nd = 0;
    if (*p == '-') { neg = 1; p++; }
    for (; p < e && H[(unsigned char) *p] >= 0; p++, nd++) w = (w << 4) | (uint64_t) H[(unsigned char) *p];
    if (nd == 0 || nd > 16 || p >= e || *p != ',') DIE("%s: relation %zu: bad a field", purged, n);
    if (w > (uint64_t) INT64_MAX) DIE("%s: relation %zu: |a| >= 2^63 (needs -large-ab)", purged, n);
    A[n] = neg ? -(int64_t) w : (int64_t) w;
    p++; w = 0; nd = 0;
    for (; p < e && H[(unsigned char) *p] >= 0; p++, nd++) w = (w << 4) | (uint64_t) H[(unsigned char) *p];
    if (nd == 0 || nd > 16 || (p < e && *p != ':' && *p != '@')) DIE("%s: relation %zu: bad b field", purged, n);
    B[n] = w; n++;
  }
  rd_close(&r);
  if (!have_hdr) DIE("%s: no '# nrows ncols' header line", purged);
  if (hdr_n != n) DIE("%s announces %" PRIu64 " relations but contains %zu (truncated input?)", purged, hdr_n, n);
  double t1 = now();
  /* index */
  rd_open(&r, index); rd_fill(&r);
  uint64_t nr = 0; int got = 0;
  /* tokenizer over lines: first token = row count, then per row: decimal count, that many hex columns */
  uint64_t *off = NULL; uint32_t *col = NULL; size_t ccap = 1u << 24, nc = 0, row = 0; uint64_t want = 0; int inrow = 0;
  col = malloc(ccap * 4); if (!col) DIE("out of memory");
  while (rd_line(&r, &s, &e)) {
    const char *p = s;
    for (;;) {
      while (p < e && (*p == ' ' || *p == '\t' || *p == '\r')) p++;
      if (p >= e) break;
      const char *q = p; uint64_t v = 0;
      if (!got) {                                        /* row count, decimal */
        for (; q < e && *q >= '0' && *q <= '9'; q++) v = v * 10 + (uint64_t) (*q - '0');
        if (q == p) DIE("%s: bad row count", index);
        nr = v; got = 1; off = malloc((nr + 1) * 8); if (!off) DIE("out of memory"); off[0] = 0;
      } else if (!inrow) {                               /* row weight, decimal */
        if (row >= nr) DIE("%s: more rows than its header %" PRIu64, index, nr);
        for (; q < e && *q >= '0' && *q <= '9'; q++) v = v * 10 + (uint64_t) (*q - '0');
        if (q == p) DIE("%s: row %zu: bad weight", index, row);
        want = v; inrow = 1;
        if (want == 0) { off[++row] = nc; inrow = 0; }
      } else {                                           /* column, hex */
        for (; q < e && H[(unsigned char) *q] >= 0; q++) v = (v << 4) | (uint64_t) H[(unsigned char) *q];
        if (q == p) DIE("%s: row %zu: bad column", index, row);
        if (v >= n) DIE("%s: row %zu: column %" PRIu64 " >= %zu purged relations", index, row, v, n);
        if (nc == ccap) { ccap *= 2; col = realloc(col, ccap * 4); if (!col) DIE("out of memory"); }
        col[nc++] = (uint32_t) v;
        if (--want == 0) { off[++row] = nc; inrow = 0; }
      }
      p = q;
    }
  }
  rd_close(&r);
  if (!got) DIE("%s: empty index", index);
  if (row != nr || inrow) DIE("%s: %zu complete rows, header says %" PRIu64 " (truncated input?)", index, row, nr);
  double t2 = now();
  /* write */
  char *tmp; if (asprintf(&tmp, "%s.tmp", out) < 0) DIE("asprintf");
  FILE *f = fopen(tmp, "wb"); if (!f) DIE("cannot create %s: %s", tmp, strerror(errno));
  uint32_t magic[2] = { 0x31584241u /* "ABX1" */, 0 }; uint64_t h3[3] = { n, nr, nc };
  put_all(f, magic, 4, 2, "magic"); put_all(f, h3, 8, 3, "header");
  put_all(f, A, 8, n, "a"); put_all(f, B, 8, n, "b"); put_all(f, off, 8, nr + 1, "off"); put_all(f, col, 4, nc, "col");
  if (fclose(f) != 0) DIE("close %s: %s", tmp, strerror(errno));
  if (rename(tmp, out) != 0) DIE("rename %s: %s", tmp, strerror(errno));
  printf("abx prep: %zu purged relations (%.1f s), index %" PRIu64 " rows / %zu entries (%.1f s), %s %.0f MB in %.1f s total\n",
         n, t1 - t0, nr, nc, t2 - t1, out, (24.0 + 16.0 * n + 8.0 * (nr + 1) + 4.0 * nc) / 1e6, now() - t0);
  return 0;
}

/* ---- deps ---- */
typedef struct { int j; uint64_t mask; char *name; size_t count; const int64_t *A; const uint64_t *B; const uint64_t *abs; size_t n; int err; int gz; } job_t;
static char *u64s(char *p, uint64_t v) { char t[24]; int k = 0; do { t[k++] = (char) ('0' + v % 10); v /= 10; } while (v); while (k) *p++ = t[--k]; return p; }
static void *writer(void *arg) {
  job_t *J = arg; FILE *f;
  if (J->gz) { char *cmd; if (asprintf(&cmd, "gzip -1 -c > '%s'", J->name) < 0) { J->err = ENOMEM; return NULL; } f = popen(cmd, "w"); free(cmd); }
  else f = fopen(J->name, "wb");
  if (!f) { J->err = errno ? errno : EIO; return NULL; }
  size_t bcap = 8u << 20; char *buf = malloc(bcap + 64), *p = buf; if (!buf) { J->err = ENOMEM; fclose(f); return NULL; }
  for (size_t i = 0; i < J->n; i++) {
    if (!(J->abs[i] & J->mask)) continue;
    int64_t a = J->A[i];
    if (a < 0) { *p++ = '-'; p = u64s(p, (uint64_t) 0 - (uint64_t) a); } else p = u64s(p, (uint64_t) a);
    *p++ = ' '; p = u64s(p, J->B[i]); *p++ = '\n'; J->count++;
    if ((size_t) (p - buf) >= bcap) { if (fwrite(buf, 1, (size_t) (p - buf), f) != (size_t) (p - buf)) { J->err = errno ? errno : EIO; break; } p = buf; }
  }
  if (!J->err && p > buf && fwrite(buf, 1, (size_t) (p - buf), f) != (size_t) (p - buf)) J->err = errno ? errno : EIO;
  if (J->gz) { int st = pclose(f); if (st != 0 && !J->err) J->err = EIO; }
  else if (fclose(f) != 0 && !J->err) J->err = errno ? errno : EIO;
  free(buf); return NULL;
}
static int cmd_deps(const char *in, const char *ker, const char *prefix, int maxdep) {
  double t0 = now();
  int gz = ends_with(prefix, ".gz");
  if (!gz && is_compressed(prefix)) DIE("prefix %s: abx writes plain or .gz dependency files only", prefix);
  FILE *f = fopen(in, "rb"); if (!f) DIE("cannot open %s: %s", in, strerror(errno));
  uint32_t magic[2]; uint64_t h3[3]; get_all(f, magic, 4, 2, "magic"); get_all(f, h3, 8, 3, "header");
  if (magic[0] != 0x31584241u) DIE("%s is not an abx file", in);
  size_t n = h3[0], nr = h3[1], nc = h3[2];
  int64_t *A = malloc(n * 8); uint64_t *B = malloc(n * 8), *off = malloc((nr + 1) * 8); uint32_t *col = malloc(nc * 4 + 4);
  if (!A || !B || !off || !col) DIE("out of memory");
  get_all(f, A, 8, n, "a"); get_all(f, B, 8, n, "b"); get_all(f, off, 8, nr + 1, "off"); get_all(f, col, 4, nc, "col"); fclose(f);
  if (off[nr] != nc) DIE("%s: inconsistent index (off[nr] %" PRIu64 " != %zu)", in, off[nr], nc);
  struct stat st; if (stat(ker, &st) != 0) DIE("stat %s: %s", ker, strerror(errno));
  if (nr == 0 || st.st_size % (off_t) nr != 0 || st.st_size / (off_t) nr < 8 || (st.st_size / (off_t) nr) % 8 != 0)
    DIE("%s: size %lld is not a multiple of 8 x %zu index rows", ker, (long long) st.st_size, nr);
  size_t stride = (size_t) (st.st_size / (off_t) nr) / 8;   /* words per row; only the first is used (CADO: first 64 deps) */
  uint64_t *K = malloc(nr * stride * 8); if (!K) DIE("out of memory");
  f = fopen(ker, "rb"); if (!f) DIE("cannot open %s", ker); get_all(f, K, 8, nr * stride, "ker"); fclose(f);
  uint64_t *abs = calloc(n, 8); if (!abs) DIE("out of memory");
  for (size_t i = 0; i < nr; i++) { uint64_t v = K[i * stride]; if (!v) continue; for (uint64_t k = off[i]; k < off[i + 1]; k++) abs[col[k]] ^= v; }
  free(K);
  uint64_t sanity = 0; for (size_t i = 0; i < n; i++) sanity |= abs[i];
  job_t J[64]; int nj = 0;
  size_t pl = strlen(prefix) - (gz ? 3 : 0);
  for (int j = 0; j < 64; j++) {
    if (!(sanity >> j & 1)) continue;
    if (maxdep > 0 && nj >= maxdep) break;
    memset(&J[nj], 0, sizeof J[nj]); J[nj].j = j; J[nj].mask = UINT64_C(1) << j;
    J[nj].name = malloc(pl + 12); sprintf(J[nj].name, "%.*s.%03d%s", (int) pl, prefix, nj, gz ? ".gz" : "");
    J[nj].A = A; J[nj].B = B; J[nj].abs = abs; J[nj].n = n; J[nj].gz = gz; nj++;
  }
  if (!nj) DIE("no non-zero dependency in %s", ker);
  pthread_t th[64];
  for (int k = 0; k < nj; k++) if (pthread_create(&th[k], NULL, writer, &J[k]) != 0) DIE("pthread_create");
  int bad = 0;
  for (int k = 0; k < nj; k++) { pthread_join(th[k], NULL); if (J[k].err) { fprintf(stderr, "abx: writing %s: %s\n", J[k].name, strerror(J[k].err)); bad = 1; } }
  if (bad) return 1;
  printf("abx deps: %d non-zero dependencies from %zu relations in %.1f s\n", nj, n, now() - t0);
  for (int k = 0; k < nj; k++) printf("%s : %zu (a,b) pairs\n", J[k].name, J[k].count);
  return 0;
}

int main(int argc, char **argv) {
  if (argc >= 5 && !strcmp(argv[1], "prep")) return cmd_prep(argv[2], argv[3], argv[4]);
  if ((argc == 5 || argc == 6) && !strcmp(argv[1], "deps")) return cmd_deps(argv[2], argv[3], argv[4], argc == 6 ? atoi(argv[5]) : 0);
  fprintf(stderr, "usage: abx prep <purged[.gz]> <index[.gz]> <out.abx>\n       abx deps <in.abx> <ker> <plain prefix> [maxdep]\n");
  return 1;
}
