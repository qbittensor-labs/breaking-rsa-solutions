#define _GNU_SOURCE
/* cofac_cpu: split cuda-sieve `--candidates` records on the CPU and emit relations.
 *
 * Input  (one record per line, as bench --pipeline --candidates writes it):
 *     [-]cof0,[-]cof1:a,b:f0hex,...:f1hex,...
 *   cof_s decimal; '-' = that side's cofactor has more than lpb_s bits and must be split.
 *   f_s = trial-division primes (hex), the special-q included on its side.
 * Output (GGNFS/CADO relation lines, exactly the bench --relations format):
 *     a,b:f0hex,...[,p,...]:f1hex,...[,p,...]
 *
 * 2LP ONLY: a split cofactor must be p*q with p,q <= 2^lpb (every prime below lim was removed by TD,
 * and 3 primes > lim do not fit in mfb <= 64 bits).  Method: primality test (deterministic
 * Miller-Rabin, 64-bit), perfect-square check, Pollard-Brent rho in 64-bit Montgomery arithmetic
 * with batched gcds.  Side 0 is tried first; a dead side 0 skips side 1 (as cuda-sieve does).
 *
 * usage: cofac_cpu -lpb0 31 -lpb1 31 -mfb0 61 -mfb1 62 -t 12 IN OUT      (IN/OUT may be - )
 *        cofac_cpu ... -follow DONE IN OUT
 *   FOLLOW MODE: IN is a file another process is still appending to (bench's candidates NAME.part;
 *   the rename to NAME keeps the inode, so the open fd follows it). Complete lines are split as they
 *   arrive; a partial last line waits for its newline. Ends once DONE exists and IN has stopped
 *   growing; a partial line left then is a torn record: dropped with a warning (bench truncates the
 *   same record on resume). Waits for IN to appear (until DONE exists).
 * stderr: one summary line  "cofac_cpu: rec R rel N dead0 D0 dead1 D1 stuck S bad B  T s"
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#define _GNU_SOURCE_MEMRCHR

typedef unsigned __int128 u128;
static int LPB[2] = {31, 31}, MFB[2] = {61, 62};
static unsigned long FBB[2] = {100000000UL, 150000000UL};   /* factor-base bounds (primality by size) */
static int USE_FACUL = 1;
#ifdef WITH_FACUL
int facul_split2(uint64_t n, unsigned long fbb, int lpb, int side, uint64_t out[2]);
#endif

/* ---------------- 64-bit Montgomery arithmetic (n odd, n < 2^63) ---------------- */
typedef struct { uint64_t n, ninv, r2, one; } mont_t;
static inline uint64_t mredc(u128 t, const mont_t *M) {
    uint64_t m = (uint64_t)t * M->ninv;                 /* ninv = -n^-1 mod 2^64 */
    u128 u = (t + (u128)m * M->n) >> 64;
    uint64_t r = (uint64_t)u;
    return r >= M->n ? r - M->n : r;
}
static inline uint64_t mmul(uint64_t a, uint64_t b, const mont_t *M) { return mredc((u128)a * b, M); }
static void mont_init(mont_t *M, uint64_t n) {
    uint64_t x = n;                                     /* Newton: x = n^-1 mod 2^64 */
    for (int i = 0; i < 6; i++) x *= 2 - n * x;
    M->n = n; M->ninv = -x;
    uint64_t r = (uint64_t)(((u128)1 << 64) % n);       /* R mod n */
    M->one = r;
    M->r2 = (uint64_t)(((u128)r * r) % n);
}
static inline uint64_t to_m(uint64_t a, const mont_t *M) { return mmul(a % M->n, M->r2, M); }

static int sprp(uint64_t n, uint64_t a, const mont_t *M) {
    uint64_t d = n - 1; int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    uint64_t x = M->one, b = to_m(a, M);
    if (b == 0) return 1;
    for (uint64_t e = d; e; e >>= 1) { if (e & 1) x = mmul(x, b, M); b = mmul(b, b, M); }
    uint64_t mone = M->n - M->one;
    if (x == M->one || x == mone) return 1;
    for (int i = 1; i < s; i++) { x = mmul(x, x, M); if (x == mone) return 1; if (x == M->one) return 0; }
    return 0;
}
static int is_prime64(uint64_t n) {
    if (n < 2) return 0;
    static const uint32_t sp[] = {2,3,5,7,11,13,17,19,23,29,31,37};
    for (int i = 0; i < 12; i++) { if (n == sp[i]) return 1; if (n % sp[i] == 0) return 0; }
    if (n < 1369) return 1;
    mont_t M; mont_init(&M, n);
    static const uint64_t B[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};  /* deterministic < 2^64 */
    for (int i = 0; i < 7; i++) if (!sprp(n, B[i], &M)) return 0;
    return 1;
}
static uint64_t gcd64(uint64_t a, uint64_t b) {
    if (!a) return b; if (!b) return a;
    int s = __builtin_ctzll(a | b); a >>= __builtin_ctzll(a);
    do { b >>= __builtin_ctzll(b); if (a > b) { uint64_t t = a; a = b; b = t; } b -= a; } while (b);
    return a << s;
}
static uint64_t isqrt64(uint64_t n) {
    uint64_t r = (uint64_t)__builtin_sqrt((double)n);
    while (r * r > n) r--;
    while ((r + 1) * (r + 1) <= n) r++;
    return r;
}
/* Pollard-Brent rho: returns a nontrivial factor or 0 after `budget` steps. */
static uint64_t rho(uint64_t n, uint64_t c0, uint64_t budget) {
    mont_t M; mont_init(&M, n);
    const uint64_t c = to_m(c0, &M);
    uint64_t y = to_m(2 + c0, &M), x = y, ys = y, q = M.one, g = 1, steps = 0;
    const uint64_t m = 128;
    for (uint64_t r = 1; g == 1; r <<= 1) {
        x = y;
        for (uint64_t i = 0; i < r; i++) { y = mmul(y, y, &M) + c; if (y >= n) y -= n; }
        for (uint64_t k = 0; k < r && g == 1; k += m) {
            ys = y;
            uint64_t lim = (r - k < m) ? r - k : m;
            for (uint64_t i = 0; i < lim; i++) {
                y = mmul(y, y, &M) + c; if (y >= n) y -= n;
                q = mmul(q, x > y ? x - y : y - x, &M);
            }
            g = gcd64(q, n);
            steps += lim;
        }
        if (steps > budget) break;
    }
    if (g == n) {                                        /* backtrack from ys */
        do { ys = mmul(ys, ys, &M) + c; if (ys >= n) ys -= n; g = gcd64(x > ys ? x - ys : ys - x, n); } while (g == 1);
    }
    return (g > 1 && g < n) ? g : 0;
}
/* Split a > lpb-bit cofactor into two primes <= 2^lpb.  1 ok, 0 dead, -1 stuck.
 * WITH_FACUL: CADO's facul strategy first (~24 us vs ~160 us for rho at c160, identical splits on 20k
 * real cofactors); its answer is re-checked (product, bounds, primality) and a MAYBE falls to rho. */
static int split2(uint64_t n, int lpb, int side, uint64_t out[2]) {
    const uint64_t L = (uint64_t)1 << lpb;
    if (n <= L) return is_prime64(n) ? 2 : 0;                    /* a '-' side that fits lpb: one prime (lpb mismatch guard) */
    if (lpb * 2 < 64 && n > (L >> 0) * L) return 0;              /* cannot be two <= lpb-bit primes */
    if (!(n & 1) || n >> 63) return 0;                           /* TD removed 2; Montgomery needs n < 2^63 */
    if (is_prime64(n)) return 0;                                 /* a prime above 2^lpb */
#ifdef WITH_FACUL
    if (USE_FACUL) {
        uint64_t f[2];
        int fr = facul_split2(n, FBB[side], lpb, side, f);
        if (fr == 0) return 0;
        if (fr == 1) {
            if (f[0] > f[1]) { uint64_t t = f[0]; f[0] = f[1]; f[1] = t; }
            if ((u128)f[0] * f[1] == n && f[1] <= L && is_prime64(f[0]) && is_prime64(f[1])) { out[0] = f[0]; out[1] = f[1]; return 1; }
            /* inconsistent answer: fall through to rho (never trust it blindly) */
        }
    }
#endif
    uint64_t r = isqrt64(n);
    if (r * r == n) { if (r > L || !is_prime64(r)) return 0; out[0] = out[1] = r; return 1; }
    for (uint64_t c = 1; c <= 4; c++) {
        uint64_t d = rho(n, c, (uint64_t)1 << 21);
        if (!d) continue;
        uint64_t e = n / d;
        if (d > e) { uint64_t t = d; d = e; e = t; }
        if (e > L) return 0;
        if (!is_prime64(d) || !is_prime64(e)) return 0;          /* > 2 primes: not a 2LP relation */
        out[0] = d; out[1] = e; return 1;
    }
    return -1;
}

/* ---------------- record processing ---------------- */
typedef struct { const char *beg, *end; char *obuf; size_t olen, ocap;
                 uint64_t nrec, nrel, dead[2], stuck, bad; } job_t;
static void put(job_t *J, const char *s, size_t n) {
    if (J->olen + n + 1 > J->ocap) { J->ocap = (J->ocap + n + 1) * 2; J->obuf = realloc(J->obuf, J->ocap); if (!J->obuf) { perror("realloc"); exit(1); } }
    memcpy(J->obuf + J->olen, s, n); J->olen += n;
}
static int parse_cof(const char **p, uint64_t *v, int *neg) {
    const char *s = *p; *neg = 0; if (*s == '-') { *neg = 1; s++; }
    uint64_t x = 0; int nd = 0;
    while (*s >= '0' && *s <= '9') { if (x > (UINT64_MAX - 9) / 10) return -1; x = x * 10 + (uint64_t)(*s - '0'); s++; nd++; }
    if (!nd) return -1; *v = x; *p = s; return 0;
}
static void do_line(job_t *J, const char *s, const char *e) {
    J->nrec++;
    uint64_t cof[2]; int neg[2]; const char *p = s;
    if (parse_cof(&p, &cof[0], &neg[0]) || *p++ != ',' || parse_cof(&p, &cof[1], &neg[1]) || *p++ != ':') { J->bad++; return; }
    const char *ab = p, *c1 = memchr(p, ':', e - p); if (!c1) { J->bad++; return; }
    const char *f0 = c1 + 1, *c2 = memchr(f0, ':', e - f0); if (!c2) { J->bad++; return; }
    const char *f1 = c2 + 1;
    uint64_t pr[2][2]; int np[2] = {0, 0};
    for (int sd = 0; sd < 2; sd++) {
        if (!neg[sd]) {
            /* bench marks a cofactor above 2^lpb with '-': an unmarked one must fit lpb (else the job's lpb
             * and ours differ -- never emit an out-of-range "prime") */
            if (cof[sd] > ((uint64_t)1 << LPB[sd])) { J->bad++; return; }
            if (cof[sd] > 1) { pr[sd][0] = cof[sd]; np[sd] = 1; }
            continue;
        }
        if (64 - __builtin_clzll(cof[sd]) > MFB[sd]) { J->dead[sd]++; return; }
        int r = split2(cof[sd], LPB[sd], sd, pr[sd]);
        if (r == 0) { J->dead[sd]++; return; }
        if (r < 0) { J->stuck++; return; }
        if (r == 2) { pr[sd][0] = cof[sd]; np[sd] = 1; continue; }
        np[sd] = 2;
    }
    char tmp[64];
    put(J, ab, (size_t)(c1 - ab)); put(J, ":", 1);
    put(J, f0, (size_t)(c2 - f0));
    for (int i = 0; i < np[0]; i++) { int n = snprintf(tmp, sizeof tmp, "%s%llx", (c2 > f0 || i) ? "," : "", (unsigned long long)pr[0][i]); put(J, tmp, n); }
    put(J, ":", 1);
    put(J, f1, (size_t)(e - f1));
    for (int i = 0; i < np[1]; i++) { int n = snprintf(tmp, sizeof tmp, "%s%llx", (e > f1 || i) ? "," : "", (unsigned long long)pr[1][i]); put(J, tmp, n); }
    put(J, "\n", 1);
    J->nrel++;
}
static void *worker(void *arg) {
    job_t *J = arg; const char *p = J->beg;
    while (p < J->end) {
        const char *nl = memchr(p, '\n', J->end - p); const char *e = nl ? nl : J->end;
        if (e > p && e[-1] == '\r') e--;
        if (e > p && *p != '#') do_line(J, p, e);
        p = nl ? nl + 1 : J->end;
    }
    return NULL;
}
static char *slurp(const char *path, size_t *len) {
    FILE *f = strcmp(path, "-") ? fopen(path, "rb") : stdin; if (!f) { perror(path); exit(1); }
    size_t cap = 1 << 24, n = 0; char *b = malloc(cap);
    for (;;) { if (n == cap) { cap *= 2; b = realloc(b, cap); if (!b) { perror("realloc"); exit(1); } }
               size_t r = fread(b + n, 1, cap - n, f); if (!r) break; n += r; }
    if (ferror(f)) { perror(path); exit(1); }
    if (f != stdin) fclose(f); *len = n; return b;
}
static int run_batch(job_t *J, int T, const char *buf, size_t len, FILE *fo) {
    pthread_t th[256]; const char *p = buf;
    for (int t = 0; t < T; t++) {
        const char *e = (t == T - 1) ? buf + len : buf + len * (t + 1) / T;
        if (e < p) e = p;
        if (t < T - 1) { const char *nl = memchr(e, '\n', buf + len - e); e = nl ? nl + 1 : buf + len; }
        J[t].beg = p; J[t].end = e; J[t].olen = 0; p = e;
        if (pthread_create(&th[t], NULL, worker, &J[t])) { perror("pthread_create"); return -1; }
    }
    for (int t = 0; t < T; t++) {
        pthread_join(th[t], NULL);
        if (J[t].olen && fwrite(J[t].obuf, 1, J[t].olen, fo) != J[t].olen) { perror("write"); return -1; }
    }
    return fflush(fo) ? -1 : 0;
}
static int follow(const char *in, const char *done, FILE *fo, job_t *J, int T, uint64_t *torn) {
    FILE *f = NULL;
    while (!(f = fopen(in, "rb"))) {
        if (access(done, F_OK) == 0) {
            if ((f = fopen(in, "rb"))) break;
            /* the writer may already have renamed NAME.part -> NAME (a band shorter than our first poll) */
            size_t l = strlen(in); char alt[4096];
            if (l > 5 && l < sizeof alt && !strcmp(in + l - 5, ".part")) {
                memcpy(alt, in, l - 5); alt[l - 5] = 0;
                if ((f = fopen(alt, "rb"))) break;
            }
            fprintf(stderr, "cofac_cpu: %s never appeared\n", in); return 0;
        }
        usleep(200000);
    }
    size_t cap = 1 << 26, have = 0; char *b = malloc(cap);
    int idle_after_done = 0;
    for (;;) {
        if (have == cap) { cap *= 2; b = realloc(b, cap); if (!b) { perror("realloc"); return -1; } }
        size_t r = fread(b + have, 1, cap - have, f);
        if (r) { have += r; idle_after_done = 0; }
        else {
            if (ferror(f)) { perror(in); return -1; }
            clearerr(f);
            if (access(done, F_OK) == 0) { if (++idle_after_done >= 2) break; usleep(50000); continue; }
            usleep(150000);
        }
        char *nl = memrchr(b, '\n', have);
        if (!nl) continue;
        size_t full = (size_t)(nl - b) + 1;
        if (!r || full >= (1u << 20)) {                    /* batch: at EOF, or 1 MB of complete lines */
            if (run_batch(J, T, b, full, fo)) return -1;
            memmove(b, b + full, have - full); have -= full;
        }
    }
    if (have) {
        char *nl = memrchr(b, '\n', have); size_t full = nl ? (size_t)(nl - b) + 1 : 0;
        if (full && run_batch(J, T, b, full, fo)) return -1;
        if (have > full) { *torn = 1; fprintf(stderr, "cofac_cpu: WARNING dropped a torn final record (%zu bytes)\n", have - full); }
    }
    fclose(f); free(b); return 0;
}
int main(int argc, char **argv) {
    int T = 8; const char *in = NULL, *out = NULL, *done = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-lpb0") && i + 1 < argc) LPB[0] = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-lpb1") && i + 1 < argc) LPB[1] = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-mfb0") && i + 1 < argc) MFB[0] = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-mfb1") && i + 1 < argc) MFB[1] = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) T = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-fbb0") && i + 1 < argc) FBB[0] = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-fbb1") && i + 1 < argc) FBB[1] = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-rho")) USE_FACUL = 0;
        else if (!strcmp(argv[i], "-follow") && i + 1 < argc) done = argv[++i];
        else if (!in) in = argv[i]; else if (!out) out = argv[i];
        else { fprintf(stderr, "usage: cofac_cpu [-lpb0 L] [-lpb1 L] [-mfb0 M] [-mfb1 M] [-t T] IN OUT\n"); return 2; }
    }
    if (!in || !out || T < 1 || T > 256 || LPB[0] > 32 || LPB[1] > 32 || MFB[0] > 64 || MFB[1] > 64) {
        fprintf(stderr, "usage: cofac_cpu [-lpb0 L<=32] [-lpb1 L<=32] [-mfb0 M<=64] [-mfb1 M<=64] [-t T] IN OUT\n"); return 2; }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    if (done) {
        FILE *fo = strcmp(out, "-") ? fopen(out, "wb") : stdout; if (!fo) { perror(out); return 1; }
        job_t *J = calloc(T, sizeof *J); uint64_t torn = 0;
        if (follow(in, done, fo, J, T, &torn)) return 1;
        if (fo != stdout && fclose(fo)) { perror(out); return 1; }
        uint64_t R = 0, N = 0, D0 = 0, D1 = 0, S = 0, B = 0;
        for (int t = 0; t < T; t++) { R += J[t].nrec; N += J[t].nrel; D0 += J[t].dead[0]; D1 += J[t].dead[1]; S += J[t].stuck; B += J[t].bad; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "cofac_cpu: rec %llu rel %llu dead0 %llu dead1 %llu stuck %llu bad %llu torn %llu  %.2f s (%d thr, follow)\n",
                (unsigned long long)R, (unsigned long long)N, (unsigned long long)D0, (unsigned long long)D1,
                (unsigned long long)S, (unsigned long long)B, (unsigned long long)torn,
                (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec), T);
        if (B) fprintf(stderr, "cofac_cpu: WARNING %llu unparsable/out-of-range record(s) skipped\n", (unsigned long long)B);
        return (B * 1000 > R) ? 3 : 0;
    }
    size_t len; char *buf = slurp(in, &len);
    if (len && buf[len - 1] != '\n') { fprintf(stderr, "cofac_cpu: FATAL %s does not end with a newline (torn input)\n", in); return 1; }
    job_t *J = calloc(T, sizeof *J); pthread_t *th = calloc(T, sizeof *th);
    const char *p = buf;
    for (int t = 0; t < T; t++) {
        const char *e = (t == T - 1) ? buf + len : buf + len * (t + 1) / T;
        if (e < p) e = p;
        if (t < T - 1) { const char *nl = memchr(e, '\n', buf + len - e); e = nl ? nl + 1 : buf + len; }
        J[t].beg = p; J[t].end = e; p = e;
        if (pthread_create(&th[t], NULL, worker, &J[t])) { perror("pthread_create"); return 1; }
    }
    FILE *fo = strcmp(out, "-") ? fopen(out, "wb") : stdout; if (!fo) { perror(out); return 1; }
    uint64_t R = 0, N = 0, D0 = 0, D1 = 0, S = 0, B = 0;
    for (int t = 0; t < T; t++) {
        pthread_join(th[t], NULL);
        if (J[t].olen && fwrite(J[t].obuf, 1, J[t].olen, fo) != J[t].olen) { perror("write"); return 1; }
        free(J[t].obuf);
        R += J[t].nrec; N += J[t].nrel; D0 += J[t].dead[0]; D1 += J[t].dead[1]; S += J[t].stuck; B += J[t].bad;
    }
    if (fflush(fo) || (fo != stdout && fclose(fo))) { perror(out); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "cofac_cpu: rec %llu rel %llu dead0 %llu dead1 %llu stuck %llu bad %llu  %.2f s (%d thr)\n",
            (unsigned long long)R, (unsigned long long)N, (unsigned long long)D0, (unsigned long long)D1,
            (unsigned long long)S, (unsigned long long)B,
            (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec), T);
    if (B) fprintf(stderr, "cofac_cpu: WARNING %llu unparsable/out-of-range record(s) skipped\n", (unsigned long long)B);
    return (B * 1000 > R) ? 3 : 0;
}
