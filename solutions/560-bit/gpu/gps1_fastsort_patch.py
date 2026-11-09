#!/usr/bin/env python3
# Copyright (C) 2026 qBitTensor Labs.
# Original author: an anonymous competition participant (Enigma / Breaking RSA competition).
# IP in custom components assigned to qBitTensor Labs under the Enigma rules.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
# details. You should have received a copy of the license with this program;
# if not, see <https://www.gnu.org/licenses/>.

"""gps1_fastsort_patch.py <gpu-siever dir> -- review 7 (docs/rsa560_codex_review7.md 5.1 item 1; review 6 6.1 / 7.1 A1):
a default-off lossless 32-bit fingerprint collision sort for gps1, the GPU polynomial stage-1 collision search.

Patches ONLY polyselect/gps1_kernels.cu of the Cognition tree at tag `latest` (b5ca9c4f). The bootstrap
(gpu/sandbox_bootstrap.sh) applies no patch to any polyselect file, so production gps1 is built from that pristine text.
Accepted inputs (sha256-pinned, fail-closed on any drift):
  pristine b5ca9c4f gps1_kernels.cu                        (stand-alone)
  the same file after gpu/gps1_stats_patch.py              (stacked: GPS1_STAGE_STATS timers keep working)
Order when stacking: gps1_stats_patch.py FIRST, then this one (the stats patch pins the pristine text and would refuse
a fastsort-patched file). Idempotent: a file that already carries the GPS1_FASTSORT marker is left alone.

Why a fingerprint sort (measured sb82, data/rsa560/audit_r6/gps1_stats_sb82.txt): the CUB pair sort is 77% of the gps1
search. Its key is (item b << 44 | u + M): offset_bits 44 is the minimum for [0, 2M) at P = 1445000 (2M = 1.67e13 <
2^44) and the 6 batch bits are the minimum for batch ~57, so 50 bits is already minimal (option (a) buys nothing:
7 onesweep passes x 24 B per entry). Batches are ~26.6M entries (57 items x ~450K slots), over L2. The fingerprint sort
moves (hash32, entry_id32) = 8 B pairs over 4 passes: 64 B per entry instead of 168 B.

  GPS1_FASTSORT unset / 0   unchanged production path (one getenv at the first gps1_run).
  GPS1_FASTSORT=1           fingerprint sort over the whole batch (pads carry a per-entry hash, so they never form runs)
  GPS1_FASTSORT=2           as 1, but padding entries (KEY_NONE) are compacted out during fill (block-aggregated atomic
                            cursor); costs one D2H + stream sync per batch for the count, sorts ~12% fewer entries
  GPS1_FASTSORT=3           as 1, but sort + scan run per chunk of whole items (GPS1_FS_CHUNK entries, default 4Mi:
                            32 MiB of (fp, id) pairs + 32 MiB alternates, so the 4 passes can run L2-resident);
                            matches never cross items
  GPS1_FS_BITS=k            debug: keep only the low k bits of the hash (1..32, default 32) -- an exactness stress
                            test (many false equal-hash pairs, equal full keys separated inside runs); the sort then
                            uses end_bit k
  GPS1_FS_CHUNK=n           mode 3 chunk target in entries (>= 1 item per chunk)

Exactness contract (review 6 6.1): the full 64-bit keys stay immutable in G->keys (written by fill, never sorted);
CUB sorts (hash32(full_key), e) pairs, e = the entry's index, into G->keys_alt (split as two uint32 buffers) and
G->vals / G->vals_alt (no new device allocation; CUB temp storage is the production buffer unless the 32-bit query
needs more). For every equal-hash run the scan compares EVERY pair (l, j), l < j, of the run on the full key, skips
KEY_NONE by full-key validity (never by a reserved hash), and emits every equal-full-key pair with distinct primes --
the same pair set as production's adjacent-run scan. p1/p2 orientation is canonical: p1 belongs to the entry with the
smaller index e, which is production's order (CUB is stable and production's sort input is in e order). The root of
an entry is recovered from e by binary search over entry_off (only for equal-key pairs); b and u come from the full
key. Long runs are handled exactly by the same all-pairs loop production uses on equal-key runs (no truncation); the
match buffer overflow stays fatal (host assert, device writes guarded). kernel_fill_fp maps threads item-major
(adjacent threads -> adjacent roots of one item -> adjacent entry slots), the entry layout is unchanged.

At exit (any mode > 0) one line on stderr:
  GPS1_FASTSORT mode= bits= chunk= batches= sorted= chunks= fp_runs= pair_checks= key_eq_pairs= false_pairs= max_run=
fp_runs = equal-hash runs of length >= 2, pair_checks = all pairs inside them, key_eq_pairs = pairs with equal valid
full keys (incl. same-prime ones), false_pairs = pair_checks - key_eq_pairs, max_run = longest equal-hash run.
A start-up line `GPS1_FASTSORT on mode=..` is printed at the first batch.
With the stats patch stacked: fill_ms = kernel_fill_fp, sort_ms = the 32-bit sort (mode 2: incl. the count D2H + sync;
mode 3: incl. the per-chunk scans, scan_ms ~0), sort_bits reports the hash width; level 2's pad count still reads the
immutable full keys. Verified locally (no nvcc/GPU here): applies to both pinned inputs, idempotent, clang -x cuda
host+device syntax check against stub CUDA/CUB headers, and a CPU emulation of the patched gps1_run (kernels run
with real warp shuffles / block barriers, CUB replaced by a stable sort) gives the production match multiset,
orientation included, for modes 1-3 at 32/20/16/8/4/1-bit hashes on synthetic instances (runs up to 2,720 entries)."""
import hashlib, os, sys

D = sys.argv[1]
F = os.path.join(D, 'polyselect', 'gps1_kernels.cu')
PINS = {'b359f4fa2a43c549059e86b8bb246a0158eaa65f11b3f06fa7ab580c1748c8ba': 'pristine latest b5ca9c4f',
        '505804d008c9c95157c14e7924fe56b0f1d9a29d50b315443cc3783724a11e3a': 'b5ca9c4f + gps1_stats_patch.py'}
src = open(F).read()
if 'GPS1_FASTSORT' in src: print('already patched'); sys.exit(0)
h = hashlib.sha256(src.encode()).hexdigest()
assert h in PINS, 'gps1_kernels.cu sha256 %s is neither pristine b5ca9c4f nor b5ca9c4f + gps1_stats_patch' % h
base = PINS[h]

def sub(a, b, cnt=1):
    global src
    assert src.count(a) == cnt, (src.count(a), a[:80])
    src = src.replace(a, b)

# ---- 1. kernels + host helpers, inserted before gps1_run ----
sub('''extern "C" uint64_t gps1_run(const gpu_t *G, uint32_t nroots, uint32_t ndq,
''', r'''/* ---- GPS1_FASTSORT begin (gpu/gps1_fastsort_patch.py, review 7) ---- */
__device__ unsigned long long d_fs_count;      /* mode 2 compaction cursor */
#define GPS1_FS_SLOTS 256                      /* counter slots (power of 2) */
__device__ unsigned long long d_fs_ctr[GPS1_FS_SLOTS][4];  /* fp_runs, pair_checks, key_eq_pairs, max_run */

static int gps1_fs_env(const char *name, int dflt, int lo, int hi)
{
    const char *e = getenv(name);
    if (!e || !*e)
        return dflt;
    int v = atoi(e);
    if (v < lo || v > hi) {
        fprintf(stderr, "%s=%s: expected %d..%d\n", name, e, lo, hi);
        exit(1);
    }
    return v;
}

static int gps1_fastsort_mode(void)
{
    static int mode = -1;
    if (mode < 0)
        mode = gps1_fs_env("GPS1_FASTSORT", 0, 0, 3);
    return mode;
}

/* murmur3 fmix64, folded to 32 bits; equal keys -> equal hashes */
static __device__ __forceinline__ uint32_t gps1_fs_hash(uint64_t k, uint32_t mask)
{
    k ^= k >> 33; k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return ((uint32_t) k ^ (uint32_t)(k >> 32)) & mask;
}

/* As kernel_fill (same keys[e] values, KEY_NONE padding included), plus
 * fp[] = hash of the full key and id[] = e for the sort. Item-major thread
 * order: t -> (b = t / nroots, c = t % nroots). compact = 0: fp/id at e,
 * padding gets the hash of its own index (spread, never a reserved value;
 * the scan drops it by full-key validity). compact = 1: only valid entries,
 * at a block-aggregated cursor d_fs_count. */
__global__ void kernel_fill_fp(const uint32_t *primes, const uint32_t *root_prime,
                               const uint64_t *roots, uint32_t nroots,
                               const uint32_t *entry_off, const uint32_t *qidx,
                               const uint64_t *rq_hi, const uint64_t *rq_lo,
                               const uint64_t *invqq, uint32_t nprimes,
                               uint32_t batch, int64_t M, uint32_t offset_bits,
                               uint64_t slots, uint64_t *keys, uint32_t *fp,
                               uint32_t *id, uint32_t mask, int compact)
{
    uint64_t t = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    bool act = t < (uint64_t) nroots * batch;
    uint32_t nv = 0, n = 0, b = 0;
    uint64_t e = 0, pp = 0;
    int64_t u = 0;
    if (act) {
        uint32_t c = t % nroots;
        b = t / nroots;
        uint32_t pi = root_prime[c];
        pp = (uint64_t) primes[pi] * primes[pi];
        uint64_t rqi = mod128(rq_hi[b], rq_lo[b], pp);
        uint64_t rp = roots[c];
        uint64_t u0 = rp >= rqi ? rp - rqi : rp + pp - rqi;
        u0 = mulmod64(u0, invqq[(uint64_t) qidx[b] * nprimes + pi], pp);
        u = (int64_t) u0 - (int64_t)(((uint64_t)((int64_t) u0 + M)) / pp) * (int64_t) pp;
        e = (uint64_t) b * slots + entry_off[c];
        n = entry_off[c + 1] - entry_off[c];
        int64_t v = u;
        for (uint32_t s = 0; s < n; s++, v += (int64_t) pp) {
            uint64_t key = v < M ? ((uint64_t) b << offset_bits) | (uint64_t)(v + M)
                                 : KEY_NONE;
            keys[e + s] = key;
            nv += v < M;
            if (!compact) {
                fp[e + s] = key != KEY_NONE ? gps1_fs_hash(key, mask)
                                            : gps1_fs_hash(~(e + s), mask);
                id[e + s] = (uint32_t)(e + s);
            }
        }
    }
    if (!compact)
        return;
    /* block-aggregated cursor (one atomic per block): warp inclusive scans,
     * then warp 0 scans the warp totals. Every thread of the block reaches
     * the shuffles and barriers (no early return in compact mode). */
    __shared__ uint32_t wsum[32];
    __shared__ unsigned long long bbase;
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned nwarps = (blockDim.x + 31) >> 5;
    uint32_t incl = nv;
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= (unsigned) o)
            incl += y;
    }
    if (lane == 31)
        wsum[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        uint32_t w = lane < nwarps ? wsum[lane] : 0, wi = w;
        for (int o = 1; o < 32; o <<= 1) {
            uint32_t y = __shfl_up_sync(0xffffffffu, wi, o);
            if (lane >= (unsigned) o)
                wi += y;
        }
        if (lane < nwarps)
            wsum[lane] = wi - w;            /* exclusive warp offset */
        if (lane == 31)
            bbase = wi ? atomicAdd(&d_fs_count, (unsigned long long) wi) : 0;
    }
    __syncthreads();
    uint64_t pos = bbase + wsum[warp] + incl - nv;
    int64_t v = u;
    for (uint32_t s = 0; s < n; s++, v += (int64_t) pp)
        if (v < M) {
            uint64_t key = ((uint64_t) b << offset_bits) | (uint64_t)(v + M);
            fp[pos] = gps1_fs_hash(key, mask);
            id[pos] = (uint32_t)(e + s);
            pos++;
        }
}

/* root index c of entry e: entry_off[c] <= e % slots < entry_off[c + 1] */
static __device__ __forceinline__ uint32_t gps1_fs_root(const uint32_t *entry_off,
                                                        uint32_t nroots, uint64_t slots,
                                                        uint64_t e)
{
    uint32_t off = (uint32_t)(e % slots), lo = 0, hi = nroots;
    while (hi - lo > 1) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (entry_off[mid] <= off)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

/* For each run of equal hashes (length >= 2), compare EVERY pair on the
 * immutable full key; emit equal valid keys with distinct primes, p1 from
 * the smaller entry index (production's stable-sort order). */
__global__ void kernel_scan_fp(const uint32_t *fp, const uint32_t *id, uint64_t n,
                               const uint64_t *keys, const uint32_t *entry_off,
                               uint32_t nroots, uint64_t slots,
                               const uint32_t *root_prime, const uint32_t *primes,
                               int64_t M, uint32_t offset_bits, match_t *matches)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i + 1 >= n)
        return;
    uint32_t h = fp[i];
    if (fp[i + 1] != h || (i > 0 && fp[i - 1] == h))
        return;                 /* not the first entry of a run of >= 2 */
    uint64_t end = i + 2;
    while (end < n && fp[end] == h)
        end++;
    unsigned long long k = end - i, eq = 0;
    unsigned long long *ctr = d_fs_ctr[i & (GPS1_FS_SLOTS - 1)];   /* spread: no hot address */
    atomicAdd(&ctr[0], 1ull);
    atomicAdd(&ctr[1], k * (k - 1) / 2);
    atomicMax(&ctr[3], k);
    for (uint64_t j = i + 1; j < end; j++) {
        uint32_t ej = id[j];
        uint64_t key = keys[ej];
        if (key == KEY_NONE)
            continue;
        for (uint64_t l = i; l < j; l++) {
            uint32_t el = id[l];
            if (keys[el] != key)
                continue;
            eq++;
            uint32_t ea = el < ej ? el : ej, eb = el < ej ? ej : el;
            uint32_t p1 = primes[root_prime[gps1_fs_root(entry_off, nroots, slots, ea)]];
            uint32_t p2 = primes[root_prime[gps1_fs_root(entry_off, nroots, slots, eb)]];
            if (p1 == p2)
                continue;
            uint64_t m = atomicAdd((unsigned long long *) &d_match_count, 1ull);
            if (m < GPS1_MAX_MATCHES)   /* overflow stays fatal on the host */
                matches[m] = { p1, p2, (uint32_t)(key >> offset_bits), 0,
                               (int64_t)(key & (((uint64_t) 1 << offset_bits) - 1)) - M };
        }
    }
    if (eq)
        atomicAdd(&ctr[2], eq);
}

struct gps1_fs_ctx_t {
    uint32_t *fp, *fp_alt, *id, *id_alt;    /* sort buffers */
    uint32_t *cur_fp, *cur_id;              /* sorted (modes 1, 2) */
    uint64_t n;                             /* entries sorted */
};
static unsigned long long gps1_fs_batches, gps1_fs_sorted, gps1_fs_chunks;

static void gps1_fs_report(void)
{
    static unsigned long long s[GPS1_FS_SLOTS][4];
    unsigned long long c[4] = {0, 0, 0, 0};
    cudaError_t err = cudaMemcpyFromSymbol(s, d_fs_ctr, sizeof s);
    for (int k = 0; err == cudaSuccess && k < GPS1_FS_SLOTS; k++) {
        c[0] += s[k][0]; c[1] += s[k][1]; c[2] += s[k][2];
        c[3] = s[k][3] > c[3] ? s[k][3] : c[3];
    }
    fprintf(stderr, "GPS1_FASTSORT mode=%d bits=%d chunk=%d batches=%llu sorted=%llu "
            "chunks=%llu fp_runs=%llu pair_checks=%llu key_eq_pairs=%llu "
            "false_pairs=%llu max_run=%llu%s\n", gps1_fastsort_mode(),
            gps1_fs_env("GPS1_FS_BITS", 32, 1, 32),
            gps1_fs_env("GPS1_FS_CHUNK", 1 << 22, 1, 1 << 30), gps1_fs_batches,
            gps1_fs_sorted, gps1_fs_chunks, c[0], c[1], c[2], c[1] - c[2], c[3],
            err == cudaSuccess ? "" : " (device counters unreadable at exit)");
}

/* CUB temp storage for (uint32, uint32) pairs: the production buffer when
 * it is large enough, else one extra allocation */
static void *gps1_fs_tmp(const gpu_t *G, size_t *bytes)
{
    static void *tmp;
    static size_t need;
    if (!tmp) {
        cub::DoubleBuffer<uint32_t> k(NULL, NULL), v(NULL, NULL);
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(NULL, need, k, v,
                                                   (uint64_t) GPS1_MAX_ENTRIES, 0, 32));
        if (need <= G->sort_tmp_bytes) {
            tmp = G->sort_tmp;
            need = G->sort_tmp_bytes;
        } else
            CUDA_CHECK(cudaMalloc(&tmp, need));
    }
    *bytes = need;
    return tmp;
}

/* fill slot: zero the cursor (mode 2), launch kernel_fill_fp */
static void gps1_fs_fill(const gpu_t *G, gps1_fs_ctx_t &X, int mode, uint32_t nroots,
                         uint32_t batch, int64_t M, uint32_t offset_bits,
                         uint64_t slots, cudaStream_t st)
{
    static bool said;
    const int bits = gps1_fs_env("GPS1_FS_BITS", 32, 1, 32);
    if (!said) {
        said = true;
        fprintf(stderr, "GPS1_FASTSORT on mode=%d bits=%d chunk=%d\n", mode, bits,
                gps1_fs_env("GPS1_FS_CHUNK", 1 << 22, 1, 1 << 30));
        atexit(gps1_fs_report);
    }
    X.fp = (uint32_t *) G->keys_alt;        /* GPS1_MAX_ENTRIES * 8 bytes = two uint32 buffers */
    X.fp_alt = X.fp + GPS1_MAX_ENTRIES;
    X.id = G->vals;
    X.id_alt = G->vals_alt;
    if (mode == 2) {
        unsigned long long z = 0;
        CUDA_CHECK(cudaMemcpyToSymbolAsync(d_fs_count, &z, 8, 0,
                                           cudaMemcpyHostToDevice, st));
    }
    uint32_t mask = bits == 32 ? 0xffffffffu : (1u << bits) - 1;
    uint64_t total = (uint64_t) nroots * batch;
    kernel_fill_fp<<<(total + 255) / 256, 256, 0, st>>>(G->primes, G->root_prime,
        G->roots, nroots, G->entry_off, G->qidx, G->rq_hi, G->rq_lo, G->invqq,
        G->nprimes, batch, M, offset_bits, slots, G->keys, X.fp, X.id, mask,
        mode == 2);
}

static void gps1_fs_scan_launch(const gpu_t *G, const uint32_t *fp, const uint32_t *id,
                                uint64_t n, uint32_t nroots, uint64_t slots, int64_t M,
                                uint32_t offset_bits, cudaStream_t st)
{
    if (n < 2)
        return;
    kernel_scan_fp<<<(n + 255) / 256, 256, 0, st>>>(fp, id, n, G->keys, G->entry_off,
        nroots, slots, G->root_prime, G->primes, M, offset_bits, G->matches);
}

/* sort slot: modes 1, 2 sort the batch; mode 3 sorts + scans each chunk */
static void gps1_fs_sort(const gpu_t *G, gps1_fs_ctx_t &X, int mode, uint64_t nentries,
                         uint32_t nroots, uint32_t batch, int64_t M,
                         uint32_t offset_bits, uint64_t slots, cudaStream_t st)
{
    const int bits = gps1_fs_env("GPS1_FS_BITS", 32, 1, 32);
    size_t tmp_bytes;
    void *tmp = gps1_fs_tmp(G, &tmp_bytes);
    gps1_fs_batches++;
    if (mode == 3) {
        uint64_t ipc = (uint64_t) gps1_fs_env("GPS1_FS_CHUNK", 1 << 22, 1, 1 << 30) / slots;
        if (ipc < 1)
            ipc = 1;
        for (uint64_t b0 = 0; b0 < batch; b0 += ipc) {
            uint64_t off = b0 * slots;
            uint64_t len = ((b0 + ipc < batch ? b0 + ipc : batch) - b0) * slots;
            cub::DoubleBuffer<uint32_t> dk(X.fp + off, X.fp_alt + off);
            cub::DoubleBuffer<uint32_t> dv(X.id + off, X.id_alt + off);
            size_t tb = tmp_bytes;
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(tmp, tb, dk, dv, len, 0,
                                                       bits, st));
            gps1_fs_scan_launch(G, dk.Current(), dv.Current(), len, nroots, slots,
                                M, offset_bits, st);
            gps1_fs_chunks++;
        }
        X.n = nentries;
        gps1_fs_sorted += nentries;
        return;
    }
    X.n = nentries;
    if (mode == 2) {    /* the compacted count gates the sort size */
        unsigned long long cnt;
        CUDA_CHECK(cudaMemcpyFromSymbolAsync(&cnt, d_fs_count, 8, 0,
                                             cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));
        if (cnt > nentries) {
            fprintf(stderr, "GPS1_FASTSORT: compacted %llu > %lu entries\n", cnt,
                    (unsigned long) nentries);
            exit(1);
        }
        X.n = cnt;
    }
    cub::DoubleBuffer<uint32_t> dk(X.fp, X.fp_alt), dv(X.id, X.id_alt);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(tmp, tmp_bytes, dk, dv, X.n, 0,
                                               bits, st));
    X.cur_fp = dk.Current();
    X.cur_id = dv.Current();
    gps1_fs_sorted += X.n;
    gps1_fs_chunks++;
}
/* ---- GPS1_FASTSORT end ---- */

extern "C" uint64_t gps1_run(const gpu_t *G, uint32_t nroots, uint32_t ndq,
''')

# ---- 2. fill slot ----
sub('''    kernel_fill<<<(total + 255) / 256, 256, 0, st>>>(G->primes, G->root_prime,
                                              G->roots,
                                              nroots, G->entry_off, G->qidx,
                                              G->rq_hi, G->rq_lo, G->invqq,
                                              G->nprimes, batch, M, offset_bits,
                                              slots, G->keys, G->vals);
''', '''    const int fs = gps1_fastsort_mode();   /* GPS1_FASTSORT, 0 = production */
    gps1_fs_ctx_t fsx = gps1_fs_ctx_t();
    if (fs)
        gps1_fs_fill(G, fsx, fs, nroots, batch, M, offset_bits, slots, st);
    else
    kernel_fill<<<(total + 255) / 256, 256, 0, st>>>(G->primes, G->root_prime,
                                              G->roots,
                                              nroots, G->entry_off, G->qidx,
                                              G->rq_hi, G->rq_lo, G->invqq,
                                              G->nprimes, batch, M, offset_bits,
                                              slots, G->keys, G->vals);
''')

# ---- 3. sort slot ----
sub('''    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(G->sort_tmp, tmp_bytes,
                                               dkeys, dvals, nentries, 0,
                                               sort_bits, st));
''', '''    if (fs) {           /* GPS1_FASTSORT; dkeys stays on the full keys */
        gps1_fs_sort(G, fsx, fs, nentries, nroots, batch, M, offset_bits, slots, st);
        sort_bits = gps1_fs_env("GPS1_FS_BITS", 32, 1, 32);
    } else
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(G->sort_tmp, tmp_bytes,
                                               dkeys, dvals, nentries, 0,
                                               sort_bits, st));
''')

# ---- 4. scan slot (mode 3 scanned per chunk inside the sort slot) ----
sub('''    kernel_scan<<<(nentries + 255) / 256, 256, 0, st>>>(dkeys.Current(),
                                                 dvals.Current(),
                                                 nentries, G->root_prime,
                                                 G->primes, M, offset_bits,
                                                 G->matches);
''', '''    if (fs) {           /* GPS1_FASTSORT */
        if (fs != 3)
            gps1_fs_scan_launch(G, fsx.cur_fp, fsx.cur_id, fsx.n, nroots, slots, M,
                                offset_bits, st);
    } else
    kernel_scan<<<(nentries + 255) / 256, 256, 0, st>>>(dkeys.Current(),
                                                 dvals.Current(),
                                                 nentries, G->root_prime,
                                                 G->primes, M, offset_bits,
                                                 G->matches);
''')

open(F, 'w').write(src)
print('gps1_fastsort_patch: patched polyselect/gps1_kernels.cu (input: %s)' % base)
