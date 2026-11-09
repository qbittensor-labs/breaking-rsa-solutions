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

"""glas_region_patch.py <gpu-siever/src/sieve_core_gpu.cu> -- two exact region-kernel changes (sandbox #23), applied
on top of the production chain (adjust, coprime, prefetch, xfuse).  Every flag defaults to off = stock code path.

GLAS_REGION_HALF=1 -- even-row half sieve.  On an even row j only odd i can give a relation (gcd(i, j) > 1 otherwise):
the survivor ballot masks the even cells with row_mask 0xaaaaaaaa, the side-0 candidate path computes only the cells of
the side-1 mask, and the rescan reads only final-mask cells, so no step reads an even cell of an even row.  The dense
and sparse small-prime walks of such a row therefore start on the first odd hit and step 2q (odd q), or skip the walk
(even q = 2^k or an even projective stride whose hits are all even).  Halves the small-prime shared-memory atomics on
even rows (-25% of both tiers, both sides).  Cell parity = parity of i: region bases, row starts, w0 and I/2 are even.
The skipped cells are never subtracted from, so no 16-bit borrow can reach an odd neighbour (none can in stock either:
BIAS 4096 log units > any sieved sum).

GLAS_REGION_RX=1 -- side-1 resieve extraction in region 1.  Stock: region 0 (side 0, after the gated side-0 fill) re-
reads the whole side-1 bucket of every region with a final survivor (~13k entries = ~106 KB per region, ~1.6-1.7 GB
per special-q, evicted from L2 long before: pure DRAM) to report the side-1 resieve hits on final survivors.  RX:
region 1, right after its survivor ballot, writes its side-1 mask words into the (free) pattern area, re-reads its own
side-1 bucket while it is L2-hot (the block read it a few microseconds earlier; GLAS_REGION_PREFETCH pulled it in) and
appends the resieve entries that hit a side-1-mask cell to a per-region list (warp-aggregated, cap
GLAS_REGION_RX_CAP, default 1024 entries); region 0 then scans only that list against its final mask.  Final mask is
a subset of the side-1 mask (CAND starts from bitmask; the non-CAND side-0 path ANDs it), so the emitted records
(position << 32 | q << 1 | side) are the same multiset; only the append order changes (the flat hit list is radix-
sorted on the whole key before the filter; per-region lists hold the same multiset).  A region whose list overflowed
the cap falls back to the full bucket scan (exact).  Memory: max_regions x (cap x 8 + 4) B per slot (128 MB at 2^29).
GLAS_REGION_RX=2 -- price probe (stock emission path): region 0 reads the side-1 bucket of each surviving region TWICE
(the first pass is folded into a value that only a volatile no-op store depends on).  wall(RX=2) - wall(stock) is
what one extra 1.6 GB re-read costs at 4 slots, i.e. the ceiling of the RX=1 saving.

GLAS_REGION_STATS=1 -- per-slot device counters (copied out per chunk on the out stream, summed on the host, printed
at exit; costs a little, use in check runs only): RX entries extracted / regions over cap / region-0 fallbacks,
regions with a final survivor and the side-1 entries region 0 re-reads (stock path), and an order-independent hash of
every chunk's survivor list and (flat) hit list.  The survivor/hit hashes of base and variant must be equal: that is
the exactness check for both flags (the relation set has a known 1-line host-side nondeterminism, the hashes do not).
Printed line: "# GLAS_REGION stats: ...".  The hash needs GLAS_RHITS=0 (production) for the hit part.

All flags need GLAS_REGION_SKIP=0 (probe bits change what the walks / apply cover); otherwise ignored with a note."""
import sys
p = sys.argv[1]; s = open(p).read()
if 'c_region_half' in s: print('already patched'); sys.exit(0)
assert 'c_region_prefetch' in s and 'GLAS_XFUSE' in s, 'apply the production chain (prefetch, xfuse) first'

def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:100])
    s = s.replace(a, b)

# ---- device constants, stats layout, helpers
sub('/* GLAS_REGION_PREFETCH: region_kernel prefetches its bucket into L2 at entry (1 per line, 2 bulk per piece). */\n'
    '__constant__ uint32_t c_region_prefetch;\n',
    '/* GLAS_REGION_PREFETCH: region_kernel prefetches its bucket into L2 at entry (1 per line, 2 bulk per piece). */\n'
    '__constant__ uint32_t c_region_prefetch;\n\n'
    '/* GLAS_REGION_HALF: even rows walk the small primes on odd cells only.  GLAS_REGION_RX: 1 = region 1 extracts the\n'
    ' * side-1 resieve entries on side-1-mask cells into per-region lists that region 0 scans; 2 = price probe (region 0\n'
    ' * reads the side-1 bucket twice). */\n'
    '__constant__ uint32_t c_region_half;\n'
    '__constant__ uint32_t c_region_rx;\n\n'
    '/* GLAS_REGION_STATS: per-slot counters, index: */\n'
    'enum { RS_RX_ENTRIES, RS_RX_OVER, RS_R0_FALLBACK, RS_R0_SURV_REGIONS, RS_R0_REREAD, RS_HASH_SURV, RS_HASH_HITS,\n'
    '       RS_N_SURV, RS_N_HITS, RS_R0_LIST, N_RSTAT = 16 };\n')

sub('/* True when the hit at row j and offset off = i + I/2 can give a\n',
    '/* GLAS_REGION_HALF: restrict a small-prime walk (first cell off, stride qq) of an even row to its odd cells: an odd\n'
    ' * stride alternates the parity, so start on the first odd cell and step 2 qq; an even stride keeps the parity, so\n'
    ' * the walk is all-odd (kept) or all-even (dropped: qq = 0). */\n'
    '__device__ __forceinline__ void dev_half_walk(uint32_t & off, uint32_t & qq)\n'
    '{\n'
    '    if (qq & 1) {\n'
    '        if (!(off & 1))\n'
    '            off += qq;\n'
    '        qq <<= 1;\n'
    '    } else if (!(off & 1)) {\n'
    '        qq = 0;\n'
    '    }\n'
    '}\n\n'
    '/* order-independent hashing of the chunk outputs (GLAS_REGION_STATS) */\n'
    '__device__ __forceinline__ unsigned long long dev_mix64(unsigned long long z)\n'
    '{\n'
    '    z += 0x9e3779b97f4a7c15ull;\n'
    '    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;\n'
    '    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;\n'
    '    return z ^ (z >> 31);\n'
    '}\n\n'
    '/* True when the hit at row j and offset off = i + I/2 can give a\n')

# ---- host context / slot fields
sub('    uint32_t region_prefetch = 0; /* GLAS_REGION_PREFETCH */\n',
    '    uint32_t region_prefetch = 0; /* GLAS_REGION_PREFETCH */\n'
    '    uint32_t region_half = 0; /* GLAS_REGION_HALF */\n'
    '    uint32_t region_rx = 0;   /* GLAS_REGION_RX: 1 extraction, 2 price probe */\n'
    '    uint32_t rx_cap = 1024;   /* GLAS_REGION_RX_CAP: list entries per region */\n'
    '    bool region_stats = false; /* GLAS_REGION_STATS */\n'
    '    std::atomic<unsigned long long> rstat[N_RSTAT] = {};\n')
sub('    uint32_t * bitmask;       /* survivor bits, 1 bit per position */\n',
    '    uint32_t * bitmask;       /* survivor bits, 1 bit per position */\n'
    '    uint64_t * rx_list = nullptr;  /* GLAS_REGION_RX: [region][rx_cap] side-1 resieve entries on side-1 mask cells */\n'
    '    uint32_t * rx_cnt = nullptr;   /* GLAS_REGION_RX: [region] list length (> cap: overflowed, full scan) */\n'
    '    unsigned long long * rstat[2] = {nullptr, nullptr};  /* GLAS_REGION_STATS: [parity][N_RSTAT] */\n'
    '    unsigned long long * h_rstat = nullptr;              /* pinned [2][N_RSTAT] */\n')

# ---- region kernel: parameters, shared counter
sub('''              uint64_t * hits, uint32_t * hit_cnt, uint64_t * hits_r,
              uint32_t * hit_rcnt, uint64_t * hits_ov)
{
    extern __shared__ __align__(8) uint32_t S[];   /* SREG 16-bit cells */''',
'''              uint64_t * hits, uint32_t * hit_cnt, uint64_t * hits_r,
              uint32_t * hit_rcnt, uint64_t * hits_ov,
              uint64_t * rx_list, uint32_t * rx_cnt, uint32_t rx_cap,
              unsigned long long * rstat)
{
    extern __shared__ __align__(8) uint32_t S[];   /* SREG 16-bit cells */''')
sub('''    __shared__ uint32_t block_surv;
    __shared__ double cj[SREG >> 12][GLAS_MAX_DEGREE + 1];''',
'''    __shared__ uint32_t block_surv;
    __shared__ uint32_t rx_n;   /* GLAS_REGION_RX: list length */
    __shared__ double cj[SREG >> 12][GLAS_MAX_DEGREE + 1];''')
sub('''    if (threadIdx.x == 0)
        block_surv = 0;
    uint32_t const region = blockIdx.x;
''', '''    if (threadIdx.x == 0) {
        block_surv = 0;
        rx_n = 0;
    }
    uint32_t const region = blockIdx.x;
''')

# ---- HALF: dense tier setup (the warp walks lane stride q, word stride 32 q: both follow the halved stride)
sub('''            if (root != SKIP_ROOT && !(use_pat && dev_pattern_entry(s.q, root)))
                qq = dev_small_walk_start(t, s.q, root, s.magic, j0 + jr, w0, off);
''', '''            if (root != SKIP_ROOT && !(use_pat && dev_pattern_entry(s.q, root)))
                qq = dev_small_walk_start(t, s.q, root, s.magic, j0 + jr, w0, off);
            if (qq && c_region_half && !((j0 + jr) & 1))
                dev_half_walk(off, qq);
''')
# ---- HALF: sparse tier
sub('''        uint32_t const qq =
            dev_small_walk_start(t, s.q, root, s.magic, j0 + js, w0, off);
        if (!qq)
            continue;
        for (uint32_t p = rs + off; p < rs + wend; p += qq)''',
'''        uint32_t qq =
            dev_small_walk_start(t, s.q, root, s.magic, j0 + js, w0, off);
        if (qq && c_region_half && !((j0 + js) & 1))
            dev_half_walk(off, qq);
        if (!qq)
            continue;
        for (uint32_t p = rs + off; p < rs + wend; p += qq)''')

# ---- RX: extraction at the end of region 1
sub('''        uint32_t const widx = (base >> 5) + wd;
        if (first) {
            bitmask[widx] = word;
            return;
        }
''', '''        uint32_t const widx = (base >> 5) + wd;
        if (first) {
            bitmask[widx] = word;
            if (!rx_list)
                return;
            /* GLAS_REGION_RX: the side-1 mask goes to the pattern area (read only by the norm phase, which is
             * over); the block re-reads its L2-hot bucket and lists the resieve entries on mask cells */
            P[wd] = word;
            if (!__syncthreads_or(word != 0)) {
                if (threadIdx.x == 0)
                    rx_cnt[region] = 0;
                return;
            }
            uint64_t * const xl = rx_list + (uint64_t)region * rx_cap;
            auto take = [&](uint64_t h, bool ok) {
                uint32_t const p = (uint32_t)h & (SREG - 1);
                bool const hit = ok && (P[p >> 5] >> (p & 31) & 1) &&
                                 (h >> BK_RESIEVE_SHIFT & 1);
                uint32_t const bal = __ballot_sync(0xffffffffu, hit);
                if (!bal)
                    return;
                uint32_t b0 = 0;
                if (lane == 0)
                    b0 = atomicAdd(&rx_n, (uint32_t)__popc(bal));
                b0 = __shfl_sync(0xffffffffu, b0, 0);
                uint32_t const sl = b0 + __popc(bal & ((1u << lane) - 1));
                if (hit && sl < rx_cap)
                    xl[sl] = h;
            };
            /* block-uniform trip count: every lane reaches every ballot */
            for (uint32_t k0 = 0; k0 < cnt; k0 += 4 * blockDim.x) {
                uint32_t const k = k0 + threadIdx.x;
                uint64_t const h0 = k < cnt ? rb(k) : 0;
                uint64_t const h1 = k + blockDim.x < cnt ? rb(k + blockDim.x) : 0;
                uint64_t const h2 = k + 2 * blockDim.x < cnt ? rb(k + 2 * blockDim.x) : 0;
                uint64_t const h3 = k + 3 * blockDim.x < cnt ? rb(k + 3 * blockDim.x) : 0;
                take(h0, k < cnt);
                take(h1, k + blockDim.x < cnt);
                take(h2, k + 2 * blockDim.x < cnt);
                take(h3, k + 3 * blockDim.x < cnt);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                rx_cnt[region] = rx_n;
                if (rstat) {
                    atomicAdd(rstat + RS_RX_ENTRIES, (unsigned long long)min(rx_n, rx_cap));
                    if (rx_n > rx_cap)
                        atomicAdd(rstat + RS_RX_OVER, 1ull);
                }
            }
            return;
        }
''')

# ---- stats: regions with a final survivor (every mode)
sub('''    __syncthreads();
    if (!block_surv)
        return;
''', '''    __syncthreads();
    if (!block_surv)
        return;
    if (rstat && threadIdx.x == 0)
        atomicAdd(rstat + RS_R0_SURV_REGIONS, 1ull);
''')

# ---- RX: region 0 rescan of the side-1 bucket
sub('''    for (int hs = 0; hs < 2; hs++) {
        uint64_t const * const bk = hs ? obuckets : buckets;
        uint32_t const cap = hs ? obucket_cap : bucket_cap;
        uint32_t const bn =
            min((hs ? obucket_cnt : bucket_cnt)[region * BK_CNT_STRIDE], cap);
        for (uint32_t k = threadIdx.x; k < bn; k += blockDim.x) {''',
'''    for (int hs = 0; hs < 2; hs++) {
        uint64_t const * const bk = hs ? obuckets : buckets;
        uint32_t const cap = hs ? obucket_cap : bucket_cap;
        uint32_t const bn =
            min((hs ? obucket_cnt : bucket_cnt)[region * BK_CNT_STRIDE], cap);
        if (hs == 1 && rx_list) {
            /* GLAS_REGION_RX: the list region 1 left holds every resieve entry of this bucket on a side-1 mask
             * cell; the final mask is a subset of that mask */
            uint32_t const xn = rx_cnt[region];
            if (xn <= rx_cap) {
                uint64_t const * const xl = rx_list + (uint64_t)region * rx_cap;
                for (uint32_t k = threadIdx.x; k < xn; k += blockDim.x) {
                    uint64_t const h = xl[k];
                    uint32_t const p = (uint32_t)h & (SREG - 1);
                    if (!(P[p >> 5] >> (p & 31) & 1))
                        continue;
                    dev_hit_emit(base + p,
                                 (uint64_t)(base + p) << 32 |
                                     (h >> BK_Q_SHIFT) << 1 |
                                     (uint32_t)(side ^ 1),
                                 max_hits, hits, hit_cnt, hits_r, hit_rcnt,
                                 hits_ov);
                }
                if (rstat && threadIdx.x == 0)
                    atomicAdd(rstat + RS_R0_LIST, (unsigned long long)xn);
                continue;
            }
            if (rstat && threadIdx.x == 0)
                atomicAdd(rstat + RS_R0_FALLBACK, 1ull);
        }
        if (hs == 1 && rstat && threadIdx.x == 0)
            atomicAdd(rstat + RS_R0_REREAD, (unsigned long long)bn);
        if (hs == 1 && c_region_rx == 2) {
            /* GLAS_REGION_RX=2 price probe: one more full read of the side-1 bucket */
            uint32_t acc = 0;
            for (uint32_t k = threadIdx.x; k < bn; k += blockDim.x)
                acc += (uint32_t)(bk[dev_bucket_idx(region, k, cap)] >> 3);
            if (acc == 0x9e3779b9u)
                *(volatile uint32_t *)&rx_n = 0;
        }
        for (uint32_t k = threadIdx.x; k < bn; k += blockDim.x) {''')

# ---- output hash kernel (GLAS_REGION_STATS), before the small resieve kernels
sub('''/* Small resieve entries: trial-divide each (survivor, entry) pair''',
'''/* GLAS_REGION_STATS: order-independent hash of a chunk's survivor positions and resieve hit records (salted with
 * the special-q and the chunk's first row), plus their counts */
__global__ void region_hash_kernel(gpu_task t, uint32_t const * survivors,
                                   uint32_t const * counters,
                                   uint32_t max_survivors,
                                   uint64_t const * hits, uint32_t max_hits,
                                   unsigned long long * rstat)
{
    uint32_t const ns = min(counters[0], max_survivors);
    uint32_t const nh = hits ? min(counters[1], max_hits) : 0;
    unsigned long long const salt = dev_mix64(t.q * 0x100000001b3ull ^ t.rho ^ (uint64_t)t.j0 << 40);
    unsigned long long a = 0, b = 0;
    uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t const stride = gridDim.x * blockDim.x;
    for (uint32_t i = gid; i < ns; i += stride)
        a += dev_mix64(survivors[i] ^ salt);
    for (uint32_t i = gid; i < nh; i += stride)
        b += dev_mix64(hits[i] ^ salt);
    for (int o = 16; o; o >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, o);
        b += __shfl_xor_sync(0xffffffffu, b, o);
    }
    if ((threadIdx.x & 31) == 0 && (a || b)) {
        atomicAdd(rstat + RS_HASH_SURV, a);
        atomicAdd(rstat + RS_HASH_HITS, b);
    }
    if (gid == 0) {
        atomicAdd(rstat + RS_N_SURV, (unsigned long long)ns);
        atomicAdd(rstat + RS_N_HITS, (unsigned long long)nh);
    }
}

/* Small resieve entries: trial-divide each (survivor, entry) pair''')

# ---- launch_region: new arguments
sub('''        ctx.rhits ? s.hits_r[par] : nullptr,
        ctx.rhits ? s.hit_rcnt[par] : nullptr,
        ctx.rhits ? s.hits_ov[par] : nullptr);
}

void sieve_one_gpu(''', '''        ctx.rhits ? s.hits_r[par] : nullptr,
        ctx.rhits ? s.hit_rcnt[par] : nullptr,
        ctx.rhits ? s.hits_ov[par] : nullptr,
        ctx.region_rx == 1 ? s.rx_list : nullptr, s.rx_cnt, ctx.rx_cap,
        ctx.region_stats ? s.rstat[par] : nullptr);
}

void sieve_one_gpu(''')

# ---- env parse (gpu_prepare, next to GLAS_REGION_CAND)
sub('''    ctx.region_cand = env_flag("GLAS_REGION_CAND", cand_ok);
''', '''    ctx.region_cand = env_flag("GLAS_REGION_CAND", cand_ok);
    {   /* glas_region_patch.py */
        uint32_t half = 0, rx = 0;
        if (char const * e = getenv("GLAS_REGION_HALF"))
            half = atoi(e) ? 1 : 0;
        if (char const * e = getenv("GLAS_REGION_RX"))
            rx = (uint32_t)std::min(2, std::max(0, atoi(e)));
        if (char const * e = getenv("GLAS_REGION_RX_CAP"))
            ctx.rx_cap = (uint32_t)std::max(32, atoi(e));
        bool const stats = env_flag("GLAS_REGION_STATS", false);
        if ((half || rx || stats) && ctx.skip) {
            fprintf(stderr, "# GLAS_REGION_HALF/RX/STATS ignored: GLAS_REGION_SKIP=%d\\n", ctx.skip);
            half = rx = 0;
        } else {
            ctx.region_stats = stats;
        }
        ctx.region_half = half;
        ctx.region_rx = rx;
        if (half)
            fprintf(stderr, "# GLAS_REGION_HALF on: even rows walk the small primes on odd cells only\\n");
        if (rx == 1)
            fprintf(stderr, "# GLAS_REGION_RX 1 on: region 1 lists the side-1 resieve entries on side-1 mask cells, "
                            "cap %u per region\\n", ctx.rx_cap);
        if (rx == 2)
            fprintf(stderr, "# GLAS_REGION_RX 2 on: price probe, region 0 reads the side-1 bucket twice\\n");
        if (ctx.region_stats)
            fprintf(stderr, "# GLAS_REGION_STATS on (the hit hash needs GLAS_RHITS=0)\\n");
    }
''')

# ---- constants upload (gpu_init, after the prefetch constant)
sub('''    CUDA_CHECK(cudaMemcpyToSymbol(c_region_prefetch, &ctx.region_prefetch, 4));
''', '''    CUDA_CHECK(cudaMemcpyToSymbol(c_region_prefetch, &ctx.region_prefetch, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(c_region_half, &ctx.region_half, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(c_region_rx, &ctx.region_rx, 4));
''')

# ---- slot memory accounting and allocation
sub('''        b += ctx.select_temp_bytes;
        if (ctx.gate_shift)''', '''        b += ctx.select_temp_bytes;
        if (ctx.region_rx == 1)   /* GLAS_REGION_RX lists */
            b += (area >> SREG_LOG) * ((uint64_t)ctx.rx_cap * 8 + 4);
        if (ctx.gate_shift)''')
sub('''        CUDA_CHECK(cudaMalloc(&s.bitmask, ctx.chunk_area / 8));
''', '''        CUDA_CHECK(cudaMalloc(&s.bitmask, ctx.chunk_area / 8));
        if (ctx.region_rx == 1) {
            CUDA_CHECK(cudaMalloc(&s.rx_list, (uint64_t)ctx.max_regions * ctx.rx_cap * 8));
            CUDA_CHECK(cudaMalloc(&s.rx_cnt, (uint64_t)ctx.max_regions * 4));
        }
        if (ctx.region_stats) {
            for (int par = 0; par < 2; par++)
                CUDA_CHECK(cudaMalloc(&s.rstat[par], N_RSTAT * 8));
            CUDA_CHECK(cudaMallocHost(&s.h_rstat, 2 * N_RSTAT * 8));
        }
''')

# ---- per chunk: reset the counters (sieve stream), hash the outputs, copy the counters out (out stream), sum them
sub('''        CUDA_CHECK(cudaStreamWaitEvent(st, s.odone[par]));
        CUDA_CHECK(cudaMemsetAsync(s.counters[par], 0, 4 * 4, st));
''', '''        CUDA_CHECK(cudaStreamWaitEvent(st, s.odone[par]));
        CUDA_CHECK(cudaMemsetAsync(s.counters[par], 0, 4 * 4, st));
        if (ctx.region_stats)
            CUDA_CHECK(cudaMemsetAsync(s.rstat[par], 0, N_RSTAT * 8, st));
''')
sub('''        mark(PH_RESIEVE);
        s.n_ev[par] = nev;
''', '''        mark(PH_RESIEVE);
        if (ctx.region_stats)
            region_hash_kernel<<<std::min(ctx.tiny_grid, 1024u), 256, 0, st>>>(
                t, s.survivors[par], s.counters[par], ctx.max_survivors,
                ctx.rhits ? nullptr : s.hits[par], ctx.max_hits, s.rstat[par]);
        s.n_ev[par] = nev;
''')
sub('''        CUDA_CHECK(cudaMemcpyAsync(s.h_counters, s.counters[par], 3 * 4,
                                   cudaMemcpyDeviceToHost, s.out_stream));
        {
            trace_scope const tr(TR_SYNC_CTR);
            CUDA_CHECK(cudaStreamSynchronize(s.out_stream));
        }
''', '''        CUDA_CHECK(cudaMemcpyAsync(s.h_counters, s.counters[par], 3 * 4,
                                   cudaMemcpyDeviceToHost, s.out_stream));
        if (ctx.region_stats)
            CUDA_CHECK(cudaMemcpyAsync(s.h_rstat + par * N_RSTAT, s.rstat[par], N_RSTAT * 8,
                                       cudaMemcpyDeviceToHost, s.out_stream));
        {
            trace_scope const tr(TR_SYNC_CTR);
            CUDA_CHECK(cudaStreamSynchronize(s.out_stream));
        }
        if (ctx.region_stats)
            for (int k = 0; k < N_RSTAT; k++)
                ctx.rstat[k] += s.h_rstat[par * N_RSTAT + k];
''')

# ---- exit report
sub('''                        (unsigned long)ctx.bucket_sum[side]);
            report_idle();''', '''                        (unsigned long)ctx.bucket_sum[side]);
            if (ctx.region_stats) {
                auto v = [](int k) { return (unsigned long long)ctx.rstat[k].load(); };
                fprintf(stderr, "# GLAS_REGION stats: half %u rx %u cap %u rhits %d | rx entries %llu, rx regions over cap %llu, "
                        "region0 list entries %llu, region0 fallbacks %llu | regions with a final survivor "
                        "%llu, side-1 entries re-read (full scans) %llu | survivors %llu hits %llu | hash surv %016llx hits %016llx\\n",
                        ctx.region_half, ctx.region_rx, ctx.rx_cap, (int)ctx.rhits, v(RS_RX_ENTRIES), v(RS_RX_OVER), v(RS_R0_LIST),
                        v(RS_R0_FALLBACK), v(RS_R0_SURV_REGIONS), v(RS_R0_REREAD), v(RS_N_SURV), v(RS_N_HITS),
                        v(RS_HASH_SURV), v(RS_HASH_HITS));
            }
            report_idle();''')

open(p, 'w').write(s); print('patched')
