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

"""glas_p1_patch.py <gpu-siever/src/sieve_core_gpu.cu> -- proposal 1 of docs/rsa560_codex_review2.md (codex-review2,
section 3.1): survivor-indexed factor recovery and a rational-first cascade in glas's GPU survivor path.

Applies ON TOP of the full production patch stack (sandbox_bootstrap.sh order: adjust, coprime, prefetch, xfuse, region,
xfgrid, session, narrow).  The pre-narrow file must be the review's G (sha256 0fd70cf0...); this patch pins the
narrow-patched file by sha256 as well (fail-closed on any drift).  Only sieve_core_gpu.cu changes.

Runtime selection (read once at gpu_init; default 0 = production behaviour, no banner):
  GLAS_P1=1  survivor-indexed layout.  Region 0 gives every final survivor a compact ID from its final bitmap: one
             atomicAdd per region block for the region base plus a block exclusive prefix scan of the per-word
             popcounts (replacing one global atomicAdd per survivor).  Large resieve hits (both bucket sides, RX
             lists) and small resieve hits (resieve_small_kernel, which already knows tile + k == ID) append their
             32-bit (prime << 1 | side) word straight into a bounded per-survivor list p1_list[ID][K]
             (GLAS_P1_K, default 16) with a counter p1_cnt[ID].  A hit that finds its list full, or whose survivor
             ID is >= the list capacity (GLAS_P1_CAP survivors, default 2^19, clamped to max_survivors), goes to the
             unchanged flat (position << 32 | prime << 1 | side) list: that EXACT FALLBACK is radix-sorted only when
             non-empty and binary-searched per survivor exactly as before.  Never truncates: every hit is in the
             survivor's list or in the flat list; the filter reads both.
  GLAS_P1=2  rational-first cascade.  The sieve stream computes tiny masks and small resieve for side 0 (rational)
             only.  The out stream sorts the flat hits so far (segment A), runs p1_rat_kernel -- the filter's own
             side-0 steps (b == 0 and q | b rejection, |F0(a,b)|, q division when sq_side == 0, division by every
             recovered side-0 prime to full valuation, tiny side-0 primes, dev_leftover) -- and keeps the survivor
             IDs whose rational leftover is not LEFT_BAD and, for a LEFT_PENDING leftover (lpb < bits <= mfb), is
             not prime: the deferred MR test that filter_prime_kernel (or the host leftover check) would apply
             anyway is run here, so a prime rational leftover is rejected before any algebraic work (exact; s90:
             PASS 4.6% + composite PENDING ~2.5% -> ~7% of survivors reach algebraic tiny/resieve).  Unresolved
             composites PASS (no speculative ECM-failure rejection).
             Side-1 tiny masks and side-1 small resieve then run on the pass list only (original IDs preserved), the
             host re-reads the hit counter (overflow -> the existing whole-q CPU fallback), sorts the new segment B,
             and the unchanged filter (narrow/IDIV specialization as production) runs over the pass list with both
             sorted segments.  The filter recomputes side 0 from the same hits, so its accept/reject decisions and
             its assembled records are those of the production filter for every survivor it visits; survivors it
             does not visit are exactly those the production filter rejects on side 0.
  GLAS_P1=3  both.
  GLAS_P1_K (1..64, default 16), GLAS_P1_CAP (default 524288): list geometry.  Extra device memory per slot:
             2 parities x CAP x (K + 1) x 4 B (layout) + max_survivors x 4 B pass list (cascade), counted in
             slot_bytes so the slot count reflects it (banner prints slots and GB/slot).

Banners: '# GLAS_P1=v on: layout L cascade C, K .., cap .., +X MB/slot' at init; at exit '# GLAS_P1 stats: chunks ..
survivors .. rational-pass .. (f0 ..) rational-reject .. list-overflow-survivors .. over-cap-survivors ..
flat-hits .. cascade-cpu-fallbacks ..'.  Requires GLAS_RHITS=0 (production) and the GPU filter; otherwise
GLAS_P1 is forced off with a banner.  Factor semantics are untouched: prime powers are never resieved (fb resieve
flag), the filter divides each recovered prime out to full valuation, projective roots are handled by the unchanged
resieve walk, q handling / signs / q|b rejection are the filter's own code."""
import hashlib, sys
p = sys.argv[1]; s = open(p).read()
if 'p1_rat_kernel' in s: print('already patched'); sys.exit(0)
assert 'GLAS_NARROW device core begin' in s, 'needs glas_narrow_patch.py (production stack) first'
NARROW_SHA = '01123ca56114c7a55c1390ae379dc6265aec2387425a958d1e10c11e72cf09f1'
h = hashlib.sha256(s.encode()).hexdigest()
if h != NARROW_SHA and '--no-pin' not in sys.argv:
    sys.exit(f'glas_p1_patch: input sha256 {h} != pinned production (narrow-patched) {NARROW_SHA}; refusing')

def sub(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (s.count(a), a[:100])
    s = s.replace(a, b)

# ---------------------------------------------------------------- task / slot / context fields
sub('''    int skip;   /* probe bits: 1 norms, 2 small walk, 4 bucket apply,
                   8 bucket stores, 64 index-derived stores */
};''', '''    int skip;   /* probe bits: 1 norms, 2 small walk, 4 bucket apply,
                   8 bucket stores, 64 index-derived stores */
    /* GLAS_P1 (gpu/glas_p1_patch.py); all zero = production */
    uint32_t p1;          /* bit 0 survivor-indexed lists, bit 1 rational-first cascade */
    uint32_t p1_k;        /* list entries per survivor */
    uint32_t p1_cap;      /* survivors with a list */
    uint32_t p1_nA;       /* filter: sorted flat hits [0, nA) segment A, [nA, n_hits) segment B */
    uint32_t p1_sides;    /* tiny_mask_kernel: sides to compute (0 = both) */
    uint32_t p1_uselist;  /* tiny / resieve / filter iterate the pass list p1_pass[0, p1_ctr[2]) */
    uint32_t * p1_cnt;    /* [cap] list lengths (may exceed k: overflowed to the flat list) */
    uint32_t * p1_list;   /* [cap][k] prime << 1 | side */
    uint32_t * p1_pass;   /* [max_survivors] survivor IDs passing the exact rational test */
    uint32_t * p1_ctr;    /* [0] list-overflow survivors [1] rational MR rejects [2] pass count [3] rational rejects (incl. MR) */
};''')

sub('''    uint64_t * hits_ov[2];    /* GLAS_RHITS: [parity] overflow tail */
    uint32_t * counters[2];''', '''    uint64_t * hits_ov[2];    /* GLAS_RHITS: [parity] overflow tail */
    uint32_t * p1_cnt[2] = {nullptr, nullptr};   /* GLAS_P1: [parity][cap] */
    uint32_t * p1_list[2] = {nullptr, nullptr};  /* GLAS_P1: [parity][cap][k] */
    uint32_t * p1_ctr[2] = {nullptr, nullptr};   /* GLAS_P1: [parity][4] */
    uint32_t * p1_pass = nullptr;                /* GLAS_P1: [max_survivors], out stream only */
    uint32_t * h_p1 = nullptr;                   /* GLAS_P1: pinned [2][4] */
    uint32_t * counters[2];''')

sub('''    bool rhits = false;       /* resieve hits land in per-region lists */''',
    '''    bool rhits = false;       /* resieve hits land in per-region lists */
    uint32_t p1 = 0;          /* GLAS_P1: bit 0 layout, bit 1 cascade */
    uint32_t p1_k = 16;       /* GLAS_P1_K */
    uint32_t p1_cap = 1u << 19;  /* GLAS_P1_CAP */
    uint32_t p1_rat_grid = 0;
    /* chunks, survivors, pass, reject, list-overflow survivors, over-cap survivors, flat hits, cascade fallbacks */
    std::atomic<unsigned long long> p1_stat[9] = {};''')

# ---------------------------------------------------------------- emit into a survivor's list
sub('''    hit_cnt[1] = 1;   /* the chunk's overflow flag */
}
''', '''    hit_cnt[1] = 1;   /* the chunk's overflow flag */
}

/* GLAS_P1 layout: the hit goes to survivor id's bounded list; a full list or an id past the list capacity takes the
 * unchanged flat path (exact fallback, the filter reads both) */
__device__ __forceinline__ void dev_p1_emit(gpu_task const & t, uint32_t id, uint32_t x, uint64_t rec,
                                            uint32_t max_hits, uint64_t * hits, uint32_t * hit_cnt,
                                            uint64_t * hits_r, uint32_t * hit_rcnt, uint64_t * hits_ov)
{
    if ((t.p1 & 1) && id < t.p1_cap) {
        uint32_t const k = atomicAdd(t.p1_cnt + id, 1);
        if (k < t.p1_k) {
            t.p1_list[(uint64_t)id * t.p1_k + k] = (uint32_t)rec;
            return;
        }
        if (k == t.p1_k)
            atomicAdd(t.p1_ctr, 1u);
    }
    dev_hit_emit(x, rec, max_hits, hits, hit_cnt, hits_r, hit_rcnt, hits_ov);
}
''')

# ---------------------------------------------------------------- region 0: compact IDs by block prefix scan
sub('''    uint32_t keep = word;
    while (word) {''', '''    bool const p1l = t.p1 & 1;   /* GLAS_P1 layout: IDs from the block scan below */
    uint32_t keep = word;
    while (word) {''')
sub('''        uint32_t const slot = atomicAdd(survivor_cnt, 1);
        if (slot < max_survivors)
            survivors[slot] = base + p;
    }''', '''        if (p1l)
            continue;
        uint32_t const slot = atomicAdd(survivor_cnt, 1);
        if (slot < max_survivors)
            survivors[slot] = base + p;
    }''')
sub('''    if (rstat && threadIdx.x == 0)
        atomicAdd(rstat + RS_R0_SURV_REGIONS, 1ull);
''', '''    if (rstat && threadIdx.x == 0)
        atomicAdd(rstat + RS_R0_SURV_REGIONS, 1ull);

    /* GLAS_P1 layout: ID = region base (one atomicAdd per block) + exclusive prefix of the final-mask popcounts in
     * word order + rank of the bit in its word.  The scores are dead (the survivor loop above was their last
     * reader), so S words [0, 1024) hold the per-word prefix and [1024, 1056) the warp bases. */
    uint32_t * const p1pre = S;
    if (p1l) {
        uint32_t const v = __popc(keep);
        uint32_t incl = v;
        for (uint32_t o = 1; o < 32; o <<= 1) {
            uint32_t const u = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= o)
                incl += u;
        }
        uint32_t const warp = threadIdx.x >> 5;
        if (lane == 31)
            S[1024 + warp] = incl;
        __syncthreads();
        if (warp == 0) {
            uint32_t const w = S[1024 + lane];
            uint32_t wi = w;
            for (uint32_t o = 1; o < 32; o <<= 1) {
                uint32_t const u = __shfl_up_sync(0xffffffffu, wi, o);
                if (lane >= o)
                    wi += u;
            }
            uint32_t rb0 = 0;
            if (lane == 31)
                rb0 = atomicAdd(survivor_cnt, wi);
            rb0 = __shfl_sync(0xffffffffu, rb0, 31);
            S[1024 + lane] = rb0 + wi - w;
        }
        __syncthreads();
        uint32_t const ex = S[1024 + warp] + incl - v;
        p1pre[wd] = ex;
        for (uint32_t m = keep; m; m &= m - 1) {
            uint32_t const b = __ffs(m) - 1;
            uint32_t const id = ex + __popc(keep & ((1u << b) - 1));
            if (id < max_survivors)
                survivors[id] = base + wd * 32 + b;
        }
        __syncthreads();
    }
    auto p1id = [&](uint32_t p) -> uint32_t {
        return p1l ? p1pre[p >> 5] + __popc(P[p >> 5] & ((1u << (p & 31)) - 1)) : 0;
    };
''')
sub('''dev_hit_emit(base + p,''', '''dev_p1_emit(t, p1id(p), base + p,''', 2)

# ---------------------------------------------------------------- small resieve: IDs, pass list
sub('''    __shared__ uint32_t sx[RS_TILE];''', '''    __shared__ uint32_t sx[RS_TILE];   /* GLAS_P1 build: survivor IDs (positions re-read on a hit) */''')
sub('''    uint32_t const n_surv = min(*survivor_cnt, max_survivors);
    uint32_t const n_rs = *rs_cnt;''', '''    uint32_t const n_surv = t.p1_uselist ? min(t.p1_ctr[2], max_survivors)
                                         : min(*survivor_cnt, max_survivors);
    uint32_t const n_rs = *rs_cnt;''')
sub('''            uint32_t const x = survivors[tile + k];
            sx[k] = x;''', '''            uint32_t const id = t.p1_uselist ? t.p1_pass[tile + k] : tile + k;
            uint32_t const x = survivors[id];
            sx[k] = id;''')
sub('''            auto emit = [&](uint32_t x) {
                dev_hit_emit(x, (uint64_t)x << 32 | (uint64_t)q << 1 | side,
                             max_hits, hits, hit_cnt, hits_r, hit_rcnt,
                             hits_ov);
            };''', '''            auto emit = [&](uint32_t id) {
                uint32_t const x = survivors[id];
                dev_p1_emit(t, id, x, (uint64_t)x << 32 | (uint64_t)q << 1 | side,
                            max_hits, hits, hit_cnt, hits_r, hit_rcnt,
                            hits_ov);
            };''')

# ---------------------------------------------------------------- tiny masks: side selection, pass list
sub('''    uint32_t const n = min(*survivor_cnt, max_survivors);
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += gridDim.x * blockDim.x) {
        uint32_t const x = survivors[idx];''', '''    uint32_t const n = t.p1_uselist ? min(t.p1_ctr[2], max_survivors)
                                    : min(*survivor_cnt, max_survivors);
    uint32_t const sides = t.p1_sides ? t.p1_sides : 3u;   /* GLAS_P1 cascade */
    for (uint32_t kk = blockIdx.x * blockDim.x + threadIdx.x; kk < n;
         kk += gridDim.x * blockDim.x) {
        uint32_t const idx = t.p1_uselist ? t.p1_pass[kk] : kk;
        uint32_t const x = survivors[idx];''')
sub('''        for (int side = 0; side < 2; side++) {
            uint64_t m = 0;''', '''        for (int side = 0; side < 2; side++) {
            if (!(sides >> side & 1))
                continue;
            uint64_t m = 0;''')

# ---------------------------------------------------------------- hit association for the filter / rational test
sub('''__device__ u128 dev_gcd128(u128 a, u128 b);
''', '''__device__ u128 dev_gcd128(u128 a, u128 b);

/* GLAS_P1: a survivor's hits = its list entries + its runs in the two sorted flat segments */
struct p1_hits {
    uint32_t x, ln;
    uint32_t const * le;
    uint32_t a0, a1, b0, b1;
};

__device__ uint32_t p1_run_end(uint64_t const * hits, uint32_t h, uint32_t end, uint32_t x)
{
    while (h < end && (hits[h] >> 32) == x)
        h++;
    return h;
}

__device__ p1_hits p1_find(gpu_task const & t, uint32_t idx, uint32_t x, uint64_t const * hits, uint32_t n_hits)
{
    p1_hits r = {};
    r.x = x;
    uint32_t const nA = min(t.p1_nA, n_hits);
    uint64_t const key = (uint64_t)x << 32;
    if (nA) {
        r.a0 = dev_hits_lower_bound(hits, nA, key);
        r.a1 = p1_run_end(hits, r.a0, nA, x);
    }
    if (n_hits > nA) {
        r.b0 = nA + dev_hits_lower_bound(hits + nA, n_hits - nA, key);
        r.b1 = p1_run_end(hits, r.b0, n_hits, x);
    }
    if (t.p1_list && idx < t.p1_cap) {
        r.ln = min(t.p1_cnt[idx], t.p1_k);
        r.le = t.p1_list + (uint64_t)idx * t.p1_k;
    }
    return r;
}

template <typename F>
__device__ void p1_each(p1_hits const & r, uint64_t const * hits, F f)
{
    for (uint32_t k = 0; k < r.ln; k++)
        f((uint64_t)r.x << 32 | r.le[k]);
    for (uint32_t h = r.a0; h < r.a1; h++)
        f(hits[h]);
    for (uint32_t h = r.b0; h < r.b1; h++)
        f(hits[h]);
}

/* GLAS_P1 cascade: the filter's side-0 steps, verbatim semantics (production-width arithmetic); a survivor the
 * production filter would reject on side 0 is left out of the pass list, every other survivor (including an
 * unresolved / deferred-MR leftover) passes. */
__global__ void p1_rat_kernel(gpu_task t, uint32_t const * survivors, uint32_t n_surv, uint64_t const * tiny,
                              uint64_t const * hits, uint32_t n_hits)
{
    uint32_t const I = 1u << t.logI;
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_surv; idx += gridDim.x * blockDim.x) {
        uint32_t const x = survivors[idx];
        int64_t const i = (int64_t)(x & (I - 1)) - I / 2;
        int64_t const j = (int64_t)t.j0 + (x >> t.logI);
        int64_t a = i * t.a0 + j * t.a1;
        int64_t b = i * t.b0 + j * t.b1;
        if (b < 0) {
            a = -a;
            b = -b;
        }
        bool pass = b != 0 && (uint64_t)b % t.q != 0;
        if (pass) {
            int const side = 0;
            wide n;
            dev_wide_norm(side, t.degree[side], a, (uint64_t)b, n);
            int nl = wide_live(n);
            if (side == t.sq_side) {
                int const nq = t.q >> 32 ? wide_divide_out(n, t.q)
                                         : wide_divide_out_u32(n, nl, (uint32_t)t.q);
                pass = nq != 0;
                nl = wide_live(n);
            }
            if (pass) {
                p1_hits const ph = p1_find(t, idx, x, hits, n_hits);
                p1_each(ph, hits, [&](uint64_t h) {
                    if ((h & 1) != (uint64_t)side)
                        return;
                    uint32_t const p = (uint32_t)h >> 1;
                    if (side == t.sq_side && p == t.q)
                        return;
                    wide_divide_out_u32(n, nl, p);
                });
                for (uint64_t m = tiny[2 * (uint64_t)idx + side]; m; m &= m - 1)
                    wide_divide_out_u32(n, nl, c_tiny_p[__ffsll((long long)m) - 1]);
                leftover_state const l = dev_leftover(n, side);
                pass = l != LEFT_BAD;
                /* the deferred primality test, done now: a prime leftover above 2^lpb (pending: lpb < bits <= mfb)
                 * is rejected by filter_prime_kernel / the host leftover check alike, so this is exact */
                if (l == LEFT_PENDING && dev_prime_lo(n.v)) {
                    pass = false;
                    atomicAdd(t.p1_ctr + 1, 1u);
                }
            }
        }
        if (pass)
            t.p1_pass[atomicAdd(t.p1_ctr + 2, 1u)] = idx;
        else
            atomicAdd(t.p1_ctr + 3, 1u);
    }
}
''')

# filter kernels (production filter_kernel and the narrow template): pass-list iteration and P1 association
sub('''    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n_surv;
         idx += gridDim.x * blockDim.x) {
        uint32_t const x = survivors[idx];
        int64_t const i = (int64_t)(x & (I - 1)) - I / 2;
        int64_t const j = (int64_t)t.j0 + (x >> t.logI);
        int64_t a = i * t.a0 + j * t.a1;''', '''    uint32_t const n_iter = t.p1_uselist ? min(t.p1_ctr[2], n_surv) : n_surv;   /* GLAS_P1 cascade */
    for (uint32_t kk = blockIdx.x * blockDim.x + threadIdx.x; kk < n_iter;
         kk += gridDim.x * blockDim.x) {
        uint32_t const idx = t.p1_uselist ? t.p1_pass[kk] : kk;
        uint32_t const x = survivors[idx];
        int64_t const i = (int64_t)(x & (I - 1)) - I / 2;
        int64_t const j = (int64_t)t.j0 + (x >> t.logI);
        int64_t a = i * t.a0 + j * t.a1;''', 2)
sub('''        uint32_t h0 = 0, h1 = 0;
        uint32_t const reg = x >> SREG_LOG;''', '''        uint32_t h0 = 0, h1 = 0;
        p1_hits ph = {};   /* GLAS_P1 */
        uint32_t const reg = x >> SREG_LOG;''', 2)
sub('''        } else {
            h0 = dev_hits_lower_bound(hits, n_hits, (uint64_t)x << 32);''', '''        } else if (t.p1) {
            ph = p1_find(t, idx, x, hits, n_hits);
        } else {
            h0 = dev_hits_lower_bound(hits, n_hits, (uint64_t)x << 32);''', 2)
sub('''        auto each_hit = [&](auto f) {
            if (hits_r) {''', '''        auto each_hit = [&](auto f) {
            if (t.p1) {
                p1_each(ph, hits, f);
                return;
            }
            if (hits_r) {''', 2)
sub('''            uint32_t nh = h1 - h0;
            if (hits_r) {''', '''            uint32_t nh = h1 - h0;
            if (hits_r || t.p1) {''', 2)

# ---------------------------------------------------------------- host: env, memory, launches
sub('''    ctx.rhits = ctx.filter && env_flag("GLAS_RHITS", true);
''', '''    ctx.rhits = ctx.filter && env_flag("GLAS_RHITS", true);
    if (char const * e = getenv("GLAS_P1"))   /* gpu/glas_p1_patch.py */
        ctx.p1 = (uint32_t)std::min(3, std::max(0, atoi(e)));
    if (char const * e = getenv("GLAS_P1_K"))
        ctx.p1_k = (uint32_t)std::min(64, std::max(1, atoi(e)));
    if (char const * e = getenv("GLAS_P1_CAP"))
        ctx.p1_cap = (uint32_t)std::max(1024, atoi(e));
    if (ctx.p1 && (!ctx.filter || ctx.rhits)) {
        fprintf(stderr, "# GLAS_P1=%u needs the GPU filter with GLAS_RHITS=0: GLAS_P1 forced off\\n", ctx.p1);
        ctx.p1 = 0;
    }
''')
sub('''        if (ctx.region_rx == 1)   /* GLAS_REGION_RX lists */''', '''        if (ctx.p1 & 1)   /* GLAS_P1 lists + counts, per parity */
            b += 2 * ((uint64_t)std::min<uint64_t>(ctx.p1_cap, area / 64) * (ctx.p1_k + 1) * 4);
        if (ctx.p1)       /* GLAS_P1 counters, pass list */
            b += 2 * 16 + ((ctx.p1 & 2) ? area / 64 * 4 : 0);
        if (ctx.region_rx == 1)   /* GLAS_REGION_RX lists */''')
sub('''    ctx.tiny_grid = full_grid(tiny_mask_kernel, 256, 0);''', '''    ctx.tiny_grid = full_grid(tiny_mask_kernel, 256, 0);
    if (ctx.p1) {
        ctx.p1_cap = std::min(ctx.p1_cap, ctx.max_survivors);
        ctx.p1_rat_grid = full_grid(p1_rat_kernel, 256, 0);
    }''')
sub('''            if (ctx.rhits) {
                CUDA_CHECK(cudaMalloc(&s.hits_r[par],''', '''            if (ctx.p1 & 1) {
                CUDA_CHECK(cudaMalloc(&s.p1_cnt[par], (uint64_t)ctx.p1_cap * 4));
                CUDA_CHECK(cudaMalloc(&s.p1_list[par], (uint64_t)ctx.p1_cap * ctx.p1_k * 4));
            }
            if (ctx.p1)
                CUDA_CHECK(cudaMalloc(&s.p1_ctr[par], 4 * 4));
            if (ctx.rhits) {
                CUDA_CHECK(cudaMalloc(&s.hits_r[par],''')
sub('''            CUDA_CHECK(cudaMalloc(&s.fcnt, 2 * 4));
        }
    }
    lap("malloc");''', '''            CUDA_CHECK(cudaMalloc(&s.fcnt, 2 * 4));
        }
        if (ctx.p1) {
            if (ctx.p1 & 2)
                CUDA_CHECK(cudaMalloc(&s.p1_pass, (uint64_t)ctx.max_survivors * 4));
            CUDA_CHECK(cudaMallocHost(&s.h_p1, 2 * 4 * 4));
        }
    }
    lap("malloc");''')
sub('''    ctx.free_mask = (1u << ctx.n_slots) - 1;
''', '''    ctx.free_mask = (1u << ctx.n_slots) - 1;
    if (ctx.p1) {
        uint64_t const extra = (ctx.p1 & 1 ? 2 * (uint64_t)ctx.p1_cap * (ctx.p1_k + 1) * 4 : 0) +
                               (ctx.p1 & 2 ? (uint64_t)ctx.max_survivors * 4 : 0) + 32;
        fprintf(stderr, "# GLAS_P1=%u on: layout %d cascade %d, list K %u x cap %u survivors (max_survivors %u), "
                "+%.1f MB/slot; slots %d, %.3f GB/slot, chunk 2^%d\\n", ctx.p1, (int)(ctx.p1 & 1),
                (int)(ctx.p1 >> 1 & 1), ctx.p1_k, ctx.p1_cap, ctx.max_survivors, extra / 1048576.0, ctx.n_slots,
                slot_bytes(chunk) / 1073741824.0, 63 - __builtin_clzll(ctx.chunk_area));
        atexit([] {
            auto v = [](int k) { return (unsigned long long)ctx.p1_stat[k].load(); };
            fprintf(stderr, "# GLAS_P1 stats: chunks %llu survivors %llu rational-pass %llu (f0 %.4f) rational-reject "
                    "%llu list-overflow-survivors %llu over-cap-survivors %llu flat-hits %llu "
                    "cascade-cpu-fallbacks %llu rational-mr-reject %llu\\n", v(0), v(1), v(2),
                    v(1) ? (double)v(2) / v(1) : 0.0, v(3), v(4), v(5), v(6), v(7), v(8));
        });
    }
''')

# launch(): per-parity task fields, clears, side-0-only tiny/resieve under the cascade
sub('''        assert(t.nregions >= 1 && t.area % SREG == 0);
''', '''        assert(t.nregions >= 1 && t.area % SREG == 0);
        t.p1 = ctx.p1;   /* GLAS_P1: this parity's buffers */
        if (ctx.p1) {
            t.p1_k = ctx.p1_k;
            t.p1_cap = ctx.p1_cap;
            t.p1_cnt = s.p1_cnt[par];
            t.p1_list = s.p1_list[par];
            t.p1_pass = s.p1_pass;
            t.p1_ctr = s.p1_ctr[par];
        }
''')
sub('''        if (ctx.rhits)
            CUDA_CHECK(cudaMemsetAsync(s.hit_rcnt[par], 0, t.nregions * 4,
                                       st));
''', '''        if (ctx.rhits)
            CUDA_CHECK(cudaMemsetAsync(s.hit_rcnt[par], 0, t.nregions * 4,
                                       st));
        if (ctx.p1) {
            CUDA_CHECK(cudaMemsetAsync(s.p1_ctr[par], 0, 4 * 4, st));
            if (ctx.p1 & 1)
                CUDA_CHECK(cudaMemsetAsync(s.p1_cnt[par], 0, (uint64_t)ctx.p1_cap * 4, st));
        }
''')
sub('''        tiny_mask_kernel<<<ctx.tiny_grid, 256, 0, st>>>(t, s.survivors[par],''', '''        gpu_task t_tiny = t;   /* GLAS_P1 cascade: side 0 here, side 1 after the rational test */
        if (ctx.p1 & 2)
            t_tiny.p1_sides = 1;
        tiny_mask_kernel<<<ctx.tiny_grid, 256, 0, st>>>(t_tiny, s.survivors[par],''')
sub('''        for (int side = 0; side < 2; side++)
            resieve_small_kernel<<<ctx.rs_grid, 256, 0, st>>>(''', '''        for (int side = 0; side < ((ctx.p1 & 2) ? 1 : 2); side++)
            resieve_small_kernel<<<ctx.rs_grid, 256, 0, st>>>(''')

# chunk loop: the cascade between the hit sort and the filter
CPU_FB = '''            CUDA_CHECK(cudaStreamSynchronize(st));
            CUDA_CHECK(cudaStreamSynchronize(s.out_stream));
            release_slot(slot);
            fprintf(stderr, "# GPU overflow at q=%lu; falling back to CPU\\n",
                    (unsigned long)task.q);'''
assert s.count(CPU_FB) == 1
fb_a = s.index(CPU_FB)
fb_b = s.index('            return;\n        }\n', fb_a) + len('            return;\n')
FALLBACK_BODY = s[fb_a:fb_b]
assert 'sieve_one_cpu(task, fb, tt)' in FALLBACK_BODY and FALLBACK_BODY.count('return;') == 1
FALLBACK_P1 = FALLBACK_BODY.replace('"# GPU overflow at q=%lu; falling back to CPU\\n"',
                                    '"# GPU overflow (GLAS_P1 cascade) at q=%lu; falling back to CPU\\n"')
FALLBACK_P1 = FALLBACK_P1.replace('            release_slot(slot);\n',
                                  '            release_slot(slot);\n            ctx.p1_stat[7]++;\n')
FALLBACK_P1 = '\n'.join(('    ' + l) if l else l for l in FALLBACK_P1.split('\n'))

sub('''        uint32_t n_out = ctx.filter ? 0 : n_surv;
        uint32_t n_hits_out = ctx.filter ? 0 : n_hits;
        if (ctx.filter && n_surv) {
            if (n_hits && !ctx.rhits) {
                size_t tmp_bytes = ctx.sort_temp_bytes;
                CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
                    s.sort_temp, tmp_bytes, s.hits[par], s.hits_sorted,
                    n_hits, 0, 64, s.out_stream));
            }
''', '''        uint32_t n_out = ctx.filter ? 0 : n_surv;
        uint32_t n_hits_out = ctx.filter ? 0 : n_hits;
        gpu_task tcf = tc;          /* GLAS_P1: the filter's view of the chunk */
        uint32_t n_hits_f = n_hits;
        tcf.p1_nA = n_hits;
        if (ctx.filter && n_surv) {
            if (n_hits && !ctx.rhits) {
                size_t tmp_bytes = ctx.sort_temp_bytes;
                CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
                    s.sort_temp, tmp_bytes, s.hits[par], s.hits_sorted,
                    n_hits, 0, 64, s.out_stream));
            }
            if (ctx.p1 & 2) {
                /* GLAS_P1 cascade: exact rational test over every survivor (sorted segment A = the flat hits so
                 * far), then side-1 tiny masks and side-1 small resieve over the pass list only */
                p1_rat_kernel<<<std::min(ctx.p1_rat_grid, (n_surv + 255) / 256), 256, 0, s.out_stream>>>(
                    tcf, s.survivors[par], n_surv, s.tiny[par], s.hits_sorted, n_hits);
                gpu_task t1 = tc;
                t1.p1_uselist = 1;
                t1.p1_sides = 2;
                tiny_mask_kernel<<<ctx.tiny_grid, 256, 0, s.out_stream>>>(t1, s.survivors[par], s.counters[par],
                                                                         ctx.max_survivors, s.tiny[par]);
                resieve_small_kernel<<<ctx.rs_grid, 256, 0, s.out_stream>>>(
                    t1, ctx.small[1], s.rs_idx[1], s.rs_cnt + 1, s.roots[1], 1, s.survivors[par],
                    s.counters[par], ctx.max_survivors, ctx.max_hits, s.hits[par], s.counters[par] + 1,
                    nullptr, nullptr, nullptr);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(s.h_counters, s.counters[par], 3 * 4,
                                           cudaMemcpyDeviceToHost, s.out_stream));
                {
                    trace_scope const tr(TR_SYNC_CTR);
                    CUDA_CHECK(cudaStreamSynchronize(s.out_stream));
                }
                uint32_t const nh2 = s.h_counters[1];
                if (s.h_counters[2] || nh2 > ctx.max_hits) {
''' + FALLBACK_P1 + '''                }
                if (nh2 > n_hits) {   /* sorted segment B: the side-1 small hits of the pass list */
                    size_t tmp_bytes = ctx.sort_temp_bytes;
                    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
                        s.sort_temp, tmp_bytes, s.hits[par] + n_hits, s.hits_sorted + n_hits,
                        nh2 - n_hits, 0, 64, s.out_stream));
                }
                n_hits_f = nh2;
                tcf.p1_uselist = 1;
            }
''')
sub('''                tc, s.survivors[par], n_surv, s.tiny[par], s.hits_sorted,
                n_hits,''', '''                tcf, s.survivors[par], n_surv, s.tiny[par], s.hits_sorted,
                n_hits_f,''', 2)
sub('''        CUDA_CHECK(cudaEventRecord(s.odone[par], s.out_stream));
        if (ctx.pipeline)
            launch_next();''', '''        if (ctx.p1)
            CUDA_CHECK(cudaMemcpyAsync(s.h_p1 + 4 * par, s.p1_ctr[par], 4 * 4,
                                       cudaMemcpyDeviceToHost, s.out_stream));
        CUDA_CHECK(cudaEventRecord(s.odone[par], s.out_stream));
        if (ctx.pipeline)
            launch_next();''')
sub('''        auto rc = std::make_shared<raw_chunk>();''', '''        if (ctx.p1) {   /* GLAS_P1 counters of this chunk (out stream synchronized above) */
            uint32_t const * const c = s.h_p1 + 4 * par;
            ctx.p1_stat[0]++;
            ctx.p1_stat[1] += n_surv;
            ctx.p1_stat[2] += c[2];
            ctx.p1_stat[3] += c[3];
            ctx.p1_stat[4] += c[0];
            if ((ctx.p1 & 1) && n_surv > ctx.p1_cap)
                ctx.p1_stat[5] += n_surv - ctx.p1_cap;
            ctx.p1_stat[6] += n_hits_f;
            ctx.p1_stat[8] += c[1];
        }
        auto rc = std::make_shared<raw_chunk>();''')

open(p, 'w').write(s); print('patched')
