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

"""glas_narrow_patch.py <gpu-siever/src/sieve_core_gpu.cu> -- proposal 2 of docs/rsa560_codex_review.md (codex-review):
bounded-width exact norms and integer exact trial division in the exact-norm survivor filter (filter_kernel).

Applies ON TOP of the full production patch stack (sandbox_bootstrap.sh order: adjust, coprime, prefetch, xfuse, region,
xfgrid, session); none of those touch filter_kernel, whose source is pinned here by sha256 (fail-closed on any drift).
The production filter_kernel is left byte-for-byte untouched; this adds a templated copy filter_kernel_n<W0, W1, IDIV>
(W0/W1 = 64-bit limbs of the side-0/side-1 norm temporaries, IDIV = Hensel exact division) and a per-task dispatcher.

  GLAS_NARROW=1  per special-q, the host computes with GMP the exact bound
                   Amax = (I/2)|a0| + (J-1)|a1|,  Bmax = (I/2)|b0| + (J-1)|b1|           (each clamped to >= 1)
                   U_side = sum_k |c_side[k]| Amax^k Bmax^(d-k)
                 which bounds every term, every partial product and every positive/negative partial sum of the
                 term-by-term norm (and every quotient after trial division), L_side = ceil(bits(U_side)/64), and
                 launches the cheapest specialization with W0 >= L0, W1 >= L1 among (2,4) (3,4) (4,4) (4,6);
                 else (any L > 6, or Amax/Bmax >= 2^63 where the kernel's int64 a, b could wrap) the 512-bit
                 (NORM_LIMBS=8) fallback, i.e. the production kernel itself when GLAS_IDIV=0.
  GLAS_IDIV=1    odd divisors < 2^32 are divided out by checked low-to-high (Hensel) exact division in radix 2^32
                 (inverse mod 2^32 by Newton, final carry == 0 <=> p | N, quotient committed only then); p = 2 by a
                 trailing-zero shift; other even p and q >= 2^32 keep the production FP64 / u128 routines.
  unset / 0      both off: the production kernel and launch, unchanged (no banner, no extra work).

Banners (only when a variant is on): '# GLAS_NARROW=n GLAS_IDIV=i' + one line per compiled specialization with
registers, local memory, occupancy grid; at exit '# GLAS_NARROW dispatch: tasks W0xW1 N ... fallback N (bound>512 N,
wrap N)' and '# GLAS_NARROW bits: side s max L'.  The CPU reference test gpu/glas_narrow_test.py compiles the
'GLAS_NARROW device core' and 'GLAS_NARROW host bound' blocks of the patched file with g++ and checks them against
Python integers."""
import hashlib, sys
p = sys.argv[1]; s = open(p).read()
if 'GLAS_NARROW' in s: print('already patched'); sys.exit(0)
assert 'fill_xfuse_kernel' in s and 'GLAS_XFUSE_GRID' in s, 'needs the production patch stack (xfuse, xfgrid) first'

def sub(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (s.count(a), a[:80])
    s = s.replace(a, b)

# ---- pin the production filter kernel (the template below is a reviewed transformation of exactly this text) ----
KA = '__global__ void filter_kernel(gpu_task t, uint32_t const * survivors,'
KB = '/* The primality tests the filter deferred'
assert s.count(KA) == 1 and s.count(KB) == 1
ka, kb = s.index(KA), s.index(KB)
K = s[ka:kb]
assert hashlib.sha256(K.encode()).hexdigest() == '2f39196df2f4215c36954175614ded6a103cd314391a07ed62dca5b2ec20092d', \
    'filter_kernel source differs from the reviewed production text'

DEVICE_CORE = r'''
/* ---- GLAS_NARROW device core begin ----
 * Width-templated copies of the wide_* helpers (W 64-bit limbs, every loop
 * fully unrolled, no runtime-indexed limb access) and the IDIV exact
 * divider.  The callers guarantee every value fits in W limbs (host bound
 * narrow_limbs_side); then each routine computes exactly what its
 * NORM_LIMBS twin computes. */
template <int W> struct widen {
    uint64_t v[W];
};

template <int W> struct wtag {
    static constexpr int value = W;
};

template <int W>
__device__ __forceinline__ int widen_cmp(widen<W> const & a, widen<W> const & b)
{
    int r = 0;
#pragma unroll
    for (int i = 0; i < W; i++)
        if (a.v[i] != b.v[i])
            r = a.v[i] < b.v[i] ? -1 : 1;
    return r;
}

template <int W>
__device__ __forceinline__ void widen_add(widen<W> & a, widen<W> const & b)
{
    uint64_t carry = 0;
#pragma unroll
    for (int i = 0; i < W; i++) {
        u128 const s = (u128)a.v[i] + b.v[i] + carry;
        a.v[i] = (uint64_t)s;
        carry = (uint64_t)(s >> 64);
    }
}

template <int W>
__device__ __forceinline__ void widen_sub(widen<W> & a, widen<W> const & b)   /* needs a >= b */
{
    uint64_t borrow = 0;
#pragma unroll
    for (int i = 0; i < W; i++) {
        u128 const s = (u128)a.v[i] - b.v[i] - borrow;
        a.v[i] = (uint64_t)s;
        borrow = (uint64_t)(s >> 64) & 1;
    }
}

template <int W>
__device__ __forceinline__ void widen_mul_u64(widen<W> & a, uint64_t m)
{
    uint64_t carry = 0;
#pragma unroll
    for (int i = 0; i < W; i++) {
        u128 const p = (u128)a.v[i] * m + carry;
        a.v[i] = (uint64_t)p;
        carry = (uint64_t)(p >> 64);
    }
}

template <int W>
__device__ __forceinline__ uint64_t widen_divrem_u64(widen<W> & a, uint64_t p)
{
    uint64_t rem = 0;
#pragma unroll
    for (int i = W - 1; i >= 0; i--) {
        u128 const cur = (u128)rem << 64 | a.v[i];
        a.v[i] = (uint64_t)(cur / p);
        rem = (uint64_t)(cur % p);
    }
    return rem;
}

template <int W>
__device__ int widen_divide_out(widen<W> & n, uint64_t p)
{
    int k = 0;
    for (;;) {
        widen<W> q = n;
        if (widen_divrem_u64(q, p))
            return k;
        n = q;
        k++;
    }
}

template <int W>
__device__ __forceinline__ int widen_live(widen<W> const & a)
{
    int l = 0;
#pragma unroll
    for (int i = 0; i < W; i++)
        if (a.v[i])
            l = i + 1;
    return l;
}

template <int W>
__device__ __forceinline__ int widen_bits(widen<W> const & a)
{
    int b = 0;
#pragma unroll
    for (int i = 0; i < W; i++)
        if (a.v[i])
            b = 64 * (i + 1) - __clzll((long long)a.v[i]);
    return b;
}

template <int W>
__device__ __forceinline__ double widen_to_double(widen<W> const & a)
{
    double v = 0.0;
#pragma unroll
    for (int i = W - 1; i >= 0; i--)
        v = v * 18446744073709551616.0 + (double)a.v[i];
    return v;
}

/* production FP64-reciprocal division over the nl live limbs (as
 * wide_divrem_u32), predicated instead of runtime-bounded */
template <int W>
__device__ __forceinline__ uint32_t widen_divrem_u32(widen<W> const & n, int nl,
                                                     widen<W> & q, div32 d)
{
    uint64_t rem = 0;
#pragma unroll
    for (int i = W - 1; i >= 0; i--) {
        if (i < nl) {
            rem = rem << 32 | n.v[i] >> 32;
            uint64_t const qh = div32_step(rem, d);
            rem = rem << 32 | (uint32_t)n.v[i];
            uint64_t const ql = div32_step(rem, d);
            q.v[i] = qh << 32 | ql;
        }
    }
    return (uint32_t)rem;
}

/* nl -= 1 when the top live limb became zero (a quotient by p < 2^32
 * loses at most one 64-bit limb) */
template <int W>
__device__ __forceinline__ int widen_drop_top(widen<W> const & n, int nl)
{
    int r = nl;
#pragma unroll
    for (int i = 0; i < W; i++)
        if (i == nl - 1 && n.v[i] == 0)
            r = nl - 1;
    return r;
}

/* p^k exactly dividing n, n /= p^k; nl tracks the live limbs.  Same
 * contract as wide_divide_out_u32 (n == 0: returns 0, n untouched). */
template <int W, bool IDIV>
__device__ int widen_divide_out_u32(widen<W> & n, int & nl, uint32_t p)
{
    if (IDIV && (p & 1)) {
        /* Hensel / exact division, radix 2^32: q_j = (N_j - c) p^-1 mod
         * 2^32, c' = (p q_j + c) >> 32.  Invariant p Q = N + c 2^(32 L),
         * so the final c == 0 iff p | N, and then Q = N / p exactly. */
        uint32_t inv = (3u * p) ^ 2u;      /* p^-1 mod 2^5 */
        inv *= 2u - p * inv;               /* mod 2^10 */
        inv *= 2u - p * inv;               /* mod 2^20 */
        inv *= 2u - p * inv;               /* mod 2^40 > 2^32 */
        int k = 0;
        for (;;) {
            if (!nl)
                return k;
            widen<W> q;
            uint64_t c = 0;
#pragma unroll
            for (int i = 0; i < W; i++) {
                q.v[i] = 0;
                if (i < nl) {
                    uint32_t const q0 = ((uint32_t)n.v[i] - (uint32_t)c) * inv;
                    c = ((uint64_t)p * q0 + c) >> 32;
                    uint32_t const q1 = ((uint32_t)(n.v[i] >> 32) - (uint32_t)c) * inv;
                    c = ((uint64_t)p * q1 + c) >> 32;
                    q.v[i] = (uint64_t)q1 << 32 | q0;
                }
            }
            if (c)
                return k;
            n = q;
            nl = widen_drop_top(n, nl);
            k++;
        }
    }
    if (IDIV && p == 2) {
        if (!nl)
            return 0;
        int k = 0;
        while (n.v[0] == 0) {              /* 2^64 | n: drop one limb */
#pragma unroll
            for (int i = 0; i + 1 < W; i++)
                n.v[i] = n.v[i + 1];
            n.v[W - 1] = 0;
            k += 64;
        }
        int const r = __ffsll((long long)n.v[0]) - 1;
        if (r) {
#pragma unroll
            for (int i = 0; i < W; i++)
                n.v[i] = n.v[i] >> r | (i + 1 < W ? n.v[i + 1] << (64 - r) : 0);
        }
        nl = widen_live(n);
        return k + r;
    }
    div32 const d = make_div32(p);
    int k = 0;
    for (;;) {
        widen<W> q = n;
        if (!nl || widen_divrem_u32(n, nl, q, d))
            return k;
        n = q;
        nl = widen_drop_top(n, nl);
        k++;
    }
}

/* |F_side(a, b)| with b >= 0, term by term as dev_wide_norm, from the
 * low W limbs of each coefficient (|c_k| <= U_side < 2^(64 W)) */
template <int W>
__device__ void dev_widen_norm(int side, int d, int64_t a, uint64_t b,
                               widen<W> & out)
{
    uint64_t const aa = a < 0 ? (uint64_t)-a : (uint64_t)a;
    widen<W> pos = {}, neg = {};
    for (int k = 0; k <= d; k++) {
        widen<W> term;
#pragma unroll
        for (int i = 0; i < W; i++)
            term.v[i] = c_flt_c[side][k][i];
        for (int m = 0; m < k; m++)
            widen_mul_u64(term, aa);
        for (int m = k; m < d; m++)
            widen_mul_u64(term, b);
        bool const tneg = c_flt_neg[side][k] ^ (a < 0 && (k & 1));
        if (tneg)
            widen_add(neg, term);
        else
            widen_add(pos, term);
    }
    if (widen_cmp(pos, neg) >= 0) {
        widen_sub(pos, neg);
        out = pos;
    } else {
        widen_sub(neg, pos);
        out = neg;
    }
}
/* ---- GLAS_NARROW device core end ---- */
'''

LEFTOVER = r'''
/* dev_leftover over W limbs (W >= 2) */
template <int W>
__device__ leftover_state dev_leftover_n(widen<W> const & n, int side)
{
    int const s = widen_bits(n);
    if (s == 0 || s > (int)c_flt_mfb[side])
        return LEFT_BAD;
    if (s <= (int)c_flt_lpb[side])
        return LEFT_OK;
    double const nd = widen_to_double(n);
    double const B = (double)c_flt_lim[side];
    double kB = B * B;
    for (uint32_t klpb = c_flt_lpb[side]; klpb < (uint32_t)s;
         klpb += c_flt_lpb[side], kB *= B)
        if (nd < kB)
            return LEFT_BAD;
    if (c_flt_defer_mr)
        return LEFT_PENDING;
    return dev_prime_any((u128)n.v[1] << 64 | n.v[0]) ? LEFT_BAD : LEFT_OK;
}
'''

# ---- the templated kernel: production text with the side loop rewritten over per-side widths ----
KN = K
def ksub(a, b):
    global KN
    assert KN.count(a) == 1, (KN.count(a), a[:80])
    KN = KN.replace(a, b)
ksub('__global__ void filter_kernel(gpu_task t,', 'template <int W0, int W1, bool IDIV>\n__global__ void filter_kernel_n(gpu_task t,')
OLD_SIDE = '''        bool ok = true;
        for (int side = 0; side < 2 && ok; side++) {
            wide n;
            dev_wide_norm(side, t.degree[side], a, (uint64_t)b, n);
            int nl = wide_live(n);
            if (side == t.sq_side) {
                nq = t.q >> 32 ? wide_divide_out(n, t.q)
                               : wide_divide_out_u32(n, nl, (uint32_t)t.q);
                if (!nq) {
                    ok = false;
                    break;
                }
                nl = wide_live(n);
            }
            each_hit([&](uint64_t h) {
                if ((h & 1) != (uint64_t)side)
                    return;
                uint32_t const p = (uint32_t)h >> 1;
                if (side == t.sq_side && p == t.q)
                    return;
                record(p, wide_divide_out_u32(n, nl, p));
            });
            for (uint64_t m = tiny[2 * (uint64_t)idx + side]; m; m &= m - 1) {
                uint32_t const p = c_tiny_p[__ffsll((long long)m) - 1];
                record(p, wide_divide_out_u32(n, nl, p));
            }
            leftover_state const l = dev_leftover(n, side);
            ok = l != LEFT_BAD;
            pend |= (uint32_t)(l == LEFT_PENDING) << side;
            if (side == 0)
                nf0 = nf;
            lo[side][0] = n.v[0];
            lo[side][1] = n.v[1];
            for (int w = 2; w < NORM_LIMBS; w++)
                if (n.v[w])
                    assemble = false;
        }
        if (!ok)
            continue;
'''
NEW_SIDE = '''        bool ok = true;
        /* one side at width W (wtag<W>); same steps as the production loop body */
        auto side_pass = [&](auto wt, int side) {
            constexpr int W = decltype(wt)::value;
            widen<W> n;
            dev_widen_norm<W>(side, t.degree[side], a, (uint64_t)b, n);
            int nl = widen_live(n);
            if (side == t.sq_side) {
                nq = t.q >> 32 ? widen_divide_out(n, t.q)
                               : widen_divide_out_u32<W, IDIV>(n, nl, (uint32_t)t.q);
                if (!nq) {
                    ok = false;
                    return;
                }
                nl = widen_live(n);
            }
            each_hit([&](uint64_t h) {
                if ((h & 1) != (uint64_t)side)
                    return;
                uint32_t const p = (uint32_t)h >> 1;
                if (side == t.sq_side && p == t.q)
                    return;
                record(p, widen_divide_out_u32<W, IDIV>(n, nl, p));
            });
            for (uint64_t m = tiny[2 * (uint64_t)idx + side]; m; m &= m - 1) {
                uint32_t const p = c_tiny_p[__ffsll((long long)m) - 1];
                record(p, widen_divide_out_u32<W, IDIV>(n, nl, p));
            }
            leftover_state const l = dev_leftover_n(n, side);
            ok = l != LEFT_BAD;
            pend |= (uint32_t)(l == LEFT_PENDING) << side;
            if (side == 0)
                nf0 = nf;
            lo[side][0] = n.v[0];
            lo[side][1] = n.v[1];
#pragma unroll
            for (int w = 2; w < W; w++)
                if (n.v[w])
                    assemble = false;
        };
        side_pass(wtag<W0>{}, 0);
        if (ok)
            side_pass(wtag<W1>{}, 1);
        if (!ok)
            continue;
'''
ksub(OLD_SIDE, NEW_SIDE)
assert 'wide ' not in KN.replace('widen', '') and 'NORM_LIMBS' not in KN, 'production-width remnant in the template'

sub(KB, DEVICE_CORE + LEFTOVER + '\n' + KN + KB)

# ---- host: per-task exact GMP bound, dispatch table, banners, counters ----
HOST = r'''
/* ---- GLAS_NARROW host bound begin ----
 * Limbs needed by side `side` for every special-q point: U_side < 2^(64 L).
 * coeff[k] = |c_k| as NORM_LIMBS little-endian limbs.  Returns 99 when the
 * kernel's int64 a = i a0 + j a1 (or b) could wrap: only the 512-bit
 * fallback reproduces the production arithmetic there. */
static int narrow_limbs_side(uint64_t const (*coeff)[NORM_LIMBS], int d,
                             int64_t a0, int64_t b0, int64_t a1, int64_t b1,
                             int logI, uint32_t J, int * bits_out)
{
    mpz_t A, B, t, u, U;
    mpz_inits(A, B, t, u, U, (mpz_ptr)0);
    auto absmul_add = [&](mpz_t r, int64_t x0, int64_t x1) {
        mpz_set_si(t, x0);
        mpz_abs(t, t);
        mpz_mul_2exp(t, t, logI - 1);               /* (I/2) |x0| */
        mpz_set_si(u, x1);
        mpz_abs(u, u);
        mpz_mul_ui(u, u, J ? J - 1 : 0);             /* (J-1) |x1| */
        mpz_add(r, t, u);
    };
    absmul_add(A, a0, a1);
    absmul_add(B, b0, b1);
    int L = 99;
    if (mpz_sizeinbase(A, 2) <= 63 && mpz_sizeinbase(B, 2) <= 63) {
        if (mpz_cmp_ui(A, 1) < 0)
            mpz_set_ui(A, 1);
        if (mpz_cmp_ui(B, 1) < 0)
            mpz_set_ui(B, 1);
        mpz_set_ui(U, 0);
        for (int k = 0; k <= d; k++) {
            mpz_import(t, NORM_LIMBS, -1, sizeof(uint64_t), 0, 0, coeff[k]);
            mpz_pow_ui(u, A, k);
            mpz_mul(t, t, u);
            mpz_pow_ui(u, B, d - k);
            mpz_mul(t, t, u);
            mpz_add(U, U, t);
        }
        int const bits = mpz_sgn(U) ? (int)mpz_sizeinbase(U, 2) : 0;
        if (bits_out)
            *bits_out = bits;
        L = bits <= 64 ? 1 : (bits + 63) / 64;
    }
    mpz_clears(A, B, t, u, U, (mpz_ptr)0);
    return L;
}

/* specializations, cheapest first; the dispatcher takes the first with
 * W0 >= L0 and W1 >= L1, else the NORM_LIMBS fallback */
constexpr int NARROW_NV = 4;
constexpr int NARROW_W[NARROW_NV][2] = {{2, 4}, {3, 4}, {4, 4}, {4, 6}};

static int narrow_pick(int L0, int L1)
{
    for (int v = 0; v < NARROW_NV; v++)
        if (NARROW_W[v][0] >= L0 && NARROW_W[v][1] >= L1)
            return v;
    return NARROW_NV;   /* fallback */
}
/* ---- GLAS_NARROW host bound end ---- */

using filter_kernel_t = void (*)(gpu_task, uint32_t const *, uint32_t, uint64_t const *, uint64_t const *, uint32_t,
                                 uint64_t const *, uint32_t const *, uint64_t const *, uint32_t const *, uint32_t *,
                                 uint64_t *, uint32_t *, uint32_t *, uint32_t *, uint64_t *, uint32_t, uint64_t *);
static struct {
    bool narrow = false, idiv = false, on = false;
    uint64_t coeff[2][GLAS_MAX_DEGREE + 1][NORM_LIMBS];
    filter_kernel_t kern[NARROW_NV + 1];   /* [NARROW_NV] = fallback */
    uint32_t grid[NARROW_NV + 1];
    std::atomic<uint64_t> tasks[NARROW_NV + 1], chunks[NARROW_NV + 1];
    std::atomic<uint64_t> over512{0}, wrap{0};
    std::atomic<int> maxbits[2];
} nar;

template <bool IDIV>
static void narrow_table()
{
    nar.kern[0] = filter_kernel_n<2, 4, IDIV>;
    nar.kern[1] = filter_kernel_n<3, 4, IDIV>;
    nar.kern[2] = filter_kernel_n<4, 4, IDIV>;
    nar.kern[3] = filter_kernel_n<4, 6, IDIV>;
    nar.kern[NARROW_NV] = IDIV ? filter_kernel_n<NORM_LIMBS, NORM_LIMBS, true> : filter_kernel;
}

/* per task: the kernel index (only with nar.on) */
static int narrow_task_variant(gpu_task const & t)
{
    int L[2], bits[2] = {-1, -1};
    for (int side = 0; side < 2; side++)
        L[side] = narrow_limbs_side(nar.coeff[side], t.degree[side], t.a0, t.b0, t.a1, t.b1, t.logI, t.J,
                                    &bits[side]);
    int v = NARROW_NV;
    if (L[0] == 99 || L[1] == 99)
        nar.wrap++;
    else if (L[0] > NORM_LIMBS || L[1] > NORM_LIMBS)
        nar.over512++;
    else if (nar.narrow)
        v = narrow_pick(L[0], L[1]);
    for (int side = 0; side < 2; side++) {
        int m = nar.maxbits[side].load();
        while (bits[side] > m && !nar.maxbits[side].compare_exchange_weak(m, bits[side])) {}
    }
    nar.tasks[v]++;
    return v;
}
'''
sub('/* Blocks of `block` threads and `smem` bytes of dynamic shared memory', HOST + '\n/* Blocks of `block` threads and `smem` bytes of dynamic shared memory')

# host copy of the coefficient magnitudes (same array the device gets)
sub('''        CUDA_CHECK(cudaMemcpyToSymbol(c_flt_c, fc, sizeof fc));
''', '''        CUDA_CHECK(cudaMemcpyToSymbol(c_flt_c, fc, sizeof fc));
        memcpy(nar.coeff, fc, sizeof fc);   /* GLAS_NARROW host bound */
''')

# init: env, dispatch table, per-specialization occupancy grid + registers/local memory, exit summary
sub('''        ctx.filter_grid = full_grid(filter_kernel, 256, 0);
        if (char const * e = getenv("GLAS_FILTER_GRID"))
            ctx.filter_grid = atoi(e);
''', '''        ctx.filter_grid = full_grid(filter_kernel, 256, 0);
        if (char const * e = getenv("GLAS_FILTER_GRID"))
            ctx.filter_grid = atoi(e);
        nar.narrow = env_flag("GLAS_NARROW", false);
        nar.idiv = env_flag("GLAS_IDIV", false);
        nar.on = nar.narrow || nar.idiv;
        if (nar.on) {
            if (nar.idiv)
                narrow_table<true>();
            else
                narrow_table<false>();
            fprintf(stderr, "# GLAS_NARROW=%d GLAS_IDIV=%d (exact per-task GMP width bound; fallback %d limbs)\\n",
                    (int)nar.narrow, (int)nar.idiv, NORM_LIMBS);
            for (int v = 0; v <= NARROW_NV; v++) {
                cudaFuncAttributes fa;
                CUDA_CHECK(cudaFuncGetAttributes(&fa, nar.kern[v]));
                nar.grid[v] = full_grid(nar.kern[v], 256, 0);
                if (char const * e = getenv("GLAS_FILTER_GRID"))
                    nar.grid[v] = atoi(e);
                if (v < NARROW_NV)
                    fprintf(stderr, "# GLAS_NARROW kernel %dx%d: %d regs, %zu B local, grid %u%s\\n",
                            NARROW_W[v][0], NARROW_W[v][1], fa.numRegs, (size_t)fa.localSizeBytes, nar.grid[v],
                            nar.narrow ? "" : " (unused: GLAS_NARROW=0)");
                else
                    fprintf(stderr, "# GLAS_NARROW kernel fallback %dx%d%s: %d regs, %zu B local, grid %u (production %u)\\n",
                            NORM_LIMBS, NORM_LIMBS, nar.idiv ? " idiv" : " production", fa.numRegs,
                            (size_t)fa.localSizeBytes, nar.grid[v], ctx.filter_grid);
            }
            atexit([] {
                fprintf(stderr, "# GLAS_NARROW dispatch: tasks");
                for (int v = 0; v < NARROW_NV; v++)
                    fprintf(stderr, " %dx%d %lu", NARROW_W[v][0], NARROW_W[v][1], (unsigned long)nar.tasks[v].load());
                fprintf(stderr, " fallback %lu (bound>512 %lu, wrap %lu); filter launches",
                        (unsigned long)nar.tasks[NARROW_NV].load(), (unsigned long)nar.over512.load(),
                        (unsigned long)nar.wrap.load());
                for (int v = 0; v <= NARROW_NV; v++)
                    fprintf(stderr, " %lu", (unsigned long)nar.chunks[v].load());
                fprintf(stderr, "\\n# GLAS_NARROW bits: side 0 max U %d, side 1 max U %d\\n", nar.maxbits[0].load(),
                        nar.maxbits[1].load());
            });
        }
''')

# per task: pick the variant once (host, before the chunk loop)
sub('''    t.skip = ctx.skip;

    trace_scope const tr_feed(TR_FEED);''', '''    t.skip = ctx.skip;
    int const nar_v = nar.on && ctx.filter ? narrow_task_variant(t) : -1;   /* GLAS_NARROW / GLAS_IDIV */

    trace_scope const tr_feed(TR_FEED);''')

# launch: the selected specialization (production launch untouched when both variants are off)
LA = '''            uint32_t const fgrid =
                std::min(ctx.filter_grid, (n_surv + 255) / 256);
            filter_kernel<<<fgrid, 256, 0, s.out_stream>>>('''
LB = '''s.h_ffac, s.h_flo, ctx.fcand_cap, s.h_hits);
'''
assert s.count(LA) == 1 and s.count(LB) == 1 and s.index(LA) < s.index(LB)
la, lb = s.index(LA), s.index(LB) + len(LB)
ARGS = s[la + len(LA):lb]
assert ARGS.count(';') == 1 and 'tc, s.survivors[par]' in ARGS
s = s[:la] + '''            if (nar_v >= 0) {   /* GLAS_NARROW / GLAS_IDIV specialization */
                nar.chunks[nar_v]++;
                nar.kern[nar_v]<<<std::min(nar.grid[nar_v], (n_surv + 255) / 256), 256, 0, s.out_stream>>>(''' + ARGS + \
    '''            } else {
''' + LA + ARGS + '''            }
''' + s[lb:]

open(p, 'w').write(s); print('patched')
