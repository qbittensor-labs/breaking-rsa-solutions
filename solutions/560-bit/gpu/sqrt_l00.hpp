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

/* sqrt_l00.hpp -- fast level-00 range product for the CADO algebraic square root (RSA-560 campaign, audit s21).
 *
 * Copied next to sqrt/sqrt.cpp and included by gpu/sqrt_l00_patch.py; selected at run time by SQRT_L00=2
 * (SQRT_L00=1 runs CADO's own product with the not-parallel mpz_poly interface; unset = CADO default).
 *
 * l00::range_product(res, ab, F) returns the product of the leaves (a - b*alpha) of one range of (a,b) pairs as
 * the cxx_mpz_polymodF that CADO's product_of_leaves(ab, cxx_mpz_polymodF_functions) returns: same pseudo-reduction
 * modulo the non-monic F (each reduction step multiplies the lower coefficients by lc(F), subtracts c_k * F shifted
 * and counts v++), so (p, v) is the same element p / lc^v with the same v, hence the same polynomial p (1, alpha, ..,
 * alpha^(d-1) is a basis) whatever the shape of the tree. Differences from CADO's path:
 *   - one thread, no OpenMP region per product (CADO's parallel-interface products open ~30 nested regions each),
 *   - no allocation per product: a binary-counter tree whose cells and scratch are thread_local and reused across
 *     ranges (capacity only grows),
 *   - the first level (two leaves) in __int128,
 *   - every polynomial product by Kronecker substitution: the coefficients are packed at limb-aligned slots of w
 *     limbs (positive and negative coefficients into two buffers, then one subtraction), one mpz_mul, and a signed
 *     (balanced-digit) unpack; replaces Toom-Cook's 9 mpz_mul plus ~100 linear evaluation/interpolation passes.
 * Requires deg F <= 8 (product degrees <= 2 deg F - 2 <= 14 slots); the sqrt keeps CADO's path otherwise.
 */
#ifndef SQRT_L00_HPP
#define SQRT_L00_HPP

#include <gmp.h>
#include "mpz_poly.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace l00 {

static constexpr int MAXC = 16;     /* coefficients of a product before reduction (deg <= 2*(8-1)) */

struct node {
    int deg = -1;                   /* -1: the empty cell */
    unsigned long v = 0;
    mpz_t c[MAXC];
    node() { for (auto & x : c) mpz_init(x); }
    ~node() { for (auto & x : c) mpz_clear(x); }
    node(node const &) = delete;
    node & operator=(node const &) = delete;
    void swap(node & o) { std::swap(deg, o.deg); std::swap(v, o.v); for (int i = 0; i < MAXC; i++) mpz_swap(c[i], o.c[i]); }
};

/* products whose packed digits exceed this many bits use CADO's Toom-Cook (mpz_poly_mul, not parallel): Kronecker
 * substitution doubles the operand size (each slot holds a product-size digit), which costs more than Toom-Cook's
 * linear passes once GMP is in its FFT range. SQRT_L00_KS_BITS overrides (log2 bits). */
static const size_t ks_max_bits = [] {
    const char * e = getenv("SQRT_L00_KS_BITS");
    return (size_t) 1 << (e ? atoi(e) : 12);
}();

struct scratch {
    std::vector<mp_limb_t> pos, neg;
    cxx_mpz_poly pa, pb, pr;
    mpz_t X, Y, Z, t;
    scratch() { mpz_init(X); mpz_init(Y); mpz_init(Z); mpz_init(t); }
    ~scratch() { mpz_clear(X); mpz_clear(Y); mpz_clear(Z); mpz_clear(t); }
};

/* X <- sum c[i] 2^(64 w i), i <= deg, signed coefficients */
static inline void pack(mpz_ptr X, node const & a, mp_size_t w, scratch & s)
{
    const mp_size_t n = (mp_size_t) (a.deg + 1) * w;
    if ((mp_size_t) s.pos.size() < n) { s.pos.resize(n); s.neg.resize(n); }
    std::memset(s.pos.data(), 0, n * sizeof(mp_limb_t));
    std::memset(s.neg.data(), 0, n * sizeof(mp_limb_t));
    bool anyneg = false, anypos = false;
    for (int i = 0; i <= a.deg; i++) {
        const int sg = mpz_sgn(a.c[i]);
        if (!sg) continue;
        mp_limb_t * dst = (sg > 0 ? s.pos.data() : s.neg.data()) + i * w;
        std::memcpy(dst, mpz_limbs_read(a.c[i]), mpz_size(a.c[i]) * sizeof(mp_limb_t));
        (sg > 0 ? anypos : anyneg) = true;
    }
    mpz_t P, N;
    mp_size_t np = n, nn = n;
    while (np > 0 && s.pos[np - 1] == 0) np--;
    while (nn > 0 && s.neg[nn - 1] == 0) nn--;
    if (!anyneg) { mpz_set(X, mpz_roinit_n(P, s.pos.data(), np)); return; }
    if (!anypos) { mpz_set(X, mpz_roinit_n(N, s.neg.data(), nn)); mpz_neg(X, X); return; }
    mpz_sub(X, mpz_roinit_n(P, s.pos.data(), np), mpz_roinit_n(N, s.neg.data(), nn));
}

/* r.c[0..nc-1] <- the balanced base-2^(64w) digits of Z (each |digit| < 2^(64w-1)) */
static inline void unpack(node & r, mpz_srcptr Z, mp_size_t w, int nc)
{
    const int sg = mpz_sgn(Z);
    const mp_limb_t * z = mpz_limbs_read(Z);
    const mp_size_t zn = mpz_size(Z);
    mp_limb_t carry = 0;
    const mp_limb_t topbit = (mp_limb_t) 1 << (GMP_NUMB_BITS - 1);
    for (int k = 0; k < nc; k++) {
        mp_limb_t * d = mpz_limbs_write(r.c[k], w);
        const mp_size_t lo = k * w;
        const mp_size_t have = zn > lo ? std::min<mp_size_t>(w, zn - lo) : 0;
        if (have) std::memcpy(d, z + lo, have * sizeof(mp_limb_t));
        if (have < w) std::memset(d + have, 0, (w - have) * sizeof(mp_limb_t));
        mp_limb_t cout = carry ? mpn_add_1(d, d, w, 1) : 0;
        bool negdigit = (d[w - 1] & topbit) != 0;
        if (negdigit) mpn_neg(d, d, w);         /* |digit| = 2^(64w) - t */
        mp_size_t m = w;
        while (m > 0 && d[m - 1] == 0) m--;
        /* digit sign: negdigit, then the sign of Z */
        const bool neg = negdigit ^ (sg < 0);
        mpz_limbs_finish(r.c[k], neg ? -m : m);
        carry = cout + (negdigit ? 1 : 0);
    }
    /* everything consumed: no carry left, no limb beyond the last slot */
    if (carry != 0 || zn > (mp_size_t) nc * w) {
        /* cannot happen when 64w - 1 >= bits(a) + bits(b) + log2(terms); keep the check cheap and fatal */
        abort();
    }
    r.deg = nc - 1;
    while (r.deg >= 0 && mpz_sgn(r.c[r.deg]) == 0) r.deg--;
}

static inline size_t maxbits(node const & a)
{
    size_t m = 0;
    for (int i = 0; i <= a.deg; i++) { const size_t b = mpz_sizeinbase(a.c[i], 2); if (b > m) m = b; }
    return m;
}

/* CADO's mpz_poly_reducemodF, step by step (same v, same polynomial) */
static inline void reduce(node & r, mpz_srcptr const * F, int d, bool lc_is_1, scratch & s)
{
    while (r.deg >= d) {
        const int k = r.deg;
        if (!lc_is_1) {
            r.v++;
            for (int i = 0; i < k; i++) mpz_mul(r.c[i], r.c[i], F[d]);
        }
        for (int i = 0; i < d; i++) mpz_submul(r.c[k - d + i], r.c[k], F[i]);
        /* cleandeg(k-1) */
        int nd = k - 1;
        while (nd >= 0 && mpz_sgn(r.c[nd]) == 0) nd--;
        r.deg = nd;
    }
    (void) s;
}

/* r <- a * b mod F (r distinct from a and b) */
static inline void mul(node & r, node const & a, node const & b, mpz_srcptr const * F, int d, bool lc_is_1, scratch & s)
{
    r.v = a.v + b.v;
    if (a.deg == 0 || b.deg == 0) {
        node const & c = a.deg == 0 ? a : b;
        node const & p = a.deg == 0 ? b : a;
        for (int i = 0; i <= p.deg; i++) mpz_mul(r.c[i], p.c[i], c.c[0]);
        r.deg = p.deg;
    } else {
        const int terms = std::min(a.deg, b.deg) + 1;
        int lg = 0; while ((1 << lg) < terms) lg++;
        const size_t bits = maxbits(a) + maxbits(b) + lg + 1;   /* |digit| < 2^bits <= 2^(64w-1) */
        if (bits > ks_max_bits) {
            /* Toom-Cook on the coefficients, which are swapped in and out (no copy) */
            mpz_poly_realloc(s.pa, a.deg + 1); mpz_poly_realloc(s.pb, b.deg + 1);
            for (int i = 0; i <= a.deg; i++) mpz_swap(s.pa->_coeff[i], const_cast<mpz_ptr>(a.c[i]));
            for (int i = 0; i <= b.deg; i++) mpz_swap(s.pb->_coeff[i], const_cast<mpz_ptr>(b.c[i]));
            s.pa->deg = a.deg; s.pb->deg = b.deg;
            mpz_poly_mul(s.pr, s.pa, s.pb);
            for (int i = 0; i <= a.deg; i++) mpz_swap(s.pa->_coeff[i], const_cast<mpz_ptr>(a.c[i]));
            for (int i = 0; i <= b.deg; i++) mpz_swap(s.pb->_coeff[i], const_cast<mpz_ptr>(b.c[i]));
            r.deg = s.pr->deg;
            for (int i = 0; i <= r.deg; i++) mpz_swap(r.c[i], s.pr->_coeff[i]);
            reduce(r, F, d, lc_is_1, s);
            return;
        }
        const mp_size_t w = (mp_size_t) (bits / GMP_NUMB_BITS + 1);
        pack(s.X, a, w, s);
        pack(s.Y, b, w, s);
        mpz_mul(s.Z, s.X, s.Y);
        unpack(r, s.Z, w, a.deg + b.deg + 1);
    }
    reduce(r, F, d, lc_is_1, s);
}

static inline void set_i128(mpz_ptr z, __int128 x)
{
    const bool neg = x < 0;
    unsigned __int128 u = neg ? (unsigned __int128) 0 - (unsigned __int128) x : (unsigned __int128) x;
    mp_limb_t * p = mpz_limbs_write(z, 2);
    p[0] = (mp_limb_t) u; p[1] = (mp_limb_t) (u >> 64);
    const mp_size_t n = p[1] ? 2 : (p[0] ? 1 : 0);
    mpz_limbs_finish(z, neg ? -n : n);
}

struct state {
    std::vector<node *> cells;      /* cells[i]: product of 2^i units, or empty */
    node unit, tmp, acc;
    scratch s;
    ~state() { for (auto * c : cells) delete c; }
};

/* per-thread storage, kept from one range to the next (its capacity only grows) until release() */
static thread_local state * tls = nullptr;

/* frees the calling thread's storage (call it once per thread after the last range) */
static inline void release() { delete tls; tls = nullptr; }

/* the product of the leaves (a - b x) for the pairs ab[0..n-1], into (deg, coeffs, v) of res, as
 * mpz_t coefficients (res.c[0..res.deg]) */
template<typename AB>
static void range_product(node & res, AB const & ab, mpz_srcptr const * F, int d)
{
    if (!tls) tls = new state;
    state & st = *tls;
    const bool lc_is_1 = mpz_cmp_ui(F[d], 1) == 0;
    for (auto * c : st.cells) c->deg = -1;
    auto push = [&](node & u) {
        /* binary counter: merge while the cell is occupied */
        size_t i = 0;
        for (;; i++) {
            if (i == st.cells.size()) st.cells.push_back(new node);
            node & c = *st.cells[i];
            if (c.deg < 0) { c.swap(u); return; }
            mul(st.tmp, c, u, F, d, lc_is_1, st.s);
            c.deg = -1;
            u.swap(st.tmp);
        }
    };
    const size_t n = ab.size();
    size_t i = 0;
    for (; i + 1 < n; i += 2) {
        /* two leaves in __int128 (|a|, b < 2^63: each product below 2^126, sums of two below 2^127) */
        const int64_t a0 = ab[i].first, a1 = ab[i + 1].first;
        const uint64_t b0 = ab[i].second, b1 = ab[i + 1].second;
        node & u = st.unit;
        u.v = 0;
        if ((b0 >> 63) || (b1 >> 63) || a0 == INT64_MIN || a1 == INT64_MIN) {
            /* out of the __int128 range: leaves as mpz, one generic product */
            node & x = st.acc;
            mpz_set_si(x.c[0], a0); x.deg = 0; x.v = 0;
            if (b0) { mpz_set_ui(x.c[1], b0); mpz_neg(x.c[1], x.c[1]); x.deg = 1; }
            node & y = st.tmp;
            mpz_set_si(y.c[0], a1); y.deg = 0; y.v = 0;
            if (b1) { mpz_set_ui(y.c[1], b1); mpz_neg(y.c[1], y.c[1]); y.deg = 1; }
            mul(u, x, y, F, d, lc_is_1, st.s);
        } else {
            /* (a0 - b0 x)(a1 - b1 x) = a0 a1 - (a0 b1 + a1 b0) x + b0 b1 x^2 */
            const __int128 c0 = (__int128) a0 * a1;
            const __int128 c1 = -((__int128) a0 * (__int128) b1 + (__int128) a1 * (__int128) b0);
            const __int128 c2 = (__int128) ((unsigned __int128) b0 * b1);
            set_i128(u.c[0], c0); set_i128(u.c[1], c1); set_i128(u.c[2], c2);
            u.deg = 2;
            while (u.deg >= 0 && mpz_sgn(u.c[u.deg]) == 0) u.deg--;
            if (d <= 2) reduce(u, F, d, lc_is_1, st.s);
        }
        push(u);
    }
    if (i < n) {
        node & u = st.unit;
        u.v = 0;
        mpz_set_si(u.c[0], ab[i].first); u.deg = 0;
        if (ab[i].second) { mpz_set_ui(u.c[1], ab[i].second); mpz_neg(u.c[1], u.c[1]); u.deg = 1; }
        push(u);
    }
    /* fold the occupied cells, small to large */
    bool have = false;
    for (auto * c : st.cells) {
        if (c->deg < 0) continue;
        if (!have) { st.acc.swap(*c); have = true; }
        else { mul(st.tmp, *c, st.acc, F, d, lc_is_1, st.s); st.acc.swap(st.tmp); }
        c->deg = -1;
    }
    if (!have) { mpz_set_ui(st.acc.c[0], 1); st.acc.deg = 0; st.acc.v = 0; }
    /* hand the result over (swap: the caller's node gets the storage) */
    res.swap(st.acc);
}

} /* namespace l00 */

#endif
