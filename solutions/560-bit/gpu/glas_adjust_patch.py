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

"""Apply the adjust-strategy port to a Cognition gpu-siever/src/glas.cpp (W2-1). Usage: glas_adjust_patch.py <glas.cpp>.
Adds GLAS_ADJUST=2|3 (default 0 = the stock fixed basis): per special-q, the q-lattice basis is replaced by the best of
CADO las's 13 unimodular shuffles (squeeze 0 only, the rectangle is unchanged) ranked by CADO's estimate_yield_in_sieve_area
(sieve/las-norms.cpp); glas's own J cap (= CADO strategy 0 / the strategy-3 second pass) then runs on the new basis."""
import sys, re
p = sys.argv[1]; s = open(p).read()
if 'GLAS_ADJUST' in s: print('already patched'); sys.exit(0)
inc = '#include "polynomial.hpp"\n'
assert inc in s
s = s.replace(inc, inc + '#include "rho.h"\n', 1)
helper = r'''
/* GLAS_ADJUST (W2-1 port of CADO las -adjust-strategy 2/3, squeeze 0): the
 * unimodular basis changes las tries, in swapped pairs (the odd ones are
 * the swaps, which do not matter for a square rectangle), and las's
 * estimate_yield_in_sieve_area: the weighted mean over a (2N+1) x (N+1)
 * grid of the product over the sides of dickman_rho(log2|F(x,y)| / lpb),
 * (x,y) mapped through the candidate matrix. */
static int glas_adjust_strategy()
{
    static int const v = getenv("GLAS_ADJUST") ? atoi(getenv("GLAS_ADJUST")) : 0;
    return v;
}
static std::atomic<unsigned long> adjust_n{0}, adjust_changed{0};
static std::atomic<double> adjust_gain_sum{0.0};

static double estimate_yield(polynomial<double> const fijd[2], int const lpb[2],
                             int logA, int const m[4], int squeeze = 0)
{
    int const N = 5;
    int const nx = 1 << (N - squeeze), ny = 1 << (N + squeeze);
    double const X = (double)(1UL << (logA - logA / 2 - squeeze));
    double const Y = (double)(1UL << (logA / 2 + squeeze + 1));
    double weightsum = 0, sum = 0;
    for (int i = -nx / 2; i <= nx / 2; i++) {
        double const x = X / nx * i;
        for (int j = 0; j <= ny / 2; j++) {
            double const y = Y / ny * j;
            double const xs = x * m[0] + y * m[2];
            double const ys = x * m[1] + y * m[3];
            double w = 1.0;
            if (i == -nx / 2 || i == nx / 2) w /= 2;
            if (j == 0 || j == ny / 2) w /= 2;
            double sprod = 1.0;
            for (int side = 0; side < 2; side++) {
                double const z = fijd[side].eval(xs, ys);
                double const a = std::log2(std::fabs(z));
                sprod *= dickman_rho_local(a / lpb[side], std::fabs(z));
            }
            weightsum += w;
            sum += w * sprod;
        }
    }
    return sum / weightsum;
}

static int const adjust_matrices[26][4] = {
    {1, 0, 0, 1}, {0, 1, 1, 0}, {1, 1, 1, 0}, {1, 0, 1, 1}, {1, 1, 0, -1}, {0, -1, 1, 1},
    {1, -1, 0, 1}, {0, 1, 1, -1}, {1, -1, 1, 0}, {1, 0, 1, -1},
    {1, 2, 0, 1}, {0, 1, 1, 2}, {1, -2, -1, 1}, {-1, 1, 1, -2}, {2, 1, 1, 1}, {1, 1, 2, 1},
    {-2, 1, 1, 0}, {1, 0, -2, 1}, {1, 0, 2, 1}, {2, 1, 1, 0}, {1, 2, 1, 1}, {1, 1, 1, 2},
    {2, -1, 1, -1}, {1, -1, 2, -1}, {-1, 2, 0, 1}, {0, 1, -1, 2}};
/* GLAS_ADJUST_SQUEEZE: largest squeeze tried (CADO: 3; each step halves I and doubles J) */
static int glas_adjust_max_squeeze()
{
    static int const v = getenv("GLAS_ADJUST_SQUEEZE") ? atoi(getenv("GLAS_ADJUST_SQUEEZE")) : 1;
    return v;
}
'''
anchor = '/* Build the task for one special-q: q-lattice basis, sieve region and\n * survivor bounds.  Returns nothing when las would discard the q. */'
assert anchor in s
s = s.replace(anchor, helper + '\n' + anchor, 1)
new = '''    int squeeze = 0;
    if (glas_adjust_strategy() >= 2) {
        /* norm polynomials for the skew-Gauss basis, then the best (shuffle, squeeze) */
        polynomial<double> base[2];
        int lpb[2];
        bool ok = true;
        for (int side = 0; side < 2 && ok; side++) {
            cxx_mpz_poly fij = cpoly[side].homography({B.a0, B.b0, B.a1, B.b1});
            if (fij.degree() < cpoly[side].degree()) { ok = false; break; }
            if (side == P.sqside)
                fij = fij.divexact(cxx_mpz(q));
            base[side] = polynomial<double>(fij);
            lpb[side] = P.side[side].lpb;
        }
        if (ok) {
            int const logA = P.logA;
            double const ref = estimate_yield(base, lpb, logA, adjust_matrices[0], 0);
            double best = ref;
            int best_r = 0, best_s = 0;
            for (int sq = 0; sq <= glas_adjust_max_squeeze(); sq++)
                for (int r = 0; r < 26; r++) {
                    if (sq == 0 && (r & 1)) continue;
                    double const e = estimate_yield(base, lpb, logA, adjust_matrices[r], sq);
                    if (e > best) { best = e; best_r = r; best_s = sq; }
                }
            adjust_n++;
            static bool const verbose = getenv("GLAS_ADJUST_VERBOSE") != nullptr;
            if (verbose)
                fprintf(stderr, "# adjust q=%" PRIu64 " rho=%" PRIu64 " B=(%" PRId64 ",%" PRId64 ")(%" PRId64 ",%" PRId64 ") best_r=%d squeeze=%d gain=%.1f%%\\n", q, rho, B.a0, B.b0, B.a1, B.b1, best_r, best_s, 100.0 * (best / ref - 1.0));
            if (best_r || best_s) {
                int const * m = adjust_matrices[best_r];
                basis64 Bn = B;
                Bn.a0 = m[0] * B.a0 + m[1] * B.a1;
                Bn.a1 = m[2] * B.a0 + m[3] * B.a1;
                Bn.b0 = m[0] * B.b0 + m[1] * B.b1;
                Bn.b1 = m[2] * B.b0 + m[3] * B.b1;
                if (fits_31bits(Bn)) {
                    B = Bn;
                    squeeze = best_s;
                    adjust_changed++;
                    double g = adjust_gain_sum.load();
                    while (!adjust_gain_sum.compare_exchange_weak(g, g + best / ref - 1.0)) {}
                }
            }
        }
    }

    int const logA = P.logA;
    int const logI = (logA + 1) / 2 - squeeze;
    uint32_t const I = 1u << logI;'''
old = '''    int const logA = P.logA;
    int const logI = (logA + 1) / 2;
    uint32_t const I = 1u << logI;'''
assert old in s
s = s.replace(old, new, 1)
# report at exit, just before glas's "# Process seconds" summary
idx = s.index('"# Process seconds')
start = s.rfind('fprintf(stderr', 0, idx)
line_start = s.rfind('\n', 0, start) + 1
ind = s[line_start:start]
hook = (ind + 'if (glas_adjust_strategy() >= 2)\n' + ind + '    fprintf(stderr, "# GLAS_ADJUST %d: basis changed on %lu of %lu special-q, '
        'mean estimated gain %.1f%%\\n", glas_adjust_strategy(), adjust_changed.load(), adjust_n.load(), '
        'adjust_changed.load() ? 100.0 * adjust_gain_sum.load() / adjust_changed.load() : 0.0);\n')
s = s[:line_start] + hook + s[line_start:]
old2 = """    J = Jmax;
    J = J / j_round * j_round;
    if (J < (1u << std::max(1, LOG_BUCKET_REGION - logI)))
        return std::nullopt;
    t.J = (uint32_t)J;"""
assert old2 in s
s = s.replace(old2, """    J = Jmax;
    J = J / j_round * j_round;
    if (getenv("GLAS_ADJUST_VERBOSE"))
        fprintf(stderr, "# adjust q=%" PRIu64 " Jmax=%u J=%" PRIu64 "%s\\n", q, Jmax, J, J < (1u << std::max(1, LOG_BUCKET_REGION - logI)) ? " DISCARD" : "");
    if (J < (1u << std::max(1, LOG_BUCKET_REGION - logI)))
        return std::nullopt;
    t.J = (uint32_t)J;""", 1)
open(p, 'w').write(s); print('patched', p)
