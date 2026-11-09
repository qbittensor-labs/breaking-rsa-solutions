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

/* la128host: the per-iteration host GF(2) algebra of gpu_la.py's 128-bit block Lanczos, in C.
 * Exact ports of gpu_la.py mul128 / tr128 / fnsV (msieve find_nonsingular_sub). Rows are 128-bit
 * values stored as (lo, hi) uint64 pairs, bit j of row i = bit j of lo | hi<<64 -- the (128,2) uint64
 * layout gpu_la.py uses. Build: gcc -O2 -march=x86-64-v3 -shared -fPIC -o la128host.so la128host.c */
#include <stdint.h>
#include <string.h>
typedef unsigned __int128 u128;
static inline u128 ld128(const uint64_t *p) { return (u128)p[0] | ((u128)p[1] << 64); }
static inline void st128(uint64_t *p, u128 v) { p[0] = (uint64_t)v; p[1] = (uint64_t)(v >> 64); }

/* c[i] = XOR over j with bit j of a[i] set of b[j] */
void la_mul128(const uint64_t *a, const uint64_t *b, uint64_t *c)
{
    u128 B[128], C[128];
    for (int j = 0; j < 128; j++) B[j] = ld128(b + 2 * j);
    for (int i = 0; i < 128; i++) {
        u128 acc = 0; uint64_t w0 = a[2 * i], w1 = a[2 * i + 1];
        while (w0) { int j = __builtin_ctzll(w0); acc ^= B[j]; w0 &= w0 - 1; }
        while (w1) { int j = __builtin_ctzll(w1) + 64; acc ^= B[j]; w1 &= w1 - 1; }
        C[i] = acc;
    }
    for (int i = 0; i < 128; i++) st128(c + 2 * i, C[i]);
}
/* transpose: out row j bit i = a row i bit j */
void la_tr128(const uint64_t *a, uint64_t *out)
{
    u128 A[128], T[128];
    for (int i = 0; i < 128; i++) { A[i] = ld128(a + 2 * i); T[i] = 0; }
    for (int i = 0; i < 128; i++) {
        u128 r = A[i];
        while (r) {
            int j = (uint64_t)r ? __builtin_ctzll((uint64_t)r) : 64 + __builtin_ctzll((uint64_t)(r >> 64));
            T[j] |= (u128)1 << i; r &= r - 1;
        }
    }
    for (int j = 0; j < 128; j++) st128(out + 2 * j, T[j]);
}
/* fnsV for VB=128. t: 128 rows (lo,hi). ls/ld: previous subspace. Outputs: returns dim (0 = failure);
 * s_out[0..dim) the chosen columns in order; winv (128 rows (lo,hi)) = low halves of the final M. */
int la_fns128(const uint64_t *t, const int *ls, int ld, int *s_out, uint64_t *winv)
{
    u128 Mh[128], Ml[128];            /* row = (Mh << 128) | Ml : Mh = t part, Ml = identity part */
    int cols[128], mask[128];
    for (int i = 0; i < 128; i++) { Mh[i] = ld128(t + 2 * i); Ml[i] = (u128)1 << i; mask[i] = 0; }
    for (int i = 0; i < ld; i++) { cols[127 - i] = ls[i]; mask[ls[i]] = 1; }
    int j = 0;
    for (int i = 0; i < 128; i++) if (!mask[i]) cols[j++] = i;
    int dim = 0;
    for (int i = 0; i < 128; i++) {
        int ci = cols[i], p = -1; u128 m = (u128)1 << ci;
        for (int jj = i; jj < 128; jj++) if (Mh[cols[jj]] & m) { p = jj; break; }
        if (p >= 0) {
            int b = cols[p]; u128 th = Mh[ci], tl = Ml[ci]; Mh[ci] = Mh[b]; Ml[ci] = Ml[b]; Mh[b] = th; Ml[b] = tl;
            u128 Pi = Mh[ci], Pl = Ml[ci];
            for (int x = 0; x < 128; x++) if (Mh[x] & m) { Mh[x] ^= Pi; Ml[x] ^= Pl; }
            Mh[ci] = Pi; Ml[ci] = Pl;
            s_out[dim++] = ci;
        } else {
            for (int jj = i; jj < 128; jj++) if (Ml[cols[jj]] & m) { p = jj; break; }
            if (p < 0) return 0;
            int b = cols[p]; u128 th = Mh[ci], tl = Ml[ci]; Mh[ci] = Mh[b]; Ml[ci] = Ml[b]; Mh[b] = th; Ml[b] = tl;
            u128 Pi = Mh[ci], Pl = Ml[ci];
            for (int x = 0; x < 128; x++) if (Ml[x] & m) { Mh[x] ^= Pi; Ml[x] ^= Pl; }
        }
    }
    for (int i = 0; i < 128; i++) st128(winv + 2 * i, Ml[i]);
    return dim;
}
