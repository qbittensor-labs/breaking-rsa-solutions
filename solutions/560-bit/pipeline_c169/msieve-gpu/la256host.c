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

/* la256host: the per-iteration host GF(2) algebra of gpu_la.py's 256-bit block Lanczos (W6), in C.
 * Exact ports of la128host.c on 256-bit rows stored as 4 uint64 words (bit j of row i = bit j%64 of
 * word j/64) -- the (256,4) uint64 layout gpu_la.py uses for ulonglong4 rows.
 * Build: gcc -O2 -march=x86-64-v3 -shared -fPIC -o la256host.so la256host.c */
#include <stdint.h>
#include <string.h>
#define W 4
#define VB 256
typedef struct { uint64_t w[W]; } r256;
static inline r256 ld256(const uint64_t *p) { r256 r; memcpy(r.w, p, 8 * W); return r; }
static inline void st(uint64_t *p, r256 v) { memcpy(p, v.w, 8 * W); }
static inline void x256(r256 *a, const r256 *b) { for (int k = 0; k < W; k++) a->w[k] ^= b->w[k]; }
static inline int bit(const r256 *a, int j) { return (a->w[j >> 6] >> (j & 63)) & 1; }
static inline void setbit(r256 *a, int j) { a->w[j >> 6] |= (uint64_t)1 << (j & 63); }
static inline int nz(const r256 *a) { return (a->w[0] | a->w[1] | a->w[2] | a->w[3]) != 0; }

/* c[i] = XOR over j with bit j of a[i] set of b[j] */
void la_mul256(const uint64_t *a, const uint64_t *b, uint64_t *c)
{
    r256 B[VB], C[VB];
    for (int j = 0; j < VB; j++) B[j] = ld256(b + W * j);
    for (int i = 0; i < VB; i++) {
        r256 acc = {{0, 0, 0, 0}};
        for (int k = 0; k < W; k++) {
            uint64_t w = a[W * i + k];
            while (w) { int j = __builtin_ctzll(w) + 64 * k; x256(&acc, &B[j]); w &= w - 1; }
        }
        C[i] = acc;
    }
    for (int i = 0; i < VB; i++) st(c + W * i, C[i]);
}
/* transpose: out row j bit i = a row i bit j */
void la_tr256(const uint64_t *a, uint64_t *out)
{
    r256 T[VB];
    memset(T, 0, sizeof T);
    for (int i = 0; i < VB; i++)
        for (int k = 0; k < W; k++) {
            uint64_t w = a[W * i + k];
            while (w) { int j = __builtin_ctzll(w) + 64 * k; setbit(&T[j], i); w &= w - 1; }
        }
    for (int j = 0; j < VB; j++) st(out + W * j, T[j]);
}
/* fnsV for VB=256 (msieve find_nonsingular_sub). t: 256 rows. ls/ld: previous subspace. Returns dim
 * (0 = failure); s_out[0..dim) the chosen columns in order; winv (256 rows) = low halves of the final M. */
int la_fns256(const uint64_t *t, const int *ls, int ld, int *s_out, uint64_t *winv)
{
    r256 Mh[VB], Ml[VB];              /* row = [Mh | Ml]: Mh = t part, Ml = identity part */
    int cols[VB], mask[VB];
    for (int i = 0; i < VB; i++) { Mh[i] = ld256(t + W * i); memset(&Ml[i], 0, sizeof(r256)); setbit(&Ml[i], i); mask[i] = 0; }
    for (int i = 0; i < ld; i++) { cols[VB - 1 - i] = ls[i]; mask[ls[i]] = 1; }
    int j = 0;
    for (int i = 0; i < VB; i++) if (!mask[i]) cols[j++] = i;
    int dim = 0;
    for (int i = 0; i < VB; i++) {
        int ci = cols[i], p = -1;
        for (int jj = i; jj < VB; jj++) if (bit(&Mh[cols[jj]], ci)) { p = jj; break; }
        if (p >= 0) {
            int b = cols[p]; r256 th = Mh[ci], tl = Ml[ci]; Mh[ci] = Mh[b]; Ml[ci] = Ml[b]; Mh[b] = th; Ml[b] = tl;
            r256 Pi = Mh[ci], Pl = Ml[ci];
            for (int x = 0; x < VB; x++) if (bit(&Mh[x], ci)) { x256(&Mh[x], &Pi); x256(&Ml[x], &Pl); }
            Mh[ci] = Pi; Ml[ci] = Pl;
            s_out[dim++] = ci;
        } else {
            for (int jj = i; jj < VB; jj++) if (bit(&Ml[cols[jj]], ci)) { p = jj; break; }
            if (p < 0) return 0;
            int b = cols[p]; r256 th = Mh[ci], tl = Ml[ci]; Mh[ci] = Mh[b]; Ml[ci] = Ml[b]; Mh[b] = th; Ml[b] = tl;
            r256 Pi = Mh[ci], Pl = Ml[ci];
            for (int x = 0; x < VB; x++) if (bit(&Ml[x], ci)) { x256(&Mh[x], &Pi); x256(&Ml[x], &Pl); }
        }
    }
    for (int i = 0; i < VB; i++) st(winv + W * i, Ml[i]);
    (void)nz;
    return dim;
}

/* ---- LA_V3 'host' (2026-10-21): faster equivalents of la_mul256 / la_tr256 / la_fns256 (same results bit for bit;
 * gpu_la.py's LA_V3 self-test compares them with the functions above before use). The host step runs while the GPU
 * idles, so every microsecond here is a microsecond of GPU time per iteration. */
/* c = a*b: Four Russians on 4-bit chunks of a's rows (64 tables x 16 entries x 32 B = 32 KB, L1-resident) */
void la_mul256f(const uint64_t *a, const uint64_t *b, uint64_t *c)
{
    static uint64_t T[64][16][W];
    for (int k = 0; k < 64; k++) {
        T[k][0][0] = T[k][0][1] = T[k][0][2] = T[k][0][3] = 0;
        for (int v = 1; v < 16; v++) {
            const int j = __builtin_ctz(v); const uint64_t *r = b + W * (4 * k + j); const uint64_t *p = T[k][v & (v - 1)];
            for (int q = 0; q < W; q++) T[k][v][q] = p[q] ^ r[q];
        }
    }
    uint64_t C[VB][W];
    for (int i = 0; i < VB; i++) {
        uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (int w = 0; w < W; w++) {
            const uint64_t x = a[W * i + w];
            for (int j = 0; j < 16; j++) {
                const uint64_t *e = T[16 * w + j][(x >> (4 * j)) & 15];
                c0 ^= e[0]; c1 ^= e[1]; c2 ^= e[2]; c3 ^= e[3];
            }
        }
        C[i][0] = c0; C[i][1] = c1; C[i][2] = c2; C[i][3] = c3;
    }
    memcpy(c, C, sizeof C);
}
/* 64x64 in-place bit transpose (bit c of A[r] <-> bit r of A[c]), Hacker's Delight */
static void tr64(uint64_t *A)
{
    uint64_t m = 0x00000000FFFFFFFFULL;
    for (int j = 32; j; j >>= 1, m ^= m << j)
        for (int k = 0; k < 64; k = ((k | j) + 1) & ~j) {
            const uint64_t t = ((A[k] >> j) ^ A[k | j]) & m;
            A[k] ^= t << j; A[k | j] ^= t;
        }
}
void la_tr256f(const uint64_t *a, uint64_t *out)
{
    uint64_t blk[64];
    for (int bi = 0; bi < 4; bi++)
        for (int bj = 0; bj < 4; bj++) {
            for (int r = 0; r < 64; r++) blk[r] = a[W * (64 * bi + r) + bj];
            tr64(blk);
            for (int r = 0; r < 64; r++) out[W * (64 * bj + r) + bi] = blk[r];
        }
}
/* la_fns256 with rows as flat [Mh | Ml] uint64[8] (same pivot order and operations) + the chosen-column mask (4 words) */
int la_fns256f(const uint64_t *t, const int *ls, int ld, int *s_out, uint64_t *winv, uint64_t *mask_out)
{
    static uint64_t M[VB][2 * W];
    int cols[VB], mk[VB];
    for (int i = 0; i < VB; i++) {
        for (int q = 0; q < W; q++) { M[i][q] = t[W * i + q]; M[i][W + q] = 0; }
        M[i][W + (i >> 6)] = (uint64_t)1 << (i & 63); mk[i] = 0;
    }
    for (int i = 0; i < ld; i++) { cols[VB - 1 - i] = ls[i]; mk[ls[i]] = 1; }
    int j = 0;
    for (int i = 0; i < VB; i++) if (!mk[i]) cols[j++] = i;
    int dim = 0;
    uint64_t msk[W] = {0, 0, 0, 0};
    for (int i = 0; i < VB; i++) {
        const int ci = cols[i], wh = ci >> 6, sh = ci & 63;
        int p = -1;
        for (int jj = i; jj < VB; jj++) if ((M[cols[jj]][wh] >> sh) & 1) { p = jj; break; }
        const int off = (p >= 0) ? 0 : W;              /* pivot found in the t part, else look in the identity part */
        if (p < 0) {
            for (int jj = i; jj < VB; jj++) if ((M[cols[jj]][W + wh] >> sh) & 1) { p = jj; break; }
            if (p < 0) return 0;
        }
        const int b = cols[p];
        uint64_t P[2 * W];
        for (int q = 0; q < 2 * W; q++) { P[q] = M[b][q]; M[b][q] = M[ci][q]; M[ci][q] = P[q]; }
        for (int x = 0; x < VB; x++) {                 /* branch-free: the bit test is a coin flip */
            const uint64_t e = (uint64_t)0 - ((M[x][off + wh] >> sh) & 1);
            for (int q = 0; q < 2 * W; q++) M[x][q] ^= P[q] & e;
        }
        if (off == 0) {
            for (int q = 0; q < 2 * W; q++) M[ci][q] = P[q];
            s_out[dim++] = ci; msk[wh] |= (uint64_t)1 << sh;
        }
    }
    for (int i = 0; i < VB; i++) for (int q = 0; q < W; q++) winv[W * i + q] = M[i][W + q];
    for (int q = 0; q < W; q++) mask_out[q] = msk[q];
    return dim;
}
