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

"""glas_xfuse_patch.py <gpu-siever/src/sieve_core_gpu.cu> -- fuse the root transform into the large-tier fill.

GLAS_XFUSE (default 0 = stock).  With lim 200M/450M the large tier (q >= the local bound 4I = 2^17, walked one thread
per entry by fill_ahead_kernel on side 1 and by the gated tail kernel on side 0) is ~35M entries per special-q, of which
the ~20M new ones in (100M, 200M] / (150M, 450M] get only ~1-4 lattice points each in the 2^29 cells.  Stock pays for
every entry, every special-q:
  transform_kernel  reads fb (16 B), writes roots (4 B), walk_x (8 B), lattice (16 B)    44 B + ~1.5k instructions
  fill_ahead_kernel reads walk_x (8 B), fb (16 B), lattice (16 B), writes walk_x (8 B)   48 B
i.e. ~92 B of DRAM traffic (the arrays are 3.2 GB per q, far above L2) for ~1.2 bucket appends.
GLAS_XFUSE=1: when the special-q fits in one chunk (J <= chunk rows: the production A 29, chunk 2^29 case), the
transform kernel only covers the entries below the local bound, and fill_xfuse_kernel replaces both large-tier fill
launches: each thread reads its fb entry once, computes the root (dev_transform_root) and the reduced basis
(dev_plattice) in registers and walks it at once.  Root, cursor and basis never touch global memory (76 of the 92 B
per entry go away; the instructions move into the request-bound fill, where they can overlap the atomics' latency).
A special-q that needs several chunks keeps the stock path (one stderr note).  Needs the production fill layout
(FILL_AHEAD 1, FILL_TAIL0, row bands, no carry / stage / probe bits); otherwise the flag is ignored with a note.

Relations identical by construction: the fused thread runs the same functions on the same inputs (gpu_task, fb[idx])
as transform_kernel + fill_ahead_kernel, starts the walk at the same cursor (I/2: lattice point (0, 0) of the first,
only chunk), applies the same keep rule, gate and bucket emit, so it appends exactly the same bucket entries; only the
order of appends inside a bucket can change, which the atomics already leave unordered.  The skipped transform outputs
(roots / walk_x / lattice of entries >= n_local) are read only by the two replaced launches (region, resieve_compact
and fill_row read entries below n_local).  Check on the box: the "GPU bucket max s: ..., entries N" totals must equal
the stock run's exactly.
Register use: fill_xfuse_kernel carries the transform's registers (int128 path included); build with
-DGLAS_XFUSE_MINBLOCKS=6 (same cap as fill_row_kernel) if ptxas -v shows it far above fill_ahead_kernel."""
import sys
p = sys.argv[1]; s = open(p).read()
if 'GLAS_XFUSE' in s: print('already patched'); sys.exit(0)

def sub(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b, 1)

sub('#include <array>\n', '#include <array>\n#include <atomic>\n')

# launch bounds knob of the fused kernel (0 leaves the register count to ptxas, as FILL_BOUNDS)
sub('/* GLAS_FILL_STAGE: threads per block of the exchange-staged fill */\n',
    '''/* GLAS_XFUSE: blocks per SM the fused transform + fill kernel is compiled to fit; 0 = no bound */
#ifndef GLAS_XFUSE_MINBLOCKS
#define GLAS_XFUSE_MINBLOCKS 0
#endif
#if GLAS_XFUSE_MINBLOCKS > 0
#define XFUSE_BOUNDS __launch_bounds__(FILL_THREADS, GLAS_XFUSE_MINBLOCKS)
#else
#define XFUSE_BOUNDS
#endif
/* GLAS_FILL_STAGE: threads per block of the exchange-staged fill */
''')

sub('    bool fill_tail0 = true;   /* GLAS_FILL_TAIL0=0 walks the side-0 tail in the row kernel */\n',
    '    bool fill_tail0 = true;   /* GLAS_FILL_TAIL0=0 walks the side-0 tail in the row kernel */\n'
    '    bool xfuse = false;       /* GLAS_XFUSE: large tier transformed inside its fill */\n')

sub('    assert(!ctx.fill_tail0 || (ctx.fill_gate && ctx.fill_row_bands));\n',
    '''    assert(!ctx.fill_tail0 || (ctx.fill_gate && ctx.fill_row_bands));
    {   /* GLAS_XFUSE: only the production fill layout, where fill_ahead_kernel (side 1) and the side-0 tail
         * kernel are the only readers of the large tier's roots / walk_x / lattice */
        bool const want = env_flag("GLAS_XFUSE", false);
        ctx.xfuse = want && ctx.fill_ahead == 1 && ctx.fill_tail0 && ctx.fill_row_bands &&
                    !ctx.carry && !ctx.fill_stage && ctx.fill_owners == 1 && ctx.skip == 0;
        if (want)
            fprintf(stderr, "# GLAS_XFUSE %s\\n", ctx.xfuse ?
                    "on: large-tier transform fused into the fill (single-chunk special-q)" :
                    "ignored: needs FILL_AHEAD 1, FILL_TAIL0, row bands, no carry/stage/owners/REGION_SKIP");
    }
''')

# the fused kernel, after fill_ahead_kernel (it uses dev_bucket_append, dev_gate_drops, the walks)
sub('__device__ void dev_bucket_append_pair(uint32_t region, uint64_t a, uint64_t b,\n',
    '''/* GLAS_XFUSE: the large tier [first, n) of a special-q sieved in one chunk.  One thread per entry: the transform
 * of transform_kernel (special-q skip, dev_transform_root, dev_plattice) and the walk of fill_ahead_kernel from the
 * cursor the transform would have stored (I/2 for an affine root with a reduced basis, the projective walk for a
 * projective one), emitting exactly fill_ahead_kernel's bucket entries; nothing is written back. */
template <bool GATED>
__global__ void XFUSE_BOUNDS
fill_xfuse_kernel(gpu_task t, gpu_fb_entry const * fb, uint32_t first,
                  uint32_t n, int side, uint32_t regions, uint32_t cap,
                  uint64_t * buckets, uint32_t * cnt, uint32_t * overflow,
                  uint32_t const * gate)
{
    uint32_t const idx = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    gpu_fb_entry const e = fb[idx];
    if (side == t.sq_side && e.q == t.q && e.r == t.rho)
        return;
    uint32_t const root = dev_transform_root(t, e.q, e.r);
    if (root == SKIP_ROOT)
        return;
    uint32_t const je = t.j0 + t.rows;
    uint64_t const xend = (uint64_t)je << t.logI;
    uint64_t const packed = dev_pack_entry(e);
    auto emit = [&](uint32_t x) {
        if constexpr (GATED)
            if (dev_gate_drops(gate, x))
                return;
        uint32_t const region = x >> SREG_LOG;
        uint32_t const bank = region & ~(regions - 1);
        dev_bucket_append(region - bank, packed | (x & (SREG - 1)), cap,
            buckets + (uint64_t)bank * cap, cnt + bank * BK_CNT_STRIDE,
            overflow);
    };
    if (root >= e.q) {
        dev_proj_walk(t, e.q, root - e.q, t.j0, je, emit);
        return;
    }
    uint32_t const I = 1u << t.logI;
    dev_plattice const P(e.q, root, I);
    if (P.i1 >> t.logI)
        return;
    uint64_t const x0 = I / 2;
    if (x0 >= xend)
        return;
    dev_affine_walk(t, make_uint4(P.mi0, P.j0, P.i1, P.j1), x0, xend, emit);
}

__device__ void dev_bucket_append_pair(uint32_t region, uint64_t a, uint64_t b,
''')

sub('    uint32_t const chunk_rows = (uint32_t)(ctx.chunk_area >> task.logI);\n',
    '''    uint32_t const chunk_rows = (uint32_t)(ctx.chunk_area >> task.logI);
    /* GLAS_XFUSE: the large tier is transformed inside its fill when the special-q is one chunk */
    bool const xfuse = ctx.xfuse && task.J <= chunk_rows;
    if (ctx.xfuse && !xfuse) {
        static std::atomic<bool> noted{false};
        if (!noted.exchange(true))
            fprintf(stderr, "# GLAS_XFUSE: q=%lu has J %u > chunk rows %u, stock path for such special-q\\n",
                    (unsigned long)task.q, task.J, chunk_rows);
    }
''')

sub('            /* roots do not depend on the chunk */\n            uint32_t const n = ctx.n_fb[side];\n',
    '            /* roots do not depend on the chunk; GLAS_XFUSE: the large tier is transformed by its fill */\n'
    '            uint32_t const n = xfuse ? ctx.n_local[side] : ctx.n_fb[side];\n')

sub('                fill_ahead_kernel<false><<<(ctx.n_fb[side] - n + FILL_THREADS - 1) /\n',
    '''                if (xfuse)
                    fill_xfuse_kernel<false><<<(ctx.n_fb[side] - n + FILL_THREADS - 1) /
                                         FILL_THREADS, FILL_THREADS, 0, st>>>(
                        next, ctx.fb[side], n, ctx.n_fb[side], side, ctx.max_regions,
                        ctx.bucket_cap[side], s.buckets[side], s.bucket_cnt[side],
                        s.counters[par] + 2, nullptr);
                else
                fill_ahead_kernel<false><<<(ctx.n_fb[side] - n + FILL_THREADS - 1) /
''')

sub('            if (tail0 && n < ctx.n_fb[side])\n                fill_ahead_kernel<true>',
    '''            if (tail0 && n < ctx.n_fb[side] && xfuse)
                fill_xfuse_kernel<true><<<(ctx.n_fb[side] - n + FILL_THREADS - 1) /
                                    FILL_THREADS, FILL_THREADS, 0, st>>>(
                    t, ctx.fb[side], n, ctx.n_fb[side], side, ctx.max_regions,
                    ctx.bucket_cap[side], buckets[side], bucket_cnt[side],
                    s.counters[par] + 2, gate);
            else if (tail0 && n < ctx.n_fb[side])
                fill_ahead_kernel<true>''')
open(p, 'w').write(s); print('patched')
