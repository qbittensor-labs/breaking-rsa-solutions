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

"""glas_xfgrid_patch.py <gpu-siever/src/sieve_core_gpu.cu> -- proposal 7 of docs/rsa560_codex_review.md (codex-review).

Applies ON TOP of the production patch stack (sandbox_bootstrap.sh: coprime, prefetch, xfuse, region). The production
fused fill kernel fill_xfuse_kernel launches one thread per large-tier entry (~ (n_fb - n_local)/FILL_THREADS blocks),
holding every SM while it runs. This gives it a grid-stride entry loop (per-entry early `return`s become `continue`;
the emit lambda's gate `return` is untouched, so no lattice walk is cut short) and a block cap for BOTH of its launches
(side-1 ahead fill, gated side-0 tail):
  GLAS_XFUSE_GRID=G      runtime cap in blocks (0 = uncapped = stock launch geometry)
  -DGLAS_XFUSE_GRID_CAP=G  compile-time default for the same cap (default 0)
Unset/0: launch geometry identical to production (each thread does exactly one loop iteration). Same entries, same
emits, same bucket multiset -> identical relations. Banners: '# GLAS_XFUSE_GRID cap=G' at init (when GLAS_XFUSE is
requested) and, once, '# GLAS_XFUSE_GRID live: side S launch B blocks (stock N)' from the first fused launch."""
import sys
p = sys.argv[1]; s = open(p).read()
if 'GLAS_XFUSE_GRID' in s: print('already patched'); sys.exit(0)
assert 'fill_xfuse_kernel' in s, 'needs glas_xfuse_patch.py first'

def sub(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (s.count(a), a[:80])
    s = s.replace(a, b)

# compile-time default
sub('''#if GLAS_XFUSE_MINBLOCKS > 0
#define XFUSE_BOUNDS''', '''/* GLAS_XFUSE_GRID: block cap of the fill_xfuse_kernel launches (grid-stride over entries); 0 = uncapped */
#ifndef GLAS_XFUSE_GRID_CAP
#define GLAS_XFUSE_GRID_CAP 0
#endif
#if GLAS_XFUSE_MINBLOCKS > 0
#define XFUSE_BOUNDS''')

# kernel: grid-stride entry loop, per-entry returns -> continue
sub('''{
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
    uint64_t const packed = dev_pack_entry(e);''',
'''{
    uint32_t const je = t.j0 + t.rows;
    uint64_t const xend = (uint64_t)je << t.logI;
    uint32_t const I = 1u << t.logI;
    /* GLAS_XFUSE_GRID: a capped grid strides over the entries (uncapped: one thread per entry, one iteration) */
    for (uint32_t idx = first + blockIdx.x * blockDim.x + threadIdx.x; idx < n;
         idx += gridDim.x * blockDim.x) {
    gpu_fb_entry const e = fb[idx];
    if (side == t.sq_side && e.q == t.q && e.r == t.rho)
        continue;
    uint32_t const root = dev_transform_root(t, e.q, e.r);
    if (root == SKIP_ROOT)
        continue;
    uint64_t const packed = dev_pack_entry(e);''')
sub('''    if (root >= e.q) {
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
''', '''    if (root >= e.q) {
        dev_proj_walk(t, e.q, root - e.q, t.j0, je, emit);
        continue;
    }
    dev_plattice const P(e.q, root, I);
    if (P.i1 >> t.logI)
        continue;
    uint64_t const x0 = I / 2;
    if (x0 >= xend)
        continue;
    dev_affine_walk(t, make_uint4(P.mi0, P.j0, P.i1, P.j1), x0, xend, emit);
    }
}
''')

# ctx field
sub('''    bool xfuse = false;       /* GLAS_XFUSE: large tier transformed inside its fill */
''', '''    bool xfuse = false;       /* GLAS_XFUSE: large tier transformed inside its fill */
    uint32_t xfuse_grid = GLAS_XFUSE_GRID_CAP;  /* GLAS_XFUSE_GRID: block cap of fill_xfuse_kernel, 0 = uncapped */
''')

# env + init banner
sub('''                    "ignored: needs FILL_AHEAD 1, FILL_TAIL0, row bands, no carry/stage/owners/REGION_SKIP");
    }
''', '''                    "ignored: needs FILL_AHEAD 1, FILL_TAIL0, row bands, no carry/stage/owners/REGION_SKIP");
        if (char const * g = getenv("GLAS_XFUSE_GRID"))
            ctx.xfuse_grid = strtoul(g, nullptr, 10);
        if (want)
            fprintf(stderr, "# GLAS_XFUSE_GRID cap=%u (0 = uncapped; compile default %u)\\n",
                    ctx.xfuse_grid, (unsigned)GLAS_XFUSE_GRID_CAP);
    }
''')

# launches
sub('''        auto fill = [&](int side, uint32_t const * gate) {
            /* with carry, fill_kernel walks the banded entries only */''',
'''        auto xf_blocks = [&](int side, uint32_t b) {   /* GLAS_XFUSE_GRID */
            uint32_t const g = ctx.xfuse_grid ? std::min(b, ctx.xfuse_grid) : b;
            static std::atomic<bool> noted[2] = {false, false};
            if (!noted[side].exchange(true))
                fprintf(stderr, "# GLAS_XFUSE_GRID live: side %d launch %u blocks (stock %u)\\n", side, g, b);
            return g;
        };
        auto fill = [&](int side, uint32_t const * gate) {
            /* with carry, fill_kernel walks the banded entries only */''')
sub('''                    fill_xfuse_kernel<false><<<(ctx.n_fb[side] - n + FILL_THREADS - 1) /
                                         FILL_THREADS, FILL_THREADS, 0, st>>>(''',
'''                    fill_xfuse_kernel<false><<<xf_blocks(side, (ctx.n_fb[side] - n + FILL_THREADS - 1) /
                                         FILL_THREADS), FILL_THREADS, 0, st>>>(''')
sub('''                fill_xfuse_kernel<true><<<(ctx.n_fb[side] - n + FILL_THREADS - 1) /
                                    FILL_THREADS, FILL_THREADS, 0, st>>>(''',
'''                fill_xfuse_kernel<true><<<xf_blocks(side, (ctx.n_fb[side] - n + FILL_THREADS - 1) /
                                    FILL_THREADS), FILL_THREADS, 0, st>>>(''')
open(p, 'w').write(s); print('patched')
