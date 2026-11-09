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

"""glas_prefetch_patch.py <gpu-siever/src/sieve_core_gpu.cu> -- idea 3 of docs/rsa560_nvidia_ideas.md.

GLAS_REGION_PREFETCH (default 0 = stock): at region_kernel entry, before the pattern / norm / small-prime phases, the
block asks L2 for this region's bucket entries, so their DRAM reads overlap the barrier-separated compute phases instead
of stalling the bucket apply that follows them.
  1 = one prefetch.global.L2 per 128-byte line (16 entries), spread over the block's threads;
  2 = one cp.async.bulk.prefetch.L2.global per 64-entry slab piece (512 B; sm_90+), lanes of the block.
Only a hint: no result changes (relation set identical by construction). Skipped when GLAS_REGION_SKIP has bit 4."""
import sys
p = sys.argv[1]; s = open(p).read()
if 'c_region_prefetch' in s: print('already patched'); sys.exit(0)

def sub(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b, 1)

sub('__constant__ uint32_t c_gate_shift;\n',
    '__constant__ uint32_t c_gate_shift;\n\n'
    '/* GLAS_REGION_PREFETCH: region_kernel prefetches its bucket into L2 at entry (1 per line, 2 bulk per piece). */\n'
    '__constant__ uint32_t c_region_prefetch;\n')
sub('    uint32_t gate_shift = 3;  /* GLAS_FILL_GATE_SHIFT=0 reads the survivor mask */\n',
    '    uint32_t gate_shift = 3;  /* GLAS_FILL_GATE_SHIFT=0 reads the survivor mask */\n'
    '    uint32_t region_prefetch = 0; /* GLAS_REGION_PREFETCH */\n')
sub('    CUDA_CHECK(cudaMemcpyToSymbol(c_gate_shift, &ctx.gate_shift, 4));\n',
    '    CUDA_CHECK(cudaMemcpyToSymbol(c_gate_shift, &ctx.gate_shift, 4));\n'
    '    if (char const * e = getenv("GLAS_REGION_PREFETCH"))\n'
    '        ctx.region_prefetch = (uint32_t)std::min(2, std::max(0, atoi(e)));\n'
    '    CUDA_CHECK(cudaMemcpyToSymbol(c_region_prefetch, &ctx.region_prefetch, 4));\n')
sub('''    if (threadIdx.x == 0)
        block_surv = 0;
    uint32_t const region = blockIdx.x;
''', '''    if (threadIdx.x == 0)
        block_surv = 0;
    uint32_t const region = blockIdx.x;
    if (c_region_prefetch && !(t.skip & 4)) {
        uint32_t const pc = min(bucket_cnt[region * BK_CNT_STRIDE], bucket_cap);
        if (c_region_prefetch == 1) {
            for (uint32_t k = threadIdx.x * 16; k < pc; k += blockDim.x * 16)
                asm volatile("prefetch.global.L2 [%0];" ::
                             "l"(buckets + dev_bucket_idx(region, k, bucket_cap)));
        } else {
            for (uint32_t k = threadIdx.x * 64; k < pc; k += blockDim.x * 64) {
                uint32_t const bytes = ((min(64u, pc - k) * 8u) + 15u) & ~15u;
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::
                             "l"(buckets + dev_bucket_idx(region, k, bucket_cap)), "r"(bytes) : "memory");
            }
        }
    }
''')
open(p, 'w').write(s); print('patched')
