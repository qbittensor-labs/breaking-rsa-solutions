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

"""la_synth.py -- time gpu_la's block Lanczos per iteration on a synthetic GF(2) matrix (W6 gate).
usage: LA_BLOCK=128|256 LA_SPMV=seg|prod LA_MAXIT=300 python3 la_synth.py <nrows> <nnz_per_row> [seed]
Random COO (rows < n, columns uniform), 64 dense rows (the CADO layout: 2 uint64 per column), no file I/O."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
n = int(sys.argv[1]); d = int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
import gpu_la
rng = np.random.default_rng(seed); t = time.time()
nnz = n * d
R = rng.integers(0, n, nnz, dtype=np.int32)          # B row (ideal)
C = np.repeat(np.arange(n, dtype=np.int32), d)        # B column (relset), non-decreasing = file order
dcol = rng.integers(0, 1 << 64, 2 * n, dtype=np.uint64)   # 64 dense rows -> bits 0..63 of word 0, word 1 zero
dcol[1::2] = 0
print(f"synthetic matrix {n}x{n}, {nnz} nnz, 64 dense rows built in {time.time()-t:.0f}s", flush=True)
ND = 64
if gpu_la.LA_BLOCK == 256: x, nd, ok, _ = gpu_la.run256(R, C, dcol, ND, n)
elif gpu_la.LA_BLOCK == 128: x, nd, ok, _ = gpu_la.run128(R, C, dcol, ND, n)
else: print("LA_BLOCK must be 128 or 256 here"); sys.exit(2)
print("result:", "stopped (timing)" if x is None else f"nd={nd} verify={ok}")
