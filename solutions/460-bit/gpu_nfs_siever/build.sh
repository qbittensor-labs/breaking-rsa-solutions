#!/usr/bin/env bash
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

# Build the GPU lattice siever for the validator GPU (RTX PRO 6000 Blackwell, sm_120).
# Needs: CUDA toolkit (nvcc, sm_120 support => CUDA >= 12.8), GMP dev, OpenMP.
#   SPB=32  -> small primes (<32) are trial-divided in the cofactor/CPU instead of resieved
#             (resieve re-walk of the tiniest primes dominated the resieve pass).
set -e
SPB="${SPB:-32}"
nvcc -O3 -arch=sm_120 -Xcompiler -fopenmp -DSPB=$SPB -o gpu_loop gpu_loop.cu -lgmp -lm  # LPB_VAL defaults to 2^29 in source
echo "built gpu_loop (SPB=$SPB)"
