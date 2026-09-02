# Copyright (C) 2026 qBitTensor Labs.
# Original author: Xdev (Enigma / Breaking RSA competition).
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

# GPU-GNFS 480-bit (c145) RSA factorizer for the Enigma validator sandbox.
# Pipeline: msieve GPU polyselect -> gpu_loop GPU lattice sieve -> memfd-backed msieve filter
#           -> GPU block-Lanczos linear algebra (gpu_la.py, CuPy+NVRTC) -> CADO multi-core sqrt.
# All multi-GB intermediates live in anonymous RAM (memfd) so /tmp stays ~100 MB under the
# 1 GiB writable-disk cap. GPU linear algebra is ~11x faster than msieve's CPU block-Lanczos;
# CADO's parallel CRT sqrt replaces msieve's serial 1-CPU sqrt. Automatic CPU fallback if the
# GPU path fails. Runs on the validator's RTX PRO 6000 (Blackwell, sm_120). Verified ~3h12m end-to-end.
# CUDA 12.8 runtime: matches the toolchain the prebuilt binaries are compiled with (sm_120 needs
# CUDA >= 12.8; the GPU msieve won't build on CUDA 13 due to the cuCtxCreate_v4 API change). Keeping
# the runtime on 12.8 also matches gpu_loop's bundled CUDA runtime, avoiding a libcudart major mismatch.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

# system deps (python 3.12 on ubuntu 24.04) + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip bash coreutils findutils \
        libgmp10 libgomp1 libstdc++6 zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Python deps for the GPU linear algebra: CuPy with bundled CUDA compilation toolkit.
# cupy-cuda12x[ctk] bundles libnvrtc + nvcc headers needed for elementwise kernel JIT.
RUN pip3 install --no-cache-dir --break-system-packages \
        numpy==2.4.6 \
        "cupy-cuda12x[ctk]" \
        gmpy2

WORKDIR /app
# GPU lattice siever (gpu_loop) + per-key driver + rels.txt memfd holder (keeps the 9GB relation
# file in RAM, off the 1 GiB /tmp tmpfs)
COPY gpu_nfs_siever/gpu_loop          /app/gpu_nfs_siever/gpu_loop
COPY gpu_nfs_siever/factor_msv.sh     /app/gpu_nfs_siever/factor_msv.sh
COPY gpu_nfs_siever/polyselect_best.sh /app/gpu_nfs_siever/polyselect_best.sh
COPY gpu_nfs_siever/relhold.py        /app/gpu_nfs_siever/relhold.py
COPY gpu_nfs_siever/pick_topk.py      /app/gpu_nfs_siever/pick_topk.py
# msieve toolchain. TWO binaries: 'msieve' = GPU build (polyselect, -g 0 -np);
# 'msieve_la' = QCB_SIZE=32 build for the downstream (-nc1/-nc2/-nc3) — the square-defect fix.
COPY msieve-gpu/msieve              /app/msieve-gpu/msieve
COPY msieve-gpu/msieve_la           /app/msieve-gpu/msieve_la
COPY msieve-gpu/msv_launcher        /app/msieve-gpu/msv_launcher
COPY msieve-gpu/msvrun_gpu.sh       /app/msieve-gpu/msvrun_gpu.sh
COPY msieve-gpu/gpu_la.py           /app/msieve-gpu/gpu_la.py
COPY msieve-gpu/cub/sort_engine.so  /app/msieve-gpu/cub/sort_engine.so
COPY msieve-gpu/stage1_core_sm120.ptx /app/msieve-gpu/stage1_core_sm120.ptx
# multi-core square root: CADO parallel sqrt binary + patched msieve (a,b) dumper + driver.
# Replaces msieve's serial 1-CPU nc3 sqrt with CADO's OpenMP all-core sqrt (~8x faster).
COPY msieve-gpu/cado_sqrt           /app/msieve-gpu/cado_sqrt
COPY msieve-gpu/msieve_dump         /app/msieve-gpu/msieve_dump
COPY msieve-gpu/cado_sqrt.sh        /app/msieve-gpu/cado_sqrt.sh
# orchestrator + challenge I/O contract
COPY enigma_challenges              /app/enigma_challenges
COPY breaking_rsa.py                /app/breaking_rsa.py

RUN chmod +x /app/gpu_nfs_siever/gpu_loop /app/gpu_nfs_siever/factor_msv.sh \
             /app/gpu_nfs_siever/polyselect_best.sh \
             /app/msieve-gpu/msieve /app/msieve-gpu/msieve_la /app/msieve-gpu/msv_launcher \
             /app/msieve-gpu/msvrun_gpu.sh /app/msieve-gpu/gpu_la.py \
             /app/msieve-gpu/cado_sqrt /app/msieve-gpu/msieve_dump /app/msieve-gpu/cado_sqrt.sh \
    && (userdel -r ubuntu 2>/dev/null || true) \
    && (getent passwd miner >/dev/null || useradd -m -u 1000 -s /usr/sbin/nologin miner)

# container runs --read-only: keep CuPy's kernel cache on the writable /tmp tmpfs (default ~/.cupy
# would crash on the read-only rootfs). gpu_la.py also sets this defensively.
# HOME + CUDA_CACHE_PATH on the writable /tmp tmpfs: rootfs is --read-only, so the CUDA driver's
# PTX->sm_120 native cache (default ~/.nv) and CuPy's cache must live on /tmp or they silently
# re-JIT every kernel load (slower) / fail to persist. CUPY_CACHE_DIR keeps CuPy's NVRTC cache too.
ENV PYTHONUNBUFFERED=1 HOME=/tmp CUPY_CACHE_DIR=/tmp/.cupy CUDA_CACHE_PATH=/tmp/.nv
USER miner
ENTRYPOINT ["python3", "/app/breaking_rsa.py"]
