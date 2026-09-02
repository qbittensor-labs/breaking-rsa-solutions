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

# GPU-GNFS 460-bit RSA factorizer for the Enigma validator sandbox.
# Pipeline: msieve GPU polyselect -> gpu_loop GPU lattice sieve -> memfd-backed msieve filter
#           -> GPU block-Lanczos linear algebra (gpu_la.py, CuPy+NVRTC) -> msieve sqrt.
# All multi-GB intermediates live in anonymous RAM (memfd) so /tmp stays ~100 MB under the
# 1 GiB writable-disk cap. GPU linear algebra is ~11x faster than msieve's CPU block-Lanczos
# (4.7 min vs 51 min for a c140 matrix), with automatic CPU fallback if the GPU path fails.
# Runs on the validator's RTX PRO 6000 (Blackwell, sm_120); CPU binaries target x86-64-v4.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# system deps (python 3.12 on ubuntu 24.04) + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip bash coreutils findutils libgmp10 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps for the GPU linear algebra: CuPy (NVRTC runtime kernel compilation) + the CUDA
# header/runtime/nvrtc wheels it compiles against (no full CUDA toolkit / nvcc needed).
RUN pip3 install --no-cache-dir --break-system-packages \
        numpy==2.4.6 \
        cupy-cuda12x \
        nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12

WORKDIR /app
# GPU lattice siever (gpu_loop) + per-key driver + rels.txt memfd holder (keeps the 9GB relation
# file in RAM, off the 1 GiB /tmp tmpfs)
COPY gpu_nfs_siever/gpu_loop        /app/gpu_nfs_siever/gpu_loop
COPY gpu_nfs_siever/factor_msv.sh   /app/gpu_nfs_siever/factor_msv.sh
COPY gpu_nfs_siever/relhold.py      /app/gpu_nfs_siever/relhold.py
COPY gpu_nfs_siever/pick_topk.py    /app/gpu_nfs_siever/pick_topk.py
# msieve toolchain: binary + GPU sort engine + memfd broker + downstream runners (CPU + GPU) + GPU-LA solver
COPY msieve-gpu/msieve              /app/msieve-gpu/msieve
COPY msieve-gpu/msv_launcher        /app/msieve-gpu/msv_launcher
COPY msieve-gpu/msvrun.sh           /app/msieve-gpu/msvrun.sh
COPY msieve-gpu/msvrun_gpu.sh       /app/msieve-gpu/msvrun_gpu.sh
COPY msieve-gpu/gpu_la.py           /app/msieve-gpu/gpu_la.py
COPY msieve-gpu/cub/sort_engine.so  /app/msieve-gpu/cub/sort_engine.so
COPY msieve-gpu/stage1_core_sm86.ptx /app/msieve-gpu/stage1_core_sm86.ptx
# orchestrator + challenge I/O contract
COPY enigma_challenges              /app/enigma_challenges
COPY breaking_rsa.py                /app/breaking_rsa.py

RUN chmod +x /app/gpu_nfs_siever/gpu_loop /app/msieve-gpu/msieve \
             /app/msieve-gpu/msv_launcher /app/msieve-gpu/msvrun.sh \
             /app/msieve-gpu/msvrun_gpu.sh /app/gpu_nfs_siever/factor_msv.sh \
    && (userdel -r ubuntu 2>/dev/null || true) \
    && (getent passwd miner >/dev/null || useradd -m -u 1000 -s /usr/sbin/nologin miner)

# container runs --read-only: keep CuPy's kernel cache on the writable /tmp tmpfs (default ~/.cupy
# would crash on the read-only rootfs). gpu_la.py also sets this defensively.
# HOME + CUDA_CACHE_PATH on the writable /tmp tmpfs: rootfs is --read-only, so the CUDA driver's
# PTX->sm_120 JIT cache (default ~/.nv) and CuPy's cache must live on /tmp or they silently re-JIT
# every kernel load (slower) / fail to persist. CUPY_CACHE_DIR keeps CuPy's NVRTC cache there too.
ENV PYTHONUNBUFFERED=1 HOME=/tmp CUPY_CACHE_DIR=/tmp/.cupy CUDA_CACHE_PATH=/tmp/.nv
USER miner
ENTRYPOINT ["python3", "/app/breaking_rsa.py"]
