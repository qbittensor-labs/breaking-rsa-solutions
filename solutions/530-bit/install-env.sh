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

# =============================================================================
# install-env.sh — reproduce the 500-bit (c151) solve environment on a fresh box
# =============================================================================
# Target hardware: NVIDIA RTX PRO 6000 (Blackwell, sm_120).
#
# Everything here targets CUDA *13.0* — EXACTLY 13.0, not 13.1/13.2 and not 12.8.
# The validator hosts run CUDA 13.0 on driver 580.159.03 / 580.173.02, and
# gpu_loop / sort_engine.so link the CUDA runtime STATICALLY, so the build
# toolkit sets the MINIMUM DRIVER each binary requires:
#     12.8 -> needs >= 570      (works, but splits the stack across two toolkits)
#     13.0 -> needs >= 580.65   -- matches the validator hosts exactly   <-- USE
#     13.1 -> needs a driver NEWER than 580.x, so it will NOT START on a
#             validator: cudaErrorInsufficientDriver, factoring nothing.
# A 13.1-built gpu_loop was tracked in this repo until 2026-08-03. Always verify
# after a rebuild:   strings gpu_loop | grep -oE 'release [0-9]+\.[0-9]+'
#
#   * gpu_loop, msieve (GPU polyselect), sort_engine.so and
#     stage1_core_sm120.ptx are all compiled against CUDA 13.0.
#     (msieve needed a one-line fix for CUDA 13's cuCtxCreate -> cuCtxCreate_v4
#     remap; see msieve/gnfs/poly/stage1/stage1_sieve_gpu.c. The older claim that
#     "the GPU msieve will NOT build on CUDA 13" was true but was a single
#     too-few-arguments error, not a structural blocker.)
#   * msieve_la / msieve_dump / msv_launcher / cado_sqrt contain NO CUDA at all
#     (no static cudart, no libcuda) and are unaffected by the toolkit choice.
#   * CuPy's runtime libs (cudart, nvrtc) are pinned to 13.0.x via cupy-cuda13x.
#   * We NEVER touch the host NVIDIA driver (it is host-injected; apt must not
#     replace it). We install the toolkit ONLY, never the `cuda`/`cuda-drivers`
#     metapackages.
#
# Usage:
#   bash install-env.sh            # runtime deps only (run the prebuilt binaries)
#   bash install-env.sh --build    # ALSO install nvcc 13.0 to rebuild gpu_loop
#
# Idempotent: safe to re-run.
# =============================================================================
set -euo pipefail

WITH_BUILD=0
[[ "${1:-}" == "--build" ]] && WITH_BUILD=1

SUDO=""
[[ "$(id -u)" -ne 0 ]] && SUDO="sudo"

PIP="pip3 install --no-cache-dir --break-system-packages"

echo "=============================================================="
echo " 500-bit (c151) environment installer  —  CUDA 13.0 pinned"
echo " build toolchain: $([[ $WITH_BUILD == 1 ]] && echo 'YES (nvcc 13.0)' || echo 'no  (runtime only)')"
echo "=============================================================="

# -----------------------------------------------------------------------------
# 1. System runtime libraries (what the prebuilt binaries dynamically need)
#    gpu_loop -> libgmp, libstdc++, libgomp ; msieve -> libcuda (host-injected)
# -----------------------------------------------------------------------------
echo "[1/4] apt runtime libraries ..."
$SUDO apt-get update -qq
$SUDO apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgmp10 libgomp1 libstdc++6 zlib1g \
        libgmp-dev
# libgmp-dev: gmp.h — needed by gmpy2's build fallback and by the gpu_loop rebuild.

# -----------------------------------------------------------------------------
# 2. (optional) CUDA 13.0 compiler — ONLY the toolkit, never the driver.
#    Required to rebuild gpu_loop (build.sh) and the GPU msieve / sort_engine.so
#    / stage1_core_sm120.ptx (msieve/Makefile, `make all CUDA=1`).
#    EXACTLY 13.0: a 13.1+ toolkit produces binaries that will not start on the
#    validators' 580.x driver. See the header.
# -----------------------------------------------------------------------------
if [[ $WITH_BUILD == 1 ]]; then
  echo "[2/4] CUDA 13.0 compiler (nvcc) ..."
  # Ensure the NVIDIA CUDA apt repo is present. On the Vast base image it already
  # is; on a bare box add it (ubuntu2404 shown — adjust distro if different).
  # NB: no pipe into `grep -q` here — it exits on first match and SIGPIPEs apt-cache,
  # which under `set -o pipefail` makes the pipeline return 141 and this guard misfire
  # precisely when the package IS available. Match on the captured string instead.
  _cuda_pol="$(apt-cache policy cuda-nvcc-13-0 2>/dev/null || true)"
  if [[ "$_cuda_pol" != *Candidate* ]]; then
    echo "  NVIDIA CUDA apt repo not found — add it, e.g.:"
    echo "    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "    \$SUDO dpkg -i cuda-keyring_1.1-1_all.deb && \$SUDO apt-get update"
    exit 1
  fi
  # nvcc 13.0 + cudart dev/static lib. NOT `cuda` / `cuda-drivers` (those pull the driver).
  $SUDO apt-get install -y cuda-nvcc-13-0 cuda-cudart-dev-13-0
else
  echo "[2/4] skipping nvcc (runtime-only). Re-run with --build to rebuild gpu_loop."
fi

# -----------------------------------------------------------------------------
# 3. Python deps for the GPU linear algebra (gpu_la.py: CuPy + NVRTC JIT).
#    cupy-cuda13x[ctk] pulls the CUDA 13.x libs via the `cuda-toolkit` metapackage
#    (CUDA 13 dropped the per-library nvidia-*-cuNN wheels the 12.x path pinned —
#    the nvidia-*-cu13 names on PyPI are 0.0.1 placeholders, do NOT use them).
#    We then PIN cuda-toolkit to the 13.0.x line so the loaded cudart/nvrtc are
#    13.0, matching the validators' driver floor. A 13.1/13.2 runtime would need
#    a driver newer than 580.x and would fail at import/JIT time.
#    Do NOT downgrade CuPy — older wheels lack Blackwell sm_120 support.
# -----------------------------------------------------------------------------
echo "[3/4] Python: numpy, gmpy2, cupy (CUDA 13.0-pinned) ..."
# PURGE the CUDA-12 wheels first. This script used to install cupy-cuda12x and
# the nvidia-*-cu12 libs; pip leaves BOTH generations installed side by side and
# CuPy then warns "multiple CuPy packages are installed ... may not function
# correctly", because `import cupy` can bind to either one. Without this the
# installer is NOT idempotent across the 12.8 -> 13.0 switch.
#
# !! cupy-cuda12x AND cupy-cuda13x UNPACK INTO THE SAME `cupy/` DIRECTORY. !!
# Uninstalling only the 12x wheel DELETES that shared directory while the 13x
# wheel's metadata still claims it, so pip reports "Requirement already
# satisfied" and `import cupy` then dies with ModuleNotFoundError. Measured
# 2026-08-03. Both CuPy wheels must therefore be removed together and the 13x
# one reinstalled from scratch below.
pip3 uninstall -y --break-system-packages \
      cupy-cuda12x cupy-cuda13x cupy \
      nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 \
      nvidia-nvjitlink-cu12 nvidia-cublas-cu12 \
      nvidia-cufft-cu12 nvidia-curand-cu12 \
      nvidia-cusolver-cu12 nvidia-cusparse-cu12 >/dev/null 2>&1 || true
$PIP numpy==2.4.6 "cupy-cuda13x[ctk]" gmpy2
$PIP "cuda-toolkit[cublas,cudart,cufft,curand,cusolver,cusparse,nvrtc]==13.0.*"
# The cuda-toolkit pin governs the METAPACKAGE; the individual libraries float.
# Measured 2026-08-03: it resolved nvidia-cublas to 13.1.1.3, and `import cupy`
# maps libcublasLt EAGERLY (before any GPU op), so a 13.1 library really does
# land in gpu_la.py's address space on the validator. Pin it back to the 13.0
# line for the same driver-floor reason as everything else here.
$PIP "nvidia-cublas==13.0.*"

# -----------------------------------------------------------------------------
# 4. Verify: imports + GPU + sm_120 NVRTC JIT + the historically-failing op.
#    CuPy needs a writable cache; keep it off any read-only rootfs.
# -----------------------------------------------------------------------------
echo "[4/4] verification ..."
export HOME="${HOME:-/tmp}" CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-/tmp/.cupy}" CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/.nv}"
python3 - <<'PY'
import ctypes, glob, sys
import numpy, gmpy2, cupy as cp
print("  numpy :", numpy.__version__)
print("  gmpy2 :", gmpy2.version())
print("  cupy  :", cp.__version__, "(build stamp CUDA %d)" % cp.cuda.runtime.runtimeGetVersion())
# actual loaded cudart version (not the build stamp):
so = sorted(glob.glob("/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13*")
          + glob.glob("/usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.13*"))
if so:
    lib = ctypes.CDLL(so[0]); v = ctypes.c_int()
    lib.cudaRuntimeGetVersion(ctypes.byref(v))
    ver = "%d.%d" % (v.value//1000, (v.value%1000)//10)
    print("  loaded cudart .so :", ver)
    # HARD GATE: 13.1+ needs a driver newer than the validators' 580.x and will
    # not start there. 13.0 is the only accepted runtime.
    if not ver.startswith("13.0"):
        sys.exit("FATAL: cudart %s loaded, expected 13.0.x — see the header of this script" % ver)
else:
    print("  loaded cudart .so : (not found under dist-packages; check the pin)")
p = cp.cuda.runtime.getDeviceProperties(0)
print("  device:", p['name'].decode(), "sm_%d%d" % (p['major'], p['minor']))
# sm_120 NVRTC JIT + the argsort-300M path that used to crash:
_ = float(cp.sqrt(cp.arange(1_000_000, dtype=cp.float64)*2+1).sum())
_ = cp.argsort(cp.arange(5_000_000, dtype=cp.int64)[::-1])
cp.cuda.Stream.null.synchronize()
print("  sm_120 JIT + argsort : OK")
print("=== ENV OK ===")
PY

echo "=============================================================="
echo " Done. To rebuild the GPU binaries (needs --build toolchain), ALL at 13.0:"
echo "   cd gpu_nfs_siever && PATH=/usr/local/cuda-13.0/bin:\$PATH bash build.sh"
echo "   cd msieve        && PATH=/usr/local/cuda-13.0/bin:\$PATH make all CUDA=1 -j\$(nproc)"
echo "     -> copy msieve, cub/sort_engine.so, stage1_core_sm120.ptx into msieve-gpu/"
echo ""
echo " build.sh is the single source of truth for the flags (calibrated config:"
echo " lpb=2^29 / LIM=50M / mfb=58-58 / FLAT_COOP2 / REDUCE_CAP=8). Do NOT hand-roll"
echo " the nvcc line — an out-of-date copy silently builds the wrong lpb, which"
echo " changes the relation crossover by ~3x."
echo ""
echo " The 13.0 PATH prefix is REQUIRED and 13.0 is a CEILING, not a floor: these"
echo " binaries link cudart STATICALLY, and a 13.1+ build needs a driver newer than"
echo " the validators' 580.x — it would not start at all. Verify every rebuild:"
echo "   strings gpu_loop | grep -oE 'release [0-9]+\\.[0-9]+'    # must be 13.0"
echo "=============================================================="
