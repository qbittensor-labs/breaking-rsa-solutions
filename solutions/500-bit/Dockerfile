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

# GPU-GNFS 500-bit (c151) RSA factorizer for the Enigma validator sandbox.
# Pipeline: msieve GPU polyselect -> gpu_loop GPU lattice sieve -> memfd-backed msieve filter
#           -> GPU block-Lanczos linear algebra (gpu_la.py, CuPy+NVRTC) -> CADO multi-core sqrt.
# All multi-GB intermediates live in anonymous RAM (memfd). /tmp is a 10 GB RAM-backed tmpfs
# (validator: "/tmp:noexec,nosuid,size=10g", constants.py:85 -- NOT the 1 GiB this header claimed
# before 2026-08-02). !! PEAK /tmp IS 6.45 GB OF 10 GB, MEASURED 2026-08-16 -- this header claimed
# "~1.5 GB" until then, understating it 4.3x. (Independently 6.69 GB on the 2026-08-06 run, so it is
# not a one-off.) That leaves only ~35% headroom: a key needing more dependency dumps could hit the
# tmpfs cap, and a tmpfs ENOSPC surfaces as a downstream tooling failure, which this pipeline
# reports as WallTimeFailure like everything else. Peak RAM is comfortable by comparison:
# 34.42 GB of 85 GB. The memfd still matters -- tmpfs bills against the 85 GB RAM cap, so an 8.2 GB
# relation file on /tmp would cost RAM, not save it. GPU linear algebra is ~11x faster than msieve's CPU block-Lanczos;
# CADO's parallel CRT sqrt replaces msieve's serial 1-CPU sqrt. Automatic CPU fallback if the
# GPU path fails. Runs on the validator's RTX PRO 6000 (Blackwell, sm_120).
# c151 params, re-read from build.sh / factor_msv.sh / polyselect_best.sh on 2026-08-11 (NOT
# transcribed from prose): gpu_loop lpb=2^29, LIM=40M, mfb=58/58, J=8192, TCOLS=4, TSPLIT=393216,
# RCOLS/REDUCE_CAP=32, SPB=311, SCAT_COOP_LEAN, FB_SORT, GRIDMUL=128, QMIN=1M, POLY_TOPK=8.
# build.sh remains the source of truth for the compile-time flags.
# !! TWO OF THESE WENT STALE AND ARE CORRECTED HERE (2026-08-21). !!  REL_TARGET is 72M and is a
# CEILING ONLY since 2026-08-20 -- it no longer sets the probe point. CKPT_SPEC_PCT=95 was REPLACED
# by the absolute breakpoint ladder CKPT_BREAKS (first break 65,075,000, step 750,000); the
# percentage survives only as a fallback used to synthesise a ladder if none is configured.
# See Configuration.md 3 -- that file is read from the code and is the one to trust.
#
# !! THE 14,604s THIS HEADER USED TO QUOTE WAS THREE ROUNDS STALE. Do not cite it. !!
# Progression on the hardest keys, each figure a FULL end-to-end at validator spec:
#   ~16,550s  pre-2026-08-09 reference config
#    14,604s  sieve round (LIM 50M->40M, POLY_TOPK 6->8, SCAT_COOP_LEAN, GRIDMUL 64->128) -- §14
#    14,485s  CKPT_SPEC_PCT 97->95 -- §16, key07, 2026-08-10
#    14,104s  key08 on THIS commit, 2026-08-11, 295s inside the wall -- hard-key/key08.json
#    13,971s  key08, EPYC 9555, 2026-08-17, after -nc2 was parallelised (Pipeline.md §14)
# ⚠️ 12,682s  key08, 2026-08-21 -- BUT ON A DIFFERENT AND FASTER BOX. Do NOT read it as progress
#             over the line above: that box is an EPYC 7642 with the RTX PRO 6000 Blackwell
#             *Workstation* Edition at 450W, ~18% faster per lattice than the Server Edition cards
#             the validator fleet is documented to run, and SLOWER on every scalar stage. It needed
#             1,618,000 lattices against the Xeon's 1,618,168 for the same 65,075,000 relations --
#             0.01% apart -- so hosts differ in ms/lat, not in the work. Daily-20260821.md.
# key07 has NOT been re-run end-to-end since the 2026-08-11 fixes; composing its measured stages
# projects ~14,367s but that is a PROJECTION, not a run -- do not quote it as one. (Its last full
# runs were 14,485s at 95%, and 15,050s at CKPT_SPEC_PCT=93, which is why 93 was reverted: 93% sits
# BELOW key07's measured purge crossover of 65,055,083, see Pipeline.md §16b-bis.)
#
# !! AND THESE ARE ONE MACHINE'S NUMBERS, NOT A VALIDATOR'S. !!
# §14a records two boxes producing BIT-IDENTICAL sieve output at +20% wall time, with the gap in
# L2-atomic-bound work -- which §17b measures as 60% of GPU time here. ~610s of the budget is also
# CPU-bound (probe 237s + matrix 228s + sqrt 130s) and the measuring box is Zen 5 while build.sh
# names EPYC 7443 (Zen 3) as a real validator CPU. Recompute the break-even on the hardware you
# actually deploy to (§15a/§15c); a 20% delta erases every margin above.
# (The old header here read lpb=2^31 / LIM=70M / mfb=62/62 / REL_TARGET=150M -- stale placeholders
#  from before calibration, never updated. build.sh is the single source of truth for the flags.)
# CUDA 13.0 runtime: matches BOTH the toolchain the prebuilt binaries are compiled with and the
# validator hosts (CUDA 13.0, driver 580.159.03 / 580.173.02).
#
# !! 13.0 IS A CEILING, NOT A FLOOR — DO NOT BUMP THIS TAG. !!
# gpu_loop and cub/sort_engine.so link cudart STATICALLY, so the build toolkit sets the minimum
# driver each binary needs: 13.0 -> >= 580.65 (fits the validators); 13.1 -> needs a driver NEWER
# than 580.x, so the binary dies with cudaErrorInsufficientDriver before doing any work and the
# solver factors nothing on every challenge. A CUDA 13.1-built gpu_loop was tracked in this repo
# until 2026-08-03 and would have failed on all three validator boxes. There is no compiler in
# this image, so that failure is unrecoverable at runtime.
# (The GPU msieve DOES build on CUDA 13 as of 2026-08-03 — it needed one line for the
#  cuCtxCreate -> cuCtxCreate_v4 remap; see msieve/gnfs/poly/stage1/stage1_sieve_gpu.c.)
# ============================================================================
# SOURCE AND PROVENANCE (submission requirements 3.3(a)-(c))
#
# This image runs PREBUILT binaries so the validator does not compile anything at
# run time. The COMPLETE source for every one of them ships in this submission, so
# any binary here can be reproduced independently:
#
#   gpu_loop        <- gpu_nfs_siever/{gpu_loop.cu,rootfind.c,u256.cuh}
#                      built by gpu_nfs_siever/build.sh, which is the SINGLE SOURCE
#                      OF TRUTH for the compile-time flags. The .cu's in-source
#                      defaults are placeholders; building without build.sh's -D
#                      flags produces a DIFFERENT siever.
#                        nvcc -O3 -use_fast_math -arch=sm_120 \
#                             -Xcompiler -fopenmp,-march=x86-64-v3 $FLAGS \
#                             -o gpu_loop gpu_loop.cu -lgmp
#   msv_launcher    <- msieve-gpu/msv_launcher.c
#                        gcc -O2 -march=x86-64-v3 -o msv_launcher msv_launcher.c
#   cado2msieve     <- cado-filter/cado2msieve.c
#                        gcc -O2 -march=x86-64-v3 -o cado2msieve cado2msieve.c
#   msieve          <- vendor/msieve/ ;  make all CUDA=1
#                      (also emits cub/sort_engine.so and stage1_core_sm120.ptx)
#   msieve_la       <- vendor/msieve/ ;  make all      (no CUDA)
#   msieve_dump     == the SAME non-CUDA binary under a second name. The (a,b)
#                      dumper is NOT a build variant: gnfs/sqrt/sqrt.c gates it on
#                      $DUMP_AB_FILE at RUNTIME and is inert otherwise.
#   freerel dup1 dup2 purge merge replay cado_sqrt
#                   <- vendor/cado-nfs-9bb8fc079.tar.xz + vendor/cado-gzip-strcmp.patch
#                      + vendor/cado-local.sh as local.sh, then
#                        make freerel dup1 dup2 purge merge replay sqrt
#                      (cado_sqrt is CADO's `sqrt`). Verified: this recipe reproduces
#                      the shipped binaries to the byte for all seven.
#
# ---- THIRD-PARTY AND MODIFICATIONS -----------------------------------------
# msieve 1.54, Jason Papadopoulos, PUBLIC DOMAIN (licenses/MSIEVE.LICENSE.txt).
#   MODIFIED BY US, therefore our component under 3.3(a); the COMPLETE modified
#   source tree ships at vendor/msieve/. Our changes:
#     * include/common.h                      MAX_COL_IDEALS 8192 -> 16384
#     * gnfs/sqrt/sqrt.c                      (a,b) dumper, gated on $DUMP_AB_FILE
#     * gnfs/poly/stage1/stage1_sieve_gpu.c   CUDA 13 cuCtxCreate -> cuCtxCreate_v4
#     * gnfs/relation.c                       block gzread + parallel relation parse
#     * common/lanczos/lanczos_pre.c          parallel reduce_matrix row remap
#   Upstream msieve is distributed via SourceForge SUBVERSION; there is no official
#   git repository and hence no upstream tag to pin. vendor/msieve-vs-1.54-reference.diff
#   is an INFORMATIONAL diff against the public 1.54 tree
#   github.com/gchilders/msieve_nfsathome @ 059b79e (itself an NFS@Home fork, so part
#   of that diff is its divergence, not ours).
#   msieve bundles, UNMODIFIED by us: CUB (BSD-3, licenses/CUB.LICENSE.txt), zlib
#   (licenses/ZLIB.LICENSE.txt), mpz_aprcl (LGPLv3, licenses/APRCL.LICENSE.txt).
#
# CADO-NFS, gitlab.inria.fr/cado-nfs/cado-nfs, pinned at 9bb8fc079
# (git describe: 2.3.0-6971-g9bb8fc079), LGPL v2.1+ (licenses/CADO-NFS.COPYING).
#   Genuine third-party. Our ONLY change is one upstream bug fix, disclosed in full
#   as vendor/cado-gzip-strcmp.patch: is_supported_compression_format() compares
#   char const* POINTERS, so -outfmt is always rejected; without compressed
#   intermediates the filter needs ~8.6 GB of the 10 GB tmpfs for the dup1 split.
#   vendor/cado-local.sh pins -O2 -march=x86-64-v3, NOT -march=native: CADO's default
#   build emits AVX-512, which SIGILLs (exit 132) on a validator CPU without it.
#   Verify with:  objdump -d purge | grep -cE '%zmm|\{%k[0-7]\}'   (MUST print 0)
#
# Runtime deps (numpy, cupy-cuda13x[ctk], gmpy2, cuda-toolkit, nvidia-cublas) are
# pulled unmodified from public indexes under their own licences.
# ============================================================================
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

# system deps (python 3.12 on ubuntu 24.04) + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip bash coreutils findutils \
        libgmp10 libgomp1 libstdc++6 zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Python deps for the GPU linear algebra: CuPy with bundled CUDA compilation toolkit.
# cupy-cuda13x[ctk] pulls libnvrtc + headers (needed for gpu_la.py's elementwise kernel JIT) via
# the `cuda-toolkit` metapackage — CUDA 13 dropped the per-library nvidia-*-cuNN wheels that the
# 12.x path pinned individually (the nvidia-*-cu13 names on PyPI are 0.0.1 placeholders).
# cuda-toolkit is pinned to the 13.0.x line for the same driver-floor reason as the base image:
# a 13.1/13.2 cudart would need a driver newer than the validators' 580.x.
RUN pip3 install --no-cache-dir --break-system-packages \
        numpy==2.4.6 \
        "cupy-cuda13x[ctk]" \
        gmpy2 \
    && pip3 install --no-cache-dir --break-system-packages \
        "cuda-toolkit[cublas,cudart,cufft,curand,cusolver,cusparse,nvrtc]==13.0.*" \
        "nvidia-cublas==13.0.*"
# nvidia-cublas is pinned SEPARATELY: the cuda-toolkit pin governs only the metapackage and let
# cublas float to 13.1.1.3. `import cupy` maps libcublasLt eagerly, before any GPU op, so that
# 13.1 library would really be loaded in gpu_la.py on a 580.x-driver validator.

WORKDIR /app
# GPU lattice siever (gpu_loop) + per-key driver + rels.txt memfd holder (keeps the 9GB relation
# file in anonymous RAM rather than on the 10 GB /tmp tmpfs, which is itself RAM-backed)
COPY gpu_nfs_siever/gpu_loop          /app/gpu_nfs_siever/gpu_loop
COPY gpu_nfs_siever/factor_msv.sh     /app/gpu_nfs_siever/factor_msv.sh
COPY gpu_nfs_siever/polyselect_best.sh /app/gpu_nfs_siever/polyselect_best.sh
COPY gpu_nfs_siever/relhold.py        /app/gpu_nfs_siever/relhold.py
COPY gpu_nfs_siever/pick_topk.py      /app/gpu_nfs_siever/pick_topk.py
# GPU-local CPU pinning. INERT AS OF 2026-08-13 -- GPU_NUMA_PIN now defaults to 0 in both
# factor_msv.sh and msvrun_gpu.sh, so neither the sieve nor block-Lanczos is wrapped and this file
# is never executed. It is kept in the image ONLY so GPU_NUMA_PIN=1 remains testable off-validator;
# it is ~7 KB and reading it costs nothing. The premise it was written on is wrong (the validator
# sets --cpus, a CFS quota, not --cpuset-cpus -- so there is no cpuset to be non-local), and it was
# the only code here that behaved differently on a multi-NUMA host than on a single-NUMA one, which
# made it the sole EPYC/Xeon divergence in the container. See the note at the top of factor_msv.sh.
COPY gpu_nfs_siever/gpu_affinity.py   /app/gpu_nfs_siever/gpu_affinity.py
# !! REQUIRED AT RUNTIME -- do not drop !!  factor_msv.sh's adaptive checkpoint ladder shells out
# to ../refilter/run_filter.sh for EVERY matrix probe. It was missing from this image until
# 2026-07-28, so the ladder could never run in the container: the probe failed with
# "No such file or directory", which (before the set -e fix in factor_msv.sh) killed the whole
# script and hung the solver with no output. Same class of defect as the memfd/readlink bug --
# invisible in dev, because dev runs the script from the source tree where refilter/ exists.
COPY refilter/run_filter.sh           /app/refilter/run_filter.sh
# ...and the SAME applies to cado_probe.sh, which is now the DEFAULT probe backend
# (run_filter.sh remains as the fallback). Dropping it would not crash the run -- the
# dispatcher in factor_msv.sh degrades to msieve -- so the image would SILENTLY lose the
# CADO filter and hard keys would fail exactly as before, with nothing in the log to say
# why. That silence is what makes it worth calling out here.
COPY refilter/cado_probe.sh           /app/refilter/cado_probe.sh
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
# CADO-NFS FILTERING (2026-08-04). msieve's filter is the binding constraint on hard
# keys: on key03 it had an excess SURPLUS and still could not build a matrix, asking
# for 1M more relations, and every msieve filter knob is inert. CADO builds a matrix
# from the SAME relations and 323s faster. See cado-filter/cado_filter.sh for the
# measurements and the two integration constraints. All six binaries are built
# -march=x86-64-v3 (NOT AVX-512 -- CADO's default build emits -mavx512f, which would
# SIGILL on a non-AVX-512 validator exactly like the old -march=native gpu_loop).
COPY cado-filter/freerel            /app/cado-filter/freerel
COPY cado-filter/dup1               /app/cado-filter/dup1
COPY cado-filter/dup2               /app/cado-filter/dup2
COPY cado-filter/purge              /app/cado-filter/purge
COPY cado-filter/merge              /app/cado-filter/merge
COPY cado-filter/replay             /app/cado-filter/replay
COPY cado-filter/cado2msieve        /app/cado-filter/cado2msieve
COPY cado-filter/cado_filter.sh     /app/cado-filter/cado_filter.sh
# !! REQUIRED AT RUNTIME, AND ITS ABSENCE IS SILENT -- same defect class as cado_probe.sh above.
# factor_msv.sh arms the pre-dedup warm-up behind `[ -x .../cado_predup.sh ]`, so a missing OR
# non-executable file does not crash anything: the warm-up simply never runs, cado_filter.sh takes
# the one-shot path, and the run is ~128s slower with NOTHING in the log to say why. That is a
# probe going 300s -> 428s on the measurement box, on the critical path (the sieve underneath the
# probe is discarded on MATRIX). The chmod below is therefore load-bearing, not hygiene.
COPY cado-filter/cado_predup.sh     /app/cado-filter/cado_predup.sh
COPY enigma_challenges              /app/enigma_challenges
COPY breaking_rsa.py                /app/breaking_rsa.py

RUN chmod +x /app/gpu_nfs_siever/gpu_loop /app/gpu_nfs_siever/factor_msv.sh /app/refilter/run_filter.sh \
             /app/refilter/cado_probe.sh \
             /app/gpu_nfs_siever/polyselect_best.sh \
             /app/msieve-gpu/msieve /app/msieve-gpu/msieve_la /app/msieve-gpu/msv_launcher \
             /app/msieve-gpu/msvrun_gpu.sh /app/msieve-gpu/gpu_la.py \
             /app/msieve-gpu/cado_sqrt /app/msieve-gpu/msieve_dump /app/msieve-gpu/cado_sqrt.sh \
             /app/cado-filter/freerel /app/cado-filter/dup1 /app/cado-filter/dup2 \
             /app/cado-filter/purge /app/cado-filter/merge /app/cado-filter/replay \
             /app/cado-filter/cado2msieve /app/cado-filter/cado_filter.sh \
             /app/cado-filter/cado_predup.sh \
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
