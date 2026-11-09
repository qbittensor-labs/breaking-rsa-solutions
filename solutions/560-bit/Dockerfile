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

# Dockerfile.c169-gpu -- the RSA-560 SUBMISSION image (Enigma SN63 "Breaking RSA", 560-bit / c169). The validator builds
# it itself from the submission zip (gpu/ops/make_submission.sh puts this file at the zip root as `Dockerfile`, next to
# the vendored CADO tree and the pipeline), so the build is kept lean and fail-closed: pinned base images (tag@digest)
# and pip versions, the CADO-NFS source VENDORED in the context (no clone at build time), no optional stages.
#
#   /app/glas/glas-gpu, gps1       Cognition glas + gps1 (LGPL-2.1, gpu-siever/ of Cognition CADO-NFS b5ca9c4), sm_120,
#                                  the production patch stack of gpu/sandbox_bootstrap.sh (same patches, same order)
#   /app/cado-filter/*, /app/msieve-gpu/*, /app/bwc, /app/pipeline_c169   CADO binaries + the c169 pipeline
#   CuPy stack pinned to the 530 image's versions (gpu_la.py)
# The cuda-sieve/gchilders bench and cofac_cpu (SIEVER=bench only) were dropped 2026-10-25: production is SIEVER=glas.
#
# build (x86_64 host; context = the flattened submission zip, or the repo root with third_party/cado-nfs-public at b5ca9c4):
#   docker build -f docker/Dockerfile.c169-gpu -t enigma-c169-gpu .
# run EXACTLY as the validator (two positional args, stdout = logs + separator + base64 zip; gpu/container_dryrun.sh):
#   docker run --rm --gpus all --network none --read-only --tmpfs /tmp:rw,noexec,nosuid,size=10g --memory 85g --cpus 24 \
#     --user miner enigma-c169-gpu <challenge_id> '{"difficulty":560,"num":"<N>","num_bits":560}'
# runtime = non-root `miner`, ENTRYPOINT breaking_rsa.py, CONTAINER=1 (the validator profile of factor_c169.sh,
# docs/rsa560_container_audit.md), HOME/caches on /tmp, RSA560_LPB1=33 (the large-prime switch, see the ENV below).
# Every binary at -march=x86-64-v3 (never -march=native / AVX-512: the validator CPU is EPYC 9555 or Xeon 6980P).
# make -j is capped at 32 (an unknown validator build host may have 192+ threads and a memory-capped builder).
ARG CUDA_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:7d2f6a8c2071d911524f95061a0db363e24d27aa51ec831fcccf9e76eb72bc92
FROM ${CUDA_IMAGE} AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libgmp-dev libhwloc-dev zlib1g-dev python3 python3-flask python3-requests \
        python3-numpy gzip bc procps util-linux coreutils perl patch ca-certificates binutils \
    && rm -rf /var/lib/apt/lists/*

# ---- CADO-NFS (Cognition tree b5ca9c4, vendored; FLAGS_SIZE 64-bit p_r values: lpb 33 needs them in freerel/renumber/
#      dup2/purge/characters/sqrt). Patches exactly as gpu/sandbox_bootstrap.sh: symlink write-through, SQRT_L00, CHARS_PRE.
#      (cado-characters-parallel.patch is NOT applied: no measured run used it.) -march=x86-64-v3, checked for zmm/kmask.
COPY third_party/cado-nfs-public /src/cado-nfs
COPY pipeline_c169/vendor/cado-symlink-writethrough-cognition.patch gpu/chars_pre_patch.py gpu/sqrt_l00_patch.py gpu/sqrt_l00.hpp gpu/purge_parts_patch.py /src/patches/
# the validator's unzip (copyfileobj) drops every mode bit: restore +x on each script (a '#!' first line) before make
# (CADO's Makefile execs scripts/call_cmake.sh); gpu/ops/make_submission.sh also dereferences the tree's symlinks
RUN set -e; cd /src/cado-nfs; python3 -c 'import os,sys; [os.chmod(q, os.stat(q).st_mode | 0o755) for t in sys.argv[1:] for r, _, fs in os.walk(t) for q in (os.path.join(r, f) for f in fs) if not os.path.islink(q) and open(q, "rb").read(2) == b"#!"]' /src/cado-nfs; \
    [ -x scripts/call_cmake.sh ] || { echo "scripts/call_cmake.sh not executable"; exit 1; }; \
    grep -q b5ca9c4fd3e331edc8978f93830f8de1e8dda70c COGNITION_COMMIT || { echo "CADO tree is not b5ca9c4"; exit 1; }; \
    p=/src/patches/cado-symlink-writethrough-cognition.patch; \
    if patch -p1 -N --dry-run -s < "$p" >/dev/null 2>&1; then patch -p1 -N -s < "$p"; echo "applied $p"; \
    elif patch -p1 -R --dry-run -s < "$p" >/dev/null 2>&1; then echo "already present $p"; \
    else echo "PATCH $p DOES NOT APPLY"; exit 1; fi; \
    python3 /src/patches/sqrt_l00_patch.py sqrt/sqrt.cpp; python3 /src/patches/chars_pre_patch.py linalg/characters.cpp; \
    python3 /src/patches/purge_parts_patch.py filter/purge.cpp; \
    grep -q CADO_PURGE_ORDERED filter/purge.cpp || { echo "PURGE_ORDERED patch missing"; exit 1; }; \
    grep -q SQRT_L00 sqrt/sqrt.cpp || { echo "SQRT_L00 patch missing"; exit 1; }; \
    grep -q CADO_CHARS_SCMAT linalg/characters.cpp || { echo "CHARS_PRE patch missing"; exit 1; }; \
    rm -f local.sh; \
    printf 'CFLAGS="-O3 -march=x86-64-v3"\nCXXFLAGS="-O3 -march=x86-64-v3 -std=c++20"\nFLAGS_SIZE="-DSIZEOF_P_R_VALUES=8"\n' > local.sh; \
    NO_PYTHON_CHECK=1 make -j"$(n=$(nproc); [ "$n" -gt 32 ] && n=32; echo "$n")"; \
    B=$(ls -d build/*/ | head -1); echo "$B" > /src/cado-build-dir; \
    for b in filter/purge filter/merge filter/dup1 filter/dup2 filter/replay sieve/las sieve/makefb sieve/freerel linalg/characters sqrt/sqrt \
             polyselect/polyselect polyselect/polyselect_ropt polyselect/sopt; do \
        [ -x "$B/$b" ] || { echo "missing $B/$b"; exit 1; }; \
        n=$(objdump -d "$B/$b" | grep -cE '%zmm|\{%k[0-7]\}' || true); echo "$b zmm/kmask insns: $n"; [ "$n" = 0 ]; done

# ---- glas-gpu + gps1 (gpu-siever/ of the same tree; the plain CADO build is what they link against) ----
#      CUDA_ARCH=120 (sm_120, RTX PRO 6000 / 5070 / 5090). No NCCL, no MPI: only the siever and the GPU polyselect.
#      Production patch stack, SAME order as gpu/sandbox_bootstrap.sh (every measured siever win; fail closed: a patch error
#      aborts the build instead of shipping a partially patched siever). Added 2026-10-23: the image had only glas_adjust.
COPY gpu/glas_adjust_patch.py gpu/glas_coprime_patch.py gpu/glas_prefetch_patch.py gpu/glas_xfuse_patch.py gpu/glas_region_patch.py \
     gpu/glas_xfgrid_patch.py gpu/glas_session_patch.py gpu/glas_narrow_patch.py gpu/glas_p1_patch.py \
     gpu/glas_empty_pattern_patch.py gpu/gps1_stats_patch.py gpu/gps1_fastsort_patch.py /src/patches/
RUN cd /src/cado-nfs/gpu-siever && set -e && python3 /src/patches/glas_adjust_patch.py src/glas.cpp && \
    python3 /src/patches/glas_coprime_patch.py src/sieve_core_gpu.cu && python3 /src/patches/glas_prefetch_patch.py src/sieve_core_gpu.cu && \
    python3 /src/patches/glas_xfuse_patch.py src/sieve_core_gpu.cu && python3 /src/patches/glas_region_patch.py src/sieve_core_gpu.cu && \
    python3 /src/patches/glas_xfgrid_patch.py src/sieve_core_gpu.cu && python3 /src/patches/glas_session_patch.py src/glas.cpp src/sieve_core_gpu.cu && \
    python3 /src/patches/glas_narrow_patch.py src/sieve_core_gpu.cu && python3 /src/patches/glas_p1_patch.py src/sieve_core_gpu.cu && \
    python3 /src/patches/glas_empty_pattern_patch.py src/sieve_core_gpu.cu && sed -i 's/^#define GLAS_EMPTY_PATTERN 0$/#define GLAS_EMPTY_PATTERN 1/' src/sieve_core_gpu.cu && \
    grep -q "^#define GLAS_EMPTY_PATTERN 1$" src/sieve_core_gpu.cu && \
    python3 /src/patches/gps1_stats_patch.py . && python3 /src/patches/gps1_fastsort_patch.py . && grep -q GPS1_FASTSORT polyselect/gps1_kernels.cu && \
    for m in GLAS_XFUSE_GRID GLAS_SESSION GLAS_IDIV GLAS_REGION_RX GLAS_P1; do grep -q "$m" src/sieve_core_gpu.cu src/glas.cpp || { echo "patch marker $m missing"; exit 1; }; done && \
    B=/src/cado-nfs/$(cat /src/cado-build-dir) && \
    make glas-gpu gps1 CADO_SRC=/src/cado-nfs CADO_BUILD="$B" CUDA_ARCH=120 \
         CXXFLAGS="-O3 -std=c++20 -march=x86-64-v3 -Wall" -j"$(n=$(nproc); [ "$n" -gt 32 ] && n=32; echo "$n")" && \
    mkdir -p /app/glas && cp glas-gpu gps1 /app/glas/ && \
    for m in GLAS_XFUSE_GRID GLAS_SESSION GLAS_IDIV GLAS_REGION_RX GLAS_P1; do strings glas-gpu | grep -q "$m" || { echo "glas-gpu lacks marker $m"; exit 1; }; done && \
    { strings gps1 | grep -q GPS1_FASTSORT || { echo "gps1 lacks GPS1_FASTSORT"; exit 1; }; } && \
    for b in glas-gpu gps1; do n=$(objdump -d "$b" | grep -cE '%zmm|\{%k[0-7]\}' || true); echo "$b zmm/kmask insns: $n"; [ "$n" = 0 ]; done

# ---- pipeline layout (as docker/Dockerfile.c169-cpu) ----
RUN B=/src/cado-nfs/$(cat /src/cado-build-dir) && mkdir -p /app/cado-filter /app/msieve-gpu && \
    for p in sieve/las sieve/makefb sieve/freerel polyselect/polyselect polyselect/polyselect_ropt polyselect/sopt \
             filter/dup1 filter/dup2 filter/purge filter/merge filter/replay; do cp "$B/$p" /app/cado-filter/; done && \
    cp "$B/linalg/characters" /app/msieve-gpu/characters && cp "$B/sqrt/sqrt" /app/msieve-gpu/sqrt && \
    cp "$B/filter/replay" /app/msieve-gpu/replay && cp -r "$B/linalg/bwc" /app/bwc
COPY pipeline_c169 /app/pipeline_c169
# the production downstream (gpu/sandbox_bootstrap.sh section 5): gpu_la_fast.py (GPU combine) + the .cu kernel sources
# gpu_la.py reads at run time + the C host algebra (ctypes-loaded FROM /app: never built or loaded from /tmp, noexec) +
# abx (SQRT_ABX) + gpu_mul.so (CADO sqrt's GPU NTT multiply, PAR_MUL_GPU; dlopen'ed from /app). Portable flags only.
RUN cc -O2 -o /app/cado-filter/cado2msieve2 /app/pipeline_c169/cado-filter/cado2msieve2.c && \
    cp /app/pipeline_c169/cado-filter/cado_filter160.sh /app/pipeline_c169/cado-filter/cado_incdup.sh /app/cado-filter/ && \
    for f in ds_cado.sh gpu_la.py gpu_la_fast.py cado_mat.py segspmv.py ks256.cu ks256g.cu ks256v2.cu ks256v3.cu; do \
        cp /app/pipeline_c169/msieve-gpu/$f /app/msieve-gpu/; done && \
    gcc -O2 -march=x86-64-v3 -shared -fPIC -o /app/msieve-gpu/la128host.so /app/pipeline_c169/msieve-gpu/la128host.c && \
    gcc -O2 -march=x86-64-v3 -shared -fPIC -o /app/msieve-gpu/la256host.so /app/pipeline_c169/msieve-gpu/la256host.c && \
    gcc -O2 -march=x86-64-v3 -pthread -o /app/msieve-gpu/abx /app/pipeline_c169/msieve-gpu/abx.c && \
    nvcc -O3 -std=c++17 -arch=sm_120 -shared -Xcompiler -fPIC,-march=x86-64-v3 -o /app/msieve-gpu/gpu_mul.so /src/cado-nfs/sqrt/gpu_mul.cu -lcudart && \
    chmod +x /app/pipeline_c169/*.sh /app/cado-filter/*.sh /app/msieve-gpu/*.sh && \
    python3 -c 'import os,sys; [os.chmod(q, os.stat(q).st_mode | 0o755) for t in sys.argv[1:] for r, _, fs in os.walk(t) for q in (os.path.join(r, f) for f in fs) if not os.path.islink(q) and open(q, "rb").read(2) == b"#!"]' /app/pipeline_c169 && \
    chmod -R a+rX /app

# ---- runtime image: CUDA runtime + the pinned CuPy stack of the 530 image (gpu_la.py) ----
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:c3fde347d52d578c84fd644bc177bc7ec333feaf11550d990da4084d7612e4c7
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip bash coreutils findutils gzip zstd bc procps util-linux perl \
        libgmp10 libgomp1 libhwloc15 libstdc++6 zlib1g \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages \
        numpy==2.4.6 "cupy-cuda13x[ctk]==14.2.0" gmpy2==2.3.1 \
    && pip3 install --no-cache-dir --break-system-packages \
        "cuda-toolkit[cublas,cudart,cufft,curand,cusolver,cusparse,nvrtc]==13.0.1" \
        "nvidia-cublas==13.0.2.14" "cuda-pathfinder==1.8.2" "nvidia-nvjitlink==13.4.92"
COPY --from=build /app /app
# the stdout protocol (engine/enigma_challenges/solution_output.py, vendored from the validator SDK)
COPY engine/enigma_challenges /app/engine/enigma_challenges
# THE large-prime switch (factor_c169.sh CONTAINER=1 profile): 33 = lpb 31/33 with lambda1 2.0379 (blind GCP keys 27/28
# 3.99 / 3.83 h); 32 = the 31/32 lambda 2.1 profile. Only this ENV turns it on; non-container runs are unchanged.
ENV RSA560_LPB1=33
# S2 (2026-10-25): the deadline telemetry (breaking_rsa.py) keeps its stage-coded status for an overtime run but fires at
# 14385 s instead of 14330 s -- 15 s before the validator's own kill at 14400 s is all the margin the emit needs; a tooling
# failure exits 3 (RunFailure) at once instead of holding. TELEMETRY=0 switches the watchdog off entirely.
ENV TELEMETRY_DEADLINE=14385
# S2: CADO's native binary renumber table (freerel -renumber renumber.bin, mmap-shared by every dup2): no 12.8 GB inflated
# text table in RAM, incremental dedup ~10x faster per step (codex-lpb, exact mapping equivalence verified, c100 PASS).
# text = the S1 path. Arm A of the S2 gate ships this Dockerfile with RENUMBER_FORMAT=text.
ENV RENUMBER_FORMAT=binary
# S3 candidate (2026-10-25 18:30 MDT, screens sb100-103 at the lpb 31/33 regime, rel/s at equal GPU span vs the S2 regime,
# production relations kept 0.9995+): mfb1 66 (+4.1..4.2% at q 12.5M/19M; admits two full 33-bit large primes) with lambda1
# 2.08 (+1% alone, +5.9% combined at q 19M), lim1 600M (+2.6..2.7% at q 12.5M/19M, +5% relations/window). Raw-rate gains:
# the relation target may rise (more two-large-prime relations); the S3 in-container e2e prices the net. S2B = this line
# absent (rsa560 1a12a47). RSA560_* switches (factor_c169.sh CONTAINER=1 profile) apply only to the c169 profile: the
# small-key gates (c100, workbench c330) set their own MFB1/LAMBDA1/ALIM (a plain ENV MFB1=66 broke the c100 gate).
# S4b (22:45 MDT): the S3 regime line is OFF -- key 3 measured the regime's weaker relations pushing the matrix ~20M raw
# later (371M: excess -4.23M vs the S2 regime's 327M: -7.39M), a wash against its +5% rel/s. S4b = S2B + the exact knobs.
# ENV RSA560_MFB1=66 RSA560_LAMBDA1=2.08 RSA560_ALIM=600000000
# S4a candidate (2026-10-25 20:30 MDT, docs/research/agents_2026-10-25/): four EXACT, measured knobs on top of S3b, validated
# by the key-4 in-container e2e. LA_SEG_HINT=2: L2 cache-priority hints, bit-identical, -0.45 ms/it ABAB on key 29's matrix
# (lahint_sb101.txt, ~30 s/key). SQRT_POOL=1 SQRT_STAGGER=30 SQRT_TPD=12: sliding sqrt window with staggered starts, side
# files identical, 174 vs 194 s/dep (sqpool_sb103.txt, ~15-30 s/key). LA_RANK64=check: the rank-64 epilogue's exact fast
# path (sqrt_tail.md #1, 22 s/key) run beside the Python path and compared (FATAL on mismatch) -- S4 ships =fast after a
# MATCH. RSA560_CPU24=1: the GPU-idle stages on all 24 vCPUs (orchestration.md #1, container.md #1; est. 40-60 s/key).
# TELEMETRY_LATE_DEADLINE: a run in its LA >= 50 % or sqrt at 14,385 s is held to 14,396 s instead of killed.
# S4c (01:55 MDT 10-26): S4b WITHOUT the sqrt pool. S4a key 4 in-container at c169 (01:42): the first pooled sqrt dependency
# ABORTED (core dump) 4 s after its start (OMP=12, pool of 2) and no wall was left for the retry ladder -> RunFailure; the
# same pool code passed c100 in-container (S4b gate: "pool of 1", FACTOR on dep 0) and the serial path passed c169
# in-container (S2B replay gate, OMP=10). Until the pool's c169 failure is understood, ship the serial sqrt at OMP=10.
# CALIBRATION SUBMISSION (06:55 MDT 10-26): RSA560_CALIBRATE=1 makes the image measure the validator host (frozen glas
# window, makefb, LA micro-benchmark; pipeline_c169/calib/) and report the predicted S4rung wall through the five status
# codes instead of factoring. OFF in production images; the calibration zip is built from a branch that sets it.
# ENV RSA560_CALIBRATE=1
# S4d (05:25 MDT 10-26): S4c minus LA_RANK64=check -- the only knob besides the pool that touches the dependencies
# handed to CADO sqrt (S4a key 4's dep 0 aborted in 4 s; its LA line showed "rank 193, 252 non-zero / 65 independent"
# vs arm B's "rank 192, 64 non-zero / 64 independent"). Fallback if S4rung's pooled sqrt aborts too.
# S5 (10-26, branch s5): the S4rung knob set (in-container gate on arm B's key FACTORED in 14,255 s, ledger 2026-10-26;
# S4rung-serial/S4d variants dropped), with LA_RANK64=fast (the S4b gate's =check printed MATCH on arm B's N, HANDOFF.md
# S4b; -22 s/key) and LA_GRAPH=1 (CUDA-graph LA iteration, -16 s, adopted 10-25), plus the overnight-cohort knobs below.
# LA_GRAPH=1 REMOVED (10:20 MDT 10-26, re-audit): gpu_la.py run256 exits FATAL when LA_GRAPH=1 meets the production
# LA_DEFER_ROWS=64 (factor_c169.sh), ds_cado.sh then retries on the 64-bit no-defer path (~2x LA) -> every c169 key misses.
# TELEMETRY_LATE_DEADLINE 14396 -> 14460 (11:25 MDT 10-26): the validator (qbittensor-labs/enigma solution_container_manager.py)
# checks every 30 s and stops a container only when now - StartedAt > 14,400 s, so a run finishing up to ~14,400 + 0..30 s
# still passes; a run in its LA or sqrt at 14,385 s now never kills itself (the validator decides: pass or WallTimeFailure =
# 'near miss in the final stages'); runs still sieving/merging at 14,385 s keep reporting their minutes-over bucket.
ENV LA_SEG_HINT=2 SQRT_POOL=1 SQRT_STAGGER=30 SQRT_TPD=12 LA_RANK64=fast RSA560_CPU24=1 TELEMETRY_LATE_DEADLINE=14460 TELEMETRY_MAP=overshoot
# S4rung regime (unchanged): lpb0 32 with lambda0 2.0352 (RSA560_LPB0 switch, factor_c169.sh CONTAINER=1 block), first probe
# at 0.92 x the regime's crossover (FIRST_PROBE_AUTO, 406M at 32/33), matrix-era probes at 12 threads (PROBE_LATE_NT) with the
# cpu.stat sampler (RES_CPU). S5 additions (docs/research/agents_2026-10-26/): STOP_FINAL_GROUPS=16 REPLAY_INDEX_PLAIN=1
# (downstream.md #1/#3: 16 one-slice dup2 groups for the stop-time incdup, plain c.index from replay; est. -25..-50 s);
# STOP_SHAPE=1 STOP_NFAC=0.246 STOP_PROBE_TAX=6 STOP_PROBE_EVERY=2 (stop_rule.md #1: one-point shape-prior stop rule, probes
# every 2 chunks after the first matrix; defaults STOP_KAPPA=9.9 STOP_SX=0.22 STOP_SHAPE_A=0.102 STOP_SHAPE_B=0.44
# STOP_MINPTS=1 apply; est. ~-35 s); RSA560_CPU_LAS=1 (idle_cpu.md #1: CADO las co-sieves q >= 20M on the idle vCPUs beside
# glas -- LAS_A 29, LAS_QMIN 20M, LAS_NCPU 24, glas host threads RSA560_CPU_LAS_GT default 4; fail-soft; est. -40..-120 s;
# RSA560_CPU_LAS_GT=16 keeps glas at -t 16 if a -t 4 penalty shows -- r13's C4 arm was not run). PS_MEMO_ROLL=1 (idle_cpu.md #2,
# PS_ADAPT extend keys only): the CPU tail runs on gps1 batches 4-5 while the GPU searches, so pass b re-runs only batch 6's
# newcomers (harness: same poly.cado; ropt miss 3 -> 1 of 20). The pass-b ropt memo itself was broken until S5 (memo v2:
# multithreaded polyselect_ropt prints all inputs first, the old parser kept ONE entry -> "ropt miss 20 of 20" on arm B).
ENV RSA560_LPB0=32 FIRST_PROBE_AUTO=1 PROBE_LATE_NT=12 RES_CPU=1 STOP_FINAL_GROUPS=16 REPLAY_INDEX_PLAIN=1 STOP_SHAPE=1 STOP_NFAC=0.246 STOP_PROBE_TAX=6 STOP_PROBE_EVERY=2 RSA560_CPU_LAS=1 PS_MEMO_ROLL=1
# STOP_LA_SCALE (slow_card.md; default 1 = unchanged): the cost rule's LA price relative to the GCP g4 card its constants were
# calibrated on; a molab-class RTX PRO 6000 measures x1.28 (laparts_sb111.txt; -33 s/key modelled there, 0 on GCP). It scales
# the secant rule's LA term AND the STOP_SHAPE kappa (set this, not a scaled STOP_KAPPA). Unset until the validator's card
# class is known from S4rung's overshoot bucket. Range [0.5, 2]; out of range dies at startup.
# ENV STOP_LA_SCALE=1.28
# CANDIDATE, default OFF (docs/research/agents_2026-10-25/container.md, idea 1): the CPU-bound stages that run with the GPU
# idle (polyselect tail, makefb, stop-time incdup + purge, merge, replay, characters/abx, sqrt) on all 24 vCPUs of the
# validator quota instead of the sieve phase's NCPU=20 budget (factor_c169.sh CONTAINER=1 profile, STAGE_NT). Gates before
# it is switched on: c100 under the official flags (the "[0s] threads:" banner line shows the resolved profile), a saved-matrix
# merge/replay/sqrt AB/BA inside the exact container, then one blind c169 e2e. Enable with the next line uncommented.
# ENV RSA560_CPU24=1
# CANDIDATES, default OFF (docs/research/agents_2026-10-26/downstream.md #1, #3): STOP_FINAL_GROUPS=16 -- the stop-time incdup
# of the last chunk ran as STOP_FINAL_GROUPS=INC_GROUPS=2 dup2 processes at 363-432 % CPU for 50 s on the 24 idle vCPUs (arm B
# docker stats, 2026-10-25; 55 s of the 136 s handoff); 16 one-slice groups (cado_incdup.sh:72 allows 16) -> est. 15-25 s.
# REPLAY_INDEX_PLAIN=1 -- replay writes c.index plain instead of through one gzip (13.0 s serial of replay's 50 s, key 28).
# Gates: c100 under the official flags, then replay_gate.sh on ckpt-key29f with EXTRA_ENV (see gpu/gcp/replay_gate.sh).
# S5 ships both on the RSA560_LPB0 line above.
# ENV STOP_FINAL_GROUPS=16 REPLAY_INDEX_PLAIN=1
# TELEMETRY_LATE_DEADLINE (candidate, default unset; container.md idea 3): a run that is in its last stages (LA >= 50 % or
# sqrt) at TELEMETRY_DEADLINE may still finish before the validator's 14,400 s; hold the telemetry kill until this later
# second instead (python t0 is < 1 s after docker's StartedAt; the payload write is milliseconds).
# ENV TELEMETRY_LATE_DEADLINE=14396
# Non-root runtime user (the validator runs --user miner --read-only); /app stays root-owned and read-only.
# chmod a+rX (here for engine/, in the build stage for the rest of /app): readable by miner whatever umask the unzip had
RUN chmod -R a+rX /app/engine && useradd -r -s /usr/sbin/nologin -d /tmp -U miner && python3 -c "import cupy, numpy; print('cupy', cupy.__version__)"
ENV CF_DIR=/app/cado-filter MSV_DIR=/app/msieve-gpu BWC_DIR=/app/bwc GLAS=/app/glas/glas-gpu \
    CONTAINER=1 HOME=/tmp TMPDIR=/tmp CUPY_CACHE_DIR=/tmp/.cupy CUDA_CACHE_PATH=/tmp/.nv \
    PYTHONPATH=/app/engine PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /tmp
USER miner
# the validator: docker run ... IMAGE <challenge_id> '<problem_json>'  (debug shell: --entrypoint bash)
ENTRYPOINT ["python3", "/app/pipeline_c169/breaking_rsa.py"]
