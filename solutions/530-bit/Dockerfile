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

# ============================================================================
# GPU-GNFS RSA factorizer for the Enigma validator sandbox — 512-bit, c155.
#
# Factors a 155-digit semiprime by the General Number Field Sieve, using the
# GPU for the two dominant stages (lattice sieve, linear algebra) and the
# otherwise-idle CPU cores for everything that parallelises off them.
#
# ---- HOW A RUN WORKS -------------------------------------------------------
# Entry: breaking_rsa.py reads N from /challenge_input/challenge_input.json and
# drives gpu_nfs_siever/factor_msv.sh through four stages.
#
# [1/4] POLYSELECT — polyselect_best.sh
#   msieve's GPU `-np` slice search sweeps the productive c5 range in parallel
#   slices, pooled and funnelled to the top 8 by msieve-E. Those 8 then go
#   through a measured yield bake-off: each is trial-sieved over three q-windows
#   and scored on UNIQUE relations per lattice (DUPSUP), because no analytic
#   score ranks candidates reliably. Emits poly.cado + the msieve factor base.
#
# [2/4] SIEVE — gpu_loop (GPU) + CADO las (CPU), concurrently
#   gpu_loop runs the lattice sieve over special-q from QSPLIT upward; las
#   covers [QMIN, QSPLIT) on the cores gpu_loop does not need. Relations stream
#   into an anonymous-memory (memfd) file, not onto /tmp.
#   The stage is RELATION-BOUND, not time-bound: an adaptive probe ladder
#   snapshots the relation set at increasing breakpoints and tries to build a
#   matrix from each, while sieving continues. The first snapshot that yields a
#   solvable matrix stops the sieve — so the run sieves as little as the key
#   actually needs rather than to a fixed target.
#
# [3/4] FILTER + LINEAR ALGEBRA — cado_filter.sh, then gpu_la.py
#   CADO's filter chain (dup1/dup2/purge/merge/replay) reduces relations to a
#   sparse matrix; block-Lanczos then solves it on the GPU (CuPy + NVRTC).
#
# [4/4] SQUARE ROOT — CADO's multi-core sqrt over the dependencies, yielding p
#   and q. Automatic CPU fallback if any GPU stage fails.
#
# Output contract: text logs, a separator line, then a base64 zip containing
# result.json, all on stdout. Runs as USER miner under --read-only with only
# /tmp writable, on one RTX PRO 6000 (Blackwell, sm_120).
#
# ---- CONSTANTS ARE COMPUTED, NOT HARD-CODED --------------------------------
# sieve_params.py and polyselect_params.py derive everything size-dependent
# from digits(N) — QMIN, QSPLIT, the relation ceiling, the probe ladder, the
# bake-off windows. Do not quote a value here; ask the generator.
# Other key sizes route to different engines and constant sets inside those two
# files: nothing here targets them, but they are live.
#
# Compile-time flags are size-independent; build.sh is their only source of
# truth: lpb=2^29, LIM=40M, mfb=58/58, J=8192, TCOLS=4, TSPLIT=393216,
# RCOLS/REDUCE_CAP=32, SPB=311, SCAT_COOP_LEAN, FB_SORT, GRIDMUL=128.
#
# ---- THREE THINGS THAT WILL BREAK THIS IMAGE -------------------------------
# 1. Bumping the CUDA tag. 13.0 is a CEILING. gpu_loop and cub/sort_engine.so
#    link cudart statically, so the build toolkit sets each binary's minimum
#    driver: 13.0 needs >= 580.65 (validators have 580.x); 13.1 needs newer,
#    and every run dies with cudaErrorInsufficientDriver before doing any work.
#    No compiler ships in this image, so that is unrecoverable at runtime.
# 2. Building anything with AVX-512. CADO's default build emits -mavx512f and
#    SIGILLs (exit 132) on a validator CPU without it. vendor/cado-local.sh and
#    build.sh both pin -march=x86-64-v3.
#      verify:  objdump -d <bin> | grep -cE '%zmm|\{%k[0-7]\}'   must print 0
# 3. Assuming /tmp is roomy. It is a 10 GB RAM-backed tmpfs, peak use ~6.5 GB.
#    Multi-GB intermediates live in anonymous RAM (memfd) precisely because
#    tmpfs bills against the 85 GB memory cap. A tmpfs ENOSPC surfaces as a
#    downstream tooling failure, indistinguishable in the log from anything
#    else that ends the run.
# ============================================================================
# SOURCE AND PROVENANCE (submission requirements 3.3(a)-(c))
#
# This image runs PREBUILT binaries so the validator compiles nothing at run
# time. Complete source for every one ships in this submission:
#
#   gpu_loop      <- gpu_nfs_siever/{gpu_loop.cu,rootfind.c,u256.cuh}, built by
#                    build.sh -- the SINGLE SOURCE OF TRUTH for the -D flags.
#                    The .cu's in-source defaults are placeholders; building
#                    without build.sh's flags produces a DIFFERENT siever.
#                      nvcc -O3 -use_fast_math -arch=sm_120 \
#                           -Xcompiler -fopenmp,-march=x86-64-v3 $FLAGS \
#                           -o gpu_loop gpu_loop.cu -lgmp
#   msv_launcher  <- msieve-gpu/msv_launcher.c   gcc -O2 -march=x86-64-v3
#   cado2msieve   <- cado-filter/cado2msieve.c   gcc -O2 -march=x86-64-v3
#   msieve        <- vendor/msieve/ ; make all CUDA=1
#                    (also emits cub/sort_engine.so and stage1_core_sm120.ptx)
#   msieve_la     <- vendor/msieve/ ; make all   (no CUDA)
#   msieve_dump   == the SAME non-CUDA binary under a second name. The (a,b)
#                    dumper is not a build variant: gnfs/sqrt/sqrt.c gates it on
#                    $DUMP_AB_FILE at RUNTIME and is inert otherwise.
#   freerel dup1 dup2 purge merge replay cado_sqrt polyselect polyselect_ropt
#                 <- vendor/cado-nfs-9bb8fc079.tar.xz + cado-gzip-strcmp.patch
#                    + cado-symlink-writethrough.patch (freerel/dup1/dup2/purge/merge/replay:
#                    utils/gzip.cpp, utils/fstream_maybe_compressed.cpp, filter/dup2.cpp write
#                    THROUGH an existing symlink instead of tmp+rename -- REQUIRED by the memfd
#                    pool; stock binaries replace the memfd links with /tmp files)
#                    + cado-local.sh as local.sh, then `make <targets>`
#                    (cado_sqrt is CADO's `sqrt`).
#   characters sqrt  (msieve-gpu/, RSA-530 CADO-native downstream, 2026-09-27)
#                 <- the SAME CADO tree @ 9bb8fc079 + BOTH vendor patches
#                    (cado-gzip-strcmp.patch + cado-symlink-writethrough.patch)
#                    + cado-local.sh as local.sh (-march=x86-64-v3), then
#                    `make characters sqrt` -- script: vendor/cado-build_chars.sh
#                    (= /tmp/rsa530/cadobuild/build_chars.sh, run in
#                    nvidia/cuda:13.0.1-devel-ubuntu24.04). CADO's linalg/characters
#                    and sqrt/sqrt; objdump shows 0 AVX-512 instructions.
#                    characters additionally carries vendor/cado-characters-parallel.patch
#                    (the exact diff of the tree it was built from, incl. the static-zlib CMake hunk
#                    and the fail-loudly-on-truncation edits; build: vendor/cado-build_chars_parallel.sh)
#                    (parallel zlib reader + threaded character evaluation; output
#                    byte-identical to stock on c111 and c160; ~15 s vs ~300 s at c160;
#                    fails loudly on truncated input; CADO_CHARS_STOCK=1 forces the
#                    stock code path, which ds_cado.sh uses as its retry).
#                    md5 characters c1ff4cbf212e88df0ed6d226bcc3c5cb
#                        sqrt       4b8509c42c6fc14c68330403a015ca0b
#
# ---- THIRD-PARTY AND MODIFICATIONS ------------------------------------------
# msieve 1.54, Jason Papadopoulos, PUBLIC DOMAIN (licenses/MSIEVE.LICENSE.txt).
#   MODIFIED BY US, therefore our component under 3.3(a); the complete modified
#   tree ships at vendor/msieve/. Our changes:
#     * include/common.h                     MAX_COL_IDEALS 8192 -> 16384
#     * gnfs/sqrt/sqrt.c                     (a,b) dumper, gated on $DUMP_AB_FILE
#     * gnfs/poly/stage1/stage1_sieve_gpu.c  CUDA 13 cuCtxCreate -> _v4
#     * gnfs/relation.c                      block gzread + parallel parse
#     * common/lanczos/lanczos_pre.c         parallel reduce_matrix row remap
#   Upstream is distributed via SourceForge Subversion -- no official git repo,
#   hence no tag to pin. vendor/msieve-vs-1.54-reference.diff is INFORMATIONAL,
#   against github.com/gchilders/msieve_nfsathome @ 059b79e (itself an NFS@Home
#   fork, so part of that diff is its divergence, not ours).
#   msieve bundles, UNMODIFIED: CUB (BSD-3), zlib, mpz_aprcl (LGPLv3) --
#   licenses/ has each.
#
# CADO-NFS, gitlab.inria.fr/cado-nfs/cado-nfs @ 9bb8fc079 (2.3.0-6971-g9bb8fc079),
# LGPL v2.1+ (licenses/CADO-NFS.COPYING). Genuine third-party. Our changes, each disclosed in full:
#   vendor/cado-gzip-strcmp.patch -- upstream bug: is_supported_compression_format() compares
#     char const* POINTERS, so -outfmt is always rejected; without compressed intermediates the
#     filter needs ~8.6 GB of the 10 GB tmpfs for the dup1 split.
#   vendor/cado-symlink-writethrough.patch -- write through symlinks (memfd pool), see above.
#   vendor/cado-characters-parallel.patch -- characters only, see above.
#   cofac_cpu links CADO's UNMODIFIED sieve/ecm facul objects (vendor/cofac/).
#
# Runtime deps (numpy, cupy-cuda13x[ctk], gmpy2, cuda-toolkit, nvidia-cublas)
# are pulled unmodified from public indexes under their own licences.
# ============================================================================
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

# system deps (python 3.12 on ubuntu 24.04) + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip bash coreutils findutils gzip zstd \
        libgmp10 libgomp1 libstdc++6 zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Python deps for the GPU linear algebra: CuPy with bundled CUDA compilation toolkit.
# cupy-cuda13x[ctk] pulls libnvrtc + headers (needed for gpu_la.py's elementwise kernel JIT) via
# the `cuda-toolkit` metapackage — CUDA 13 dropped the per-library nvidia-*-cuNN wheels that the
# 12.x path pinned individually (the nvidia-*-cu13 names on PyPI are 0.0.1 placeholders).
# cuda-toolkit is pinned to the 13.0.x line for the same driver-floor reason as the base image:
# a 13.1/13.2 cudart would need a driver newer than the validators' 580.x.
# ALL PINNED to the stack every RSA-530 measurement ran on (pip list of image rsa530:v4, 2026-09-27): the
# validator builds this Dockerfile itself, so an unpinned cupy/cuda-pathfinder release between now and
# validation would put an untested NVRTC-discovery/RawModule stack under gpu_la.py + segspmv.py.
RUN pip3 install --no-cache-dir --break-system-packages \
        numpy==2.4.6 \
        "cupy-cuda13x[ctk]==14.2.0" \
        gmpy2==2.3.1 \
    && pip3 install --no-cache-dir --break-system-packages \
        "cuda-toolkit[cublas,cudart,cufft,curand,cusolver,cusparse,nvrtc]==13.0.1" \
        "nvidia-cublas==13.0.2.14" \
        "cuda-pathfinder==1.8.2" \
        "nvidia-nvjitlink==13.4.92"
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
# !! REQUIRED -- polyselect_params.py makes the polyselect constants functions of digits(N).
# Without it the fallback MIN_EVALUE=3e-12 banks NOTHING at 512-bit (the best candidate scores
# 2.679e-12) and the stage exits 3 after ~10 min.
# cado_top_raw.py / cado_extract_polys.py serve the CADO engine, which c155 does not use.
# Kept because polyselect_ropt -inputpolys silently parses only ONE polynomial from unformatted
# input ("1 polynomial(s) read") -- not an error, just a search that evaluated one candidate.
COPY gpu_nfs_siever/polyselect_params.py    /app/gpu_nfs_siever/polyselect_params.py
COPY gpu_nfs_siever/cado_top_raw.py         /app/gpu_nfs_siever/cado_top_raw.py
COPY gpu_nfs_siever/cado_extract_polys.py   /app/gpu_nfs_siever/cado_extract_polys.py
# !! REQUIRED -- sieve_params.py emits QMIN/QSPLIT/REL_TARGET/first_break/PROBE_WINS for this
# key size. Without it the fallback puts the checkpoint ladder ~8 M relations from the crossover:
# the run over- or under-sieves and nothing in the log says why.
COPY gpu_nfs_siever/sieve_params.py         /app/gpu_nfs_siever/sieve_params.py
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
# CADO POLYNOMIAL SELECTION. Same source, local.sh and no-AVX-512 constraint as the filter
# binaries above -- two more `make` targets.
# ⚠️ NOT ON THE c155 PATH. c155 selects with msieve: scored on unique yield over the range the
# sieve actually runs, msieve's pool wins on both metrics and its stage is 396 s cheaper.
# These stay for the other engines polyselect_best.sh can select, and for POLYSEL_ENGINE=both,
# which scores the union of both pools -- the tool for settling the engine question on a new
# key size. Absence is handled: polyselect_best.sh logs a WARNING and uses msieve. Verify:
#     objdump -d cado-filter/polyselect | grep -cE '%zmm|\{%k[0-7]\}'   -> must print 0
COPY cado-filter/polyselect         /app/cado-filter/polyselect
COPY cado-filter/polyselect_ropt    /app/cado-filter/polyselect_ropt
# CONCURRENT CPU SIEVER. gpu_loop needs only a few of the 24 cores, so las sieves the low
# special-q band on the rest, concurrently, for free -- the GPU pays nothing for those cores.
# !! ABSENCE IS SILENT-ish: las_sieve.sh logs a WARNING and the run continues GPU-only, slower.
COPY cado-filter/las                /app/cado-filter/las
COPY cado-filter/makefb             /app/cado-filter/makefb
COPY gpu_nfs_siever/las_sieve.sh    /app/gpu_nfs_siever/las_sieve.sh
# CPU CO-SIEVE (LAS=1, factor_c160.sh default, 2026-09-28): CADO las (the vendored tree, unmodified) sieves
# algebraic special-q from LAS_QMIN=12M upward on the cores the pipeline leaves idle (affinity-confined every
# poll via las_ctl.py, SIGSTOPped during probes), validated outputs ride into the next GPU chunk's slot.
# FAIL-SOFT: any las failure disables it with a WARNING and the run continues GPU-only. Measured c160:
# 626 rel/s at -t 14 with no GPU slowdown, ~46% unique.
COPY gpu_nfs_siever/las_ctl.py      /app/gpu_nfs_siever/las_ctl.py
COPY enigma_challenges              /app/enigma_challenges
# ---- RSA-530 (c160) pipeline: cuda-sieve + RAM-backed CADO filter (see gpu_nfs_siever/factor_c160.sh)
# cuda-sieve (github.com/kyleaskine/cuda-sieve @ db85a39, CC0), built in nvidia/cuda:13.0.1-devel with
#   make bench fbgen fbgen_gpu normscan GPU_ARCH=120 CF_LMAX=3 BN_LIMBS=12 HOST_TUNE=-march=x86-64-v3 CUDART_LINK=static
# (static cudart, CUDA 13.0 -> runs on the 580.x driver; objdump shows 0 AVX-512 instructions).
# bench is built with BN_LIMBS=8 (256-bit exact norms; c160 needs <= ~210: side-1 log2(maxnorm) <= 186 + log2 q <= 24,
# checked on 3 key-3 polys at q 2.2M and 12M, 0 skipped) -- measured -2.2% ms/q vs the default 12, byte-identical
# output. fbgen_gpu keeps the make line above (BN_LIMBS=12). A special-q whose norm did not fit would be SKIPPED with a
# warning (never wrapped); factor_c160.sh reports it. bench also carries vendor/cuda-sieve-fast-emit.patch (2026-09-27): (1) the host join loop builds each relation /
# candidate line in a buffer and writes it with one fwrite (was ~20 fprintf + a 12-limb bn_to_dec per line:
# 1.13 -> 0.40 ms per special-q); (2) without --cofactor the loop only PACKS each line's raw fields and one
# writer thread sorts/formats/writes them in FIFO order (0.40 -> 0.07 ms/q; -1.1% wall per q); every
# checkpoint and the final close drain the writer first -- kill -9 + resume is byte-identical (tested).
# CS_SYNC_EMIT=1 selects the synchronous path. (3) under --fill-concurrent the per-q trial-division small-prime
# tables are built AFTER both sides' sieve work is issued (they overlap the ~15 ms sieve instead of running with the
# GPU drained) -- with --bkthresh 65536 (factor_c160.sh default) 25.1 -> 24.1 ms/q; relations identical at a given
# bkthresh. (4) (2026-09-28, look-ahead prepare) under --fill-concurrent on an unslabbed band the NEXT special-q
# is selected and the host half of its per-q prepare (small-prime tables, norms, staging -> a second pinned buffer
# set per side) is built inside the current q's trial division, just before its final sync while the GPU is still
# busy; the loop top then only issues the device half. 24.1-24.2 -> 23.1 ms/q (-4.3%; A/B at q 5M and 8.5M, two
# repeats each; GPU-accounted/wall 0.94 -> 0.98); relations + candidates byte-identical to (3); kill -9 + resume
# byte-identical. CS_NO_LOOKAHEAD=1 disables it (timing then equal to (3)). (Its setvbuf(NULL, 4 MiB) calls are size no-ops on glibc -- the streams keep
# the default buffer; harmless, and what every measurement ran with.) Output BYTE-IDENTICAL to the unpatched build
# (relations, --candidates and --cofactor relations; cmp on a 30k-q c160 band). Same make line as above.
COPY cuda-sieve/bench                  /app/cuda-sieve/bench
# cofac_cpu (COFAC_CPU=1, the factor_c160.sh default): splits bench's --candidates on the CPU while the band
# runs. Our code (vendor/cofac/cofac_cpu.c: parser/emitter, Brent rho + deterministic Miller-Rabin fallback,
# follow mode) + vendor/cofac/facul_split.cpp calling CADO-NFS facul (sieve/ecm, LGPL v2.1+, the vendored
# CADO tree above), linked with that tree's facul objects + libutils.a: vendor/cofac/build_cofac.sh in
# nvidia/cuda:13.0.1-devel-ubuntu24.04, -march=x86-64-v3, 0 AVX-512 instructions. Relation set identical
# to bench --cofactor on 1.22M + 1.0M real c160 candidates.
COPY cuda-sieve/cofac_cpu              /app/cuda-sieve/cofac_cpu
COPY cuda-sieve/fbgen_gpu              /app/cuda-sieve/fbgen_gpu
COPY gpu_nfs_siever/factor_c160.sh     /app/gpu_nfs_siever/factor_c160.sh
COPY gpu_nfs_siever/polysel160.py      /app/gpu_nfs_siever/polysel160.py
COPY gpu_nfs_siever/memhold.py         /app/gpu_nfs_siever/memhold.py
# polytest160.sh (POLY_TS=1 default): GPU test-sieve of the top-5 root-optimized polys after ropt, pick by modelled
# sieve time (polysel160.py pick); never fatal. Measured key 3: MurphyE-top -> #2 = -2.9..-6% sieve time.
COPY gpu_nfs_siever/polytest160.sh     /app/gpu_nfs_siever/polytest160.sh
COPY cado-filter/cado_filter160.sh     /app/cado-filter/cado_filter160.sh
COPY cado-filter/cado2msieve2          /app/cado-filter/cado2msieve2
# INCREMENTAL DEDUP (INCDUP=1, the factor_c160.sh default): per-chunk dup1+dup2 off the critical path.
# !! ABSENCE IS FATAL BY NAME: factor_c160.sh's preflight dies "missing .../cado_incdup.sh".
COPY cado-filter/cado_incdup.sh        /app/cado-filter/cado_incdup.sh
# CADO-NATIVE DOWNSTREAM (DS_MODE=cado, the factor_c160.sh default): replay -> gpu_la.py on the CADO
# matrix (cado_mat.py loader; msieve-gpu/gpu_la.py above IS the version that reads .sparse.bin) ->
# CADO characters -> CADO sqrt. characters/sqrt provenance: see the header (built from the vendored
# CADO tree + both patches). Preflight in factor_c160.sh dies by name if any of these is missing.
COPY msieve-gpu/ds_cado.sh             /app/msieve-gpu/ds_cado.sh
COPY msieve-gpu/cado_mat.py            /app/msieve-gpu/cado_mat.py
# 128-BIT SEGMENTED LA (2026-09-27, LA_BLOCK=128 LA_SPMV=seg = the ds_cado.sh defaults): gpu_la.py
# imports segspmv.py from its own directory. Hot ideals (columns [ND,6400) of B) are gathered from a
# shared-memory window, the cold rest in L2-sized column parts; 128-bit vectors halve the iteration
# count. Measured c160: 7.9 ms per 64 dims vs 15.8 (0.50x). If the plan build fails, gpu_la.py logs a
# WARNING and continues on the column-split / production SpMV; ds_cado.sh's LA try 2 is 64-bit prod.
COPY msieve-gpu/segspmv.py             /app/msieve-gpu/segspmv.py
# la128host.so (our code, msieve-gpu/la128host.c): the 128-bit Lanczos per-iteration host GF(2) algebra (mul128,
# tr128, msieve find_nonsingular_sub) in C, loaded by gpu_la.py via ctypes (LA_HOST_C=1 default; missing .so ->
# the Python versions). Exact ports: 3000 randomized cases identical, and a c160 Lanczos run gives the identical
# state checksum at every checkpoint; host time 1.6 -> 0.25 ms/iteration. Built in nvidia/cuda:13.0.1-devel-
# ubuntu24.04: gcc -O2 -march=x86-64-v3 -shared -fPIC -o la128host.so la128host.c (0 AVX-512 instructions).
COPY msieve-gpu/la128host.so           /app/msieve-gpu/la128host.so
COPY msieve-gpu/la128host.c            /app/msieve-gpu/la128host.c
COPY msieve-gpu/characters             /app/msieve-gpu/characters
COPY msieve-gpu/sqrt                   /app/msieve-gpu/sqrt
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
             /app/cado-filter/polyselect /app/cado-filter/polyselect_ropt \
             /app/cado-filter/las /app/cado-filter/makefb \
             /app/gpu_nfs_siever/las_sieve.sh /app/gpu_nfs_siever/las_ctl.py \
             /app/cuda-sieve/bench /app/cuda-sieve/fbgen_gpu /app/cuda-sieve/cofac_cpu /app/gpu_nfs_siever/factor_c160.sh \
             /app/gpu_nfs_siever/polysel160.py /app/gpu_nfs_siever/memhold.py /app/gpu_nfs_siever/polytest160.sh \
             /app/cado-filter/cado_filter160.sh /app/cado-filter/cado2msieve2 \
             /app/cado-filter/cado_incdup.sh \
             /app/msieve-gpu/ds_cado.sh /app/msieve-gpu/cado_mat.py /app/msieve-gpu/segspmv.py \
             /app/msieve-gpu/characters /app/msieve-gpu/sqrt \
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
