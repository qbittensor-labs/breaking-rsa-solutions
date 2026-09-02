#!/usr/bin/env bash
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

# Build the GPU lattice siever for the validator GPU (RTX PRO 6000 Blackwell, sm_120).
# Needs: CUDA toolkit 13.0 (nvcc with sm_120), libgmp-dev, OpenMP. NO GPU needed to BUILD.
#
# !! TOOLKIT IS PINNED AT EXACTLY 13.0 -- NOT 13.1, NOT 13.2, NOT 12.8. !!
#   The validator hosts run CUDA 13.0 on driver 580.159.03 / 580.173.02. gpu_loop links the CUDA
#   runtime STATICALLY, so the toolkit it is built with sets the MINIMUM DRIVER the binary needs:
#     12.8 -> needs >= 570   (runs, but is a different toolkit from the rest of the stack)
#     13.0 -> needs >= 580.65 -- matches the validator hosts exactly            <-- USE THIS
#     13.1 -> needs a driver NEWER than 580.x -> cudaErrorInsufficientDriver, the binary does not
#             start at all and the solver factors NOTHING on every challenge.
#   A CUDA 13.1-built binary was tracked in this repo until 2026-08-03 and would have failed on
#   every validator box. Same unrecoverable class as the -march=native trap below: the Dockerfile
#   COPYs the prebuilt binary and the image has no nvcc, so it cannot be recovered at runtime.
#   Build with:  PATH=/usr/local/cuda-13.0/bin:$PATH bash build.sh
#   Verify with: strings gpu_loop | grep -oE 'release [0-9]+\.[0-9]+'   -> must print 13.0
#
# Default build = kernel-optimized + lpb=29 (the MEASURED-best config: builds a matrix at 66M raw,
# the fewest relations of any lpb). See Pipeline.md for the full lpb sweep and measurements.
#
# Kernel wins (all correctness-neutral; relations byte-identical to the plain build, verified by
# sorted-set diff). Measured on this Max-Q box (c151 vetted poly): GPU 15.5 -> 11.7 ms/lat (~1.32x).
#   -DFLAT_COOP2   cooperative uniform-work resieve (5.3 -> 2.7 ms/lat); fixes the 34.6% warp
#                  utilisation of the one-thread-per-line box walk via per-warp shared-mem work
#                  redistribution. (Box scatter kept; coop scatter measured a regression.)
#   Montgomery MR  device Miller-Rabin in the Montgomery domain (cofactor MR 1.47 -> 0.83 ms/lat).
#                  Unconditional in gpu_loop.cu.
#   fast u_divmod  single-limb / leading-zero fast paths in u256 trial division. Unconditional.
#   int32 bounds   lattice bounds+walk in int32 (basis <2^14, det <2^27); +occupancy. Unconditional.
# FAST_FINAL / FUSED_SCAN: correctness-neutral pipeline flags (see gpu_loop.cu).
#
# DO NOT add -DSCAN_EARLYOUT: it is LOSSY (drops ~0.02% of relations). Against the relation cliff
# that is dangerous. It is NOT in the default flags.
#
# lpb config choices (LPB_VAL, LIM, MFB) — pick via FLAGS; see Pipeline.md §2 for crossovers:
#   lpb=29 (default): LPB_VAL=2^29 (536870912)  LIM=50M  MFB=58/58  -> matrix at ~66M raw (BEST)
#   lpb=30:           LPB_VAL=2^30 (1073741824) LIM=60M  MFB=60/60  -> matrix at ~130M raw
#   lpb=31 (shipped): LPB_VAL=2^31 (2147483648) LIM=70M  MFB=62/62  -> matrix at ~230M raw
# NOTE: lpb=29/66M is validated to MATRIX FORMATION (filter cycles), not yet end-to-end to p*q=N.
#
# Experimental variants (off by default; -D to A/B, all relation-identical): -DFLAT_RESIEVE /
# -DFLAT_WALK / -DFLAT_COOP / -DSCAT_COOP / -DSCAT_SPLIT (all slower than FLAT_COOP2).
# -DBUCKET_SIEVE is a +24% regression. -DSLOW_REDUCE reverts to int64 reduction.
set -e
# !! LIM=35M TRIED AND REVERTED (2026-08-09). LIM STAYS 40M. !!
# The fixed-q bench predicted +2.2% and PRODUCTION DELIVERED ZERO. Recorded in full because the
# bench numbers look convincing and someone will otherwise re-run this exact sweep.
#     bench rel/s vs LIM=40M (warmup run discarded at each point):
#     LIM     q=11M    q=19M    q=25M    weighted (23/42/35% of sieve time)
#     30M     +5.9%    +2.8%    -0.8%       +2.3%
#     35M     +5.8%    +1.6%    +0.5%       +2.2%
#     45M     +2.5%    -1.8%    -1.4%       -0.7%
#     e2e reality at LIM=35M, matched RELATION COUNT against the LIM=40M run:
#       22.57M rels: 35M t=3,311s vs 40M t=3,318s   (+0.2%)
#       28.45M rels: 35M t=4,367s vs 40M t=4,364s   (-0.07%)   -> a WASH, not +2.2%
#
# WHY THE BENCH LIED, and how to not repeat it: the windows were q=11M/19M/25M and the +5.9% at
# q=11M was extrapolated across the whole 23% of sieve time BELOW it, which was never measured.
# Production starts at QMIN=1M. **Any future compile-time sweep must include low-q windows (2M, 5M)
# before its weighted number is believed.** Note also that LIM=35M measurably LOWERS yield --
# the bake-off winner scored 41.40 rel/lat vs 43.06 at 40M (-3.9%) -- which means more large primes
# per relation and a HIGHER purge crossover. So 35M is a wash on speed while adding crossover risk:
# strictly worse than 40M, whose crossover is the only one verified end-to-end (MATRIX at 65,760,018).
#
# (The in-source LIM note further down does not reflect the production value set here.)
#
# SPB 256 -> 311 (2026-08-09, -0.2% at q=19M / -0.8% at q=11M, 6-rep alternating wall-time A/B,
# RELATION-IDENTICAL: same count and same sorted md5 at q=11M and q=19M). 311 is the hard ceiling:
# hSP[] holds 64 entries and pi(311)=64, so 256 (pi=54) was leaving 10 slots unused. Small, but free.
#
# !! SCAT_SKIP=256 REJECTED (2026-08-09) -- do not retry without reading this. !!
# gpu_loop.cu:35 says "Pair SCAT_SKIP with SPB", and SCAT_SKIP was never re-paired when SPB went
# 32 -> 256 on 2026-08-01. Pairing it is a 2-D problem: SCATOFF compensates the un-scattered log mass,
# which roughly DOUBLES going from p<32 (sum log2(p)/p ~ 3.5) to p<256 (~6.5), so SCATOFF=10 is badly
# mistuned there -- it drops 19% of relations. Swept SCATOFF = 10/14/18/22/26/30/34 at SCAT_SKIP=256:
#     SCATOFF   10     14     18     22     26     30     34
#     q=11M   -15.6%  -4.9%  +1.2%  +2.5%  -0.1%  -7.3%  -17.5%   (rel/s vs SCAT_SKIP=32)
#     q=19M                         -1.7%  -6.3%  -11.4% -21.0%
#     q=25M                         -2.2%
# The optimum (SCATOFF~22) is POSITIVE at low q and NEGATIVE at every production q above it, and the
# sieve spends ~77% of its time above q=11M. Net negative. SCAT_SKIP stays 32.
# (Methodology: the FIRST rep after a build always reads high -- discard it. An early +2.9% for
#  SCATOFF=22 was that artifact; it settles to +2.5% at q=11M and negative everywhere higher.)
# REDUCE_CAP=8 added 2026-07-20: Blackwell measured cap=8 -> 7.34 vs cap=64 -> 7.39 ms/lat (the doc's
# "cap=8 identical" was Max-Q). Correctness-neutral (mu is unimodular -> same lattice, relations
# byte-identical).
# SUPERSEDED 2026-08-08: REDUCE_CAP/RCOLS are now 32 (-0.5%, 5-rep A/B, relation-identical) and the
# GRIDMUL default in gpu_loop.cu is 128, not 64 (-4.9%, 3-rep A/B). Both of the older numbers above
# were measured against the pre-SCAT_COOP_LEAN scatter; re-measure this pairing after any scatter
# change rather than trusting either table. See Pipeline.md §14b.
# 2026-07-25 additions to the default build (all relation-BYTE-IDENTICAL, sorted-set diff = 0 over
# q=[1.0M,1.01M]: 42,595 relations, same md5; verified at three q-windows). Together 7.22 -> 6.07
# ms/lat (-15.9%). See Pipeline.md §5.
#   -DTILE_COL      shared-memory tiled scatter for the dense groups (p<=T) + column recurrence for
#                   j0 + FUSED INIT (the kernel stores its tile, so the 2 x 33.5 MB memset is gone).
#                   scatter 4.50 -> 3.75.  TCOLS must divide W (static_assert enforces it).
#   -DTILE_RESIEVE  the same column recurrence in the dense-group resieve. resieve 1.58 -> 1.40.
#   -DPRIM_SCAN     primitive-cell (i,j) wheel mask on the scan (was already measured +3.7% in
#                   Pipeline.md §3.8 but had never been switched on by default). cofactor 0.77->0.58.
#   -DBA_VEC        int4 reduced-basis store/load. Measured NEUTRAL here; kept because it is the
#                   correct access shape and costs nothing.
# REDUCE_CAP=8 re-confirmed optimal against the new kernel mix (6=6.19, 12=6.13, and 4 EXPLODES to
# 27.6 ms/lat AND loses relations -- do not lower it).
# 2026-07-27: SIEVE REGION J 4096 -> 8192 (with TCOLS=4). End-to-end 5.161h -> 4.240h (-17.8%);
# sieve alone 4.630h -> 3.654h (-21.1%). Factored c151, p*q==N verified, dep 1. See Pipeline.md §17.
#
# WHY THIS WAS AVAILABLE: the old "J=4096 optimal" calibration was measured on the PRE-2026-07-25
# kernel and never re-swept after -DTILE_COL/-DTILE_RESIEVE changed how cost scales with area.
# Rebuilding both kernels shows the ordering invert (ms per raw relation, q=40M):
#     old kernel: J=4096 0.3555  vs J=6144 0.3582   -> 4096 wins (reproduces the old record)
#     new kernel: J=4096 0.3085  vs J=6144 0.2971   -> 6144+ wins
# Mechanism: ~71% of ms/lat is FIXED per-lattice cost (the large-prime scatter walks the factor base;
# for p > area the cost is per-prime, not per-cell), so area is far cheaper than the old sweep assumed.
# Measured at q=40M: ms ~= 4.41 + 0.0546 * area(Mcells). Doubling area costs +30% time for +44% NEW
# relations -- and the DUPLICATE RATE FALLS (44.5% -> 38.2%), because a bigger box reaches (a,b)
# points the small-q lattices never cover.
#
# THE OPTIMUM IS INTERIOR AND BOUNDED BY L2 (ms per NEW relation at q=40M; NEW = not already owned
# by a smaller special-q, measured with the DUPSUP oracle):
#     J= 4096  arrays 2x32MiB   6.19 ms/lat   NEW/lat  9.92   0.6232
#     J= 6144  arrays 2x48MiB   7.05 ms/lat   NEW/lat 12.33   0.5722
#     J= 8192  arrays 2x64MiB   8.04 ms/lat   NEW/lat 14.28   0.5630  <-- DEFAULT
#     J=12288  arrays 2x96MiB  10.06 ms/lat   NEW/lat 17.39   0.5785
#     J=16384  arrays 2x128MiB 28.86 ms/lat   NEW/lat 19.85   1.4539  <-- L2 CLIFF (2x the 128MB L2)
#
# !! TCOLS=4 IS MANDATORY AT J=8192 !!  scat_tile's shared array is `uint8_t tile[TCOLS*J]`, and
# static __shared__ is capped at 48 KB. TCOLS=8 x J=8192 = 64 KB and will NOT compile/launch.
# TCOLS must also divide W (=2*I2=8192); 4 does. Measured free: J=6144 TCOLS=8 -> 7.10 ms/lat vs
# TCOLS=4 -> 7.05, i.e. TCOLS=4 is if anything marginally better.
#
# !! PORTABILITY !! J=8192 puts the two sieve arrays at 2x64MiB = 128 MiB = EXACTLY this card's
# (RTX PRO 6000, 128 MB) L2. It is tuned to that L2 and is NOT assumed portable: on a smaller-L2 GPU
# it may land on the cliff above. J=6144 (2x48MiB, real margin) costs ~1.6% more ms/NEW and is the
# safe fallback -- build with FLAGS="... -DJ=6144" (TCOLS=8 is fine there: 8*6144 = 48 KB exactly).
#
# !! REL_TARGET IS COUPLED TO THIS !!  factor_msv.sh's REL_TARGET is a RAW relation count and the
# crossover moved 72.5M -> ~68M. Changing J here without changing REL_TARGET there over-sieves past
# the crossover and throws the win away. They must move together.
# TSPLIT 262144 -> 327680 (2026-07-31). Scatter is 58% of the 7.7 ms/lattice GPU time (PROF: scatter
# 4.45 | resieve 2.16 | cofactor 0.56 | scan 0.32 | project 0.21), and TSPLIT is the boundary that
# divides work inside it. 131072 was swept properly at J=4096; when J doubled the value was DOUBLED
# BY ANALOGY to 262144 and A/B'd, never swept -- everything above it was unmeasured. Full sweep at
# J=8192 (mean ms/lat over q=5M and 15M):
#     131072 7.967 | 196608 7.765 | 262144 7.675 | 327680 7.647 | 393216 7.655 | 524288 7.715
# The optimum is a broad plateau from ~262k to ~393k, so this is a small move, and the first sweep's
# 0.37% was close to the noise floor. Re-measured with 18 INTERLEAVED reps each over 3 windows:
#     262144  mean 7.6761 ms  min 7.6000     327680  mean 7.6378 ms  min 7.5800
# = +0.50% on the mean, +0.26% on the min, i.e. 33-62s of a 12,509s sieve. Bank the low end: the
# mean gap exceeding the min gap means some of it is variance reduction, not throughput.
# RELATION-IDENTICAL by construction and verified: sorted-set md5 d0d4503a71040773 at both values.
# SPB 32 -> 256 (2026-07-31). SPB is the small-prime RESIEVE SKIP: primes < SPB are not resieved
# (a per-cell walk costing NCELL/p each) but trial-divided out of each survivor's norm instead.
# Resieve was 28% of sieve GPU time (PROF: scatter 4.45 | resieve 2.16 | cofactor 0.56 | scan 0.32 |
# project 0.21) and had never been touched. The shipped 32 is documented in gpu_loop.cu as the
# the in-source SPB default is a compile-time placeholder -- tuned for a SMALLER problem, superseded for
# c151, whose lpb/LIM/region all differ. Measured sweep (2 windows, PROF):
#   SPB    GPU     resieve  cofactor
#    32   7.630     2.115     0.500
#    64   7.510     1.965     0.540
#   128   7.505     1.895     0.590
#   256   7.490     1.775     0.690   <- resieve -16%, cofactor +38%, net GPU -1.8%
# Confirmed with 18 INTERLEAVED reps over 3 production-range windows:
#   SPB=32  mean 7.6439 ms  min 7.5900     SPB=256  mean 7.4711 ms  min 7.4100
# = +2.26% mean / +2.37% min (they agree, unlike the TSPLIT sweep) -> ~275s of a 12,156s sieve.
# RELATION-NEUTRAL BY CONSTRUCTION: trial division is the ground truth that resieve merely optimises,
# so unlike SCAT_SKIP this needs no SCATOFF threshold compensation. Verified: sorted-set md5
# d0d4503a71040773 at both SPB=32 and SPB=256, yield unchanged at 46.6 rel/lat.
# CPU stays ~1.4 ms/lat against 7.5 ms of GPU, so the extra cofactor work remains fully hidden.
# CEILING: hSP[] holds at most 64 primes and pi(311)=64, so SPB must not exceed ~311.
FLAGS="${FLAGS:--DSPB=311 -DFB_SORT -DTSPLIT=393216 -DFAST_FINAL -DFUSED_SCAN -DFLAT_COOP2 -DSCAT_COOP_LEAN -DLPB_VAL=536870912ULL -DLIM=40000000u -DCFG_MFB1=58 -DCFG_MFB0=58 -DMAXSURV_VAL=600000 -DREDUCE_CAP=32 -DRCOLS=32 -DTILE_COL -DTILE_RESIEVE -DPRIM_SCAN -DBA_VEC -DJ=8192 -DTCOLS=4}"
# -maxrregcount tested 2026-07-20 and DROPPED: rigorous 4-rep A/B on a fixed q-window showed
# cap8+rreg96=7.235 vs cap8-only=7.240 ms/lat = within noise. An early small-window reading (7.27)
# was run-to-run noise. Keep the compiler's default register allocation.
# -march=x86-64-v3, NOT -march=native (changed 2026-07-22 -- see Pipeline.md §1.4).
# -march=native emits whatever ISA the BUILD box has. The previously tracked binary was built on an
# AVX-512 machine and SIGILLs (exit 132) on any CPU without it -- e.g. AMD EPYC 7443 (Zen 3). The
# Dockerfile COPYs this binary and the image has no nvcc, so that is an unrecoverable
# factor-nothing failure in deployment. x86-64-v3 (AVX2+BMI2) runs everywhere modern and measured
# FREE: relations byte-identical, GPU 7.32 vs 7.31 ms, CPU 1.25 vs 1.26 ms (both noise) -- the CPU
# stage is fully hidden under the GPU anyway. DO NOT "restore" -march=native for speed.
nvcc -O3 -use_fast_math -arch=sm_120 -Xcompiler -fopenmp,-march=x86-64-v3 $FLAGS -o gpu_loop gpu_loop.cu -lgmp
echo "built gpu_loop (FLAGS=$FLAGS)"
