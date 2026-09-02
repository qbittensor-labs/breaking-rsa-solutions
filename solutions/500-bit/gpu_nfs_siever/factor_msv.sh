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

# ============================================================================
#
# Per-key 500-bit (c151, ~151-digit) RSA-semiprime factorizer for the validator
# sandbox (1 GiB writable /tmp, 85 GiB RAM, 24 CPU, 1 GPU, --network none, 4h).
#   N  ->  msieve GPU polyselect  ->  gpu_loop GPU sieve  ->  memfd-backed msieve
#          (filter + in-RAM Lanczos + CADO multi-core sqrt)  ->  p, q
# All multi-GB intermediates live in anonymous RAM (memfd); /tmp stays ~100 MB.
# usage: factor_msv.sh <N> [workdir]   (set GNFS_CACHE=<dir> for dev stage-caching)
# ============================================================================
set -e
T_START=$(date +%s)
N="$1"; WORK="${2:-/tmp/fk}"
SIEVER=$(cd "$(dirname "$0")" && pwd)
# msieve toolchain dir (msieve + cub/sort_engine.so + msv_launcher + msvrun_gpu.sh).
# default: sibling 'msieve-gpu' next to the siever dir, else env MSV_DIR.
MSV="${MSV_DIR:-$(cd "$SIEVER/../msieve-gpu" 2>/dev/null && pwd || echo /root/msieve-gpu)}"
# ---------------------------------------------------------------------------
# GPU-LOCAL CPU PINNING (2026-08-12). A submission that SUCCEEDED on an AMD EPYC 9555 returned
# WallTimeFailure on an Intel Xeon 6980. The EPYC is single-socket, so a 24-CPU cpuset is one NUMA
# domain and is local to the GPU by accident; the Xeon 6980P is 128C and commonly dual-socket /
# SNC-partitioned, so the same cpuset can land on the socket that does NOT own the GPU's PCIe root.
# The sieve is 88% of the wall and transfers per lattice (~2.1M times a run), so that placement is
# charged over and over. gpu_affinity.py narrows our own mask to the GPU's node before exec.
# It is a NO-OP on a single-node host by construction, so it cannot regress the EPYC result, and it
# only ever narrows a mask we already hold -- it cannot grant CPUs the cgroup withheld.
# ONLY the GPU stages are wrapped: Pipeline.md §14b measured gpu_loop at 8.215 ms/lat with OMP=4 vs
# 8.230 with OMP=24, so losing threads here costs ~0.2%. polyselect, the filter probe and the sqrt
# ARE thread-scaling and are deliberately left on the full cpuset.
# It MUST exec (not fork): the phase-B loop stops the sieve with `kill "$SVPID"`, and a wrapper that
# forked would swallow that signal and let the sieve run on to REL_TARGET.
# !! DEFAULT FLIPPED TO OFF, 2026-08-13. THE PREMISE ABOVE IS FACTUALLY WRONG. !!
# The validator does NOT hand us a cpuset. run_solution.docker_run_security_args() emits
# `--cpus 24` (VALIDATOR_DOCKER_CPU_LIMIT_DEFAULT), which is a CFS BANDWIDTH QUOTA (cpu.max),
# not --cpuset-cpus. Verified inside a container launched with the validator's exact flags:
#     cpu.max = "2400000 100000"      cpuset.cpus = ""      cpuset.cpus.effective = 0-29
#     os.sched_getaffinity(0) -> EVERY host CPU
# So there is no 24-CPU cpuset, and nothing for one to "straddle". Two consequences:
#   * NOT a no-op on a multi-NUMA host. `allowed` is the whole machine and `local` is one node,
#     so pin != allowed ALWAYS holds and it pins on every multi-node box.
#   * The pin can be NARROWER than the 24-core quota. Exercised against synthetic topologies:
#     a 24-vCPU host presented as 2 NUMA nodes pins to 12; as 4 nodes, to 6. MIN_CPUS=4 does not
#     catch this -- that floor is calibrated on "gpu_loop is fine at OMP=4", measured with 24 CPUs
#     FREE, whereas phase B runs the pinned sieve against the UNPINNED filter probe at NT=20.
# It is also the only code in this image that executes on a multi-node host and not on a
# single-node one -- i.e. the sole EPYC/Xeon behavioural divergence in the container (msieve's
# own GenuineIntel/AuthenticAMD branch in common/util.c writes obj->cpu, which is never read;
# its cache_size2 only sets CPU-Lanczos blocking on the path msvrun_gpu.sh kill -9's).
# It shipped untested on the only class of host where it does anything: hard-key/
# container_verify_20260812.txt records `affinity --print: []` on a 1-NUMA-node box.
# GPU_NUMA_PIN=1 re-enables. NOTE the validator's `docker run` passes NO -e at all, so in
# production the value of this default IS the setting -- there is no other way to reach it.
GPU_PIN=""
if [ "${GPU_NUMA_PIN:-0}" != "0" ] && [ -r "$SIEVER/gpu_affinity.py" ]; then
  GPU_PIN="python3 $SIEVER/gpu_affinity.py --"      # deliberately UNQUOTED at the call sites
fi
POLYSELECT_SECS="${POLYSELECT_SECS:-540}"      # GPU polyselect budget (~18 min)
ADMAX="${ADMAX:-200000}"
# QMAX is intentionally LARGE: the sieve stops at REL_TARGET (good keys, early) or at the wall-time
# budget (low-yield keys), NOT at the q-range. A small QMAX used to make low-yield keys quit with too
# few relations while budget remained -> doomed downstream. GNFS_WALL = total seconds the GNFS may use
# (the caller's remaining wall budget); DOWNSTREAM_RESERVE is held back for filter+LA+sqrt.
# c151 (500-bit) params: gpu_loop is built lpb=2^29 / LIM=50M / mfb=58-58 / J=8192 TCOLS=4 (was
# J=4096 until 2026-07-27) / SCAT_SKIP=32 / SCATOFF=10 / GRIDMUL=64 / REDUCE_CAP=8 (see build.sh +
# Pipeline.md §1). lpb=29 is measured-best (lpb 28/29/30 all conserve to ~7-8h; 29 needs the fewest
# relations). Downstream = CADO filter (refilter/cado_probe.sh) -> msvrun_gpu.sh -nc2.
#
# ⚠️ SUPERSEDED CLAIM. This header used to read "CADO filter measured NO crossover advantage on
# this N (Pipeline.md §6.2)". That is contradicted by direct measurement on 2026-08-04, and the
# §6.2 it cited NO LONGER EXISTS in Pipeline.md, so whatever backed it is not in the repo and
# could not be re-checked. What IS measured now, twice, end-to-end to p*q==N:
#   key03 65,496,292 relations -> msieve -nc1 FAILS ("wants 1000000 more relations") while its own
#     E_core (245,809) EXCEEDS target_excess (211,895); CADO builds a 3.99M matrix from the same
#     relations, 323s faster, and the key factors in 3.806h.
#   key01 68,500,009 relations -> both work; CADO path total 3.871h.
# If the old claim was measured, it was on a different N, a different lpb/J, or a CADO whose
# filtering has since changed. Do not restore it without a fresh measurement.
# DOWNSTREAM_RESERVE=3000: measured 2026-07-27 at J=8192 (matrix 4.28Mx4.28M) = 35:11 end-to-end
# (-nc1 9:29 + -nc2 5:47 + GPU-LA 17:21 + sqrt 2:34). 3000s leaves ~9% headroom -- do not lower it.
# NOTE the downstream got ~3.3min SLOWER at J=8192 (LA 856s -> 943s despite a smaller matrix); the
# sieve saving of ~59min dwarfs it, but the reserve must cover the larger figure.
#
# QMIN=1M is TIME-OPTIMAL (measured end-to-end, calibration dumps). time-to-solvable-matrix:
#   3M -> 72M in 7.38h | 1M -> 74M in 5.67h (BEST) | 700k -> 75M in 5.69h | 500k -> 76M in 5.74h
# Monotonic: below 1M is progressively WORSE -- lower QMIN raises rel/s but the duplicate rate rises
# FASTER (crossover climbs 74->75->76M), so do NOT drop below 1M. Above 1M loses rel/s (3M = +30%).
# The optimum is confirmed on both sides; the QMIN lever is closed.
#
# REL_TARGET=77M: CALIBRATED (refilter/matrix_health.sh on a 76M QMIN=1M dump). msieve crossover is
# 74M raw (72M FAILS "wants more"; 74M = +0.14% cycle excess; 76M = +0.49%) -- END-TO-END VERIFIED
# p*q==N at 76M. 77M = 74M + margin (74M is razor-thin; N-to-N crossover spread ~6%). Was a STALE
# 150M = 2x over-sieving (wasted ~1.5h; and >4.5x over-sieve can yield an UNSOLVABLE matrix, §4).
# The crossover is N-specific: re-run refilter/matrix_health.sh against a dump to retune REL_TARGET per key.
# 2026-07-25: with the adaptive checkpoint ladder below, REL_TARGET is the FIRST PROBE point, not
# a target with a safety margin baked in -- the filter itself decides whether to buy more. Set to
# the measured crossover (74M) rather than 77M; the 3M margin is now bought only on demand.
# 2026-07-26: 74M was LOOSE, not wrong. "74M works / 72M fails" only brackets the crossover to
# (72M,74M]; measuring it properly puts it at 72.5M, i.e. near the BOTTOM of that interval:
#   * 72.00M raw: E_core=  85,114  target_excess=206,242  -> deficit 121,128  SHORT
#   * 75.57M raw: E_core= 904,721  target_excess=215,709  -> surplus 689,012  matrix OK (p*q==N)
#   interpolated zero crossing 72.53M; independently 72.58M by extrapolating a 62M..72M cap sweep
#   (E_core +426k per 2M raw vs target_excess +5.5k per 2M). Two methods agree to 0.07%.
# First probe therefore drops 74M -> 73M (+0.5M over the measured crossover). ~1M fewer relations
# ~= 5 min less sieving; the ladder still buys more on demand if this N runs above 72.5M.
#
# ===========================================================================
# 2026-07-27: REL_TARGET 73M -> 68.5M. THIS IS COUPLED TO build.sh's J=8192.
# ===========================================================================
# The sieve region doubled (J 4096 -> 8192), which moved the crossover DOWN in raw relations and
# the duplicate rate down with it (44.5% -> 38.2%). Measured on a 70M-raw J=8192 dump:
#   cap 56M: deficit 1,808,896  SHORT      cap 64M: deficit   239,066  SHORT
#   cap 60M: deficit 1,103,007  SHORT      cap 68M: deficit   -35,523  MATRIX  (4,302,842/4,282,355
#                                                                       cycles = +0.48% surplus)
# 68M ran END-TO-END to p*q==N (dep 1), matrix 4,280,564 x 4,280,731. Sieve to 68M = 3.654h vs the
# J=4096 baseline's 4.630h; end-to-end 4.240h vs 5.161h. Full data: Pipeline.md §17.
#
# 68.5M = the measured matrix point (68M) + 0.5M margin, matching the prior convention.
#
# !! IF YOU REVERT build.sh TO J=4096, REVERT THIS TO 73000000. !!  A raw relation count is only
# meaningful for the region size that produced it: at J=4096, 68.5M raw is ~4M SHORT of a matrix.
# Conversely leaving 73M with J=8192 over-sieves ~5M relations past the crossover (~20 min wasted),
# which is most of the win. These two numbers must always move together.
# 2026-08-08: key07's crossover measured directly off a dump -- 62.0M gives purge excess -751,416
# and 63.5M gives -419,161 (+0.2215 excess/relation), so its zero crossing is ~65.4M.
# TEMPTING BUT REVERTED: dropping REL_TARGET to 68.0M moves the 97% probe to 65.96M and saves key07
# ~485k relations (~100s). It also moves the probe BELOW seed 20260728's measured 66.0M crossover --
# and 66.445M was chosen deliberately to sit just ABOVE it (see the PROBE POINT block below). That
# key would then SHORT on probe 1, sieve on to REL_TARGET and pay a second, unhidden filter:
# ~300s extra sieve + ~271s filter = ~570s worse, to buy 100s on one key. Left at 68.5M.
QMIN="${QMIN:-1000000}"; QMAX="${QMAX:-220000000}"; REL_TARGET="${REL_TARGET:-72000000}"
# REL_TARGET IS NOW A CEILING, NOT A PROBE POINT (2026-08-20). It used to be both: the probe fired
# at CKPT_SPEC_PCT% of it, so raising it for safety ALSO pushed the first probe later, and lowering
# it to save sieve time dragged the probe BELOW the measured crossover. Configuration.md 5 item 5
# records that coupling as a trap ("Lowering REL_TARGET is not a free saving -- it lowers the probe
# point with it"). The probe schedule is now absolute (CKPT_BREAKS, below), so this value only
# answers "how far do we sieve if EVERY breakpoint came back SHORT" and can be raised on its own.
# 68.5M -> 72M: it must sit above the last breakpoint or the walk-up is cut off by the ceiling.
# E_core gained per 1M raw relations, net of target_excess drift. Drives the deficit-based probe
# increment in the ladder; override per key if a sweep measures a different slope.
# 2026-07-27 re-measured on the J=8192 cap sweep (deficit closed per 1M raw, the same quantity):
#   56M->60M 176,472/1M      60M->64M 215,985/1M   (the slope accelerates toward the crossover)
# 216,000 is the near-crossover value. Note it is close to J=4096's 210,000 -- the region change
# barely moves this constant, because it is a per-RAW-relation rate on both sides.
#
# !! CAUTION, measured 2026-07-27 !!  Do NOT extrapolate this rate across the SHORT->MATRIX
# boundary. msieve's singleton peeling is a MULTI-ROUND CASCADE and matrix_health.sh scrapes the
# LAST "reduce to" line, so a run that clears the excess check keeps cascading while one that aborts
# stops earlier -- R_core/I_core/E_core/deficit are then read at DIFFERENT cascade depths and are not
# commensurable. Extrapolating 60M->64M forward predicted a 65.0M crossover; the truth was 68M.
# The 1.20 margin below exists partly for this; treat a ladder probe as possibly landing short.
ECORE_PER_MREL="${ECORE_PER_MREL:-216000}"
# ---------------------------------------------------------------------------
# DUPSUP guard. gpu_loop's DUPSUP=1 emits only relations no SMALLER special-q already
# owned, so the dump becomes duplicate-free (measured: 44.5% -> 0% duplicates on c151).
# BOTH constants above are denominated in RAW relations. Under DUPSUP=1 a "relation" is
# a UNIQUE relation, so leaving them alone would sieve to 73M *unique* -- roughly 1.8x
# the work, hours of overshoot. Refuse rather than silently overshoot.
#   Derived (NOT measured) starting points at the J=8192 measured 38.2% duplicate rate:
#     REL_TARGET     68M raw matrix point x 0.618 ~= 42_000_000 unique
#     ECORE_PER_MREL 216k per 1M raw / 0.618      ~=    350_000 per 1M unique
#   Re-measure both with refilter/matrix_health.sh on a DUPSUP=1 dump before trusting them.
# ---------------------------------------------------------------------------
if [ "${DUPSUP:-0}" != "0" ]; then
  if [ "$REL_TARGET" = "68500000" ] || [ "$ECORE_PER_MREL" = "216000" ]; then
    echo "FATAL: DUPSUP=$DUPSUP with raw-denominated REL_TARGET=$REL_TARGET / ECORE_PER_MREL=$ECORE_PER_MREL." >&2
    echo "       Under DUPSUP these count UNIQUE relations; the raw defaults would over-sieve ~1.8x." >&2
    echo "       Set both explicitly, e.g. REL_TARGET=42000000 ECORE_PER_MREL=350000 (derived, unverified)." >&2
    exit 1
  fi
fi
# !! WALL BUDGET, measured 2026-07-27 -- READ BEFORE TRUSTING THE 14400 DEFAULT !!
# SIEVE_SECS = GNFS_WALL - ELAPSED - DOWNSTREAM_RESERVE. With GNFS_WALL=14400 (4h) that leaves the
# sieve ~11400s, and c151 does NOT reach its crossover in that window at any region size:
#     J=4096  11400s ->  54.2M raw   (needs 72.5M -- short by 18.3M)
#     J=8192  11400s ->  61.3M raw   (needs 68.0M -- short by  6.8M)   <- much closer, still short
# So 14400 is NOT sufficient for c151; it is a legacy default inherited from smaller keys, and the
# 2026-07-27 J=8192 change narrows the gap a lot but does not close it. What c151 actually needs:
#     polyselect ~770s + sieve ~13158s (68M at J=8192) + downstream 2111s ~= 16040s
#     => GNFS_WALL >= ~16800 (4.7h) for a single-shot run with margin.
#   NB: POLYSELECT_SECS above is DEAD on this path -- line ~144 calls polyselect_best.sh without
#   forwarding it. Real cost = 16-slice parallel search (SLICE_TMO=600, exhausts ~540s) + a
#   POLY_TOPK=6 x 3-window yield bake-off (~230s). ~770s is ESTIMATED, not measured -- the one
#   unmeasured stage in Pipeline.md §17's timeline. Measure it before trusting this budget.
# breaking_rsa.py passes the caller's real timeout, so this default only bites on direct invocation.
# If the caller's budget truly is 4h, the run is time-bound: the sieve stops early and the downstream
# will report SHORT. That is a budget problem, not a config problem -- do not "fix" it by lowering
# REL_TARGET below the crossover, which only guarantees an unusable relation set sooner.
# =============================================================================================
# !! DOWNSTREAM_RESERVE 3000 -> 2000 (2026-07-29). THIS WAS CAUSING 100% VALIDATOR FAILURE. !!
# =============================================================================================
# The validator passes NO environment into the container (docker run has no -e), so WALL_TIME is
# unset and breaking_rsa.py falls back to its 14400s default. The reserve is then subtracted UP
# FRONT, and the sieve is capped at:
#     SIEVE_SECS = 14400 - 5(Tier0) - 60(slack) - ~622(polyselect) - 3000 = 10,713s
# Every key measured needs MORE than that just to reach the probe point:
#     seed20260732 11,429s | seed20260730 11,446s | seed20260729 ~11,676s | seed20260736 ~12,348s
# So the sieve stopped early on EVERY key, no matrix could be built, and the solver exited cleanly
# at ~3.31h -- 2,495s of its own budget UNUSED -- emitting {"status":"failed"}. The validator reads
# that valid payload and reports IncorrectFailure (NOT a timeout: the container never went overdue).
#
# The 3000 was sized on 2026-07-27 for a downstream of 2,111s that included a full final -nc1.
# The speculative probe (CKPT_SPEC) SKIPS that filter. Measured post-sieve time, run #4:
#     matrix -nc2 250s + GPU Lanczos ~850s + CADO sqrt 245s + gaps = 1,203s
#     (worst observed sqrt was 396s -> worst-case downstream ~1,500s)
# 2000 covers the worst case with ~500s margin and returns 1,000s to the sieve, which is what the
# measured keys were short by.
#
# !! DO NOT raise this back to 3000 without re-checking the arithmetic above. !!
# !! DO NOT "fix" a shortfall by lowering REL_TARGET below the crossover -- that only guarantees an
#    unusable relation set sooner. !!
# =============================================================================================
# 2026-07-31: GNFS_WALL UNSET => UNBOUNDED (relation-bound) MODE. This is now the normal path.
# =============================================================================================
# breaking_rsa.py no longer imposes an internal deadline, so it exports no GNFS_WALL, and none of
# the budget arithmetic above applies: the sieve runs until the filter reports a matrix, the
# downstream runs, and if the total exceeds the validator's wall the container is killed
# (WallTimeFailure). That is preferred to the old behaviour, where the solver quit early on its own
# arithmetic and emitted {"status":"failed"} -- scored IncorrectFailure, i.e. a WRONG ANSWER, while
# the container had never actually gone overdue.
# Two measured reasons the old split lost time even on keys that would have fitted:
#   * the solver self-terminated at 3.983h, but the container manager polls for overdue containers
#     every 5 min, so the real kill is 4h + up to 5 min -- that grace was forfeited;
#   * DOWNSTREAM_RESERVE was subtracted UP FRONT, truncating the sieve on any key whose downstream
#     is cheaper than the reserve (measured post-sieve: 1,203s easy key vs 2,299s hard key, so no
#     single constant is correct).
# Set GNFS_WALL explicitly to restore bounded behaviour (dev harness / timeout regression test).
GNFS_WALL="${GNFS_WALL:-}"; DOWNSTREAM_RESERVE="${DOWNSTREAM_RESERVE:-2000}"
UNBOUNDED=0; [ -z "$GNFS_WALL" ] && UNBOUNDED=1
# SIEVE_SECS and LEFT stay EMPTY in unbounded mode. Every gpu_loop call site below uses
# `env ${VAR:+GPULOOP_MAX_SECS=$VAR}` so an empty value passes NO budget at all rather than a zero
# one -- which matters, because gpu_loop treats an ABSENT GPULOOP_MAX_SECS as "no limit" but would
# treat an explicit 0 or negative value as "no limit" too (see the guard `if(SIEVE_MAX>0 ...)`).
# Passing nothing is the unambiguous form. All LEFT-based guards are skipped when LEFT is empty.
rm -rf "$WORK"; mkdir -p "$WORK"
# ---- optional STAGE CACHE (dev iteration only; INERT unless GNFS_CACHE is set). Keyed by N so a
# re-run reuses already-computed stages (polyselect -> sieve relations -> matrix) instead of redoing
# them. Lets you iterate on filter/LA/sqrt/poly-quality without the ~2.6h sieve each time.
# ---------------------------------------------------------------------------
# FILTER BACKEND (2026-08-04). msieve's -nc1 is the binding constraint on hard
# keys, NOT the sieve: on key03 it had an excess SURPLUS (E_core 245,809 vs
# target 211,895) and still could not build a matrix, asking for 1M more
# relations -- and every msieve filter knob is inert on this build
# (filter_maxrels, filter_lpbound, target_density: five configs, byte-identical).
# CADO builds a matrix from the SAME relations and 323s faster. Measured cliff
# 45.03 -> 42.47 rel/lat. See cado-filter/Pipeline.md §11.
#
# CADO_FILTER=0 forces the old msieve path. The dispatcher also falls back to
# msieve automatically if the CADO tooling is missing or errors (rc other than
# 2), so a broken/absent cado-filter/ degrades to the previous behaviour rather
# than failing the run. rc=2 means purge found no matrix = a GENUINE shortfall,
# and is NOT retried on msieve (msieve would not do better, and a wasted -nc1
# probe costs ~570s of wall).
CADO_FILTER="${CADO_FILTER:-1}"
run_probe() {   # $1=rels $2=fb $3=N $4=outdir ; writes $4/m.dat.cyc on MATRIX
  local _r="$1" _fb="$2" _n="$3" _d="$4" _rc
  if [ "$CADO_FILTER" = 1 ] && [ -r "$SIEVER/../refilter/cado_probe.sh" ]; then
    POLY="${POLY:-$WORK/poly.cado}" bash "$SIEVER/../refilter/cado_probe.sh" "$_r" "$_fb" "$_n" "$_d"
    _rc=$?
    [ "$_rc" = 0 ] && return 0
    [ "$_rc" = 2 ] && return 2
    echo "  [filter] CADO probe unusable (rc=$_rc) -> msieve -nc1 fallback"
  fi
  bash "$SIEVER/../refilter/run_filter.sh" "$_r" "$_fb" "$_n" "$_d"
}
# DS_RELS for a CADO verdict must be the PURGE-ORDERED relation file, not the raw
# dump: the cycle file indexes purge order, so handing the downstream the original
# relations mismatches every cycle. Same class of coupling as the memfd snapshot.
ds_rels_for() { # $1=probe dir  $2=fallback rels
  if [ -s "$1/m.dat.rels" ]; then printf '%s\n' "$1/m.dat.rels"; else printf '%s\n' "$2"; fi
}

# NEVER set GNFS_CACHE on the validator (its /tmp is 1 GiB; the relation set is ~9 GB). Point it at
# roomy disk, e.g.  GNFS_CACHE=/root/work/enigma/temp/cache bash factor_msv.sh <N>
CACHE=""
if [ -n "${GNFS_CACHE:-}" ]; then
  NHASH=$(printf '%s' "$N" | md5sum | cut -c1-12)
  CACHE="$GNFS_CACHE/$NHASH"; mkdir -p "$CACHE"
  echo "### STAGE CACHE on: $CACHE  (N hash $NHASH) — completed stages are reused"
fi

# =============================================================================================
# PREFLIGHT (2026-08-23) -- FAIL FAST AND BY NAME, NEVER BY RUNNING OUT OF WALL
# =============================================================================================
# POLICY: a WallTimeFailure should mean "we were making progress and ran out of time". It must
# NOT be how the validator learns that a binary was missing, a directory was not writable, the
# GPU was absent, or the workdir had no space. Every one of those used to surface hours later --
# breaking_rsa.py's no-factors path deliberately HOLDS until the container manager kills it, so a
# tooling failure and a genuine time shortage are indistinguishable in the validator's report.
# Measured examples this exists to catch: a CuPy cache dir owned by another user (GPU Lanczos
# silently degrades to CPU, ~11x, guaranteeing the wall); cado_probe.sh / cado_predup.sh absent
# from the image (documented in the Dockerfile as SILENT -- the filter quietly downgrades).
# Anything fixable is fixed here; anything fatal exits NOW with a named reason, ~5 s in, while a
# retry still fits inside the wall.
preflight() {
  local fatal=0 _d _b
  # 1. binaries the run cannot proceed without ------------------------------------------------
  for _b in "$SIEVER/gpu_loop" "$MSV/msieve" "$MSV/msieve_la" "$MSV/msv_launcher" \
            "$MSV/msvrun_gpu.sh" "$SIEVER/polyselect_best.sh"; do
    [ -x "$_b" ] || { echo "  [preflight] FATAL: missing or not executable: $_b" >&2; fatal=1; }
  done
  # 2. binaries whose ABSENCE IS SILENT -- the run continues but degrades (Dockerfile note) ----
  for _b in "$SIEVER/../refilter/cado_probe.sh" "$SIEVER/../refilter/run_filter.sh" \
            "$SIEVER/../cado-filter/cado_filter.sh" "$SIEVER/../cado-filter/cado_predup.sh" \
            "$MSV/cado_sqrt" "$MSV/msieve_dump" "$MSV/cado_sqrt.sh" "$MSV/gpu_la.py"; do
    [ -r "$_b" ] || echo "  [preflight] WARNING: $_b missing -- that stage will silently degrade" >&2
  done
  for _b in freerel dup1 dup2 purge merge replay cado2msieve; do
    [ -x "$SIEVER/../cado-filter/$_b" ] \
      || { echo "  [preflight] WARNING: cado-filter/$_b missing -- CADO filter unavailable," \
                "hard keys will fail at the matrix" >&2; }
  done
  # 3. writability: workdir + every cache dir the downstream needs -----------------------------
  #    HOME/CUPY_CACHE_DIR/CUDA_CACHE_PATH are REPOINTED rather than failed on: a cache is a
  #    convenience, and re-JITting is far cheaper than losing the GPU LA path.
  if ! ( : > "$WORK/.wtest" ) 2>/dev/null; then
    echo "  [preflight] FATAL: workdir $WORK is not writable" >&2; fatal=1
  else rm -f "$WORK/.wtest"; fi
  for _v in HOME CUPY_CACHE_DIR CUDA_CACHE_PATH; do
    eval _d=\"\${$_v:-}\"
    [ -n "$_d" ] || continue
    mkdir -p "$_d" 2>/dev/null || true
    if ! ( : > "$_d/.wtest" ) 2>/dev/null; then
      local _alt="$WORK/.$(echo "$_v" | tr 'A-Z' 'a-z')"
      if mkdir -p "$_alt" 2>/dev/null && ( : > "$_alt/.wtest" ) 2>/dev/null; then
        rm -f "$_alt/.wtest"; export "$_v=$_alt"
        echo "  [preflight] $_v=$_d is not writable -- repointed to $_alt" >&2
      else
        echo "  [preflight] WARNING: $_v=$_d is not writable and no fallback worked" >&2
      fi
    else rm -f "$_d/.wtest"; fi
  done
  # 4. storage: refuse a workdir that cannot hold the run at all -------------------------------
  local _fkb; _fkb=$(df -Pk "$WORK" 2>/dev/null | awk 'NR==2{print $4}')
  if [ -n "${_fkb:-}" ] && [ "$_fkb" -lt "${PREFLIGHT_MIN_FREE_KB:-6291456}" ]; then
    echo "  [preflight] FATAL: only $(( _fkb / 1024 )) MB free on $(df -P "$WORK" 2>/dev/null | awk 'NR==2{print $6}')" \
         "-- the filter needs several GB and will fail hours from now. Refusing at t+0 instead." >&2
    fatal=1
  fi
  # 5. RAM-backed relation file: relhold refuses disk, so no memfd = no run --------------------
  python3 -c 'import os,sys; sys.exit(0 if hasattr(os,"memfd_create") else 1)' 2>/dev/null \
    || echo "  [preflight] WARNING: python has no memfd_create -- relhold falls back to /dev/shm" >&2
  # 6. GPU: 94% of the wall is GPU work; without it nothing below matters ----------------------
  if ! python3 -c 'import cupy; cupy.zeros(1); cupy.cuda.runtime.deviceSynchronize()' >/dev/null 2>&1; then
    echo "  [preflight] WARNING: python3 cannot run a CuPy op -- GPU block-Lanczos will fall back" \
         "to CPU msieve -nc (~11x slower, i.e. the wall). Check the interpreter and the driver." >&2
  fi
  [ "$fatal" = 0 ] || {
    echo "### PREFLIGHT FAILED -- refusing to start. This is a SETUP failure, NOT a time shortage," >&2
    echo "###   and NOT a relation shortfall. Fix the lines marked FATAL above." >&2
    exit 4
  }
  echo "  [preflight] ok $(date +%T)"
}
preflight

echo "### [1/4] msieve GPU polyselect (parallel wide-range search) + yield bake-off $(date +%T)"
# polyselect_best.sh runs a PARALLEL wide search over the productive c5 zone [1,32000] (16 slices, each
# its own -nf so no msieve.fb collision) + a top-K=3 MEASURED-yield bake-off. This finds cand0 (c5=18360,
# +4.8% yield over the old timeout-truncated poly). ~10 min on 24 CPU / 1 GPU. COARSE_N=0 skips the sparse
# high-c5 zone (measured worse). Emits $WORK/poly.cado + $WORK/c.fb. See Pipeline.md.
if [ -n "$CACHE" ] && [ -s "$CACHE/poly.cado" ] && [ -s "$CACHE/c.fb" ]; then
  echo "  [cache HIT] poly.cado + c.fb — skipping polyselect"; cp "$CACHE/poly.cado" "$CACHE/c.fb" "$WORK/"
else
  COARSE_N=0 bash "$SIEVER/polyselect_best.sh" "$N" "$WORK"
  if [ -n "$CACHE" ]; then cp "$WORK/poly.cado" "$WORK/c.fb" "$CACHE/" 2>/dev/null && echo "  [cache SAVE] poly.cado + c.fb" || true; fi
fi

# PRECOMPUTE freerel WHILE THE SIEVE RUNS (2026-08-09). freerel depends only on (poly, lpb), costs
# 11.07s measured, and used to run inside every probe -- on the critical path each time. The poly is
# final right here; the first probe is ~3.6h away. Backgrounded so it costs the sieve nothing (it is
# 11s of CPU against a GPU-bound stage that has ~20 idle cores). cado_filter.sh symlinks the result;
# a miss just recomputes, so a failure here is invisible rather than fatal.
export CADO_FREEREL_CACHE="${CADO_FREEREL_CACHE:-$WORK/frcache}"
mkdir -p "$CADO_FREEREL_CACHE" 2>/dev/null || true
if [ -s "$WORK/poly.cado" ] && [ -x "$SIEVER/../cado-filter/freerel" ]; then
  (
    _k=$(cat "$WORK/poly.cado" | md5sum | cut -c1-16)-lpb${LPB:-29}
    _t="$CADO_FREEREL_CACHE/.pre.$$"; mkdir -p "$_t" || exit 0
    "$SIEVER/../cado-filter/freerel" -poly "$WORK/poly.cado" -lpb0 "${LPB:-29}" -lpb1 "${LPB:-29}" \
        -out "$_t/freerel.gz" -renumber "$_t/renumber.gz" -t 8 > "$_t/freerel.log" 2>&1 \
      && grep -qa 'nprimes=' "$_t/freerel.log" \
      && { rm -rf "${CADO_FREEREL_CACHE:?}/$_k"; mv -f "$_t" "$CADO_FREEREL_CACHE/$_k"; \
           echo "  [freerel] precomputed into cache ($_k) $(date +%T)"; } \
      || rm -rf "$_t"
  ) >/dev/null 2>&1 &
fi

# Wall-time budget for the sieve = remaining GNFS budget minus a downstream reserve. The sieve uses
# this whole window for a low-yield key (gathering as many relations as possible) instead of quitting
# early at a fixed QMAX with budget to spare (the seed-42 failure mode).
NOW=$(date +%s); ELAPSED=$(( NOW - T_START ))
if [ "$UNBOUNDED" = 1 ]; then
  SIEVE_SECS=""
  echo "### [2/4] GPU lattice sieve -> $REL_TARGET relations (q<=$QMAX, NO TIME BUDGET -- relation-bound) $(date +%T)"
else
  SIEVE_SECS=$(( GNFS_WALL - ELAPSED - DOWNSTREAM_RESERVE ))
  [ "$SIEVE_SECS" -lt 600 ] && SIEVE_SECS=600   # floor so a late start still attempts a sieve
  echo "### [2/4] GPU lattice sieve -> $REL_TARGET relations (q<=$QMAX, wall<=${SIEVE_SECS}s, reserve ${DOWNSTREAM_RESERVE}s) $(date +%T)"
fi
# Bug-1 fix: the full relation set is ~9 GB but the validator's /tmp is a 1 GiB tmpfs, which would
# silently truncate rels.txt. Back rels.txt with an anonymous memfd (RAM) via relhold.py so the
# sieve output never touches /tmp.  (downstream .lp/.mat already use the msv_launcher memfd broker.)
RELPID=""
if [ -n "$CACHE" ] && [ -s "$CACHE/rels.txt" ]; then
  echo "  [cache HIT] rels.txt — skipping sieve"; cp "$CACHE/rels.txt" "$WORK/rels.txt"
  # DS_RELS is otherwise assigned ONLY inside the sieve branch below (line ~294), so on a cache
  # hit it stayed empty and the downstream died instantly with `line 560: : No such file or
  # directory` -> "NO FACTOR". That made GNFS_CACHE -- the documented dev iteration path
  # (Pipeline.md §6) -- unable to reach the filter/LA/sqrt stages at all. Found 2026-08-08.
  DS_RELS="$WORK/rels.txt"; SPEC_CYC=""; CKPT_CYC=""; RELPID=""
  # Run the SAME filter production runs (CADO by default via run_probe). Without this the cache
  # path fell through to msieve -nc1 -- the filter Pipeline.md §11 documents as failing on hard
  # keys -- so a cached-relation run was not representative of production at all.
  mkdir -p "$WORK/ckpt0"
  if run_probe "$WORK/rels.txt" "$WORK/c.fb" "$N" "$WORK/ckpt0"; then
    CKPT_CYC="$WORK/ckpt0/m.dat.cyc"; DS_RELS=$(ds_rels_for "$WORK/ckpt0" "$WORK/rels.txt")
    echo "  [cache] filter produced a matrix -> reusing cycles (SKIP_NC1)"
  else
    echo "  [cache] filter reported SHORT on the cached relations"
  fi
else
  # >/dev/null 2>&1 IS LOAD-BEARING (measured 2026-07-28): relhold must NOT inherit our
  # stdout. It is a long-lived holder that survives a mid-run failure of this script (it gets
  # reparented to init), and for as long as it holds the pipe open, breaking_rsa.py's
  # `for line in proc.stdout` never sees EOF -- so a crash here turns into an INFINITE HANG
  # with no result payload, instead of a reported failure. The trap makes the holder's death
  # follow ours, so the memfd is released even on an abnormal exit.
  python3 "$SIEVER/relhold.py" "$WORK/rels.txt" >/dev/null 2>&1 & RELPID=$!
  trap 'kill "$RELPID" 2>/dev/null || true; [ -n "${SNAPPID:-}" ] && kill "$SNAPPID" 2>/dev/null || true; [ -n "${PREDUPPID:-}" ] && kill "$PREDUPPID" 2>/dev/null || true' EXIT INT TERM
  for i in $(seq 1 100); do [ -L "$WORK/rels.txt" ] && break; sleep 0.1; done
  cd "$SIEVER"
  # Retry on a transient GPU-init miss (cudaErrorNoDevice / 0 relations) — belt-and-suspenders behind the
  # breaking_rsa.py keep-alive, for the de-init window before the holder warms. Capped so it can't eat the wall.
  # ===========================================================================================
  # SPECULATIVE EARLY PROBE (CKPT_SPEC, default on).  Measured 2026-07-28: saves ~1,000-1,400s.
  # ===========================================================================================
  # The sieve is GPU-BOUND (7.9 ms GPU vs 1.1 ms CPU per lattice), so ~85% of the CPU idles for
  # 3.3 hours while the 364s filter waits to run strictly AFTER it. But simply running the filter
  # concurrently at the END saves nothing: its verdict gates the downstream either way.
  #
  # What DOES pay is probing EARLY. Sieve to SPEC_TARGET (< REL_TARGET), snapshot, and run the
  # filter on the idle CPU while the sieve carries on toward REL_TARGET:
  #   * MATRIX at SPEC_TARGET -> stop sieving NOW. Saves the remaining sieve AND the final filter.
  #   * SHORT                 -> cost is ~0 wall time; the ladder proceeds exactly as before.
  # Evidence this has real headroom: seed 20260728 crossed at 66M against REL_TARGET=68.5M, and
  # seed 20260729 finished with E_core surplus +16% over target_excess at 68.5M.
  #
  # The SNAPSHOT IS MANDATORY, for two independent reasons:
  #   1. msieve -nc1 MUTATES its input in place (it appends free relations: 68,500,027 ->
  #      68,621,459 on seed 20260729), and
  #   2. the sieve is concurrently APPENDING to rels.txt.
  # Either alone corrupts a filter reading rels.txt directly. The winning snapshot therefore also
  # becomes the relation set handed to the downstream (DS_RELS), because m.dat.cyc indexes exactly
  # the file the filter mutated.
  # Set CKPT_SPEC=0 to disable and fall back to the plain sieve-then-probe behaviour.
  DS_RELS="$WORK/rels.txt"; SPEC_CYC=""; SPECPID=""; SNAPPID=""
  # TRIED AND REJECTED 2026-08-08: co-sieving with CADO las on the "idle" CPU cores.
  # The premise (GPU 9.11 ms/lat vs 0.96 ms CPU => ~20 of 24 cores free) is true ON AVERAGE and
  # wrong in practice: the CPU work is BURSTY -- the GPU stalls while all 24 threads factor the
  # cofactors after each lattice. Live e2e: GPU 6,236,917 rels at t=885s vs 6,523,032 GPU-only
  # (-4.4%) while las added only +3.9% -> NET NEGATIVE. Independently: OMP=24 beats OMP=12 by
  # 4.3% over 3 reps. las relations are also worth only 0.129 excess/relation vs 0.222 for GPU
  # ones (rare large-prime ideals that peel as singletons). Do not retry without fixing both.
  # PROBE POINT, chosen from the saving/loss economics (measured 2026-07-28):
  #   MATRIX at X -> saving = S, the sieve seconds between X and REL_TARGET.
  #   SHORT  at X -> the ladder must still filter at REL_TARGET, so the probe saves nothing;
  #                  it LOSES max(0, 364-S) if S is too small to hide the filter behind.
  # At the end-of-sieve rate (~4,400 rel/s) and a 364s filter:
  #     95% -> S=778s safe | 96% -> 623s safe | 97% -> 467s safe | 98% -> 311s LOSES 53s
  # Measured crossovers: seed20260728 at 66.0M (96.4% of REL_TARGET), 2026-07-27 key at 68.0M
  # (99.3%). 97% (=66.4M) sits just ABOVE the one key known to cross early -- so it wins there --
  # while keeping S=467s > 364s, i.e. ZERO downside on a key that crosses late.
  # !! Only 2 crossover measurements back this. Re-check with refilter/matrix_health.sh if more
  # keys become available; do NOT push past 97% without re-deriving the table above. !!
  # RE-PROBE ON SHORT (2026-08-09). Until now a SHORT verdict meant the sieve ran all the way to
  # REL_TARGET before anything filtered again -- on key07 that is ~13,760s + a 261s filter ~= 13,944s,
  # versus 13,261s for the MATRIX path. The sieve is GPU-bound and the probe is CPU-bound, so there
  # is no reason to wait: on SHORT, release the snapshot, take a new one at the CURRENT relation
  # count and probe again, with the sieve still running underneath.
  # This matters twice over, because it also makes the FIRST probe point cheap to lower:
  #   MATRIX probe = 261s   (freerel 11 + strip/dup1 50 + dup2 82 + purge 41 + merge/replay 77)
  #   SHORT  probe = 184s   (exits at purge -- never runs merge/replay/cado2msieve)
  # so a first probe that misses costs ~12s net, while one that hits at 96% saves ~177s.
  # Measured 2026-08-09; the 184s figure is the 62M probe on this box, the 261s the key07 e2e.
  spec_launch() {   # snapshot the current relations and start a probe in the background
    # `|| true` matches the idiom used elsewhere in this file. NOT load-bearing here: bash's set -e
    # exempts a failing command inside an && list unless it is the one after the final && (verified
    # 2026-08-09), so an empty SNAPPID on the first call would not abort. Kept for consistency.
    { [ -n "${SNAPPID:-}" ] && { kill "$SNAPPID" 2>/dev/null || true; SNAPPID=""; }; } || true   # free the old 8 GB memfd
    # m.dat.rels MUST be cleared too: ds_rels_for() hands it to the downstream as DS_RELS whenever
    # it is non-empty, so a leftover from an earlier probe in this same SPEC_DIR would be fed to the
    # matrix build INSTEAD of the winning snapshot's relations -- a wrong-input failure that shows up
    # only as a bad factorization at the very end. (cado_filter.sh writes it last, on the MATRIX path
    # only, so today it should never survive a SHORT -- this is belt-and-braces for the re-probe loop.)
    rm -f "$SPEC_DIR/.finished" "$SPEC_DIR/m.dat.cyc" "$SPEC_DIR/m.dat.purged" \
          "$SPEC_DIR/m.dat.rels" 2>/dev/null || true
    rm -rf "$SPEC_DIR/cf" 2>/dev/null || true
    local _n="$1"
    # !! DELETE THE OLD SYMLINK BEFORE STARTING THE NEW HOLDER -- 2026-08-11 !!
    # THIS IS THE RE-PROBE-LOOP BUG. relhold.py publishes its memfd as a symlink at $SNAP pointing
    # at /proc/<holder pid>/fd/N. The kill above frees the memfd but leaves the SYMLINK behind, so
    # on every re-probe a stale link to a DEAD pid is still sitting at $SNAP. The readiness test
    # below only asked "is a symlink present", which the stale one satisfies INSTANTLY -- so the
    # `head` below wrote through a dangling /proc entry, failed with "Permission denied", and the
    # `|| true` swallowed it. The probe then ran on a non-existent snapshot and came back SHORT in
    # SIX SECONDS. That bogus verdict also starved the SPEC_MIN_GAIN gate below (6s accrues ~27k
    # relations against a 300k threshold), so the walk-up was abandoned after 2 of its 4 probes and
    # the sieve ran all the way to REL_TARGET.
    # MEASURED 2026-08-11, key07 at CKPT_SPEC_PCT=93: sieve stage 13,799s instead of ~12,900s, total
    # 15,050s = 4.18h. Probe #1 was never affected ($WORK is freshly rm -rf'd, so no stale link),
    # which is why the loop looked healthy for as long as the first probe kept returning MATRIX --
    # i.e. THE WALK-UP HAS NEVER ONCE WORKED, and the -235s credited to it in Pipeline.md §16a came
    # entirely from runs where probe #1 hit.
    rm -f "$SNAP" 2>/dev/null || true
    python3 "$SIEVER/relhold.py" "$SNAP" >/dev/null 2>&1 & SNAPPID=$!
    # -L alone is not sufficient even after the rm: the new holder could be mid-publish, and a
    # symlink whose /proc target is not a live fd of ours is not writable. Require BOTH.
    local i; for i in $(seq 1 100); do { [ -L "$SNAP" ] && [ -w "$SNAP" ]; } && break; sleep 0.1; done
    { [ -L "$SNAP" ] && [ -w "$SNAP" ]; } || { echo "  [spec] snapshot memfd never appeared — skipping probe" >&2; SPECPID=""; return 1; }
    echo "  [spec] snapshotting $_n relations $(date +%T)"
    # A FAILED SNAPSHOT MUST NOT BE LAUNDERED INTO A "SHORT" FILTER VERDICT. That is the same class
    # of defect as Pipeline.md §11's tooling-failure-reported-as-SHORT: the caller cannot tell a
    # real excess shortfall from a broken input, so it draws the wrong conclusion and keeps going.
    # Report the launch as failed instead; the caller retries at the next relation step.
    if ! head -n "$_n" "$WORK/rels.txt" > "$SNAP" 2>/dev/null; then
      echo "  [spec] snapshot WRITE FAILED — skipping this probe (this is NOT a SHORT verdict)" >&2
      { [ -n "${SNAPPID:-}" ] && { kill "$SNAPPID" 2>/dev/null || true; SNAPPID=""; }; } || true
      SPECPID=""; return 1
    fi
    # O(1) plausibility check on the result. `wc -l` here would re-read ~8 GB on the probe's
    # critical path; stat on the memfd is free. Relations run ~129 bytes, so 64 is a safe floor.
    local _sz; _sz=$(stat -Lc %s "$SNAP" 2>/dev/null || echo 0)
    if [ "$_sz" -lt $(( _n * 64 )) ]; then
      echo "  [spec] snapshot is only $_sz bytes for $_n relations — skipping this probe (NOT a SHORT verdict)" >&2
      { [ -n "${SNAPPID:-}" ] && { kill "$SNAPPID" 2>/dev/null || true; SNAPPID=""; }; } || true
      SPECPID=""; return 1
    fi
    echo "  [spec] probe #$SPEC_N started on the snapshot; sieve continues to $REL_TARGET $(date +%T)"
    # THREAD SPLIT (measured 2026-07-28). The probe and the still-running sieve are BOTH
    # CPU-hungry, and on a 24-core validator each defaulting to 24 threads means 48 threads
    # thrashing 24 cores: the sieve degraded 8.08 -> 18-28 ms/lat (GPU time unchanged at 7.9,
    # so it was pure CPU starvation) and a SHORT verdict cost ~222s of lost sieving -- NOT the
    # "~0" this design originally assumed.
    # Cheap to fix because the sieve barely needs CPU: the OMP sweep measured 8.215 ms/lat at
    # OMP=4 vs 8.230 at OMP=24 (its CPU stage hides under a 7.9 ms GPU stage). NT+OMP = 24.
    # !! The 20/4 ratio is REASONED, NOT MEASURED -- msieve -nc1 thread scaling was not profiled.
    # !! `|| true` IS LOAD-BEARING (2026-08-06) -- do not remove !!
    # This subshell inherits `set -e` (line 12) and run_probe is NOT in a condition context
    # here, so ANY non-zero exit from cado_probe.sh killed the subshell ON THE SPOT: `_rc=$?`
    # inside run_probe was never reached, so the CADO -> msieve fallback that cado_probe.sh's
    # rc=3 exists to trigger could not run at this call site at all (the ladder's site gets it
    # for free from its own `&& _prc=0 || _prc=$?`), and `.finished` was never written.
    # Putting the call in a condition context suspends errexit for the whole function body.
    ( NT="${SPEC_FILTER_NT:-20}" run_probe "$SNAP" "$WORK/c.fb" "$N" "$SPEC_DIR" \
        >"$SPEC_DIR/filter.out.$SPEC_N" 2>&1 || true; echo done > "$SPEC_DIR/.finished" ) & SPECPID=$!
    SPEC_AT="$_n"
    return 0
  }
  # 97 -> 96 (2026-08-09): safe only BECAUSE of the re-probe loop above. The old 97% was chosen to
  # sit above the one key known to cross early, since a SHORT then forfeited everything; now a miss
  # costs ~12s. Measured crossovers: 66.0M / 68.0M (old keys), ~65.5M extrapolated on key07 from
  # 62M excess=-751,416 and 63M excess=-534,730 (+216,686 per 1M).
  # PROBE POINT: 97 -> 96 -> 95 -> 93 (2026-08-10). Three e2e runs on key07 give the MARGINAL curve.
  # The sieve saving per 685k-relation step is CONSTANT (~176s); the downstream charge-back GROWS,
  # because fewer relations means less excess, a bigger matrix and more Lanczos iterations:
  #     probe point        sieve   matrix   Lanczos           net/step   LA iters
  #     66,445,014 (base)    --      --       --                 --       65,562
  #     65,760,018 (96%)   -177s    +13s    +31s               -133s      68,325
  #     65,075,012 (95%)   -176s    +30s    +88s (vs base)     -102s      71,790
  # Extrapolating the charge-back growth (44 -> 74 -> ~110 -> ~155 per step) against the flat -176s:
  #     94% -> -66s   93% -> -87s cumulative   92% -> turns over and starts LOSING
  # so ~93% is the optimum and 95% stopped two steps short. THE OPTIMUM IS INTERIOR -- probing ever
  # earlier is NOT monotonically better, which is why this is pinned to a measured curve.
  #
  # ⚠️ 93 WAS TESTED END-TO-END ON 2026-08-11 AND IT LOST. REVERTED TO 95. ⚠️
  # The extrapolation above is only valid while the probe still HITS, and at 93% it does not:
  # probe #1 at 63,705,010 ran a full 141s purge and returned a GENUINE SHORT. Together with the
  # MATRIX at 65,075,012, that BRACKETS key07's purge crossover in (63,705,010, 65,075,012] --
  # which is the measurement the note this replaces called "unknown". 95% sits above it and hits
  # on the first probe; 93% does not, so the -87s it was chosen for does not exist on this key.
  # (The run also cost 15,050s = 4.18h, but most of that was the stale-symlink bug in spec_launch
  # above, not the probe point itself -- do not read the 4.18h as the cost of 93%.)
  # Restoring a lower probe point is only worth revisiting once the crossover is known per-key,
  # because the walk-up covers at most SPEC_MAX_PROBES x ~675k = ~2.7M relations of ground.
  # !! RE-DERIVE THIS CURVE IF J, LIM, lpb OR THE FILTER CHANGE -- it is a property of the matrix,
  #    not of the sieve, and every one of those moves the matrix. !!
  # =============================================================================================
  # 2026-08-12: PROBE POINT 95 -> 93, AND SPEC_MAX_PROBES 4 -> 8. THESE TWO MUST MOVE TOGETHER.
  # =============================================================================================
  # WHY THIS IS NOT A REPEAT OF THE 93% THAT WAS REVERTED ON 2026-08-11. That revert is recorded
  # above and in §16b-bis, and its evidence was a 15,050s e2e -- but §17a then established that the
  # run was dominated by the stale-symlink bug in spec_launch, and says in terms: "do not read the
  # 4.18h as the cost of 93%". The walk-up was fixed the same day and HAS NEVER BEEN e2e-TESTED AT
  # A LOW PROBE POINT. So the case against 93% rests on a measurement taken with the mechanism that
  # makes 93% safe switched off. Both numbers below are re-derived against a WORKING walk-up.
  #
  # WHAT A MISS COSTS NOW. A SHORT probe is ~184s and exits at purge; it runs on the idle CPU while
  # the GPU-bound sieve continues underneath, so the sieve advances ~675k relations (173s x 3,900)
  # during it and the net charge is ~12s. The walk-up therefore lands within ONE 675k step above
  # the true crossover no matter where it starts -- starting lower buys per-key adaptivity for ~12s
  # a step instead of forfeiting the whole saving.
  #
  # THE ARITHMETIC, against every crossover this repo has measured or bracketed:
  #     key07           65,055,083  = 94.97% of REL_TARGET   (§16b-bis, measured off a dump)
  #     key08          <=65,075,000 = 95.00%  (MATRIX on probe #1 -- an UPPER BOUND, not a value)
  #     seed20260728    66,000,000  = 96.35%  (§16a)
  #     yd2             66,567,000  = 97.18%  (Pipeline.md §17, crossed by +0.163%)
  #     2026-07-27 key  68,000,000  = 99.27%  (§16a)
  #   coverage = first probe + (SPEC_MAX_PROBES-1) x 675k:
  #     95% x 4  -> 65,075,000 .. 67,100,000   FALLS THROUGH on the 68.0M key   <-- SHIPPED TODAY
  #     93% x 4  -> 63,705,000 .. 65,730,000   FALLS THROUGH on three of five   <-- NEVER DO THIS
  #     93% x 8  -> 63,705,000 .. 68,430,000   covers every known key
  #
  # !! THE 4-PROBE BUDGET IS ALREADY TOO SMALL AT 95%. !!  That is a defect in the shipped config,
  # not a new risk introduced by 93: a key whose crossover sits at 68.0M exhausts the walk-up,
  # falls through to REL_TARGET and lands ~868s late (the §16a failure mode). Widening to 8 fixes
  # that on its own and is the reason this change is a net safety IMPROVEMENT rather than a trade.
  #
  # WHY 8 IS CHEAP. An unused probe costs ~12s; a fall-through costs 250-870s. Eight probes reach
  # 68,430,000 -- i.e. REL_TARGET -- so the walk-up can no longer exhaust before the fixed target
  # it would otherwise fall back to, which makes the fall-through path unreachable by construction.
  # Cost if every probe misses: ~96s. Thread contention is bounded by the existing 20/4 split, and
  # 8 x 184s = 1,472s of a ~12,500s sieve at SPEC_SIEVE_OMP=4 is ~0.02% by §14b's OMP curve.
  # Memory is unchanged: probes are strictly sequential, so it is still ONE 8 GB snapshot at a time.
  #
  # HONEST EV ON THE 95 -> 93 HALF. It is NOT free, and it is smaller than the SPEC_MAX_PROBES half:
  #   * key07 (crossover 94.97%): 93% goes SHORT twice and lands at ~65,169,000 instead of
  #     65,075,000 -- 114k / ~29s WORSE. This is the worst case among known keys.
  #   * any key whose crossover is genuinely below 65.055M: saves ~160-350s. key08 is the candidate
  #     -- probe #1 hit, so all we know is <=65,075,000, and how far below is UNMEASURED.
  #   * high-crossover keys: unaffected in landing point, ~24s for the two extra SHORT probes.
  # So this half is EV-positive only if crossovers really do spread below 65.055M. THAT IS THE
  # THING TO VERIFY, and §16b-bis gives a 5-minute method for it that does not need a 4h run:
  # truncate a saved dump with `head -n K` and run cado_probe.sh at two points; interpolate.
  # key08's dump is saved (see hard-key/key08.json "saved_relations", 65,898,392 raw relations).
  # !! IF key08's CROSSOVER COMES BACK AT OR ABOVE 65.0M, REVERT THIS HALF TO 95 AND KEEP THE 8. !!
  # ❌ 93 WAS TESTED ON key08, 2026-08-12, AND IT LOST. REVERTED TO 95. (§19f)
  # Measured, not modelled: probes #1-#3 at 63,705,003 / 64,180,444 / 64,713,244 all returned SHORT
  # (136s each), MATRIX on probe #4 at 65,243,879. The 95% run lands on probe #1 at 65,075,000, so
  # 93% oversieved 168,879 relations = 43s at the measured 3,902 rel/s; the reconstructed total was
  # 14,160s against the 14,104s baseline (+56s).
  # key08's crossover is therefore bracketed to (64,713,244, 65,075,000] = 94.48%-95.00% of
  # REL_TARGET -- i.e. 95% sits DIRECTLY above it and is near-optimal on this key by construction.
  # THE GENERAL LESSON, which is why this is not just a revert: the probe point is a GRID PHASE, not
  # a margin. The landing point is the first grid step at or above a KEY-SPECIFIC crossover, so
  # lowering the percentage does not "reduce margin" -- it re-phases the grid and can land HIGHER.
  # Both keys with a measured crossover (key07 94.97%, key08 94.48-95.00%) sit just under 95%.
  # !! Do not lower this again without measuring the crossover FIRST -- refilter/crossover.sh does
  #    it in ~5 minutes from a saved dump, versus the 4h this revert cost. !!
  # ---- ABSOLUTE PROBE BREAKPOINTS (2026-08-20) -----------------------------------------------
  # The probe point used to be a PERCENTAGE of REL_TARGET, which tied two unrelated decisions
  # together: "where is this key's purge crossover" and "how far do we sieve if we never find it".
  # 19f/19g measured the crossovers directly (key07 94.97%, key08 94.48-95.00%) -- those are
  # RELATION COUNTS, and expressing them as a percentage of a ceiling that also wants to move is
  # what made CKPT_SPEC_PCT=93 land HIGHER than 95 rather than lower (the "grid phase, not a
  # margin" finding). Breakpoints are absolute, so each one means exactly what it says.
  #
  # Behaviour is otherwise UNCHANGED and still fully concurrent with the sieve: the sieve runs to
  # the first breakpoint, snapshots, and keeps sieving while the filter probes the snapshot in the
  # background. On MATRIX the sieve is killed immediately; on SHORT the sieve has meanwhile carried
  # the count past the next breakpoint (or we wait for it) and the next probe launches. Nothing
  # here serialises the sieve behind a probe.
  #
  # The FIRST breakpoint defaults to 65,075,000 -- the exact point the 2026-08-16/17 cold e2e runs
  # verified end-to-end -- so an unconfigured build probes where the verified one did.
  SPEC_PCT="${CKPT_SPEC_PCT:-95}"          # retained ONLY to derive a first break if none is given
  SPEC_MAX_PROBES="${SPEC_MAX_PROBES:-8}"  # bound the re-probe loop; 0 verdicts is not a failure mode
  SPEC_MIN_GAIN="${SPEC_MIN_GAIN:-300000}" # floor: never re-probe on fewer new relations than this
  # VERIFIED END-TO-END 2026-08-20 on key08, cold, p*q==N in 12,791.5s. The run was forced with
  # CKPT_FIRST_BREAK=62000000 (below key08's crossover) so the walk-up HAD to run: probes at
  # 62,000,029 / 63,021,096 / 64,152,448 all SHORT, MATRIX at 65,334,209 on probe #4, sieve killed
  # early. The gate selected the next un-passed breakpoint each time and correctly SKIPPED 64,250,000
  # once the sieve had already run past it. No waiting occurred -- probes ran back-to-back.
  CKPT_FIRST_BREAK="${CKPT_FIRST_BREAK:-65075000}"
  CKPT_BREAK_STEP="${CKPT_BREAK_STEP:-750000}"
  # ~750k is one probe-duration of sieving (a purge is 140-260s; production is ~3.9k rel/s), so the
  # ladder stays probe-paced -- a breakpoint is normally already behind us when the previous verdict
  # lands, and the walk-up runs back-to-back exactly as it does today. Widening the step trades
  # probe count for coarser resolution around the crossover; it does not idle the sieve either way.
  # The ladder runs to the CEILING, not to SPEC_MAX_PROBES entries. Each snapshot is taken at the
  # live count, which is already PAST the breakpoint that triggered it (the sieve never pauses), so
  # SPEC_AT outruns the ladder and _break_after skips entries. Generating only SPEC_MAX_PROBES of
  # them would therefore exhaust the ladder while probes were still in the budget -- the exact
  # "silently sieve to REL_TARGET with probes left" failure 16a warns about. SPEC_MAX_PROBES still
  # bounds how many are actually spent; the 64-entry cap is only a runaway guard on a tiny step.
  if [ -z "${CKPT_BREAKS:-}" ]; then
    CKPT_BREAKS=""; _b="$CKPT_FIRST_BREAK"; _k=0
    while [ "$_b" -le "$REL_TARGET" ] && [ "$_k" -lt 64 ]; do
      CKPT_BREAKS="$CKPT_BREAKS $_b"; _b=$(( _b + CKPT_BREAK_STEP )); _k=$(( _k + 1 ))
    done
  fi
  # First breakpoint strictly greater than $1, or "" when the ladder is exhausted. Stateless on
  # purpose -- an index would have to be kept in step with a sieve that can blow past several
  # breakpoints while one probe runs, and that is precisely the class of drift that produced the
  # stale-symlink walk-up bug (17a). ALWAYS returns 0: this runs under `set -e` (line 12) and is
  # used in an assignment, where a non-zero command substitution would abort the run.
  _break_after(){ local _c="$1" _x; for _x in $CKPT_BREAKS; do
                    if [ "$_x" -gt "$_c" ]; then echo "$_x"; return 0; fi; done; echo ""; return 0; }
  SPEC_N=1; SPEC_AT=0
  SPEC_TARGET=$(_break_after 0)
  if [ -z "$SPEC_TARGET" ]; then
    # Ladder came out empty -- only reachable by misconfiguration (CKPT_FIRST_BREAK above the
    # ceiling, or an explicitly empty CKPT_BREAKS). Fall back to the old percentage rule for the
    # first point and REBUILD a ladder from it, rather than leaving CKPT_BREAKS empty: an empty
    # ladder makes _break_after return "" forever, which would silently disable the whole walk-up
    # and sieve straight to REL_TARGET with the full probe budget unspent.
    SPEC_TARGET=$(( REL_TARGET * SPEC_PCT / 100 ))
    CKPT_BREAKS=""; _b="$SPEC_TARGET"; _k=0
    while [ "$_b" -le "$REL_TARGET" ] && [ "$_k" -lt 64 ]; do
      CKPT_BREAKS="$CKPT_BREAKS $_b"; _b=$(( _b + CKPT_BREAK_STEP )); _k=$(( _k + 1 ))
    done
    echo "  [spec] no usable breakpoint ladder -- rebuilt from ${SPEC_PCT}% of REL_TARGET:$CKPT_BREAKS" >&2
  fi
  if [ "${CKPT_SPEC:-1}" = 1 ] && [ "$SPEC_TARGET" -gt 1000000 ]; then
    echo "  [spec] phase A: sieve to breakpoint $SPEC_TARGET, then probe while sieving on (ladder:$CKPT_BREAKS ceiling $REL_TARGET)"
    GPU_SPEC_TARGET="$SPEC_TARGET"
    # ---- PRE-DEDUP WARM-UP (2026-08-21) -----------------------------------------------------
    # Half of a MATRIX probe is dup1+dup2 re-reading every relation (132s of 261s on the EPYC,
    # more on a Xeon), and the probe is on the critical path -- the sieve running under it is
    # thrown away the moment the verdict is MATRIX. Relations are append-only and CADO's dup2
    # takes already-renumbered input at 6.65x the rate of raw input, so that half can be paid
    # EARLY, under the GPU-bound sieve, on cores Pipeline.md §14 measured sitting idle.
    #
    # This backgrounds cado_predup.sh to deduplicate everything up to PREDEDUP_LEAD relations
    # BELOW the first breakpoint. At the production rate (~3,900 rel/s) a 6M lead leaves ~25 min
    # between the warm-up finishing and the probe firing, so the two never overlap and the probe
    # never waits on it. cado_filter.sh re-verifies the prefix against its own snapshot and
    # silently falls back to the one-shot path on any mismatch, so a warm-up that fails, is
    # killed, or finishes late costs nothing but the CPU it used.
    #
    # ON BY DEFAULT as of 2026-08-21, after an A/B on the captured key08 corpus at the real
    # breakpoint: probe 428s -> 300s, with dup1 99->18s and dup2 83->29s, and identical unique
    # count / purge first+final state / matrix-within-purge-noise. CADO_PREDEDUP=0 disables it.
    #
    # ⚠️ THE ONE THING NOT MEASURED IN A LIVE RUN is this warm-up competing with the sieve; both
    # A/B legs ran standalone. It is enabled anyway because the arithmetic survives the pessimistic
    # case: the warm-up is 12 threads at nice 10 against a sieve whose real CPU demand is ~3 of 24
    # cores (Pipeline.md §14), i.e. strictly LIGHTER than the NT=20 probe this design already runs
    # concurrently for 184-385s. Even a 10% ms/lat penalty for the whole warm-up is ~+30s against
    # -128s. And every failure mode is graceful: a warm-up that is slow, killed, or never finishes
    # simply leaves no READY file, and cado_filter.sh takes the one-shot path.
    # ---- 2026-08-23: THIS BROKE THE FILTER, AND THE FAULT WAS THE GUARD, NOT THE WARM-UP ----
    # A cold e2e at validator spec on a FRESH RANDOM KEY (not key08) found the CADO filter
    # refusing to start: the warm-up leaves ~2,172 MB of slices in $WORK, and cado_filter.sh's
    # headroom guard demanded 7,842 MB free against 7,694 MB -- short by 148 MB, exit 3, which
    # cado_probe.sh maps to "TOOLING FAILURE -> falling back to msieve". The CADO filter -- the
    # whole reason hard keys work -- never ran.
    #
    # THE GUARD WAS CHARGING THE ONE-SHOT BUDGET TO THE INCREMENTAL PATH. It ran BEFORE the cache
    # check and always demanded ~1.00x RELKB, the cost of splitting EVERY relation -- work the
    # incremental path does not do. Decomposed against the same Xeon calibration:
    #     one-shot     0.52x RELKB full gz split + 0.50x purge/merge   = ~1.02x RELKB  (~8,690 MB)
    #     incremental  prefix slices ALREADY WRITTEN + 1.5x the DELTA + 0.50x purge/merge
    #                                                                  = ~5,030 MB
    # i.e. the warm-up REDUCES peak $WORK by ~1,400 MB, because 2,172 MB of prefix slices replace
    # a 4,280 MB full split. Pre-dedup is storage-POSITIVE. The guard only ever saw it as negative
    # because it measured free space after the cache was written and then compared it against work
    # that would never happen.
    # Fixed in cado-filter/cado_filter.sh: the cache check now runs FIRST and the budget is
    # path-aware, with a last-resort self-heal that drops the cache if even the incremental budget
    # does not fit. Re-checked against the captured numbers: need 5,028 MB, free 7,694 MB -> PASS
    # with 2,666 MB to spare, where the old rule refused by 148 MB.
    # So the warm-up stays ON (it is worth ~-123 s on an EPYC 9555, Daily-20260821.md 9) and the
    # trigger below no longer loses half its lead to a wrong bytes/relation constant.
    # ⚠️ Daily-20260821.md 9's caveat still stands and is now the ONLY open item here: the warm-up
    # running CONCURRENTLY with the sieve has still never been timed end-to-end. Measured during
    # the audit: it costs 457 s under a live sieve, not the 170 s measured standalone (2.7x).
    if [ "${CADO_PREDEDUP:-1}" = 1 ] && [ -x "$SIEVER/../cado-filter/cado_predup.sh" ]; then
      export CADO_PREDEDUP_DIR="${CADO_PREDEDUP_DIR:-$WORK/predup}"
      _pd_at=$(( SPEC_TARGET - ${PREDEDUP_LEAD:-6000000} ))
      if [ "$_pd_at" -gt 1000000 ]; then
        (
          # !! THE OLD `_sz / 132` CONVERSION WAS WRONG AND COST HALF THE DESIGNED LEAD. !!
          # It divided the file size by a hardcoded 132 bytes/relation, called "a deliberate
          # OVER-estimate (measured ~129), so the trigger can only ever fire late, never early".
          # It fires late, but nobody measured by how much. Measured live 2026-08-23 on a fresh
          # key: relations are 126.4 B, not 132, so the trigger fired at 61,926,525 relations
          # instead of the designed 59,075,000 -- it ate 3,186,905 of the 6,000,000-relation lead
          # (53%), leaving ~10 min instead of the ~25 min the design is sized for, against a
          # warm-up that takes 457s under a live sieve (not the 170s measured standalone).
          # WORSE, the stand-down guard below used the SAME constant, so it could only trip at
          # 68.2M relations -- 3.1M PAST the breakpoint it exists to protect. The one safeguard
          # against the warm-up running head-to-head with the probe was unreachable by
          # construction. Count lines instead: one pass over a RAM-backed memfd is ~0.8s of one
          # core out of the ~21 the GPU-bound sieve leaves idle, polled at 60s -- the same order
          # as the `wc -l` the SHORT walk-up loop below already does every 20s.
          # TWO-PHASE, so being exact costs almost nothing. `stat` is free but needs a
          # bytes/relation constant; `wc -l` is exact but re-reads a multi-GB memfd. So: coast on
          # a DELIBERATELY LOW divisor (120 vs the measured 126.4) which can only make us start
          # counting EARLY, then count lines on the final approach. That is ~4-6 `wc -l` calls for
          # the whole warm-up instead of ~150, and the trigger lands on the exact relation.
          # BOTH LOOPS MUST BE ABLE TO GIVE UP. If the sieve stops early (the probe returned
          # MATRIX and the sieve was killed) or stalls, the trigger relation count is never
          # reached and an unbounded `while :;` would spin for the rest of the run -- cheap for
          # `stat`, NOT cheap for `wc -l` on an 8 GB memfd. Stand down when the file stops
          # growing, and re-check the stand-down condition on every pass rather than only once
          # after the wait.
          _pd_stall=0; _pd_last=0; _pd_give_up=0
          while :; do
            sleep 60
            _sz=$(stat -Lc %s "$WORK/rels.txt" 2>/dev/null || echo 0)
            if [ "$_sz" -le "$_pd_last" ]; then
              _pd_stall=$(( _pd_stall + 1 ))
              [ "$_pd_stall" -ge "${PREDEDUP_STALL_POLLS:-10}" ] && { _pd_give_up=1; break; }
            else
              _pd_stall=0; _pd_last="$_sz"
            fi
            [ $(( _sz / 120 )) -ge "$_pd_at" ] && break
          done
          if [ "$_pd_give_up" = 0 ]; then
            _pd_tries=0
            while :; do
              _nr=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
              [ "$_nr" -ge "$_pd_at" ] && break
              [ "$_nr" -ge "$SPEC_TARGET" ] && { _pd_give_up=1; break; }
              _pd_tries=$(( _pd_tries + 1 ))
              [ "$_pd_tries" -ge "${PREDEDUP_COUNT_POLLS:-60}" ] && { _pd_give_up=1; break; }
              sleep 20
            done
          fi
          if [ "$_pd_give_up" = 1 ]; then
            echo "  [predup] stood down -- the sieve stopped or stalled before the warm-up point" >&2
            exit 0
          fi
          # If the sieve is already at the breakpoint the warm-up has missed its
          # window: it would then run head-to-head with the probe it exists to
          # speed up, which is a straight loss. Stand down instead.
          if [ "$_nr" -ge "$SPEC_TARGET" ]; then
            echo "  [predup] stood down -- sieve already at the breakpoint" >&2
            exit 0
          fi
          NT="${PREDEDUP_NT:-12}" LPB="${LPB:-29}" \
            nice -n 10 bash "$SIEVER/../cado-filter/cado_predup.sh" \
                 "$WORK/rels.txt" "$WORK/poly.cado" "$CADO_PREDEDUP_DIR" "$_pd_at" & _pdchild=$!
          # Forward a kill from the EXIT trap so the warm-up's 8 GB prefix memfd is
          # released with the run, rather than leaking into a reparented orphan.
          trap 'kill "$_pdchild" 2>/dev/null || true' TERM INT
          wait "$_pdchild"
        ) & PREDUPPID=$!
        echo "  [predup] warm-up armed: pre-deduplicate the first $_pd_at relations (nt=${PREDEDUP_NT:-12})"
      fi
    fi
    for attempt in 1 2 3; do
      env ${SIEVE_SECS:+GPULOOP_MAX_SECS=$SIEVE_SECS} $GPU_PIN ./gpu_loop "$WORK/poly.cado" $QMIN $QMAX "$WORK/rels.txt" $GPU_SPEC_TARGET && rc=0 || rc=$?
      nrel=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
      [ "$rc" = 0 ] && [ "$nrel" -gt 1000 ] && break
      echo "  sieve attempt $attempt: rc=$rc nrel=$nrel — retrying after GPU re-probe"
      python3 -c "import cupy; cupy.zeros(1); cupy.cuda.runtime.deviceSynchronize()" >/dev/null 2>&1 || sleep 5
    done
    NOW=$(date +%s)
    if [ "$UNBOUNDED" = 1 ]; then LEFT=""; else LEFT=$(( GNFS_WALL - (NOW - T_START) - DOWNSTREAM_RESERVE )); fi
    nrel=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
    if [ "$nrel" -ge "$SPEC_TARGET" ] && { [ -z "$LEFT" ] || [ "$LEFT" -gt 600 ]; }; then
      SPEC_DIR="$WORK/spec"; mkdir -p "$SPEC_DIR"; SNAP="$WORK/snap.txt"
      # snapshot into a SECOND memfd (RAM): the validator's /tmp is 1 GiB and this is ~8 GB.
      spec_launch "$nrel" || true
    fi
  fi
  # Phase B: carry on to REL_TARGET, but stop the moment the speculative probe reports MATRIX.
  nrel=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
  if [ "$nrel" -lt "$REL_TARGET" ]; then
    NOW=$(date +%s)
    if [ "$UNBOUNDED" = 1 ]; then LEFT=""; else LEFT=$(( GNFS_WALL - (NOW - T_START) - DOWNSTREAM_RESERVE )); fi
    # !! REMAINING, not cumulative !!  gpu_loop.cu declares `long long tot_rel=0` PER INVOCATION and
    # stops on `tot_rel>=TARGET`, so a GPULOOP_APPEND=1 resume handed the cumulative REL_TARGET is
    # being asked for that many ADDITIONAL relations. At 66.4M already banked, passing 68,500,000
    # asked for another 68.5M -- ~2.1M lattices, ~3.5h -- instead of the ~2.1M it actually needs.
    # This is unreachable, and with no time budget nothing bounds it: on a SHORT probe verdict the
    # polling loop below only breaks on m.dat.cyc appearing, so the sieve would simply run on.
    # Fires on any key whose crossover sits above the 97% probe point (measured: 1 of 4 keys).
    NEED_MORE=$(( REL_TARGET - nrel ))
    NEXTQ=$(cat "$WORK/rels.txt.nextq" 2>/dev/null || echo "$QMIN")
    for attempt in 1 2 3; do
      if [ -n "$SPECPID" ]; then
        # OMP capped while the probe runs (see the thread-split note above); gpu_loop's CPU stage
        # is hidden under the GPU stage, so this costs ~0.2% and frees ~20 cores for the filter.
        env OMP_NUM_THREADS="${SPEC_SIEVE_OMP:-4}" GPULOOP_APPEND=1 ${LEFT:+GPULOOP_MAX_SECS=$LEFT} \
            $GPU_PIN ./gpu_loop "$WORK/poly.cado" "$NEXTQ" $QMAX "$WORK/rels.txt" $NEED_MORE & SVPID=$!
        while kill -0 "$SVPID" 2>/dev/null; do
          if [ -f "$SPEC_DIR/.finished" ]; then
            if [ -s "$SPEC_DIR/m.dat.cyc" ]; then
              echo "  [spec] >>> MATRIX at $SPEC_AT relations — stopping the sieve EARLY $(date +%T)"
              kill "$SVPID" 2>/dev/null || true; wait "$SVPID" 2>/dev/null || true
              SPEC_CYC="$SPEC_DIR/m.dat.cyc"; DS_RELS=$(ds_rels_for "$SPEC_DIR" "$SNAP"); break
            fi
            # SHORT -- and the sieve is STILL RUNNING. Do not wait for REL_TARGET before filtering
            # again (that was the old behaviour and cost ~680s on key07); re-probe right here at the
            # relation count that has accumulated while this probe ran (~717k at 3.9k rel/s x 184s).
            wait "$SPECPID" 2>/dev/null || true; SPECPID=""
            _now=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
            # HISTORY (2026-08-11), because the PRINCIPLE outlives the mechanism: this branch once
            # fell through to "no further probes" whenever a probe came back too fast, silently
            # converting that into "sieve all the way to REL_TARGET" -- the ~868s-late failure 16a
            # warns about, reached with probes still in the budget. The rule that fixed it still
            # holds: when a verdict lands early, let the sieve CATCH UP, never abandon the walk-up.
            # Each poll costs one ~8 GB `wc -l`, which is why the wait polls at 20s, not the 5s of
            # the outer loop.
            # ---- NEXT BREAKPOINT, not "SPEC_MIN_GAIN more relations" ----------------------
            # The old gate re-probed on a fixed DELTA from wherever the previous probe happened
            # to fire, so every probe point after the first drifted with probe duration and none
            # of them was a count anyone had measured. With an absolute ladder, probe k lands on
            # breakpoint k however long probe k-1 took. SPEC_MIN_GAIN survives as a pure FLOOR
            # against an anomalously fast verdict: a real purge is 140-260s, so a 6-second one
            # means a broken snapshot (17a), and honouring a breakpoint that close would burn a
            # probe on the same relations twice.
            # The snapshot is still taken at $_now, not at the breakpoint -- the sieve never stops,
            # so by the time a verdict lands we are normally past it. The breakpoint is the
            # TRIGGER; whatever has accumulated by then is what gets filtered.
            _nextb=$(_break_after "$SPEC_AT")
            if [ -n "$_nextb" ] && [ $(( _nextb - SPEC_AT )) -lt "$SPEC_MIN_GAIN" ]; then
              _nextb=$(_break_after $(( SPEC_AT + SPEC_MIN_GAIN )))
            fi
            if [ "$SPEC_N" -lt "$SPEC_MAX_PROBES" ] && [ -n "$_nextb" ] \
               && [ "$_nextb" -le "$REL_TARGET" ] && [ "$_now" -lt "$_nextb" ]; then
              echo "  [spec] probe #$SPEC_N SHORT at $SPEC_AT — sieving on to breakpoint $_nextb (now $_now) $(date +%T)"
              _tries=0
              while [ "$_now" -lt "$_nextb" ] && [ "$_tries" -lt "${CKPT_BREAK_WAIT_POLLS:-30}" ]; do
                kill -0 "$SVPID" 2>/dev/null || break
                sleep 20; _tries=$(( _tries + 1 ))
                _now=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
              done
            fi
            if [ "$SPEC_N" -lt "$SPEC_MAX_PROBES" ] && [ -n "$_nextb" ] \
               && [ "$_now" -ge "$_nextb" ] && [ "$_now" -lt "$REL_TARGET" ]; then
              echo "  [spec] probe #$SPEC_N SHORT at $SPEC_AT — breakpoint $_nextb reached, re-probing at $_now $(date +%T)"
              SPEC_N=$(( SPEC_N + 1 ))
              # A FAILED LAUNCH IS NOT A VERDICT. spec_launch now returns non-zero only for a
              # BROKEN SNAPSHOT, and it clears $SPEC_DIR/.finished on entry -- so a bare
              # `|| true` here would leave the polling loop with no probe running AND no
              # .finished to react to, i.e. silently sieving to REL_TARGET with probes still in
              # the budget. That is the same outcome as the bug this whole block fixes, just
              # quieter. Retry once, then fail loudly and take the REL_TARGET path deliberately.
              if ! spec_launch "$_now"; then
                sleep 10
                _now=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
                if ! spec_launch "$_now"; then
                  echo "  [spec] snapshot unusable on two attempts — sieving to REL_TARGET $(date +%T)" >&2
                  break
                fi
              fi
            else
              echo "  [spec] probe #$SPEC_N SHORT at $SPEC_AT — breakpoints exhausted (ladder:$CKPT_BREAKS, budget $SPEC_MAX_PROBES); sieving to REL_TARGET $REL_TARGET $(date +%T)"
              { [ -n "${SNAPPID:-}" ] && { kill "$SNAPPID" 2>/dev/null || true; SNAPPID=""; }; } || true  # free the 8 GB memfd
              break
            fi
          fi
          sleep 5
        done
        wait "$SVPID" 2>/dev/null || true; rc=0
      else
        env GPULOOP_APPEND=1 ${LEFT:+GPULOOP_MAX_SECS=$LEFT} $GPU_PIN ./gpu_loop "$WORK/poly.cado" "$NEXTQ" $QMAX "$WORK/rels.txt" $NEED_MORE && rc=0 || rc=$?
      fi
      nrel=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
      [ -n "$SPEC_CYC" ] && break
      [ "$rc" = 0 ] && [ "$nrel" -gt 1000 ] && break
      echo "  sieve attempt $attempt: rc=$rc nrel=$nrel — retrying after GPU re-probe"
      python3 -c "import cupy; cupy.zeros(1); cupy.cuda.runtime.deviceSynchronize()" >/dev/null 2>&1 || sleep 5
      NEXTQ=$(cat "$WORK/rels.txt.nextq" 2>/dev/null || echo "$NEXTQ")
    done
  fi
  # Collect the speculative probe if it is still running / already finished without an early stop.
  if [ -n "$SPECPID" ] && [ -z "$SPEC_CYC" ]; then
    echo "  [spec] sieve reached $REL_TARGET; waiting on the probe verdict $(date +%T)"
    wait "$SPECPID" 2>/dev/null || true
    if [ -s "$SPEC_DIR/m.dat.cyc" ]; then
      echo "  [spec] >>> MATRIX on the snapshot — using it, final filter SKIPPED $(date +%T)"
      SPEC_CYC="$SPEC_DIR/m.dat.cyc"; DS_RELS=$(ds_rels_for "$SPEC_DIR" "$SNAP")
    else
      echo "  [spec] snapshot was SHORT — falling through to the normal ladder (cost ~0 wall time)"
      [ -n "$SNAPPID" ] && kill "$SNAPPID" 2>/dev/null || true   # release the 8 GB snapshot memfd
      SNAPPID=""
    fi
  fi
  # ---- ADAPTIVE FILTER CHECKPOINTS ------------------------------------------------------------
  # REL_TARGET used to carry a fixed safety margin over the measured crossover (77M vs 74M) because
  # a short relation set costs a whole re-sieve. That margin is ~3M relations of pure over-sieving
  # (~20 min at the end-of-sweep yield). Instead: sieve to the crossover, ASK the filter, and only
  # buy more relations if it says it wants them (msieve prints "filtering wants NNN more relations").
  # The winning checkpoint's -nc1 output (m.dat.cyc) is handed to the downstream via SKIP_NC1=1, so
  # a successful first probe costs nothing extra -- the filter runs exactly once, as before.
  # Set REL_CKPT=0 to disable and keep the old fixed-target behaviour.
  # The speculative probe already produced a usable matrix -> the ladder has nothing to do and the
  # final filter is skipped entirely. This is where the ~364s of filter time is actually saved.
  if [ -n "$SPEC_CYC" ]; then
    echo "  [ckpt] SKIPPED — speculative probe already yielded a matrix at $(wc -l < "$DS_RELS") relations"
    CKPT_CYC="$SPEC_CYC"
  fi
  if [ "${REL_CKPT:-1}" = 1 ] && [ -z "$SPEC_CYC" ]; then
    CK_DIR="$WORK/ck"; mkdir -p "$CK_DIR"
    for step in $(seq 1 "${REL_CKPT_MAX:-4}"); do
      NOW=$(date +%s)
    if [ "$UNBOUNDED" = 1 ]; then LEFT=""; else LEFT=$(( GNFS_WALL - (NOW - T_START) - DOWNSTREAM_RESERVE )); fi
      nrel=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
      echo "  [ckpt $step] testing matrix at $nrel relations ($(date +%T), ${LEFT}s sieve budget left)"
      run_probe "$WORK/rels.txt" "$WORK/c.fb" "$N" "$CK_DIR" >"$CK_DIR/filter.$step.out" 2>&1 && _prc=0 || _prc=$?
      # Distinguish "the filter ran and said SHORT" from "the filter never ran at all".
      # Only the first is answerable by buying more relations; the second is a broken input
      # path (see the memfd/readlink note in run_filter.sh) and more relations cannot fix it.
      # Without this, a filter that cannot open its .dat looks exactly like a short relation
      # set, and the ladder burns the entire remaining sieve budget re-probing a dead path.
      #
      # !! BACKEND-AGNOSTIC !!  This health check used to grep msieve.log unconditionally.
      # CADO never writes msieve.log, so a PERFECTLY GOOD CADO matrix was read as "filter did
      # not run" and the ladder aborted -- measured 2026-08-05 in the first full run, after
      # CADO had already produced m.dat.cyc. Order matters: a cycle file is proof of success
      # for EITHER backend, and rc=2 is CADO reporting a genuine shortfall (which the ladder
      # SHOULD answer by buying more relations, not by aborting).
      _filter_dead=0
      if [ -s "$CK_DIR/m.dat.cyc" ]; then
        _filter_dead=0                        # success, whichever backend produced it
      elif [ "$_prc" = 2 ]; then
        _filter_dead=0                        # CADO ran and said SHORT -- a real verdict
      elif ! grep -aq 'commencing relation filtering' "$CK_DIR/msieve.log" 2>/dev/null \
         || grep -aq 'error: cannot open' "$CK_DIR/msieve.log" 2>/dev/null; then
        _filter_dead=1
      fi
      if [ "$_filter_dead" = 1 ]; then
        echo "  [ckpt $step] FILTER DID NOT RUN -- aborting the ladder (this is NOT a relation shortfall)" >&2
        echo "  [ckpt $step] first lines of $CK_DIR/filter.$step.out:" >&2
        sed -n '1,12p' "$CK_DIR/filter.$step.out" >&2 2>/dev/null || true
        break
      fi
      WANT=$(grep -aoE 'filtering wants [0-9]+ more relations' "$CK_DIR/msieve.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || true)
      # Diagnose WHY, not just whether. msieve logs the 2-core left after recursive singleton
      # peeling as "reduce to R relations and I ideals in P passes"; E_core = R - I is the only
      # quantity that decides whether more filtering effort could ever help:
      #   E_core < 0  -> excess-limited. No purge/merge strategy and no LA algorithm can invent
      #                  the missing dependency; only more (or better) relations will do.
      #   E_core > 0 but no matrix -> merge-limited. The relations suffice and the purge/merge
      #                  parameters (target_density, max_weight) are what failed -- retrying the
      #                  filter is far cheaper than buying millions more relations.
      RED=$(grep -aoE 'reduce to [0-9]+ relations and [0-9]+ ideals' "$CK_DIR/msieve.log" 2>/dev/null | tail -1)
      RC=$(echo "$RED" | grep -oE '[0-9]+' | head -1); IC=$(echo "$RED" | grep -oE '[0-9]+' | tail -1)
      # msieve's own bar for "enough excess to build cycles". It is NOT zero and it GROWS with the
      # relation count (178,611 @62M -> 206,242 @72M -> 215,709 @75.6M on c151), so E_core>0 is not
      # sufficient -- see DEFICIT below.
      # !! `|| true` IS LOAD-BEARING -- do not remove (measured 2026-07-28) !!
      # This pipeline ENDS in grep, which exits 1 when the log has no "target excess" line
      # (e.g. the filter never ran). Its neighbours on the lines above end in tail/head and
      # always succeed, so this was the only exposed one. Under `set -e` (line 12) the bare
      # form did not just skip the diagnosis -- it KILLED factor_msv.sh outright, mid-ladder.
      # Because relhold.py is orphaned holding breaking_rsa.py's stdout pipe, the solver then
      # blocked forever on `for line in proc.stdout` and emitted NO result payload at all:
      # the validator would see "no solution output found" after the full wall clock.
      TGTX=$(grep -aoE 'target excess is [0-9]+' "$CK_DIR/msieve.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || true)
      ECORE=""; DEFICIT=""
      if [ -n "${RC:-}" ] && [ -n "${IC:-}" ]; then
        ECORE=$(( RC - IC ))
        # 2026-07-26 CORRECTION. The old test was `E_core > 0 -> merge-limited, tune the filter`.
        # That is too lenient and actively misleads: E_core turns positive at ~71.6M on c151, so a
        # 72M set reports E_core=+85,114 and trips the "merge-limited" branch while still being
        # 121,128 SHORT of target_excess. Acting on it means tuning a filter that cannot succeed --
        # measured: max_weight {150,120,100,80} x target_density {120,170} at 72M gave SEVEN
        # byte-identical results (E_core/target_excess/deficit unchanged to the digit).
        # Two mechanisms make that a structural dead end, not just an empirical miss:
        #   * max_weight is a STARTING FLOOR ("ideals of max weight >= X"), not a cap, and the
        #     heaviest ideal here has weight 121 -- so the default <=200 cut discards NOTHING and
        #     there is no ideal for a lower setting to remove (same inert class as filter_maxrels).
        #   * msieve prints "filtering wants N more relations" and ABORTS AT THE EXCESS CHECK, before
        #     the merge stage -- which is the only place target_density applies. At a SHORT verdict
        #     that knob is unreachable by construction.
        # So the real test is E_core >= target_excess. Below it, only relations help.
        if [ -n "${TGTX:-}" ]; then
          DEFICIT=$(( TGTX - ECORE ))
          if [ "$DEFICIT" -gt 0 ]; then
            DIAG="excess-limited (E_core=$ECORE < target_excess=$TGTX, short by $DEFICIT) -> needs relations"
          else
            DIAG="E_core=$ECORE >= target_excess=$TGTX -> excess sufficient; a failure here IS merge-limited"
          fi
        else
          # target_excess unparseable: fall back to the sign test, but say so rather than pretend.
          if [ "$ECORE" -le 0 ]; then DIAG="excess-limited (E_core=$ECORE; target_excess unparsed)"
          else DIAG="E_core=$ECORE > 0 but target_excess unparsed -- cannot classify, treating as short"; fi
        fi
        echo "  [ckpt $step] 2-core: $RC rels / $IC ideals; $DIAG"
      fi
      if [ -z "$WANT" ] && [ -s "$CK_DIR/m.dat.cyc" ]; then
        echo "  [ckpt $step] MATRIX OK at $nrel relations — skipping the fixed over-sieve margin"
        cp -f "$CK_DIR/m.dat.cyc" "$WORK/rels.txt.cyc" 2>/dev/null || true
        CKPT_CYC="$CK_DIR/m.dat.cyc"; DS_RELS=$(ds_rels_for "$CK_DIR" "$DS_RELS"); break
      fi
      # Short. How many more relations to buy?
      #
      # msieve's "wants NNN more" is USELESS as a quantity: it is a CLAMPED placeholder when far
      # short (verified: exactly 1,000,000 at a 1.30M-relation set, and again exactly 1,000,000 at
      # EVERY cap from 62M to 72M on c151 -- ten measurements, one constant). It says "short", not
      # "how short". The old code multiplied that constant by 1.25 and floored it at 3% of current,
      # i.e. it stepped blind: from 62M it would probe 63.9, 65.8, 67.8, 69.9M and hit
      # REL_CKPT_MAX=4 having never reached the 72.5M crossover.
      #
      # DEFICIT (= target_excess - E_core) is the real distance, and it closes at a MEASURED rate
      # (ECORE_PER_MREL, ~210k per 1M raw on c151). So invert it: relations needed ~= deficit/rate.
      # 20% is added so a probe lands just ABOVE the crossover rather than exactly on it. It is 20%
      # and not less because the E_core slope ACCELERATES with relation count (measured per 2M raw:
      # +294k at 62M, +334k, +369k, +400k, +426k at 72M). ECORE_PER_MREL is calibrated NEAR the
      # crossover, so it is optimistic when applied from far below -- at 1.15x a 62M start landed at
      # 72.50M, i.e. 0.03M SHORT of the 72.53M crossover, buying a whole extra ~6.5min -nc1 probe.
      # Checked against the c151 data (crossover 72.53M) with the 1.20x in place:
      #   from 72M: deficit   121,128 -> +0.69M -> probe 72.69M  (clears)
      #   from 70M: deficit   541,991 -> +3.10M -> probe 73.10M  (clears)
      #   from 62M: deficit 1,916,683 -> +10.9M -> probe 72.95M  (clears, in ONE step)
      #   at 75.57M: deficit -689,012 -> SUFFICIENT, no further sieving
      # CAVEAT: run_filter.sh symlinks m.dat at rels.txt, so each -nc1 probe appends msieve's
      # ~121,612 free relations INTO rels.txt (that mutation is load-bearing -- SKIP_NC1's cycle
      # indices depend on it). So `nrel` on probe 2+ counts those as if sieved, and this step buys
      # ~121k fewer real relations than nominal. 0.16% per probe -- inside the 1.15x margin, but do
      # not remove that margin without accounting for it.
      if [ -n "${DEFICIT:-}" ] && [ "$DEFICIT" -gt 0 ]; then
        NEED=$(( DEFICIT * 1200000 / (ECORE_PER_MREL * 1000) ))   # (deficit/rate) * 1.20, in relations
        NEED=$(( NEED * 1000 ))
        EST="deficit=$DEFICIT -> +$NEED (measured $ECORE_PER_MREL E_core per 1M rel)"
      else
        # No usable E_core/target_excess reading (parse failure). Fall back to the old blind step.
        NEED=$(( ${WANT:-2000000} * 5 / 4 ))
        FLOOR=$(( nrel * 3 / 100 )); [ "$NEED" -lt "$FLOOR" ] && NEED=$FLOOR
        EST="no deficit reading -> blind step +$NEED"
      fi
      [ "$NEED" -lt 500000 ] && NEED=500000
      # Cap one step at 25% of current so a bad parse cannot trigger a runaway over-sieve (>4.5x
      # over-sieving can itself yield an UNSOLVABLE matrix -- see the §4 note above).
      CAP=$(( nrel / 4 )); [ "$NEED" -gt "$CAP" ] && { NEED=$CAP; EST="$EST (capped at 25%)"; }
      REL_TARGET=$(( nrel + NEED ))
      # ---- STORAGE CEILING ON THE LADDER (2026-08-23) --------------------------------------
      # The 25% step above is bounded by nothing but the relation count, so from 72M it targets
      # 90M and from there 112M. That is not merely slow -- past a certain size the relation set
      # CANNOT BE FILTERED AT ALL on a 10 GiB tmpfs, because cado_filter.sh requires free >= RELKB
      # and RELKB grows with the set while the tmpfs does not. Measured 2026-08-23 at 126.4 B per
      # relation with frcache occupying 372 MB:
      #     ~77M relations  peak $WORK reaches the 10 GiB budget      -> ENOSPC mid-filter
      #     ~82M relations  RELKB exceeds ALL free space              -> guard can never pass
      # So a ladder step past that converts "SHORT, needs more relations" into "unfilterable, and
      # every remaining probe fails as a TOOLING failure" -- the sieve keeps running and the run
      # ends at the wall with nothing saying why. Clamp to what the CURRENT free space supports,
      # measured from THIS key's bytes/relation rather than a constant, and say so out loud.
      _fkb=$(df -Pk "$WORK" 2>/dev/null | awk 'NR==2{print $4}')
      _szb=$(stat -Lc %s "$WORK/rels.txt" 2>/dev/null || echo 0)
      if [ -n "${_fkb:-}" ] && [ "$_szb" -gt 0 ] && [ "${nrel:-0}" -gt 0 ]; then
        _bpr=$(( _szb / nrel )); [ "$_bpr" -lt 64 ] && _bpr=64      # floor: never divide by ~0
        # 0.90 of free: the filter needs RELKB free AND peaks ~1.06x RELKB while running.
        _maxrel=$(( _fkb / _bpr * 1024 * 90 / 100 ))
        if [ "$REL_TARGET" -gt "$_maxrel" ]; then
          echo "  [ckpt $step] STORAGE CEILING: $REL_TARGET relations cannot be filtered on" \
               "$(df -P "$WORK" 2>/dev/null | awk 'NR==2{print $6}') ($(( _fkb / 1024 )) MB free," \
               "$_bpr B/relation) -- clamping to $_maxrel" >&2
          REL_TARGET="$_maxrel"
        fi
        if [ "$nrel" -ge "$_maxrel" ]; then
          echo "  [ckpt $step] STORAGE CEILING REACHED at $nrel relations -- more relations cannot" \
               "be filtered in $(( _fkb / 1024 )) MB. This is a STORAGE limit, not a relation" \
               "shortfall; further sieving cannot help. Proceeding with what we have." >&2
          break
        fi
      fi
      echo "  [ckpt $step] $EST -> sieving to $REL_TARGET"
      if [ -n "$LEFT" ] && [ "$LEFT" -lt 300 ]; then echo "  [ckpt $step] out of sieve budget — proceeding with what we have"; break; fi
      NEXTQ=$(cat "$WORK/rels.txt.nextq" 2>/dev/null || echo "$QMIN")
      env GPULOOP_APPEND=1 ${LEFT:+GPULOOP_MAX_SECS=$LEFT} $GPU_PIN ./gpu_loop "$WORK/poly.cado" "$NEXTQ" $QMAX "$WORK/rels.txt" $NEED || true
    done
  fi
  # cache the relation set (the expensive ~2.6h artifact) for re-use by downstream experiments
  if [ -n "$CACHE" ]; then cp "$DS_RELS" "$CACHE/rels.txt" 2>/dev/null && echo "  [cache SAVE] rels.txt ($(wc -l < "$CACHE/rels.txt") relations)" || true; fi
  # ---- DEV-ONLY RELATION DUMP (GNFS_SAVE_DIR). INERT unless set; NEVER set on the validator.
  # Same job as GNFS_CACHE's save above, with two differences that matter for a TIMED run:
  #   1. it copies the RAW sieve output ($WORK/rels.txt), not the purge-ordered DS_RELS, so the
  #      dump is a superset that any later probe point can be reproduced from by truncation;
  #   2. it runs in the BACKGROUND at nice 19, overlapping the CPU/GPU-bound downstream instead
  #      of sitting on the critical path -- the blocking `cp` above is ~8 GB of pure I/O and
  #      would be charged straight to the wall clock.
  # Layout is GNFS_CACHE's ($DIR/<md5(N)[0:12]>/) so a later run with GNFS_CACHE=$GNFS_SAVE_DIR
  # picks these stages straight back up and skips polyselect + the ~3.5h sieve.
  # Waited on before the memfd holder is killed (line ~715) -- by then it is long finished.
  SAVEPID=""
  if [ -n "${GNFS_SAVE_DIR:-}" ] && [ -s "$WORK/rels.txt" ]; then
    SAVE="$GNFS_SAVE_DIR/$(printf '%s' "$N" | md5sum | cut -c1-12)"
    if mkdir -p "$SAVE" 2>/dev/null; then
      _srel=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
      cp -f "$WORK/poly.cado" "$WORK/c.fb" "$SAVE/" 2>/dev/null || true
      cp -f "${SPEC_DIR:-/nonexistent}/m.dat.cyc"    "$SAVE/spec.m.dat.cyc"    2>/dev/null || true
      cp -f "${SPEC_DIR:-/nonexistent}/m.dat.purged" "$SAVE/spec.m.dat.purged" 2>/dev/null || true
      printf '{"N":"%s","rels_raw":%s,"spec_pct":%s,"spec_at":%s,"spec_probes":%s,"rel_target":%s,"saved_utc":"%s"}\n' \
        "$N" "$_srel" "${SPEC_PCT:-0}" "${SPEC_AT:-0}" "${SPEC_N:-0}" "$REL_TARGET" "$(date -u +%FT%TZ)" \
        > "$SAVE/meta.json" 2>/dev/null || true
      # cp to a .part then rename, so an interrupted dump can never be mistaken for a complete one.
      ( nice -n 19 cp -f "$WORK/rels.txt" "$SAVE/rels.txt.part" && mv -f "$SAVE/rels.txt.part" "$SAVE/rels.txt" ) \
        >/dev/null 2>&1 & SAVEPID=$!
      echo "  [save] backgrounded dump of $_srel raw relations -> $SAVE (pid $SAVEPID)"
    fi
  fi
fi
echo "  relations: $(wc -l < "$DS_RELS")   (source: $(basename "$DS_RELS"))"

echo "### [3/4] memfd-backed msieve downstream (filter + in-RAM Lanczos + sqrt) $(date +%T)"
cd "$MSV"
# GNFS_CACHE_DS (when caching is on) lets msvrun_gpu.sh reuse a cached matrix and skip the filter
# (-nc1/-nc2) so you can iterate on LA/sqrt without re-filtering. Inert when unset.
# SKIP_NC1: the checkpoint ladder already ran -nc1 and produced the cycle file; reuse it so the
# filter is not run twice (saves ~8 min whenever the first probe succeeds).
if [ -n "${CKPT_CYC:-}" ] && [ -s "$CKPT_CYC" ]; then
  mkdir -p "$WORK/ds"; cp -f "$CKPT_CYC" "$WORK/ds/m.dat.cyc"; export SKIP_NC1=1
  # A CADO verdict also carries the purge index map. msvrun_gpu.sh self-detects it and
  # adds cado_filter=1 to -nc2; WITHOUT it msieve reads the cycle file as relation
  # indices and silently builds garbage. It must sit beside m.dat, which msv_launcher
  # materialises in this directory. DS_RELS was already switched to the purge-ordered
  # relations by ds_rels_for -- the two MUST travel together.
  if [ -s "$(dirname "$CKPT_CYC")/m.dat.purged" ]; then
    cp -f "$(dirname "$CKPT_CYC")/m.dat.purged" "$WORK/ds/m.dat.purged"
    echo "### downstream: CADO-filtered cycles + purge map (-nc2 cado_filter=1)"
  elif [ -s "$(dirname "$CKPT_CYC")/m.dat.rels" ]; then
    # !! CADO CYCLES WITHOUT THE PURGE MAP = A GARBAGE MATRIX, SILENTLY (2026-08-24). !!
    # m.dat.rels only exists when the CADO backend produced this verdict (msieve's run_filter.sh
    # never writes it), so reaching here means the cycles ARE purge-line-numbered but the map that
    # translates them is gone. msvrun_gpu.sh self-detects on the map's presence, so it would omit
    # `cado_filter=1`, msieve would read the cycles as RELATION INDICES, and -nc2 would build a
    # matrix out of the wrong rows -- with NO error at any layer, all the way to a sqrt that finds
    # no factor 40 min later. Pipeline.md §12 class B lists this as the dangerous unguarded case.
    # Drop the checkpoint instead: without SKIP_NC1 the downstream re-runs -nc1 itself (~8 min) and
    # builds a matrix that is at least internally consistent. Costs time; never produces garbage.
    echo "### downstream: WARNING -- CADO cycles present but m.dat.purged is MISSING." >&2
    echo "###   Refusing to reuse them (msieve would misread purge line numbers as relation" >&2
    echo "###   indices and silently build a WRONG matrix). Falling back to a full -nc1." >&2
    rm -f "$WORK/ds/m.dat.cyc"; unset SKIP_NC1
  fi
fi
# DS_RELS is the snapshot when the speculative probe won -- msieve mutated THAT file and
# m.dat.cyc indexes it, so handing the downstream the (larger, still-growing) rels.txt
# would mismatch the cycles. Falls back to rels.txt on every other path.
# !! `|| DS_RC=$?` IS LOAD-BEARING (2026-08-06) -- do not drop the guard !!
# This call was unguarded under `set -e` (line 12), so ANY non-zero exit from msv_launcher or
# msvrun_gpu.sh killed factor_msv.sh RIGHT HERE -- before the [4/4] block below. The solver then
# saw neither a `p=` line nor the `NO FACTOR` line, breaking_rsa.py emitted {"status":"failed"},
# and the validator scored that well-formed payload as IncorrectFailure: a WRONG ANSWER reported
# for what was actually a downstream tooling failure, with nothing in the log naming the stage.
# Always fall through to the factor extraction, which reports what really happened.
DS_RC=0
GNFS_CACHE_DS="$CACHE" SKIP_NC1="${SKIP_NC1:-0}" ./msv_launcher "$DS_RELS" "$WORK/c.fb" "$WORK/ds" ./msvrun_gpu.sh "$N" "$WORK/ds" || DS_RC=$?
[ "$DS_RC" = 0 ] || echo "### downstream exited rc=$DS_RC -- continuing to factor extraction anyway"
# The GNFS_SAVE_DIR dump reads $WORK/rels.txt, which is a symlink into the holder's memfd -- so it
# MUST finish before the holder is killed on the next line or the dump is silently truncated. It
# was started ~1,000s ago against a ~60s copy, so this wait is normally instant; it is here to make
# "dump exists" mean "dump is complete" rather than "the timing happened to work out".
if [ -n "${SAVEPID:-}" ]; then
  _sw=$(date +%s); wait "$SAVEPID" 2>/dev/null || true
  echo "  [save] dump complete (waited $(( $(date +%s) - _sw ))s)"
fi
[ -n "$RELPID" ] && kill "$RELPID" 2>/dev/null || true   # release the rels memfd (skipped on cache hit)
[ -n "${SNAPPID:-}" ] && kill "$SNAPPID" 2>/dev/null || true   # release the speculative snapshot memfd (~8 GB)
# Surface the filter outcome (unique relations vs ideals / excess / shortfall) so a failure is
# diagnosable from the container log instead of silently discarded.
if [ -f "$WORK/ds/msieve.log" ]; then
  grep -aE 'unique relations|keeping .* ideals|begin with|reduce to|filtering wants|matrix is' "$WORK/ds/msieve.log" | tail -6 | sed 's/^/  [filter] /'
fi

echo "### [4/4] FACTORS $(date +%T)"
# `|| true`: this is the LAST command, so its status is the script's. A missing/unreadable
# msieve.log (downstream died early) used to raise here and exit non-zero with no verdict line
# printed at all -- indistinguishable, from the orchestrator's side, from a wrong answer.
python3 - "$N" "$WORK/ds/msieve.log" <<'PY' || true
import sys,os,re
N=int(sys.argv[1])
fs=set()
if not os.path.isfile(sys.argv[2]):
    print("NO FACTOR (downstream produced no msieve.log -- see the stage logs above)"); raise SystemExit(0)
for m in re.findall(r'factor:\s*([0-9]+)', open(sys.argv[2]).read()):
    f=int(m)
    if 1<f<N and N%f==0: fs.add(f)
if fs:
    p=min(fs); q=N//p
    print(f"p={p}")     # line-start p=/q= for the orchestrator to parse
    print(f"q={q}")
    print(f"FACTORED: p*q==N {p*q==N}")
else:
    print("NO FACTOR (insufficient relations / matrix)")
PY
