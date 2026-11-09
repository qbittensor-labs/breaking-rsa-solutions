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
# factor_msv.sh -- GNFS driver: polyselect -> GPU lattice sieve (+ las on the
# idle cores) -> adaptive filter ladder -> matrix -> block Lanczos -> sqrt.
#
# Sized for 512-bit N (154/155 digits) inside a 4 h wall on one GPU and 24
# cores. Every size-dependent constant comes from sieve_params.py.
# ---------------------------------------------------------------------------
GPU_PIN=""
if [ "${GPU_NUMA_PIN:-0}" != "0" ] && [ -r "$SIEVER/gpu_affinity.py" ]; then
  GPU_PIN="python3 $SIEVER/gpu_affinity.py --"      # deliberately UNQUOTED at the call sites
fi
POLYSELECT_SECS="${POLYSELECT_SECS:-540}"      # GPU polyselect budget (~18 min)
ADMAX="${ADMAX:-200000}"
# QMAX is intentionally large: the sieve stops on REL_TARGET or the wall budget, never on q.
#
# REL_TARGET is RAW-denominated and is a CEILING, not the probe point -- the adaptive ladder
# below decides where to stop. It answers only "how far do we sieve if every probe came back
# SHORT", so it can be raised on its own without dragging the probe point with it.
# Do NOT lower it below the crossover to save time: that guarantees an unusable relation set.
#
# All size-dependent sieve constants come from sieve_params.py. Below 152 digits it emits the
# shipped values verbatim, so the live path is untouched by construction. Env still wins.
SV_DIGITS=0 SV_QMIN= SV_QSPLIT= SV_LAS_SPAN= SV_REL_TARGET= SV_CKPT_FIRST_BREAK= SV_ECORE_PER_MREL=
SV_UNIQUE_NEEDED= SV_EXPECT_RAW= SV_SPEC_MAX_PROBES=
if _sp=$(python3 "$SIEVER/sieve_params.py" "$N" 2>/dev/null) && [ -n "$_sp" ]; then
  eval "$_sp"
else
  echo "  [sieve] WARNING: sieve_params.py failed -- using the c151 constants" >&2
  SV_DIGITS=${#N} SV_QMIN=1000000 SV_QSPLIT=3800000 SV_LAS_SPAN=2800000 SV_REL_TARGET=72000000
  SV_CKPT_FIRST_BREAK=65075000 SV_ECORE_PER_MREL=216000 SV_UNIQUE_NEEDED=39900000 SV_EXPECT_RAW=65000000
  SV_SPEC_MAX_PROBES=8
fi
# ---- SIEVER BINARY, SELECTED BY SIZE -------------------------------------------------------
# LIM is a COMPILE-TIME flag, so a different factor-base bound means a different binary.
# MEASURED 2026-09-14 at c155, scored on UNIQUE yield over the q-range the sieve actually runs:
#     LIM  40M  12.59 unique rel/lat  5.25 ms  2.398 unique/ms   <- what ships below 155 digits
#     LIM  60M  14.31                 5.49     2.607  (+8.7%)
#     LIM 100M  16.39                 6.14     2.669  (+11.3%)  <- c155
#     LIM 140M  17.46                 6.85     2.549  (+6.3%)   -- past the peak, so 100M is interior
#
# ⚠️⚠️ DISABLED 2026-09-14 AFTER AN END-TO-END RUN. The table above is REAL but it is NOT the
# whole cost. Measured on key 20260930, same key, only LIM differing:
#   * ms/lat in PRODUCTION rose +25.5% (5.1 -> 6.4), not the +17% the three probe windows read.
#     Those windows sit at the 25/50/75 lattice quantiles of the 40M cost curve and never sample
#     q = 16.5-27 M, which is exactly where a bigger factor base costs most. Delivered gain was
#     +3.8% unique rel/ms, not +11.3%.
#   * THE RELATION REQUIREMENT ROSE. 40M found its matrix at 55,031,237 relations; 100M walked
#     all NINE ladder rungs past 55,757,505 without one. LIM fixes the factor-base column count
#     (2.43M -> 5.76M primes) and those columns must be covered for positive excess. A
#     150k-relation dump CANNOT see this -- it touches almost none of them -- so the
#     ideals-per-relation check that cleared this change (-0.7%) was structurally blind to it.
# Net: WORSE than 40M. FINDINGS 5b's conclusion was right even though its method was not.
# Do not re-enable without an END-TO-END run; no bake-off-scale probe can price LIM.
# FINDINGS.md 5b closed this axis at 40M using RAW rel/lat at q = 4/11/19 M -- windows outside the
# production range, where factor-base work dominates the lattice and the cost of a bigger base is
# overstated (it read +17% ms/lat there; at production q the same step is far smaller).
# ⚠️ c151 and c154 are NOT measured at LIM=100M and keep the 40M binary.
GPULOOP_BIN="${GPULOOP_BIN:-}"
if [ -z "$GPULOOP_BIN" ]; then
  if [ "${SV_DIGITS:-0}" -ge 999 ] && [ -x "$SIEVER/gpu_loop.lim100m" ]; then
    GPULOOP_BIN=./gpu_loop.lim100m
  else
    GPULOOP_BIN=./gpu_loop
  fi
fi
echo "  [sieve] siever binary: $GPULOOP_BIN"

QMIN="${QMIN:-$SV_QMIN}"; QMAX="${QMAX:-220000000}"; REL_TARGET="${REL_TARGET:-$SV_REL_TARGET}"
# One line, up front, naming every size-derived constant. On the validator, container stdout is the
# ONLY diagnostic there is (Pipeline.md §12), and a run that mis-sizes its ladder is otherwise
# indistinguishable from a run that was merely slow.
echo "  [params] digits=$SV_DIGITS QMIN=$QMIN las_split=$SV_QSPLIT REL_TARGET=$REL_TARGET" \
     "first_break=$SV_CKPT_FIRST_BREAK probes=${SV_SPEC_MAX_PROBES:-8} ecore=$SV_ECORE_PER_MREL" \
     "(expect ~${SV_UNIQUE_NEEDED} unique from ~${SV_EXPECT_RAW} raw)"
# REL_TARGET is a ceiling only; the probe schedule is absolute (CKPT_BREAKS below). Keeping them
# separate is what stops "raise the target for safety" from also pushing the first probe later.
ECORE_PER_MREL="${ECORE_PER_MREL:-$SV_ECORE_PER_MREL}"
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
# WALL BUDGET. GNFS_WALL is the whole-run budget; the sieve gets what is left after the
# downstream reserve. Unset WALL_TIME means UNBOUNDED -- the run is relation-bound and reports
# its true time rather than self-terminating, because a validator timeout is preferable to a
# self-inflicted wrong answer. DOWNSTREAM_RESERVE must cover filter + matrix + LA + sqrt; if the
# sieve eats it, the run ends with relations it never gets to use.
GNFS_WALL="${GNFS_WALL:-}"; DOWNSTREAM_RESERVE="${DOWNSTREAM_RESERVE:-2000}"
UNBOUNDED=0; [ -z "$GNFS_WALL" ] && UNBOUNDED=1
# SIEVE_SECS and LEFT stay EMPTY in unbounded mode. Every gpu_loop call site below uses
# `env ${VAR:+GPULOOP_MAX_SECS=$VAR}` so an empty value passes NO budget at all rather than a zero
# one -- which matters, because gpu_loop treats an ABSENT GPULOOP_MAX_SECS as "no limit" but would
# treat an explicit 0 or negative value as "no limit" too (see the guard `if(SIEVE_MAX>0 ...)`).
# Passing nothing is the unambiguous form. All LEFT-based guards are skipped when LEFT is empty.
rm -rf "$WORK"; mkdir -p "$WORK"
# ---- optional STAGE CACHE (dev only; INERT unless GNFS_CACHE is set) ----
# Keyed on N so a re-run reuses relations/matrix and you can iterate on one stage.
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
            "$MSV/cado_sqrt" "$MSV/msieve_dump" "$MSV/cado_sqrt.sh" "$MSV/gpu_la.py" \
            "$SIEVER/polyselect_params.py" "$SIEVER/cado_top_raw.py" "$SIEVER/cado_extract_polys.py" \
            "$SIEVER/sieve_params.py"; do
    [ -r "$_b" ] || echo "  [preflight] WARNING: $_b missing -- that stage will silently degrade" >&2
  done
  # CADO polyselect: absent => polyselect_best.sh warns and falls back to msieve -np. That still
  # produces a polynomial, so the run completes -- ~11.7% slower at c154 (31.30 -> 28.02 rel/lat,
  # ~1,900s of extra sieve). Worth a named line here rather than only inside the stage.
  for _b in polyselect polyselect_ropt; do
    [ -x "$SIEVER/../cado-filter/$_b" ] \
      || echo "  [preflight] WARNING: cado-filter/$_b missing -- polyselect degrades to msieve" >&2
  done
  # CPU siever: absent => GPU-only sieve, ~8.2% slower (~1,350 s at c154). Named here rather than
  # only inside the stage, for the same reason as cado_predup.sh: the degradation is otherwise silent.
  for _b in las makefb; do
    [ -x "$SIEVER/../cado-filter/$_b" ] \
      || echo "  [preflight] WARNING: cado-filter/$_b missing -- concurrent CPU siever disabled" >&2
  done
  [ -r "$SIEVER/las_sieve.sh" ] \
    || echo "  [preflight] WARNING: las_sieve.sh missing -- concurrent CPU siever disabled" >&2
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

echo "### [1/4] polyselect (size-aware: CADO at >=152 digits, msieve below) + yield bake-off $(date +%T)"
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
# ---- CONCURRENT CPU SIEVER (las) ----------------------------------------------------------------
# The sieve is GPU-bound: gpu_loop uses ~1.05 ms of CPU per 6.07 ms lattice, so ~20 of the 24 cores
# idle for the whole run. MEASURED 2026-09-04 at c154, las -t 20 on cores 4-23 CONCURRENT with a
# running gpu_loop: 594 rel/s (183,691 relations, 45.5 rel/sq, 309 s at q=3.0-3.06M) against the
# GPU's ~7,258 rel/s at the same q = +8.2 %, and the GPU pays NOTHING for the cores:
#     cores idle : GPU 6.06 ms  CPU 1.54 ms  wall 6.20 ms
#     cores busy : GPU 6.09 ms  CPU 1.49 ms  wall 6.20 ms      <- wall identical
# The two sievers also agree on rel/special-q to ~1 %, which cross-validates both.
#
# !! las gets its OWN file. gpu_loop opens its dump with buffered stdio (fopen(dumpf,"a"),
# gpu_loop.cu:2712), NOT O_APPEND write()-per-line, so a shared path interleaves partial buffers and
# corrupts relations silently. Do not "simplify" this into one file. !!
LAS_RELS="$WORK/rels.las.txt"
# las writes "<out>.tmp.<pid>" and renames only when it COMPLETES its q-range, so every reader
# that looks at the final name is blind to a running las. Two silent consequences:
#   1. the sieve cannot stop on the true total -- the stop is gated by las FINISHING, not by the
#      relation count, so a crossover below where las happens to finish is unreachable;
#   2. a las KILLED before it finishes loses everything -- ~9 % of the run's relations, dropped
#      with no log line.
# las_live() resolves whichever file las is actually writing. Prefer the final name; else the
# newest tmp. Never fails, because it is used in command substitution under set -e.
las_live() {
  local _f
  if [ -s "$LAS_RELS" ]; then printf '%s\n' "$LAS_RELS"; return 0; fi
  _f=$(ls -1t "$LAS_RELS".tmp.* 2>/dev/null | head -n 1)
  [ -n "$_f" ] && [ -s "$_f" ] && printf '%s\n' "$_f"
  return 0
}
LAS_ENABLE="${LAS_ENABLE:-1}"
# q split: las takes [QMIN, QMIN+LAS_SPAN), gpu_loop starts at the split, so neither re-finds the
# other's relations. Sized so las FINISHES its range: it runs at 13.06 special-q/s (measured), and
# pi(2.8M)-pi(1M) ~ 124k special-q ~ 9,500 s -- comfortably inside a ~13,000 s sieve. Deliberately
# conservative: if las overruns, the unswept tail of its range is a HOLE at the highest-yield end of
# the q-range, and the GPU has to make it up at ~3x the cost. Idling early is the cheap failure.
LAS_SPAN="${LAS_SPAN:-$SV_LAS_SPAN}"
LASPID=""
# Relation count across BOTH sievers. las emits '#' comment lines, so they must be excluded or
# the ladder fires early on a count that includes them.
nrel_all() {
  local _a _b _f
  _a=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
  _a=${_a:-0}
  _b=0
  # las_live(), not LAS_RELS: while las runs, its output is at "<out>.tmp.<pid>" (see above).
  _f=$(las_live)
  if [ -n "$_f" ]; then
    _b=$(grep -cv '^#' "$_f" 2>/dev/null || echo 0)
    # The last line of a LIVE file can be half-written, so never count it. One relation of
    # undercount against a multi-million total is free; a truncated relation in a snapshot is not.
    [ "${_b:-0}" -gt 0 ] && _b=$(( _b - 1 ))
  fi
  echo $(( _a + _b ))
}
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
  trap 'kill "$RELPID" 2>/dev/null || true; [ -n "${SNAPPID:-}" ] && kill "$SNAPPID" 2>/dev/null || true; [ -n "${PREDUPPID:-}" ] && kill "$PREDUPPID" 2>/dev/null || true; [ -n "${LASPID:-}" ] && kill "$LASPID" 2>/dev/null || true' EXIT INT TERM
  for i in $(seq 1 100); do [ -L "$WORK/rels.txt" ] && break; sleep 0.1; done
  cd "$SIEVER"
# Retry a transient GPU-init miss (cudaErrorNoDevice / 0 relations); capped so it cannot eat the
# wall. Belt-and-braces behind breaking_rsa.py's keep-alive.
  DS_RELS="$WORK/rels.txt"; SPEC_CYC=""; SPECPID=""; SNAPPID=""
# An early attempt at co-sieving was net-negative because the GPU's CPU demand is BURSTY, not the
# flat ~3 cores the average suggests. It works now only because gpu_loop is pinned to its own
# cores (see the 20/4 split below); do not re-enable las without that pin.
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
# Delete the old symlink BEFORE starting the new holder. relhold publishes its memfd as a symlink
# to /proc/<pid>/fd/N; killing the holder frees the memfd but leaves the symlink, so a readiness
# test that only asks "is a symlink present" is satisfied instantly by a link to a DEAD pid. The
# probe then runs on a non-existent snapshot and returns SHORT in seconds.
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
    # Both sievers' output, comment-stripped, truncated to the probe point. `cat` of the two is
    # safe here (unlike a shared output file) because this is a single reader building a snapshot,
    # not two concurrent writers.
    local _lf=""
    if ! { cat "$WORK/rels.txt"; _lf=$(las_live); [ -n "$_lf" ] && grep -v '^#' "$_lf" | head -n -1; } 2>/dev/null \
         | head -n "$_n" > "$SNAP" 2>/dev/null; then
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
# THREAD SPLIT. The probe and the running sieve are both CPU-hungry; each defaulting to 24
# threads thrashes 24 cores and starves the sieve. Cheap to fix because the sieve barely needs
# CPU -- its stage hides under the GPU. NT + OMP = 24.
    ( NT="${SPEC_FILTER_NT:-20}" run_probe "$SNAP" "$WORK/c.fb" "$N" "$SPEC_DIR" \
        >"$SPEC_DIR/filter.out.$SPEC_N" 2>&1 || true; echo done > "$SPEC_DIR/.finished" ) & SPECPID=$!
    SPEC_AT="$_n"
    return 0
  }
# THE ADAPTIVE PROBE LADDER.
# The sieve is GPU-bound and the filter is CPU-bound, so probing costs almost nothing: snapshot
# at a breakpoint and filter it while the sieve carries on underneath.
#   MATRIX -> stop sieving now; saves the remaining sieve AND the final filter.
#   SHORT  -> release the snapshot, re-probe at the current count. ~20 s net.
# The snapshot is mandatory: msieve -nc1 mutates its input, and the sieve is concurrently
# appending. The winning snapshot therefore also becomes the relation set handed downstream.
#
# BREAKPOINTS ARE ABSOLUTE, not a percentage of REL_TARGET -- they are relation counts, and
# expressing them as a fraction of a ceiling that also moves ties two unrelated decisions
# together. The probe point is a GRID PHASE, not a margin: the run lands on the first breakpoint
# at or above a KEY-SPECIFIC crossover, so lowering the start can land HIGHER, not lower.
#
# The budget must reach the ceiling. A ladder that exhausts mid-walk falls through to REL_TARGET
# and over-sieves by minutes; an unused probe costs nothing. sieve_params.py sizes it.
  SPEC_PCT="${CKPT_SPEC_PCT:-95}"          # retained ONLY to derive a first break if none is given
  # 2026-09-12: SIZE-AWARE. sieve_params.py emits a budget long enough for the ladder it just
  # generated -- it lowered the first breakpoint to the bottom of the modelled crossover band for
  # d >= 152, and a longer ladder with the old 8-probe budget would exhaust mid-walk and fall
  # through to REL_TARGET, which is the 250-870 s failure 16a describes. c151 still gets exactly 8
  # (emitted verbatim), so the live 500-bit path is unchanged. Env still wins over both.
  SPEC_MAX_PROBES="${SPEC_MAX_PROBES:-${SV_SPEC_MAX_PROBES:-8}}"  # bound the re-probe loop; 0 verdicts is not a failure mode
  SPEC_MIN_GAIN="${SPEC_MIN_GAIN:-300000}" # floor: never re-probe on fewer new relations than this
  # VERIFIED END-TO-END 2026-08-20 on key08, cold, p*q==N in 12,791.5s. The run was forced with
  # CKPT_FIRST_BREAK=62000000 (below key08's crossover) so the walk-up HAD to run: probes at
  # 62,000,029 / 63,021,096 / 64,152,448 all SHORT, MATRIX at 65,334,209 on probe #4, sieve killed
  # early. The gate selected the next un-passed breakpoint each time and correctly SKIPPED 64,250,000
  # once the sieve had already run past it. No waiting occurred -- probes ran back-to-back.
  CKPT_FIRST_BREAK="${CKPT_FIRST_BREAK:-$SV_CKPT_FIRST_BREAK}"
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
# ---- PRE-DEDUP WARM-UP ---------------------------------------------------------------------
# Half of a MATRIX probe is dup1+dup2 re-reading every relation, and the probe is on the critical
# path. Relations are append-only and dup2 takes already-renumbered input far faster, so that
# half is paid EARLY, under the GPU-bound sieve, on otherwise idle cores.
# Every failure mode is graceful: a warm-up that is slow, killed or never finishes simply leaves
# no READY file and cado_filter.sh takes the one-shot path.
    if [ "${CADO_PREDEDUP:-1}" = 1 ] && [ -x "$SIEVER/../cado-filter/cado_predup.sh" ]; then
      export CADO_PREDEDUP_DIR="${CADO_PREDEDUP_DIR:-$WORK/predup}"
      _pd_at=$(( SPEC_TARGET - ${PREDEDUP_LEAD:-6000000} ))
      if [ "$_pd_at" -gt 1000000 ]; then
        (
# TWO-PHASE TRIGGER. Coast on a deliberately LOW bytes/relation divisor -- which can only make us
# start counting EARLY -- then count lines exactly on the final approach. That is a few wc -l
# calls instead of one per poll, and the trigger lands on the exact relation.
# Both loops must be able to give up: if the sieve stops early the trigger count is never reached
# and an unbounded loop would spin for the rest of the run.
          _pd_stall=0; _pd_last=0; _pd_give_up=0
          while :; do
            sleep 60
            _sz=$(stat -Lc %s "$WORK/rels.txt" 2>/dev/null || echo 0)
            # ...PLUS las's live file, for the same reason nrel_all() had to change: rels.txt is
            # gpu_loop's alone, so on a run where las carries ~9 % of the relations this trigger
            # coasts ~2.4 M relations LATE and eats a third of the 6 M lead the warm-up is sized
            # for. Still two stat() calls, i.e. still free -- the whole point of the coast phase.
            _pdlf=$(las_live); _lsz=0
            [ -n "$_pdlf" ] && _lsz=$(stat -Lc %s "$_pdlf" 2>/dev/null || echo 0)
            _sz0="$_sz"                      # gpu_loop's own bytes, kept for the split estimate
            _sz=$(( _sz + _lsz ))            # combined, for the stall detector
            if [ "$_sz" -le "$_pd_last" ]; then
              _pd_stall=$(( _pd_stall + 1 ))
              [ "$_pd_stall" -ge "${PREDEDUP_STALL_POLLS:-10}" ] && { _pd_give_up=1; break; }
            else
              _pd_stall=0; _pd_last="$_sz"
            fi
            # Divisor 120 is a deliberate UNDER-estimate of bytes/relation so the coast can only
            # break EARLY. It was calibrated on gpu_loop's ~132 B; las's are ~162 B (measured
            # 2026-09-12: 995,949,788 B / 6,112,557 relations), so on the combined size 120
            # overestimates the relation count by ~13 % rather than ~10 % and the break lands ~5 M
            # relations early. Split the estimate per file -- still both under, so still early,
            # but by a margin the wait loop below can absorb.
            [ $(( _sz0 / 126 + _lsz / 150 )) -ge "$_pd_at" ] && break
          done
          if [ "$_pd_give_up" = 0 ]; then
            _pd_tries=0
            while :; do
              _nr=$(nrel_all)
              [ "$_nr" -ge "$_pd_at" ] && break
              [ "$_nr" -ge "$SPEC_TARGET" ] && { _pd_give_up=1; break; }
              _pd_tries=$(( _pd_tries + 1 ))
              # !! 60 POLLS (1,200 s) WAS SIZED FOR A COAST PHASE THAT COULD NOT SEE las. !!
              # 2026-09-12: the coast loop above now adds las's bytes, so it breaks ~6 M relations
              # EARLIER than it used to (it used to overshoot _pd_at in TOTAL terms, which was the
              # bug). The wait from there to the exact count is ~5.3 M relations ~= 1,330 s at the
              # production rate -- just past 1,200 s, so the warm-up stood down and the probe paid
              # the one-shot path (~428 s) instead of the warm-started one (~300 s). Measured live
              # on the 2026-09-12 c155b run: "[predup] stood down", ~130 s. There is no runaway
              # risk in raising it: the loop also exits on _nr >= SPEC_TARGET, i.e. the moment the
              # sieve reaches the breakpoint the warm-up exists to precede.
              [ "$_pd_tries" -ge "${PREDEDUP_COUNT_POLLS:-150}" ] && { _pd_give_up=1; break; }
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
    # ---- launch the concurrent CPU siever over the LOW end of the q-range -------------------
    # It must start before gpu_loop so the two ranges are disjoint from the first lattice. Backgrounded
    # and left to the EXIT trap / process-group kill; if las or makefb is missing, las_sieve.sh warns
    # and returns 0, and everything below still works GPU-only (Class B, Pipeline.md §12).
    GPU_QMIN=$QMIN; GPU_TASKSET=""; GPU_LAS_OMP="${GPU_LAS_OMP:-4}"
    if [ "$LAS_ENABLE" = 1 ] && [ -x "$SIEVER/las_sieve.sh" ]; then
      GPU_QMIN=$(( QMIN + LAS_SPAN ))
      bash "$SIEVER/las_sieve.sh" "$WORK/poly.cado" "$QMIN" "$GPU_QMIN" "$LAS_RELS" \
           "$WORK/las.log" >/dev/null 2>&1 & LASPID=$!
      echo "  [las] CPU siever armed on q=[$QMIN,$GPU_QMIN); GPU starts at $GPU_QMIN (pid $LASPID)"
# CONFINE gpu_loop TO ITS OWN CORES WHILE las RUNS -- worth +10.3 % of relation rate.
# gpu_loop needs ~4 cores to stay GPU-bound; below that its CPU stage stops hiding under the GPU.
# Measured: 20/4 is the optimum, 21/3 is +2.0 %, 22/2 is -1.0 %.
# Derived from sched_getaffinity, never hard-coded: the validator gives a CPU quota and a mask
# covering every host CPU, so absolute pinning could collide with another tenant.
      GPU_LAS_OMP="${GPU_LAS_OMP:-4}"
      GPU_TASKSET=""
      if command -v taskset >/dev/null 2>&1; then
        _gcpus=$(python3 -c "import os;c=sorted(os.sched_getaffinity(0))[:${GPU_LAS_OMP}];print(','.join(map(str,c)))" 2>/dev/null || echo "")
        [ -n "$_gcpus" ] && GPU_TASKSET="taskset -c $_gcpus"
        echo "  [las] gpu_loop confined to CPUs ${_gcpus:-<unpinned>} at OMP=$GPU_LAS_OMP; las owns the rest"
      fi
    fi
# Phase A must stop on the TOTAL, not on gpu_loop's own count. SPEC_TARGET is a total derived
# from REL_TARGET and is compared against nrel_all() everywhere else, but gpu_loop counts PER
# INVOCATION -- so handing it that figure overshoots by las's entire contribution.
# gpu_loop's own target stays as a CEILING (correct when las is off, and a backstop if the
# watcher dies); the watcher is what actually stops it once las is running.
    for attempt in 1 2 3; do
      env ${SIEVE_SECS:+GPULOOP_MAX_SECS=$SIEVE_SECS} ${LASPID:+OMP_NUM_THREADS=$GPU_LAS_OMP} \
          $GPU_TASKSET $GPU_PIN "$GPULOOP_BIN" "$WORK/poly.cado" $GPU_QMIN $QMAX "$WORK/rels.txt" $GPU_SPEC_TARGET & GLPID=$!
      _hit_target=0
      if [ -n "${LASPID:-}" ]; then
        while kill -0 "$GLPID" 2>/dev/null; do
          if [ "$(nrel_all)" -ge "$GPU_SPEC_TARGET" ]; then
            echo "  [spec] TOTAL (gpu+las) reached $GPU_SPEC_TARGET -- stopping the GPU sieve $(date +%T)"
            _hit_target=1; kill "$GLPID" 2>/dev/null || true; break
          fi
          sleep 10
        done
      fi
      # !! `wait "$GLPID"; rc=$?` IS WRONG HERE -- `set -e` (line 12) EXITS THE WHOLE RUN when
      # wait reports the SIGTERM (143) the watcher above just sent. The `&& ||` form is the only
      # set -e-safe way to capture a non-zero status, which is why the original gpu_loop call used
      # it. Verified: `set -e; sleep 30 & P=$!; kill $P; wait $P; rc=$?` never reaches the echo.
      wait "$GLPID" && rc=0 || rc=$?
      # A deliberate stop is SUCCESS, not a crash: without this the guard below sees rc!=0 and
      # burns the two remaining retries re-sieving from scratch.
      [ "$_hit_target" = 1 ] && rc=0
      # gpu_loop's OWN output, not nrel_all: this guard asks "did this binary produce relations?",
      # and las's file would mask a gpu_loop that emitted nothing at all.
      nrel=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
      [ "$rc" = 0 ] && [ "$nrel" -gt 1000 ] && break
      echo "  sieve attempt $attempt: rc=$rc nrel=$nrel — retrying after GPU re-probe"
      python3 -c "import cupy; cupy.zeros(1); cupy.cuda.runtime.deviceSynchronize()" >/dev/null 2>&1 || sleep 5
    done
    NOW=$(date +%s)
    if [ "$UNBOUNDED" = 1 ]; then LEFT=""; else LEFT=$(( GNFS_WALL - (NOW - T_START) - DOWNSTREAM_RESERVE )); fi
    nrel=$(nrel_all)
    if [ "$nrel" -ge "$SPEC_TARGET" ] && { [ -z "$LEFT" ] || [ "$LEFT" -gt 600 ]; }; then
      SPEC_DIR="$WORK/spec"; mkdir -p "$SPEC_DIR"; SNAP="$WORK/snap.txt"
      # snapshot into a SECOND memfd (RAM): the validator's /tmp is 1 GiB and this is ~8 GB.
      spec_launch "$nrel" || true
    fi
  fi
  # ---- fold the CPU siever's relations into the downstream stream --------------------------
  # DS_RELS defaults to $WORK/rels.txt, which is gpu_loop's file ONLY. Every path that does not go
  # through the speculative snapshot (which already concatenates, see spec_launch) would therefore
  # hand the filter the GPU's relations and silently discard las's -- a ~9 % relation loss with
  # nothing in any log to say so, i.e. exactly the Class B failure mode Pipeline.md §12 warns about.
  # Appending here is safe: las has been reaped by then, or is still writing to its OWN file and the
  # only reader is this one.
  las_merge() {
    # las_live(), not LAS_RELS: las has just been KILLED, so unless it happened to finish its range
    # first there is no final name at all and the old test dropped every CPU-sieved relation.
    local _f; _f=$(las_live); [ -n "$_f" ] || return 0
    local _n; _n=$(grep -cv '^#' "$_f" 2>/dev/null || echo 0)
    [ "$_n" -gt 0 ] || return 0
    if grep -v '^#' "$_f" | head -n -1 >> "$WORK/rels.txt" 2>/dev/null; then
      echo "  [las] merged $(( _n - 1 )) CPU-sieved relations into the downstream set (from $(basename "$_f"))"
      # RENAME OUT OF las_live()'s GLOB, do not truncate. `: > "$_f"` was safe only because the old
      # code could never reach a file las was still writing; if a straggler survives the kill above
      # it holds an open fd at its own offset, and truncating under it yields a sparse file full of
      # NULs that the next nrel_all() would happily count. The new name does not match
      # "$LAS_RELS".tmp.* , so las_live() stops returning it and nothing double-counts.
      mv -f "$_f" "$WORK/rels.las.merged" 2>/dev/null || : > "$_f"
    else
      echo "  [las] WARNING: could not merge $_n relations -- proceeding GPU-only" >&2
    fi
  }

  # Phase B: carry on to REL_TARGET, but stop the moment the speculative probe reports MATRIX.
  nrel=$(nrel_all)
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
            $GPU_PIN "$GPULOOP_BIN" "$WORK/poly.cado" "$NEXTQ" $QMAX "$WORK/rels.txt" $NEED_MORE & SVPID=$!
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
            _now=$(nrel_all)
# A tooling failure must never be laundered into a SHORT verdict: the caller cannot tell a real
# shortfall from a broken input, so it draws the wrong conclusion and keeps going.
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
                _now=$(nrel_all)
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
                _now=$(nrel_all)
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
        env GPULOOP_APPEND=1 ${LEFT:+GPULOOP_MAX_SECS=$LEFT} $GPU_PIN "$GPULOOP_BIN" "$WORK/poly.cado" "$NEXTQ" $QMAX "$WORK/rels.txt" $NEED_MORE && rc=0 || rc=$?
      fi
      nrel=$(nrel_all)
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
  # ---- the CPU siever is done contributing: reap it and fold its relations in -----------------
  # Everything below hands $WORK/rels.txt (or a truncation of it) to a filter, and rels.txt is
  # gpu_loop's file alone. Merge here, once, at the single boundary between "still sieving" and
  # "filtering": before this line nrel_all() counts both files; after it, las's file is empty and
  # rels.txt holds everything, so both the ladder and the final downstream see the full set.
  if [ -n "${LASPID:-}" ]; then
    # LASPID is las_sieve.sh, the WRAPPER -- `las` itself is its child (verified live 2026-09-12:
    # pid 342 = bash las_sieve.sh, pid 2737 = las). bash does not forward SIGTERM to children, so
    # killing the wrapper alone leaves `las` running and still WRITING to the file las_merge is
    # about to consume. That was harmless while las_merge could not see the live file at all; now
    # that it can, a surviving writer is a correctness problem, so reap the child too.
    pkill -TERM -P "$LASPID" 2>/dev/null || true
    kill "$LASPID" 2>/dev/null || true
    wait "$LASPID" 2>/dev/null || true
    LASPID=""
  fi
  las_merge

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
      nrel=$(nrel_all)
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
# E_core above target_excess does NOT mean the filter can build a matrix -- msieve can hold a
# genuine excess surplus and still fail to find enough cycles. Test the cycle count, not E_core.
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
# SHORT. Buy more relations, then re-probe.
# NEED_MORE is REMAINING, not cumulative: gpu_loop counts per invocation and stops on its own
# target, so handing a resume the cumulative figure asks for that many ADDITIONAL relations.
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
      env GPULOOP_APPEND=1 ${LEFT:+GPULOOP_MAX_SECS=$LEFT} $GPU_PIN "$GPULOOP_BIN" "$WORK/poly.cado" "$NEXTQ" $QMAX "$WORK/rels.txt" $NEED || true
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
      _srel=$(nrel_all)
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
