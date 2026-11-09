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
# polyselect_best.sh -- best-yield polynomial selection, size-aware.
#
# Search a RANGE in parallel, pool everything, then funnel the top few into a
# MEASURED-yield bake-off: no analytic score ranks candidates reliably enough to
# skip it. A single timed search is cut off by its timeout, not exhausted, and
# the best polynomials live past that cutoff.
#
# THREE ENGINES, selected by size (POLYSEL_ENGINE=auto):
#   cado    CADO polyselect + ropt. +11.7% yield at 512-bit, and the gap is
#           structural, not budget. Used at/above POLYSEL_CADO_MINDIGITS.
#   msieve  the GPU -np slice search. Used below that, where it is the engine
#           two cold end-to-end runs validated. Also the fallback if the CADO
#           binaries are missing.
#   both    run the two CONCURRENTLY -- CADO on the cores, msieve on the otherwise
#           idle GPU -- and bake off the UNION of their two top-K pools. Costs one
#           extra pool of bake-off probes and no extra search wall-clock. Used
#           at/above POLYSEL_BOTH_MINDIGITS (155), i.e. c155 only.
#
# Every size-dependent constant comes from polyselect_params.py; at 151 digits
# the generator reproduces the shipped values exactly, so that path is unchanged.
#
# usage: polyselect_best.sh <N> [workdir]
# output: $WORK/poly.cado (best-yield poly) and $WORK/c.fb (msieve factor base)
# ============================================================================
set -u
N="$1"; WORK="${2:-/tmp/polysel}"
SIEVER=$(cd "$(dirname "$0")" && pwd)
MSV="${MSV_DIR:-$(cd "$SIEVER/../msieve-gpu" 2>/dev/null && pwd || echo /root/msieve-gpu)}"
CADOBIN="${CADOBIN:-$(cd "$SIEVER/../cado-filter" 2>/dev/null && pwd || echo /nonexistent)}"

# ---------------- size-aware defaults (env still wins) ----------------------
# polyselect_params.py emits PS_*; every knob below then reads ${KNOB:-$PS_KNOB},
# so an explicit environment value still overrides -- which matters for the dev
# harness, and costs nothing in production where the validator passes no -e.
PS_DIGITS=0 PS_MIN_EVALUE= PS_FINE_HI= PS_FINE_W= PS_POLY_BREAKEVEN=
PS_CADO_P= PS_CADO_ADMIN= PS_CADO_ADMAX= PS_CADO_INCR= PS_CADO_NQ= PS_CADO_NRKEEP= PS_CADO_SOPTEFFORT=
if _pp=$(python3 "$SIEVER/polyselect_params.py" "$N" 2>/dev/null) && [ -n "$_pp" ]; then
  eval "$_pp"
else
  # Degrade to the c151 constants rather than to nothing: a broken helper must not
  # take the run down, and at c151 these ARE what the generator emits.
  echo "### polyselect WARNING: polyselect_params.py failed -- falling back to c151 constants" >&2
  PS_DIGITS=${#N} PS_MIN_EVALUE=3e-12 PS_FINE_HI=40000 PS_FINE_W=1250 PS_POLY_BREAKEVEN=43.3
  PS_CADO_P=312000 PS_CADO_ADMIN=2160 PS_CADO_ADMAX=142000
  PS_CADO_INCR=420 PS_CADO_NQ=156250 PS_CADO_NRKEEP=72 PS_CADO_SOPTEFFORT=4
fi

# The bake-off has to score candidates over the q-range the SIEVE will actually run, and that
# range is sieve_params.py's business (QMIN/QSPLIT), not polyselect_params.py's. Pull it from
# there. Absent/failed => SV_PROBE_WINS stays empty and the historical default applies, which is
# also what happens below 152 digits where sieve_params deliberately emits neither.
SV_PROBE_WINS= SV_QMIN=
if _sp=$(python3 "$SIEVER/sieve_params.py" "$N" 2>/dev/null) && [ -n "$_sp" ]; then
  eval "$(printf '%s\n' "$_sp" | grep -E '^SV_(PROBE_WINS|QMIN)=')"
fi

# ------------------------- CONFIG (proven defaults) -------------------------
POLYSEL_ENGINE="${POLYSEL_ENGINE:-auto}"        # auto | cado | msieve | both
POLYSEL_CADO_MINDIGITS="${POLYSEL_CADO_MINDIGITS:-152}"
# `both` is gated SEPARATELY from `cado`, and deliberately higher, because it is not free:
# it adds a second pool of bake-off probes (~150 s measured) to the critical path in exchange
# for a yield that can only be >= the cado-only winner (the bake-off takes the max).
#   c155 (23.2% of 512-bit keys) MISSES the wall on typical keys -- 27.64 rel/lat -> 15,657 s
#     against a 14,400 s wall. It has nothing to lose and 150 s is cheap against ~1,250 s.
#   c154 (76.8%) PASSES at 13,910 s with only ~490 s of margin. Spending 150 s of that on a
#     pool that has never been measured at c154 would be trading a sure thing for a maybe.
# So: 155 until someone measures the c154 union. MEASURED at c155/20260916 (2026-09-14):
# cado pool best 27.64 (identical to its solo run -- the concurrency costs CADO nothing),
# msieve pool best 28.81 and second-best 28.69, union winner 28.81 = +4.2%.
# ⚠️ DEFAULT OFF (999 = never in auto). MEASURED END-TO-END 2026-09-14 on key 20260916, the
# hardest c155 key on record, against its own cado-only baseline (c155d) as a controlled A/B --
# same relation target, same ladder, same predup prefix, same las q-range, only the poly differs:
#
#             raw rel/lat   unique frac   MATRIX at      probes   GPU lattices   TOTAL
#   cado-only     27.64        0.7602     57,306,550        4      ~2,755,000    15,655 s
#   union         28.81        0.7422     59,559,566        7       2,808,000    15,866 s  (+211)
#
# The union DID find a better polynomial by the metric the bake-off scores (+4.2% rel/lat) and it
# DID cost nothing in search wall-clock. It still LOST, because raw yield is the wrong metric: the
# higher-yielding poly deduplicated 2.4% worse, so it needed 3.9% MORE raw relations, and the two
# effects cancelled to within 2% in LATTICES -- which is what sieve time actually tracks. The run
# then lost by exactly the ~168 s of extra bake-off probes the second pool costs.
#
# Do NOT re-enable this on the strength of a rel/lat number. It is worth re-enabling only once the
# bake-off scores UNIQUE yield (see the DUPSUP=2 note in the bake-off below), which is the change
# that would let a 16-candidate pool actually pay for itself.
# RE-ENABLED at c155 2026-09-14, AFTER the selector above was fixed -- not before. The 999 that
# stood here for one afternoon was correct at the time: with the old raw-yield-at-4/11/19M score
# the union picked a poly that LOST the end-to-end by 211 s (see the table above). Re-scored with
# unique yield on the production windows, the SAME 16-candidate pool's best is cand8 [msieve],
# c5=2640 -- which the old selector ranked 27.38 vs the cado winner's 27.64 and therefore threw
# away. Measured over the range the run actually swept, [9.6M, 58M]:
#
#   cand0 [cado]   (what ran, 15,655 s)   11.88 unique rel/lat
#   cand8 [msieve] (what it should pick)  12.67 unique rel/lat   +6.7%  -> ~-865 s of sieve
#
# The winner comes from the msieve pool, so cado-only cannot reach it. The union costs ~+168 s of
# extra bake-off; missing a 6.7 % poly costs five times that. ⚠️ Still c155-only: c154 passes at
# 13,910 s and its union has never been measured.
# DEFAULT OFF (999 = never in auto). The union was re-enabled at 155 once the selector was fixed,
# then turned off again 2026-09-14 when msieve-only measured BETTER AND CHEAPER at c155 (see the
# engine table below): the union's winner comes from the msieve half, and CADO's 750 s is the
# binding constraint on the stage, so `both` pays 1,032 s for a polynomial msieve finds in 441 s.
# Kept because it is the right tool for deciding an engine question on a NEW size -- run it once,
# score the union, see which half wins -- not for production.
# ⚠️ This branch is tested FIRST, so setting it to 155 silently shadows the msieve branch below.
POLYSEL_BOTH_MINDIGITS="${POLYSEL_BOTH_MINDIGITS:-999}"
# The bake-off must probe with the SAME siever the sieve will use -- scoring candidates with a
# different LIM than production runs would rank them on a factor base that never gets used.
GPULOOP_BIN="${GPULOOP_BIN:-}"
if [ -z "$GPULOOP_BIN" ]; then
  if [ "$PS_DIGITS" -ge 999 ] && [ -x "$SIEVER/gpu_loop.lim100m" ]; then
    GPULOOP_BIN=./gpu_loop.lim100m
  else
    GPULOOP_BIN=./gpu_loop
  fi
fi
POLY_TOPK="${POLY_TOPK:-8}"                     # funnel width into the yield bake-off
POLY_THREADS="${POLY_THREADS:-24}"              # validator cgroup is --cpus 24

# ---- msieve engine ----
MIN_EVALUE="${MIN_EVALUE:-$PS_MIN_EVALUE}"      # c151 3e-12 | c154 1.78e-12  (was hard-coded 3e-12)
FINE_HI="${FINE_HI:-$PS_FINE_HI}"               # c151 40000 | c154 69100     (was hard-coded 40000)
FINE_W="${FINE_W:-$PS_FINE_W}"                  # 32 slices over [1,FINE_HI]
COARSE_HI="${COARSE_HI:-200000}"
COARSE_N="${COARSE_N:-0}"                       # coarse high-c5 slices (0 = off; measured worse)
# SIZE-AWARE 2026-09-14. MEASURED on key 20260916, msieve engine, corrected bake-off:
#     SLICE_TMO=300 -> 114,510 raw polys -> winner c5=2640 at 12.87 unique rel/lat
#     SLICE_TMO=600 -> 232,648 raw polys -> winner c5=2640 at 12.87 unique rel/lat
# 2.03x the pool, IDENTICAL winner, IDENTICAL score: the second 300 s buys nothing and is 300 s
# of a 14,400 s wall. ⚠️ 600 STAYS BELOW 155 DIGITS -- c151 is the live 500-bit path, it runs
# this same engine, and its slice budget has never been swept. Do not "simplify" this to one value.
if [ "$PS_DIGITS" -ge 155 ]; then SLICE_TMO="${SLICE_TMO:-300}"; else SLICE_TMO="${SLICE_TMO:-600}"; fi

# ---- CADO engine ----
# nq is the one parameter that CANNOT be taken from CADO's own params.cNNN.
# MEASURED 2026-09-04: params.c155's `-nq 156250 -incr 420` at validator parity
# (24 cores, 300s) produced **ZERO** polynomials -- not one `ad` value completed.
# Those files are written for a distributed run with adrange=1260 work units and
# many machine-hours; at 300s the search never emits. `-nq 3125 -incr 60` is the
# regime that was actually measured to work here: 2,981 raw polys in 7 minutes on
# 24 cores, containing the best polynomial any method has found for this key
# (31.30 rel/lat, c5=50040, MurphyE 3.275e-12).
CADO_P="${CADO_P:-$PS_CADO_P}"
CADO_NQ="${CADO_NQ:-3125}"
CADO_INCR="${CADO_INCR:-60}"
CADO_ADMIN="${CADO_ADMIN:-0}"
CADO_ADMAX="${CADO_ADMAX:-200000}"
CADO_SOPTEFFORT="${CADO_SOPTEFFORT:-$PS_CADO_SOPTEFFORT}"
CADO_NRKEEP="${CADO_NRKEEP:-40}"                # raw polys handed to the rootsieve
CADO_ROPTEFFORT="${CADO_ROPTEFFORT:-10}"
# 600 s, not 300. Yield multiplies into the sieve, which is ~88% of the budget, so a small
# relative gain here is a large absolute one: +4.73% yield measured = -682 s of sieve for +285 s
# of search. The candidate pool SATURATES at 600 s (900 s gave an identical top-8), so this is a
# plateau edge, not a slope -- do not raise it without re-measuring.
# It cannot rescue a weak key: each extra 300 s raises the yield needed to fit by ~0.7 rel/lat.
# Unreachable below POLYSEL_CADO_MINDIGITS, where the msieve engine runs instead.
CADO_SEARCH_SECS="${CADO_SEARCH_SECS:-600}"
CADO_ROPT_SECS="${CADO_ROPT_SECS:-150}"

# PROBE_WINS: the bake-off MUST measure INSIDE the production q-range and must be
# scored as sum(rel)/sum(lat), not as a flat mean of per-window rates.
#   1M/10M/25M -> 4M/11M/19M + prod-weighted mean (2026-08-02). On the yd2 hard key
#   the flat mean picked a poly 3.12% WORSE in production (~-378s of sieve): cand3
#   wins only at q=1M and loses at every q production actually spends its time at.
#   Windows are unchanged at c154 -- production there sweeps to q~45M, so 4M/11M/19M
#   still sits inside the range, and the c154 bake-off reproduces the full-sweep
#   ranking (FINDINGS-20260904 s10).
# DERIVED, not constant -- see sieve_params.probe_windows() for the measurement that forced
# this. The literal below is the pre-2026-09-14 value and is now only the <152-digit fallback.
PROBE_WINS="${PROBE_WINS:-${SV_PROBE_WINS:-4000000:4015000 11000000:11015000 19000000:19015000}}"
# Score candidates on UNIQUE yield (DUPSUP=2 counts relations already reachable from a smaller
# special-q and reports NEW/lat) rather than raw rel/lat. MEASURED 2026-09-14: raw yield picks
# the LOSING polynomial of the c155d/c155e A/B at either window placement; unique yield on the
# production windows picks the winner. Set BAKEOFF_UNIQUE=0 to fall back to raw.
BAKEOFF_UNIQUE="${BAKEOFF_UNIQUE:-1}"
[ -n "$SV_QMIN" ] || BAKEOFF_UNIQUE=0     # no production QMIN => no meaningful dup baseline
# Score unique relations per MILLISECOND, not per LATTICE.
# MEASURED 2026-09-14: a lattice does NOT cost a constant amount -- `scatter` steps up ~30% once
# the special-q passes the polynomial's SKEW, so ms/lat varies BETWEEN CANDIDATES IN ONE POOL
# (4.82-5.15 on key 20260916, tracking skew). Sieve time is lattices x ms/lat, so ranking on
# rel/LATTICE over-values high-skew candidates whose extra relations cost more to produce.
# Ruled out as the cause: LIM (40M and 60M builds step at the same q) and the sieve region
# (J=8192 and J=4096 likewise). On that pool the two metrics pick DIFFERENT winners, rel/ms by
# +4.0%. Free -- gpu_loop already prints `wall ... ms` on the per-lattice line the probe emits.
# ⚠️⚠️ DEFAULT OFF — MEASURED REGRESSION, 2026-09-15. The derivation above is arithmetically
# right (sieve time IS lattices x ms/lat) and the probe DID measure a real cost difference, but
# scoring it from the three probe windows gets the answer WRONG IN SIGN:
#
#                        probe (27/38/49M)        real range [16.5,57M]
#     yield gap               -2.6%                     -7.4%
#     cost  gap               -5.3%                     -3.9%   <- probe got this one right
#     verdict            cand1 +2.8%  PICKED        cand1 -3.6%
#
# The windows sit ABOVE both candidates' steps, i.e. exactly where the incumbent looks worst,
# and the two candidates' yield curves have different SHAPES, so the windows are not neutral
# between them. End-to-end on key 20260916: 14,368.7 s with rel/lat vs ~14,772 s projected with
# rel/ms — +403 s WORSE. (The low-skew winner did deduplicate better, 0.8152 vs 0.7908, which is
# why it was only +403 s and not more.)
# Re-enable ONLY with full-range scoring (8+ q-points integrated against lattice density), which
# is the method that predicted the 2026-09-14 selector fix at -865 s and delivered -903 s.
BAKEOFF_RATE="${BAKEOFF_RATE:-0}"
[ "$BAKEOFF_UNIQUE" = 1 ] || BAKEOFF_RATE=0
# ---------------------------------------------------------------------------

rm -rf "$WORK"; mkdir -p "$WORK"

# ---- engine choice -----------------------------------------------------------
CADO_PS="$CADOBIN/polyselect"; CADO_ROPT="$CADOBIN/polyselect_ropt"
# ⚠️ c155 IS ON THE MSIEVE ENGINE, NOT CADO. MEASURED 2026-09-14 on key 20260916, scoring the
# full 16-candidate union pool on unique yield at the production windows:
#
#            pool mean   pool best   polyselect stage
#   cado        11.18       12.23        842 s
#   msieve      12.07       12.87        441 s   (SLICE_TMO=300, POLY_TOPK=8)
#
# msieve's pool is better AND cheaper -- it has no separate rootsieve stage. This reverses
# FINDINGS.md §1 ("CADO beats msieve by +11.7%, 31.30 vs 28.02 rel/lat"), which was measured in
# RAW rel/lat at probe windows that sit outside the production q-range; at the production windows
# msieve leads on raw too (22.41 vs 19.69 mean). `both` costs 1,032 s here -- CADO's 750 s is the
# binding constraint -- for a winner msieve finds alone, so the union is OFF.
#
# ⚠️⚠️ n=1 KEY. The CADO engine was adopted on a c154 measurement and validated by two c154
# end-to-end runs; this overrides it for c155 ONLY, on one key, and c154 is untouched. The 9-key
# unique-yield screen is what would make this general. Until then a second c155 key landing on a
# weak msieve pool is a live risk.
ENGINE="$POLYSEL_ENGINE"
if [ "$ENGINE" = auto ]; then
  if   [ "$PS_DIGITS" -ge "$POLYSEL_BOTH_MINDIGITS" ]; then ENGINE=both
  elif [ "$PS_DIGITS" -ge 155 ]; then ENGINE=msieve
  elif [ "$PS_DIGITS" -ge "$POLYSEL_CADO_MINDIGITS" ]; then ENGINE=cado
  else ENGINE=msieve; fi
fi
# `both` needs BOTH engines present. Without the CADO binaries it is just the msieve engine
# with extra steps, so say so and degrade to that rather than half-running a union.
if [ "$ENGINE" = both ] && { [ ! -x "$CADO_PS" ] || [ ! -x "$CADO_ROPT" ]; }; then
  echo "### polyselect WARNING: union engine requested but $CADO_PS / $CADO_ROPT missing" >&2
  echo "###   -> falling back to msieve alone." >&2
  ENGINE=msieve
fi
if [ "$ENGINE" = cado ] && { [ ! -x "$CADO_PS" ] || [ ! -x "$CADO_ROPT" ]; }; then
  # Class B degrade (Pipeline.md s12): say so LOUDLY, then carry on with the engine
  # that is present. Silent fallback here would cost ~11.7% of yield with nothing in
  # the log to explain a run that came in 1,900s slow.
  echo "### polyselect WARNING: CADO engine requested but $CADO_PS / $CADO_ROPT missing" >&2
  echo "###   -> falling back to msieve. Expect ~11.7% less yield at c154." >&2
  ENGINE=msieve
fi
echo "### polyselect_best: engine=$ENGINE  digits=$PS_DIGITS  $(date +%T)"

# ============================ engine: msieve =================================
# $1 = directory to emit cand<k>.cado into (default $WORK). Everything else is unchanged;
# the parameter exists so the `both` engine can keep the two pools apart.
msieve_search() {
  local OUTD="${1:-$WORK}"; mkdir -p "$OUTD"
  cd "$MSV"
  echo "###   msieve wide-range search: MIN_EVALUE=$MIN_EVALUE FINE_HI=$FINE_HI FINE_W=$FINE_W"
  idx=0; SLICE_PIDS=()
  launch() {  # $1=lo $2=hi
    local lo=$1 hi=$2 d; d=$(printf "%s/inst%02d" "$WORK" "$idx"); mkdir -p "$d"
    # --foreground IS LOAD-BEARING (2026-07-28). Without it GNU timeout puts the managed command in
    # its OWN process group, so the slices escape breaking_rsa.py's killpg and SURVIVE as orphans
    # holding the GPU. Measured: a wall-time-expiry test left all 16 msieve processes running.
    timeout --foreground "$SLICE_TMO" ./msieve -g 0 -np "min_evalue=$MIN_EVALUE $lo,$hi" \
       -s "$d/p.dat" -nf "$d/msieve.fb" -l "$d/p.log" "$N" >"$d/out.txt" 2>&1 &
    SLICE_PIDS+=($!)
    idx=$((idx+1))
  }
  lo=1; while [ "$lo" -lt "$FINE_HI" ]; do hi=$((lo+FINE_W)); [ "$hi" -gt "$FINE_HI" ] && hi=$FINE_HI; launch "$lo" "$hi"; lo=$hi; done
  if [ "$COARSE_N" -gt 0 ]; then
    cw=$(( (COARSE_HI-FINE_HI)/COARSE_N )); lo=$FINE_HI
    for j in $(seq 1 "$COARSE_N"); do hi=$((lo+cw)); [ "$j" = "$COARSE_N" ] && hi=$COARSE_HI; launch "$lo" "$hi"; lo=$hi; done
  fi
  # WAIT ON OUR OWN PIDS, NOT A BARE `wait` (2026-09-14). A bare `wait` reaps EVERY child of
  # this shell, and under `engine=both` that includes the backgrounded CADO subshell -- so this
  # blocked on CADO as well, and the later `wait "$cado_pid"` then failed with "not a child"
  # and reported the union as `cado rc=127`. The pools were fine (CADO had in fact completed),
  # but the exit status was fiction and the stage timings were attributed to the wrong engine.
  echo "###   launched $idx slices, waiting"; wait "${SLICE_PIDS[@]}"
  NSLICE=$idx
  echo "###   search done $(date +%T); pooling + top-$POLY_TOPK by msieve-E"
  python3 "$SIEVER/pick_topk.py" "$N" "$OUTD" "$POLY_TOPK" "$WORK"/inst*/p.dat.p
}

# ============================= engine: cado ==================================
# $1 = directory to emit cand<k>.cado into (default $WORK). See msieve_search().
cado_search() {
  local OUTD="${1:-$WORK}"; mkdir -p "$OUTD"
  echo "###   CADO polyselect: P=$CADO_P nq=$CADO_NQ incr=$CADO_INCR ad=[$CADO_ADMIN,$CADO_ADMAX]" \
       "t=$POLY_THREADS budget=${CADO_SEARCH_SECS}s"
  # --foreground for the same process-group reason as the msieve slices above.
  # polyselect streams candidates to stdout as it finds them, so a timeout keeps
  # everything found so far -- rc=124 is the NORMAL exit here, not a failure.
  timeout --foreground "$CADO_SEARCH_SECS" \
    "$CADO_PS" -N "$N" -degree 5 -t "$POLY_THREADS" \
      -P "$CADO_P" -nq "$CADO_NQ" -incr "$CADO_INCR" \
      -admin "$CADO_ADMIN" -admax "$CADO_ADMAX" -sopteffort "$CADO_SOPTEFFORT" \
      >"$WORK/cado.raw" 2>"$WORK/cado.raw.err"
  NSLICE=$(grep -c '^Y1' "$WORK/cado.raw" 2>/dev/null || echo 0)
  echo "###   search done $(date +%T): $NSLICE raw polynomials"
  [ "$NSLICE" -gt 0 ] || { echo "###   CADO search produced nothing" >&2; return 1; }

  # Pool -> top-NRKEEP by exp_E -> rootsieve. cado_top_raw.py exists because
  # polyselect_ropt -inputpolys needs BLANK-LINE-separated blocks with the trailing
  # `# exp_E` comment stripped; fed the raw output verbatim it silently parses ONE
  # polynomial and reports "1 polynomial(s) read", which is not an error you notice.
  python3 "$SIEVER/cado_top_raw.py" "$WORK/cado.top" "$CADO_NRKEEP" "$WORK/cado.raw" || return 1
  echo "###   rootsieve: nrkeep=$CADO_NRKEEP ropteffort=$CADO_ROPTEFFORT budget=${CADO_ROPT_SECS}s"
  timeout --foreground "$CADO_ROPT_SECS" \
    "$CADO_ROPT" -inputpolys "$WORK/cado.top" -t "$POLY_THREADS" -ropteffort "$CADO_ROPTEFFORT" \
      >"$WORK/cado.ropt" 2>"$WORK/cado.ropt.err"
  # Same story as the search: a timeout still leaves every polynomial rootsieved so far.
  python3 "$SIEVER/cado_extract_polys.py" "$WORK/cado.ropt" "$OUTD" "$POLY_TOPK" || return 1
  # cado_extract_polys.py writes b<k>.cado; the bake-off below consumes cand<k>.cado.
  for k in $(seq 0 $((POLY_TOPK-1))); do
    [ -s "$OUTD/b$k.cado" ] && mv -f "$OUTD/b$k.cado" "$OUTD/cand$k.cado"
  done
  [ -s "$OUTD/cand0.cado" ]
}

# ============================= engine: both ==================================
# UNION OF TWO INDEPENDENT POOLS (PLAN-20260913 s2).
#
# The two searches do not contend: CADO's `polyselect`/`polyselect_ropt` are pure CPU and
# msieve's `-np` slice search is GPU. Measured on key 20260916, the GPU sat IDLE for ~750 s
# of an 842 s polyselect stage -- 4.8% of the whole budget, more than half that key's
# deficit, spent before the sieve starts. Running msieve in that window is free wall-clock.
#
# WHY A UNION AND NOT A BETTER RANKING: what decides the outcome is POOL COMPOSITION, not
# ranking within a pool. Key 20260912's bake-off winner was third by MurphyE and was absent
# from the 300 s pool entirely; key 20260916's 8 candidates spanned only 23.96-27.64. A
# second, structurally different search is the cheapest way to thicken a thin pool.
#
# THE POOLS ARE NOT CROSS-RANKABLE. msieve-E and CADO MurphyE are different quantities on
# different scales -- msieve-E is ~uncorrelated with yield here (rho~0.21). So this does NOT
# merge-and-sort. Each engine contributes its own top-POLY_TOPK and the BAKE-OFF ranks the
# union, which it can do because rel/lat is an exact relation count, not a score. Cost is one
# extra bake-off pool (~8 probes, ~150 s) against a sieve of ~13,800 s.
both_search() {
  local mrc=1 crc=1
  mkdir -p "$WORK/pool_cado" "$WORK/pool_msieve"
  echo "###   engine=both: CADO (CPU) and msieve (GPU) concurrently"
  # CADO first and in the background: it owns the cores, msieve owns the GPU. msieve's slices
  # are launched by msieve_search's own `wait`, so it must NOT be backgrounded as a group with
  # CADO -- run CADO in the background and msieve in the foreground.
  ( cado_search "$WORK/pool_cado" ) >"$WORK/cado.engine.log" 2>&1 &
  local cado_pid=$!
  msieve_search "$WORK/pool_msieve"; mrc=$?
  echo "###   msieve pool done $(date +%T); waiting for CADO"
  wait "$cado_pid"; crc=$?
  sed 's/^/###   [cado] /' "$WORK/cado.engine.log" 2>/dev/null
  local n=0 k f
  # CADO first: it is the engine two c155 end-to-end runs validated, so if the bake-off ties
  # (awk `>` keeps the FIRST of equal scores) the incumbent wins and `both` cannot regress.
  for src in "$WORK/pool_cado" "$WORK/pool_msieve"; do
    for k in $(seq 0 $((POLY_TOPK-1))); do
      f="$src/cand$k.cado"; [ -s "$f" ] || continue
      cp "$f" "$WORK/cand$n.cado"
      echo "$n $(basename "$src" | sed 's/^pool_//')" >> "$WORK/cand.engine"
      n=$((n+1))
    done
  done
  UNION_N=$n
  echo "###   union pool: $n candidates (cado rc=$crc, msieve rc=$mrc)"
  [ "$n" -gt 0 ]
}

UNION_N=0
rm -f "$WORK/cand.engine"
if [ "$ENGINE" = both ]; then
  both_search || {
    echo "### polyselect WARNING: union engine produced no candidate -- retrying with msieve alone" >&2
    ENGINE=msieve; rm -f "$WORK"/cand*.cado "$WORK/cand.engine"; msieve_search
  }
elif [ "$ENGINE" = cado ]; then
  cado_search || {
    echo "### polyselect WARNING: CADO engine produced no candidate -- retrying with msieve" >&2
    ENGINE=msieve; rm -f "$WORK"/cand*.cado; msieve_search
  }
else
  msieve_search
fi
# The bake-off sweeps cand0..cand$((BAKEOFF_N-1)); the union makes that wider than POLY_TOPK.
BAKEOFF_N=$POLY_TOPK
[ "$UNION_N" -gt 0 ] && BAKEOFF_N=$UNION_N

# ----- YIELD BAKE-OFF: probe each candidate over 3 q-windows, keep best -----
# DO NOT REMOVE THIS, even under the CADO engine. FINDINGS.md s4b measured MurphyE
# "monotone with measured yield" on one ordering of six polynomials and that reads as
# a licence to ship b0 blind. It is not: on the 2026-09-04 candidate set the poly with
# the LOWEST MurphyE of the top three had the HIGHEST measured yield, beating b0 by
# 2.4% -- and rel/lat is an exact count, so that is not the 1.6% timing noise floor.
# MurphyE is a good FUNNEL (it is what makes POLY_TOPK=8 sufficient); it is not a
# selector. The bake-off costs ~25s per candidate against a sieve of ~16,000s.
echo "### yield bake-off $(date +%T)"
best=-1; bestf=""
for k in $(seq 0 $((BAKEOFF_N-1))); do
  f="$WORK/cand$k.cado"; [ -f "$f" ] || continue
  # ONE gpu_loop process per candidate: GPULOOP_WINS sweeps every probe window on a SINGLE factor
  # base (build_fb() is special-q independent and costs 4.9s), so this pays 1 FB build per candidate
  # instead of 3. Measured 2026-07-31: 39.5s -> 28.7s with per-window counts IDENTICAL.
  # OMP_NUM_THREADS MUST BE PINNED (2026-07-31): factor_msv.sh calls this script with no OMP setting,
  # and the same 6-candidate bake-off is 171s at OMP=24 vs 794s at OMP=1.
  # !! 24 -> 8, RE-MEASURED 2026-09-07. THE 2026-07-31 SWEEP ONLY TRIED 1 AND 24. !!
  # gpu_loop enters its per-lattice `process` OMP region every ~6 ms with only nship (~36-65)
  # items at schedule(dynamic,8) -- 4-8 chunks. 24 threads is pure barrier overhead on that, and
  # the optimum is a broad plateau at 4-12 (see the thread-cap block in gpu_loop.cu main()).
  # Measured here on the c154 winner poly, exact shipped invocation, 3 probe windows, quiet box:
  #     OMP=24  9.24 + 14.56 + 13.91 = 37.7 s
  #     OMP=8   6.09 +  5.75 +  5.50 = 17.3 s      -> 2.2x, i.e. ~171s -> ~78s over 6 candidates
  # RELATION COUNTS ARE BYTE-IDENTICAL between the two (38808/26922/21132, avg 31.2 rel/lat), so
  # this is a pure wall-clock win and CANNOT reorder the bake-off. gpu_loop now self-caps at 8,
  # but this export still wins over that cap, so it has to be set here too.
  qa0=${PROBE_WINS%% *}; qa0=${qa0%%:*}
  # stderr is KEPT (2026-08-08): with it discarded, a crashed/timed-out gpu_loop was indistinguishable
  # from a candidate that genuinely measured zero yield. That is how the float-overflow zero-relation
  # bug stayed invisible -- 4 of 6 candidates scored "0.00" and the run looked normal.
  eb="$WORK/bakeoff_cand$k.err"
  # DUPSUP=2 emits every relation and additionally reports how many were already reachable from
  # a special-q below DUP_QMIN, i.e. the duplicates a production sweep starting at QMIN would
  # throw away. It costs nothing measurable (13.5 s/candidate against the ~18 s the raw probe
  # was taking) and it aggregates across all three windows weighted by lattices, which is
  # exactly the average the raw path was computing by hand.
  _dup=""; [ "$BAKEOFF_UNIQUE" = 1 ] && _dup="DUPSUP=2 DUP_QMIN=$SV_QMIN"
  out=$(cd "$SIEVER" && env $_dup GPULOOP_WINS="$PROBE_WINS" OMP_NUM_THREADS="${BAKEOFF_OMP:-8}" \
        timeout 900 "$GPULOOP_BIN" "$f" "$qa0" "$((qa0+1))" /dev/null 0 2>"$eb"); grc=$?
  r=""
  if [ "$BAKEOFF_UNIQUE" = 1 ]; then
    # "[DUPSUP] valid N  dup-of-smaller-q M (P%)  new K  -> NEW/lat X.XX  (dup_qmin=Q)"
    r=$(printf '%s' "$out" | grep -oE 'NEW/lat[[:space:]]+[0-9.]+' | grep -oE '[0-9.]+' | head -1)
  fi
  if [ -z "$r" ]; then
    # Weight each window by the lattices it actually contributed: sum raw counts, divide once.
    r=$(printf '%s' "$out" | awk '
      /\[WIN /{ for(i=1;i<=NF;i++){ if($i=="lattices") l=$(i+1); if($i=="relations") r=$(i+1) }
                L+=l; R+=r; n++ }
      END{ if(n>0 && L>0) printf "%.2f", R/L }')
  fi
  if [ -z "$r" ]; then
    r=$(printf '%s' "$out" | grep -oE 'avg [0-9.]+ rel/lattice' | grep -oE '[0-9.]+' | head -1)
  fi
  if [ -z "$r" ] || [ "$grc" -ne 0 ]; then
    echo "  !! cand$k PROBE FAILED (exit=$grc, no yield parsed) -- scored 0 but NOT measured." >&2
    echo "     stderr tail: $(tail -c 300 "$eb" | tr '\n' ' ')" >&2
    r=0
  fi
  _ms=$(printf '%s' "$out" | grep -a 'per-lattice:' | grep -oE 'wall [0-9.]+' | grep -oE '[0-9.]+' | head -1)
  _perlat="$r"
  if [ "$BAKEOFF_RATE" = 1 ] && [ -n "$r" ] && [ -n "$_ms" ] \
     && awk -v m="$_ms" 'BEGIN{exit !(m>0)}' 2>/dev/null; then
    r=$(awk -v u="$r" -v m="$_ms" 'BEGIN{printf "%.4f", u/m}')
  fi
  c5=$(awk '/^c5/{print $2}' "$f")
  eng=""; [ -f "$WORK/cand.engine" ] && eng=" $(awk -v k="$k" '$1==k{print "["$2"]"}' "$WORK/cand.engine")"
  _unit="rel/lat"; [ "$BAKEOFF_UNIQUE" = 1 ] && _unit="UNIQUE rel/lat"
  if [ "$BAKEOFF_RATE" = 1 ]; then
    _sk=$(awk '/^skew/{printf "%.0f",$2}' "$f")
    echo "  cand$k$eng (c5=$c5, skew=$_sk): $r UNIQUE rel/ms  [$_perlat rel/lat / $_ms ms]"
  else
    echo "  cand$k$eng (c5=$c5): $r $_unit"
  fi
  awk -v a="$r" -v b="$best" 'BEGIN{exit !(a>b)}' && { best="$r"; bestf="$f"; }
done
[ -z "$bestf" ] && bestf="$WORK/cand0.cado"
# NAME THE FAILURE (2026-08-06). No `set -e` here, so a search that produced no candidate at all used
# to surface three lines later as a bare KeyError from the .fb heredoc; factor_msv.sh's `set -e` then
# killed the run at ~12 min and the solver reported {"status":"failed"} -- a WRONG ANSWER scored
# IncorrectFailure, for a polyselect failure nothing in the log identified as such.
if [ ! -s "$bestf" ]; then
  echo "### polyselect FATAL: no candidate polynomial produced ($bestf missing/empty)." >&2
  echo "###   engine=$ENGINE  digits=$PS_DIGITS  MIN_EVALUE=$MIN_EVALUE  FINE_HI=$FINE_HI" >&2
  if [ "$ENGINE" = cado ]; then
    echo "###   cado.raw polys: $(grep -c '^Y1' "$WORK/cado.raw" 2>/dev/null || echo 0)" >&2
    [ -s "$WORK/cado.raw.err" ] && { echo "###   cado stderr:" >&2; tail -c 600 "$WORK/cado.raw.err" >&2; }
    [ -s "$WORK/cado.ropt.err" ] && { echo "###   ropt stderr:" >&2; tail -c 600 "$WORK/cado.ropt.err" >&2; }
  else
    echo "###   slices that emitted a pool file: $(ls "$WORK"/inst*/p.dat.p 2>/dev/null | wc -l)/${NSLICE:-?}" >&2
    for _d in "$WORK"/inst00 "$WORK"/inst01; do
      [ -f "$_d/out.txt" ] && { echo "###   $_d/out.txt:" >&2; sed -n '1,8p' "$_d/out.txt" >&2; }
    done
  fi
  exit 3
fi
cp "$bestf" "$WORK/poly.cado"
_unit="rel/lat"; [ "$BAKEOFF_UNIQUE" = 1 ] && _unit="UNIQUE rel/lat"
[ "$BAKEOFF_RATE" = 1 ] && _unit="UNIQUE rel/ms"
echo "### bake-off winner: $(basename "$bestf") at $best $_unit -> $WORK/poly.cado"
echo "###   scored over PROBE_WINS=$PROBE_WINS  (DUP_QMIN=${SV_QMIN:-none})"
# ---- YIELD-BAND WARNING -----------------------------------------------------------------
# REDEFINED 2026-09-04. The shipped 43.3 meant "below this the sieve cannot reach the crossover
# inside the wall". At c154 NOTHING reaches that (the ceiling is 31.3 rel/lat over ~25,700
# polynomials from four search regimes), so as a per-key discriminator the old threshold fires
# every time and stops meaning anything. It is now "clearly below the achievable band FOR THIS
# SIZE" (0.92 x the size's ceiling: 43.3 at c151, 28.5 at c154), which is what the warning was
# actually useful for -- catching a key whose polynomial search went wrong. Configuration.md 5.2
# already records that this metric does not predict wall time. Advisory only: a marginal key can
# still land inside the poll grace, and refusing would forfeit a run that might succeed.
# ⚠️ POLY_BREAKEVEN is calibrated in RAW rel/lat (43.3 at c151, 28.5 at c154). A UNIQUE score is
# ~40-55 % lower at the production windows, so comparing the two would fire this warning on every
# single run and make it noise. Suppressed until someone re-derives the band in unique units.
_be="${POLY_BREAKEVEN:-$PS_POLY_BREAKEVEN}"
if [ "$BAKEOFF_UNIQUE" = 1 ]; then _be=""; fi
if [ -n "$_be" ] && [ -n "$best" ] && awk -v a="$best" -v b="$_be" 'BEGIN{exit !(a<b)}' 2>/dev/null; then
  echo "### polyselect WARNING: winning yield $best rel/lat is BELOW the ~$_be rel/lat band" >&2
  echo "###   expected at $PS_DIGITS digits. This key's polynomial search underperformed; the" >&2
  echo "###   sieve will be correspondingly slower. Cause is POLYNOMIAL QUALITY, not the sieve" >&2
  echo "###   or the filter. Engine=$ENGINE, top-$POLY_TOPK bake-off." >&2
fi

# emit msieve .fb for postprocessing
python3 - "$N" "$WORK/poly.cado" "$WORK/c.fb" <<'PY'
import sys,re
N,cado,fb=sys.argv[1],sys.argv[2],sys.argv[3]; p={}
for line in open(cado):
    m=re.match(r'(skew|c\d|Y\d):\s*(\S+)',line.strip())
    if m: p[m.group(1)]=m.group(2)
open(fb,'w').write("N %s\nSKEW %s\nR0 %s\nR1 %s\n"%(N,p['skew'],p['Y0'],p['Y1'])
                   +"".join("A%d %s\n"%(i,p['c'+str(i)]) for i in range(6)))
print("  c.fb written: skew=%s c5=%s"%(p['skew'],p['c5']))
PY
