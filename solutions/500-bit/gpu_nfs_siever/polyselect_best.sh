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
# polyselect_best.sh  --  best-yield polynomial selection for a 500-bit (c151) N
#
# WHY: a single timed `msieve -np` is CUT OFF by its timeout, not exhausted.
#   For our N msieve logged "expecting poly E 1.98e-11..2.27e-11" but a 540s run
#   delivered only 1.142e-11 -- it died at c5~3120 (~1.5% of the range). The best
#   polys actually live PAST that cutoff (measured: c5 = 5040 / 18360 / 21420).
#
# WHAT: search the full PRODUCTIVE c5 range in parallel slices (each its own
#   -nf/-s/-l so no msieve.fb collision), pool every candidate, take the top-K by
#   msieve-E, then a 3-window gpu_loop YIELD bake-off picks the best MEASURED
#   rel/lattice (E over-predicts yield, so the bake-off is the real selector).
#
# This is a drop-in replacement for the polyselect stage of factor_msv.sh and
# works on ANY same-size N -- it searches a RANGE, so it adapts to
# wherever that N's good polys sit.
#
# usage: polyselect_best.sh <N> [workdir]
# output: $WORK/poly.cado  (best-yield poly) and $WORK/c.fb (msieve factor base)
# ============================================================================
set -u
N="$1"; WORK="${2:-/tmp/polysel}"
SIEVER=$(cd "$(dirname "$0")" && pwd)
MSV="${MSV_DIR:-$(cd "$SIEVER/../msieve-gpu" 2>/dev/null && pwd || echo /root/msieve-gpu)}"

# ------------------------- CONFIG (proven defaults) -------------------------
# Dev-box (384c/1GPU) research profile. For the validator (24 CPU / 1 GPU) set a
# smaller SLICES/THREADS so total threads <= 24 and GPU isn't over-subscribed:
#   e.g. SLICES=4 OMP_NUM_THREADS=6  (see "validator profile" note at bottom).
MIN_EVALUE="${MIN_EVALUE:-3e-12}"   # c151: lower E threshold (larger N -> intrinsically worse polys)
FINE_HI="${FINE_HI:-40000}"         # c151: wider productive c5 zone (raised from 32000)
# FINE_W 2500 -> 1250 (2026-07-31): slice count is ceil(FINE_HI/FINE_W), so this is 16 -> 32 slices.
# The search costs the SLOWEST slice, and instrumenting the 16-slice run showed all slices EXHAUST
# naturally (302-582s, none hits SLICE_TMO) while GPU utilisation averaged just 29% -- i.e. ~190s of
# the wall was one straggler with the box idle. Narrower slices pack that tail. Measured on
# seed20260736, each verified by a full yield bake-off against the ground-truth winner:
#     16 slices  580s   winner c5=31080  47.5 rel/lat   (reference)
#     24 slices  475s   winner c5=30600  46.5 rel/lat   REJECTED: -2.1% yield is ~263s of extra
#                                                       sieve to save 105s -- a net loss
#     32 slices  485s   winner c5=31080  47.5 rel/lat   TAKEN: -95s at identical quality
# CAVEAT: the 24-slice miss shows the search is NOT saturated -- which candidates surface depends on
# where the slice boundaries fall. 16 and 32 both found c5=31080 and 24 did not, which reads as
# partition luck, not a monotone trend. 32 is taken because it is faster AND pools more candidates
# (a strictly wider net than 16), but this rests on ONE key's ground truth. Re-verify on any new key
# before trusting it. Throughput saturates by 24 slices: 24 and 32 are within 10s of each other.
FINE_W="${FINE_W:-1250}"            # c151: 32 slices over [1,FINE_HI]
COARSE_HI="${COARSE_HI:-200000}"    # coarse slices over [FINE_HI, COARSE_HI]
COARSE_N="${COARSE_N:-0}"           # coarse high-c5 slices (0 = off; measured worse, off by default. set >0 for insurance)
SLICE_TMO="${SLICE_TMO:-600}"       # per-slice safety cap (s). 900->600 (2026-07-01 bench): on a CPU-limited host the 16 slices oversubscribe and run to this cap; measured A/B on one key found the BYTE-IDENTICAL winning poly (cand0 c5=16020, 90.2 rel/lat) at 600s as at 900s, saving ~5min. Inert on the 24-CPU validator (slices exhaust in ~540s before either cap).
POLY_TOPK="${POLY_TOPK:-8}"          # funnel width into the yield bake-off. 6->12->8 (2026-08-08):
                                    # measured on key07, the winner is cand6 -- E-RANK 7, so a top-6
                                    # funnel structurally cannot see it and picks cand5 (43.23 vs
                                    # 45.71, a 5.3% yield loss = ~670s of extra sieve). TOPK=8 is the
                                    # narrowest funnel that still contains it; cand8..cand11 measured
                                    # 42.25/42.98/41.96/41.69, i.e. 12 buys nothing here and costs
                                    # ~92s (4 extra 3-window probes at ~23s each).
                                    # !! RISK: E-rank predicts yield poorly. In a 32-candidate sweep
                                    # the SECOND-best poly was cand15 (44.06) -- outside even TOPK=12.
                                    # A key whose winner sits at rank 9-12 loses ~5% of sieve (~650s)
                                    # to save 92s. Raise this if a new key's bake-off looks flat.
# PROBE_WINS: the bake-off MUST measure at the PRODUCTION QMIN, and must span the production q-range.
# BUG (fixed 2026-07-21): was 3M/30M/80M while production sieves from QMIN=1M -> the bake-off sampled
# where polys look TIED and missed the low-q region where they differ AND where much yield is collected.
# Measured: current poly vs a fresh candidate were 27.4 vs 27.2 at 3M/30M/80M (indistinguishable) but
# 55.0 vs 48.9 at q=1M (+12.5%). So the old bake-off would happily pick the 12.5%-worse poly. Now the
# windows start at QMIN=1M and span to ~the crossover q (~40M).
# 40M -> 25M (2026-07-31): a c151 hard key finishes its sieve at q~25M, so the 40M probe measured
# where production NEVER GOES and cost 78s of the bake-off. Re-checked against ground truth: for
# seed20260736 the production winner is cand3 (c5=31080), confirmed best over a real 66M-relation
# sieve. All of 1M/10M/40M, 1M/10M, 1M/10M/20M, 1M/10M/25M, 1M/5M/10M and 1M/10M/15M/20M/25M rank
# cand3 first, on both a flat and a lattice-weighted average -- so this is a pure wall-time cut, not
# a selection change. Kept at THREE windows: 1M/10M alone saves a further 44s but halves the
# insurance on the one decision the whole 12,509s sieve depends on.
#
# 1M/10M/25M -> 4M/11M/19M, and FLAT MEAN -> sum(rel)/sum(lat) (2026-08-02). This is the fix
# Pipeline.md §11 specified but never shipped, and on the yd2 hard key the defect BIT: the shipped
# metric picked the poly that is 3.12% WORSE in production.
#
# WHY the flat mean is the wrong statistic: production sieves q continuously 1M -> ~25M and lattice
# density is ~uniform in q, so ~95% of production lattices sit ABOVE q=2M -- but averaging three
# per-window RATES hands q=1M a full third of the weight. Measured on yd2's real 6-candidate set
# (deterministic, identical to 3 decimals over 3 interleaved reps -- relation counts are exact, so
# the ~1.6% TIMING noise floor does not apply here):
#
#     cand   c5      shipped(flat)   fixed(prod-w)   4M / 11M / 19M
#     cand0  21420   44.53  (5th)    45.83  (1st)    54.6 / 44.8 / 37.0
#     cand3  21420   46.13  (PICKED) 44.44  (3rd)    56.9 / 41.2 / 34.2
#
# cand3 wins ONLY at low q and loses everywhere production actually sieves. The shipped metric
# picked it; the real 3.996h run then sieved 12,485s on a poly 3.12% off the best available.
# Sieve time scales ~inversely with yield => ~-378s. The gain is if anything understated: cand0's
# margin GROWS with q (+8% at both 11M and 19M), and production runs on past 19M to ~25M.
#
# Cost-neutral: same three windows, same 15k-q span each, same one gpu_loop process per candidate.
# Windows sit INSIDE the production range; scoring by summed counts weights each window by the
# lattices it actually contributed instead of averaging rates over unequal denominators.
PROBE_WINS="${PROBE_WINS:-4000000:4015000 11000000:11015000 19000000:19015000}"  # inside production q-range
# ---------------------------------------------------------------------------

rm -rf "$WORK"; mkdir -p "$WORK"; cd "$MSV"
echo "### polyselect_best: parallel wide-range search  $(date +%T)"

idx=0
launch() {  # $1=lo $2=hi
  local lo=$1 hi=$2 d; d=$(printf "%s/inst%02d" "$WORK" "$idx"); mkdir -p "$d"
  # --foreground IS LOAD-BEARING (added 2026-07-28). Without it GNU timeout puts the managed
  # command in its OWN PROCESS GROUP, so these 16 slices escape a process-group kill of the
  # pipeline and SURVIVE as orphans holding the GPU. Measured: a wall-time-expiry test left all
  # 16 msieve processes running after the solver had exited. --foreground keeps each slice in
  # our group so breaking_rsa.py's killpg reaps the whole tree. Safe here: each slice is a single
  # process, so timeout's own SIGTERM still reaches exactly what it needs to.
  timeout --foreground "$SLICE_TMO" ./msieve -g 0 -np "min_evalue=$MIN_EVALUE $lo,$hi" \
     -s "$d/p.dat" -nf "$d/msieve.fb" -l "$d/p.log" "$N" >"$d/out.txt" 2>&1 &
  idx=$((idx+1))
}
# fine slices over the productive low zone
lo=1; while [ "$lo" -lt "$FINE_HI" ]; do hi=$((lo+FINE_W)); [ "$hi" -gt "$FINE_HI" ] && hi=$FINE_HI; launch "$lo" "$hi"; lo=$hi; done
# coarse slices over the sparse high zone (skipped when COARSE_N=0)
if [ "$COARSE_N" -gt 0 ]; then
  cw=$(( (COARSE_HI-FINE_HI)/COARSE_N )); lo=$FINE_HI
  for j in $(seq 1 "$COARSE_N"); do hi=$((lo+cw)); [ "$j" = "$COARSE_N" ] && hi=$COARSE_HI; launch "$lo" "$hi"; lo=$hi; done
fi
echo "### launched $idx slices, waiting"; wait
echo "### search done $(date +%T); pooling + top-$POLY_TOPK by E"

python3 "$SIEVER/pick_topk.py" "$N" "$WORK" "$POLY_TOPK" "$WORK"/inst*/p.dat.p

# ----- YIELD BAKE-OFF: probe each candidate over 3 q-windows, keep best -----
echo "### yield bake-off $(date +%T)"
best=-1; bestf=""
for k in $(seq 0 $((POLY_TOPK-1))); do
  f="$WORK/cand$k.cado"; [ -f "$f" ] || continue
  # ONE gpu_loop process per candidate. GPULOOP_WINS sweeps every probe window on a SINGLE factor
  # base: build_fb() is special-q independent and costs 4.9s, so the old loop paid for 3 FB builds
  # per candidate (18 across the bake-off) where 1 suffices. Measured 2026-07-31 on seed20260736:
  # 3 windows as 3 processes 39.5s -> as 1 process 28.7s, with per-window rel/lattice IDENTICAL
  # (1119/75728, 944/41018, 893/23934).
  # SCORING (2026-08-02): sum(relations)/sum(lattices) across the windows, NOT the flat mean of the
  # per-window rates -- see the PROBE_WINS block above for the measurement that forced this.
  # OMP_NUM_THREADS MUST BE PINNED HERE (2026-07-31). gpu_loop's cofactorisation is OpenMP and this
  # script is called by factor_msv.sh with NO OMP setting, so the bake-off silently inherited the
  # caller's value. Measured on seed20260736: the same 6-candidate bake-off is 171s at OMP=24 and
  # 794s at OMP=1 -- a 4.6x swing on a variable nothing was setting deliberately. The msieve slices
  # above are single-threaded by design and unaffected.
  qa0=${PROBE_WINS%% *}; qa0=${qa0%%:*}
  # stderr is KEPT (2026-08-08). It used to go to /dev/null, which combined with the `r=0` fallback
  # below made a crashed / timed-out / erroring gpu_loop indistinguishable from a candidate that
  # genuinely measured zero yield -- with no evidence left to tell them apart. That is exactly how
  # the float-overflow zero-relation bug (see gpu_loop.cu d_clog2off) stayed invisible: 4 of 6
  # candidates scored "0.00" and the run looked normal.
  eb="$WORK/bakeoff_cand$k.err"
  out=$(cd "$SIEVER" && GPULOOP_WINS="$PROBE_WINS" OMP_NUM_THREADS="${BAKEOFF_OMP:-24}" \
        timeout 900 ./gpu_loop "$f" "$qa0" "$((qa0+1))" /dev/null 0 2>"$eb"); grc=$?
  # Weight each window by the lattices it actually contributed: sum the raw counts, then divide
  # once. Averaging per-window RATES instead divides by unequal denominators and over-weights the
  # cheapest (lowest-q) window -- the defect this replaces.
  r=$(printf '%s' "$out" | awk '
    /\[WIN /{ for(i=1;i<=NF;i++){ if($i=="lattices") l=$(i+1); if($i=="relations") r=$(i+1) }
              L+=l; R+=r; n++ }
    END{ if(n>0 && L>0) printf "%.2f", R/L }')
  # single-window PROBE_WINS prints no [WIN] lines (MULTIWIN is off) -- fall back to the summary
  if [ -z "$r" ]; then
    r=$(printf '%s' "$out" | grep -oE 'avg [0-9.]+ rel/lattice' | grep -oE '[0-9.]+' | head -1)
  fi
  # Distinguish "measured zero" from "failed to measure". A non-zero exit or unparseable output is
  # a BROKEN PROBE, not a zero-yield poly: say so loudly instead of silently scoring it 0.00.
  if [ -z "$r" ] || [ "$grc" -ne 0 ]; then
    echo "  !! cand$k PROBE FAILED (exit=$grc, no yield parsed) -- scored 0 but NOT measured." >&2
    echo "     stderr tail: $(tail -c 300 "$eb" | tr '\n' ' ')" >&2
    r=0
  fi
  c5=$(awk '/^c5/{print $2}' "$f"); echo "  cand$k (c5=$c5): $r rel/lat"
  awk -v a="$r" -v b="$best" 'BEGIN{exit !(a>b)}' && { best="$r"; bestf="$f"; }
done
[ -z "$bestf" ] && bestf="$WORK/cand0.cado"
# NAME THE FAILURE (2026-08-06). This script has no `set -e`, so when the search produced no
# candidate at all (every GPU slice died, pick_topk.py found no pool) the missing cand0.cado used
# to surface three lines later as a bare KeyError from the .fb heredoc -- factor_msv.sh's `set -e`
# then killed the whole run at ~12 min and the solver reported {"status":"failed"}, i.e. a wrong
# answer, for a polyselect failure that nothing in the log identified as such.
if [ ! -s "$bestf" ]; then
  echo "### polyselect FATAL: no candidate polynomial produced ($bestf missing/empty)." >&2
  echo "###   slices that emitted a pool file: $(ls "$WORK"/inst*/p.dat.p 2>/dev/null | wc -l)/$idx" >&2
  for _d in "$WORK"/inst00 "$WORK"/inst01; do
    [ -f "$_d/out.txt" ] && { echo "###   $_d/out.txt:" >&2; sed -n '1,8p' "$_d/out.txt" >&2; }
  done
  exit 3
fi
cp "$bestf" "$WORK/poly.cado"
echo "### bake-off winner: $(basename "$bestf") at $best rel/lat -> $WORK/poly.cado"
# ---- BREAK-EVEN YIELD WARNING (2026-08-23) ----------------------------------------------
# Configuration.md 5.2 puts the break-even at ~43.3 rel/lat on this build: below it the sieve
# cannot reach the purge crossover inside the 4h wall, so the run is already lost HERE, ~8 min in,
# and every later stage is wasted. Nothing said so -- the pipeline sailed on and the validator saw
# an unexplained WallTimeFailure 4 hours later. This does not abort (a marginal key can still land
# inside the poll grace, and refusing would forfeit a run that might succeed); it makes a doomed
# run diagnosable from its first minutes. Override the threshold with POLY_BREAKEVEN.
_be="${POLY_BREAKEVEN:-43.3}"
if [ -n "$best" ] && awk -v a="$best" -v b="$_be" 'BEGIN{exit !(a<b)}' 2>/dev/null; then
  echo "### polyselect WARNING: winning yield $best rel/lat is BELOW the ~$_be rel/lat break-even." >&2
  echo "###   This key is unlikely to reach the purge crossover inside the wall. Expect a" >&2
  echo "###   WallTimeFailure -- the cause is POLYNOMIAL QUALITY, not the sieve or the filter." >&2
  echo "###   Candidates tried: $idx slices -> top-${POLY_TOPK:-8} bake-off." >&2
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

# ---------------------------------------------------------------------------
# The defaults above (16 fine slices over [1,32000], COARSE_N=0, top-3 bake-off)
# are validator-legal: ~16 slices share the 1 GPU (the real limiter, ~99% util),
# so peak CPU stays under 24. Validated end-to-end 2026-06-29: ~10 min, finds the
# same cand0 (c5=18360) as a 384-core research sweep. See Pipeline.md.
# ---------------------------------------------------------------------------
