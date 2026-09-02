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

# Multi-core square root via CADO (replaces msieve's serial -nc3).
# Flow: for each msieve dependency, dump its (a,b) pairs (patched msieve, CPU, fast),
#       run CADO's OpenMP parallel sqrt (all cores) until a non-trivial factor appears.
# Writes "factor: <p>" / "factor: <q>" to $LOG (msieve format) so the orchestrator parses it.
# args: <N> <m.dat> <m.fb> <log> <nthreads>
set -u
N="$1"; S="$2"; FB="$3"; LOG="$4"; NT="${5:-24}"
HERE=$(cd "$(dirname "$0")" && pwd)
MDUMP="${MDUMP:-$HERE/msieve_dump}"      # patched msieve (a,b dumper)
CSQRT="${CSQRT:-$HERE/cado_sqrt}"        # CADO parallel sqrt binary
PY="${PYTHON:-python3}"
MAXDEP="${CADO_MAXDEP:-16}"
WD=$(dirname "$S"); POLY="$WD/poly_cado.txt"; DEPPFX="$WD/cadodep"

# m.fb (msieve) -> CADO poly
"$PY" - "$N" "$FB" "$POLY" <<'PYEOF'
import sys
N,fb,out=sys.argv[1],sys.argv[2],sys.argv[3]
d={}
for line in open(fb):
    p=line.split()
    if len(p)==2: d[p[0]]=p[1]
with open(out,"w") as f:
    f.write("n: %s\nskew: %s\n"%(N,d.get('SKEW','1')))
    for i in range(6): f.write("c%d: %s\n"%(i,d['A%d'%i]))
    f.write("Y0: %s\nY1: %s\n"%(d['R0'],d['R1']))
PYEOF
[ -s "$POLY" ] || { echo "[cado-sqrt] poly conversion failed"; exit 2; }

# ---------------------------------------------------------------------------------------------
# PIPELINED DUMPS (2026-07-28). Measured: each dependency costs ~27s of SINGLE-THREADED dump plus
# ~71s of OMP-parallel sqrt = ~98s. Dependencies were tried strictly serially, so an unlucky key
# paid the dump 4 times on the critical path: seed 20260729 needed 4 deps = 396s, vs 98s for
# seed 20260728 which hit on dep 1. (The msieve -nc3 FALLBACK path below already ran 4 deps
# concurrently; the preferred CADO path did not.)
#
# The dump is 1-thread and every dependency's dump is INDEPENDENT, so they can all run at once
# alongside the parallel sqrt, using cores the sqrt is not saturating. That takes the dump off the
# critical path for every dependency after the first:
#     before: 4 x (27 dump + 71 sqrt) = 392s
#     after :  27 + 4 x 71            = 311s      (-81s, and 0s regression when dep 1 wins)
# Strictly no worse than serial: dep 1 still starts immediately, we only pre-fetch the rest.
PREDUMP="${CADO_PREDUMP:-4}"      # how many dependency dumps to run ahead; 0 disables
# !! COMPLETION MUST BE SIGNALLED BY A SENTINEL, NOT BY THE DUMP FILE EXISTING !!
# msieve creates DUMP_AB_FILE when it OPENS it, so testing `-e <dumpfile>` would happily hand a
# HALF-WRITTEN dump to the sqrt -- a silent-corruption race. Each pre-dump therefore writes to
# <prefix>.pre<d>.part and only renames to <prefix>.pre<d> after the dumper exits; the reader waits
# on the renamed name, which cannot appear early because rename(2) is atomic.
predump_start() {   # $1 = dep index
  local d="$1"
  [ "$d" -gt "$MAXDEP" ] && return 0
  [ -e "$DEPPFX.pre$d" ] || [ -e "$DEPPFX.pre$d.part" ] && return 0
  # !! CLAIM THE SLOT SYNCHRONOUSLY, BEFORE FORKING (2026-08-25). !!
  # The guard above tests for .part, but .part used to be created INSIDE the subshell below, by
  # msieve, asynchronously -- so two calls for the same d could both pass the guard and fork two
  # dumpers onto the SAME .part. Measured live on seed20260824: deps 1 and 5 each got two dumpers
  # (msieve.log.dump1/dump5 are exactly 2x the size of the others and each contains two "DUMPED
  # 8556046 (a,b) pairs" lines -- BOTH dumpers succeeded). The loser's `mv` then found no .part and
  # fell through to `: > pre$d`, TRUNCATING a complete 8.5M-pair dump to 0 bytes, which sqrt_one
  # reads as "not a square, skip". Two of six dependencies were silently discarded, invisibly,
  # because that message is also what a genuine non-square prints.
  # It races because line 63 pre-forks deps 1..PREDUMP and the window loop then immediately calls
  # predump_start on the same indices (and on dd+PREDUMP), so the two calls are microseconds apart.
  : > "$DEPPFX.pre$d.part"
  (
    DUMP_AB_FILE="$DEPPFX.pre$d.part" OMP_NUM_THREADS=1 "$MDUMP" -nc3 "$d,$d" \
        -s "$S" -nf "$FB" -l "$LOG.dump$d" -v "$N" >/dev/null 2>&1 || true
    # rename only on completion; an empty .part means "dep d is not a square" -> leave a 0-byte
    # sentinel so the reader learns that instead of waiting forever.
    # The `[ -e ... ]` arm is load-bearing: NEVER truncate a pre$d that already exists. Even with
    # the claim above, any future path that races here must degrade to a no-op, not to data loss.
    mv -f "$DEPPFX.pre$d.part" "$DEPPFX.pre$d" 2>/dev/null \
      || [ -e "$DEPPFX.pre$d" ] || : > "$DEPPFX.pre$d"
  ) &
}
for k in $(seq 1 "$PREDUMP"); do predump_start "$k"; done

# ---------------------------------------------------------------------------------------------
# CONCURRENT SQRT WINDOW (2026-07-31). The dumps were already pipelined, but the SQRTS themselves
# still ran strictly one dependency at a time, so the cost of the run is decided by how lucky the
# first dependency is. Both outcomes measured on seed20260736, same key, same matrix:
#     2026-07-30: dep 1 not a square -> skipped instantly, dep 2 hit          = 165s
#     2026-07-31: dep 1 WAS a square -> full 99s sqrt, trivial gcd, dep 2 hit = 260s
# Same work, +95s of pure draw. Each dependency is independent and a CADO sqrt at OMP=24 does not
# saturate 24 cores, so running a small window of them at once converts that variance into a fixed
# cost: the answer arrives when the FIRST successful dependency finishes, not when all the failures
# ahead of it have been retired in order.
#
# The msieve -nc3 FALLBACK path further down has always done this (deps 1-4 concurrent). This just
# brings the preferred CADO path in line. SQRT_WIN=1 restores the old strictly-serial behaviour.
#
# DEFAULT 1 -> 3 (2026-08-02). THE 2026-07-31 BLOCKER IS EXPLAINED; it was never a defect in the
# window code. The note it replaces read: "a cache-HIT test run reported all 16 dependencies 'not a
# square', and a hand-run msieve_dump of dep 1 against the same .dep reproduced exactly that, which
# contradicts the 10:45 run that found the factor on dep 2 from that very file."
#
# That symptom is the SEPARATE m.dat.lp bug, documented in Pipeline.md §9 the following day
# (2026-08-01) as having "cost two failed investigations before being identified" -- this was one of
# them. msieve_dump -nc3 needs the large-prime file from filtering; without it EVERY dependency
# reports "algebraic side is not a square!". And msvrun_gpu.sh:33-35 shows the cache-HIT path copies
# ONLY m.dat.mat and m.dat.cyc, never m.dat.lp:
#     cp "${GNFS_CACHE_DS}/m.dat.mat" "$S.mat"; cp "${GNFS_CACHE_DS}/m.dat.cyc" "$S.cyc"
# So a cache-HIT run CANNOT reach a working sqrt at any SQRT_WIN, and the 10:45 non-cache run that
# did find a factor is not a contradiction -- it had m.dat.lp. The window code was never falsified.
#
# WHY IT MATTERS MORE THAN THE "~50s expected" the old note weighed: expected value is the wrong
# statistic for a hard deadline. Measured on the yd2 hard key (2026-08-02, 3.996h end-to-end, 14s
# inside the wall) the serial path needed SIX dependencies -- dep 1 not a square, deps 2-5 trivial
# gcd, factor on dep 6 -- for 443s against the 161/243/260s previously seen. Needing dep 6 is a ~3%
# draw, and it cost 443-186 = 257s on the single run where the margin was 14s. The window converts
# that tail into a fixed cost: the answer arrives when the FIRST successful dependency finishes.
SQRT_WIN="${CADO_SQRT_WIN:-6}"
sqrt_one() {   # $1 = dep index, $2 = dump file ; writes "<p>" to $DEPPFX.win$1.factor on success
  local d="$1" dmp="$2" tpd=$(( NT/SQRT_WIN>0 ? NT/SQRT_WIN : 1 ))
  [ -s "$dmp" ] || { echo "[cado-sqrt] dep $d: not a square, skip"; return 0; }
  cp -f "$dmp" "$DEPPFX.w$d.000" 2>/dev/null || return 0
  echo "[cado-sqrt] dep $d: CADO sqrt (OMP=$tpd) $(date +%T)"
  OMP_NUM_THREADS="$tpd" "$CSQRT" -poly "$POLY" -prefix "$DEPPFX.w$d" -dep 0 \
      -side0 -side1 -gcd >"$LOG.cado$d" 2>&1 || true
  "$PY" - "$N" "$LOG.cado$d" > "$DEPPFX.win$d.factor" <<'PYEOF'
import sys,re
N=int(sys.argv[1])
for line in open(sys.argv[2]):
    s=line.strip()
    if re.fullmatch(r"[0-9]+", s):
        f=int(s)
        if 1<f<N and N%f==0:
            print(f); break
PYEOF
  rm -f "$DEPPFX.w$d".* 2>/dev/null || true
}
if [ "$SQRT_WIN" -gt 1 ]; then
  d=1
  while [ "$d" -le "$MAXDEP" ]; do
    wpids=(); wdeps=()
    for k in $(seq 0 $((SQRT_WIN-1))); do
      dd=$((d+k)); [ "$dd" -gt "$MAXDEP" ] && break
      predump_start "$dd"
      _w=0; while [ ! -e "$DEPPFX.pre$dd" ] && [ "$_w" -lt 600 ]; do sleep 1; _w=$((_w+1)); done
      [ -e "$DEPPFX.pre$dd" ] || continue
      rm -f "$DEPPFX.win$dd.factor"
      sqrt_one "$dd" "$DEPPFX.pre$dd" & wpids+=($!); wdeps+=("$dd")
      predump_start "$((dd+PREDUMP))"
    done
    [ "${#wpids[@]}" = 0 ] && break
    # EARLY EXIT ON THE FIRST WINNER (2026-08-11).
    # This used to be `for p in "${wpids[@]}"; do wait "$p"; done` -- wait for EVERY dependency in
    # the batch, THEN look at the .factor files. So a batch cost max(t_1..t_WIN) even when dep 1
    # had the factor in a third of that, and the header of this file was simply wrong when it said
    # "the answer arrives when the FIRST successful dependency finishes". It does now.
    # WHY IT MATTERS MORE THAN ITS MEAN: sqrt is the last stage before the wall and its spread is
    # the largest left in the run (measured 91s / 119s / 218s across three e2e runs). At a hard
    # deadline the tail is what fails you, and Configuration.md §5 records that this spread now
    # EXCEEDS the wall-time margin. Cutting max() to first-success attacks exactly that tail.
    WINNER=""
    while :; do
      for dd in "${wdeps[@]}"; do [ -s "$DEPPFX.win$dd.factor" ] && { WINNER="$dd"; break; }; done
      [ -n "$WINNER" ] && break
      _alive=0; for p in "${wpids[@]}"; do kill -0 "$p" 2>/dev/null && { _alive=1; break; }; done
      [ "$_alive" = 0 ] && break          # whole batch finished with nothing -- fall through
      sleep 1
    done
    if [ -n "$WINNER" ]; then
      # Reap the WINNER properly before reading its file: the .factor redirect creates the file
      # before python writes to it, so `-s` can fire on a partially written line. Waiting on just
      # that one pid costs nothing (it is at its exit) and makes the read race-free.
      for _i in "${!wdeps[@]}"; do
        [ "${wdeps[$_i]}" = "$WINNER" ] && { wait "${wpids[$_i]}" 2>/dev/null || true; }
      done
      # Stop the siblings -- their verdict cannot change the answer and each is holding NT/SQRT_WIN
      # cores. `kill $p` alone would only take out the sqrt_one subshell and orphan the cado_sqrt
      # child it is blocked on, so kill the children first.
      for p in "${wpids[@]}"; do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done
      wait 2>/dev/null || true
    fi
    for dd in "${wdeps[@]}"; do
      F=$(cat "$DEPPFX.win$dd.factor" 2>/dev/null)
      if [ -n "$F" ]; then
        rm -f "$DEPPFX".pre* "$DEPPFX".pre*.part "$DEPPFX".win*.factor 2>/dev/null || true
        Q=$("$PY" -c "print($N//$F)")
        { echo "factor: $F"; echo "factor: $Q"; } >> "$LOG"
        echo "[cado-sqrt] FACTOR on dep $dd: p=$F"
        exit 0
      fi
      rm -f "$DEPPFX.pre$dd" 2>/dev/null || true
    done
    d=$((d+SQRT_WIN))
  done
  rm -f "$DEPPFX".pre* "$DEPPFX".pre*.part "$DEPPFX".win*.factor 2>/dev/null || true
  echo "[cado-sqrt] no factor found in deps 1-$MAXDEP (concurrent window=$SQRT_WIN)"
  exit 1
fi

for d in $(seq 1 "$MAXDEP"); do
  rm -f "$DEPPFX.000"
  if [ "$PREDUMP" -gt 0 ]; then
    predump_start "$d"                       # no-op if already queued/complete
    # Wait for THIS dependency only (never a blanket `wait`, which would serialise the window).
    # Bounded so a dumper that dies cannot hang the pipeline: fall back to a direct dump.
    _w=0
    while [ ! -e "$DEPPFX.pre$d" ] && [ "$_w" -lt 600 ]; do sleep 1; _w=$((_w+1)); done
    if [ -e "$DEPPFX.pre$d" ]; then
      mv -f "$DEPPFX.pre$d" "$DEPPFX.000" 2>/dev/null || true
    else
      echo "[cado-sqrt] dep $d: pre-dump timed out, dumping inline"
      DUMP_AB_FILE="$DEPPFX.000" OMP_NUM_THREADS=1 "$MDUMP" -nc3 "$d,$d" \
          -s "$S" -nf "$FB" -l "$LOG.dump" -v "$N" >/dev/null 2>&1 || true
    fi
    predump_start "$((d+PREDUMP))"           # keep the look-ahead window full
  else
    DUMP_AB_FILE="$DEPPFX.000" OMP_NUM_THREADS=1 "$MDUMP" -nc3 "$d,$d" \
        -s "$S" -nf "$FB" -l "$LOG.dump" -v "$N" >/dev/null 2>&1 || true
  fi
  [ -s "$DEPPFX.000" ] || { echo "[cado-sqrt] dep $d: not a square, skip"; continue; }
  echo "[cado-sqrt] dep $d: CADO multi-core sqrt (OMP=$NT) $(date +%T)"
  OMP_NUM_THREADS="$NT" "$CSQRT" -poly "$POLY" -prefix "$DEPPFX" -dep 0 \
      -side0 -side1 -gcd >"$LOG.cado" 2>&1 || true
  F=$("$PY" - "$N" "$LOG.cado" <<'PYEOF'
import sys,re
N=int(sys.argv[1])
for line in open(sys.argv[2]):
    s=line.strip()
    if re.fullmatch(r"[0-9]+", s):
        f=int(s)
        if 1<f<N and N%f==0:
            print(f); break
PYEOF
)
  if [ -n "$F" ]; then
    rm -f "$DEPPFX".pre* "$DEPPFX".pre*.part 2>/dev/null || true   # drop unused pre-dumps (~108 MB each)
    Q=$("$PY" -c "print($N//$F)")
    { echo "factor: $F"; echo "factor: $Q"; } >> "$LOG"
    echo "[cado-sqrt] FACTOR on dep $d: p=$F"
    exit 0
  fi
  echo "[cado-sqrt] dep $d: trivial gcd, next"
done
rm -f "$DEPPFX".pre* "$DEPPFX".pre*.part 2>/dev/null || true
echo "[cado-sqrt] no factor found in deps 1-$MAXDEP"
exit 1
