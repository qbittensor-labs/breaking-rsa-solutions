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

# msieve's sqrt refuses a negative leading algebraic coefficient, and msieve_dump IS msieve -nc3,
# so it exits in 0.2 s having written a ZERO-BYTE dump -- which sqrt_one() below reports as
# "not a square". Every dependency then fails and the run prints NO FACTOR (insufficient
# relations), naming the one thing that was not wrong. CADO polyselect emits negative c5 routinely
# and the bake-off ranks on yield alone, so this fires on roughly one run in three.
#
# Negating every algebraic coefficient is exact: -f has the same roots, the same ideals and the
# same relations; only the norm's sign flips, and N | f(m) => N | -f(m).
# The normalised copy is used ONLY by the dumper. $FB itself is left alone because POLY below is
# derived from it and CADO's own sqrt handles c5<0 correctly -- that pairing is the verified one.
FBDUMP="$FB"
_a5=$(awk '$1=="A5"{print $2; exit}' "$FB" 2>/dev/null || echo "")
case "$_a5" in
  -*) FBDUMP="$WD/m.fb.dumpsign"
      if "$PY" - "$FB" "$FBDUMP" <<'PYEOF'
import sys
src,dst=sys.argv[1],sys.argv[2]
out=[]
for line in open(src):
    f=line.split()
    if len(f)==2 and len(f[0])==2 and f[0][0]=="A" and f[0][1].isdigit():
        out.append("%s %d" % (f[0], -int(f[1])) + chr(10))
    else:
        out.append(line)
open(dst,"w").writelines(out)
PYEOF
      then
        echo "[cado-sqrt] leading algebraic coefficient is NEGATIVE ($_a5) -- msieve -nc3 refuses it;"
        echo "[cado-sqrt]   dumping against a sign-normalised copy of the factor base (relations are unchanged)."
      else
        echo "[cado-sqrt] WARNING: could not sign-normalise $FB -- dumps will be EMPTY and every" >&2
        echo "[cado-sqrt]   dependency will report 'not a square'. THIS IS NOT A RELATION SHORTFALL." >&2
        FBDUMP="$FB"
      fi ;;
esac

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
# PIPELINED DUMPS. Each dep costs ~27 s of single-threaded dump plus ~71 s of parallel sqrt. The
# dumps are independent, so pre-fetching them takes the dump off the critical path for every dep
# after the first: 4x(27+71)=392 s serial vs 27+4x71=311 s. Never worse -- dep 1 still starts now.
#
# PREDUMP MUST TRACK SQRT_WIN. Pinned to a constant while SQRT_WIN grew, the window launched six
# concurrent sqrts against four prefetched dumps and deps 5-6 each blocked ~25 s on a dump the
# window had only just started -- re-introducing the stagger on exactly the tail draw the window
# exists to remove. The floor of 4 keeps the look-ahead on the serial path (SQRT_WIN=1).
PREDUMP="${CADO_PREDUMP:-}"        # how many dependency dumps to run ahead; 0 disables
if [ -z "$PREDUMP" ]; then
  PREDUMP=4
  # Guard the comparison: `[ abc -gt 4 ]` is a bash usage error, and CADO_SQRT_WIN is caller-supplied.
  case "${CADO_SQRT_WIN:-}" in
    ''|*[!0-9]*) : ;;
    *) [ "$CADO_SQRT_WIN" -gt "$PREDUMP" ] && PREDUMP="$CADO_SQRT_WIN" || true ;;
  esac
  # ...and match the SQRT_WIN DEFAULT (6) when the caller sets nothing at all. The check next to
  # SQRT_WIN itself WARNS if the two ever disagree again, so the drift shows up in container stdout
  # instead of as an unexplained ~44s at the end of the run.
  [ -z "${CADO_SQRT_WIN:-}" ] && PREDUMP=6
  # `|| true` is LOAD-BEARING, same class as the two set -e incidents in Pipeline.md 12C: the line
  # above is the last statement of this block, so when CADO_SQRT_WIN IS set it evaluates false and
  # the whole `if` exits non-zero. Harmless while this script is set -u only, fatal the day anyone
  # adds set -e -- and one of those incidents scored a WRONG ANSWER for a tooling failure.
  true
fi
# !! COMPLETION MUST BE SIGNALLED BY A SENTINEL, NOT BY THE DUMP FILE EXISTING !!
# msieve creates DUMP_AB_FILE when it OPENS it, so testing `-e <dumpfile>` would happily hand a
# HALF-WRITTEN dump to the sqrt -- a silent-corruption race. Each pre-dump therefore writes to
# <prefix>.pre<d>.part and only renames to <prefix>.pre<d> after the dumper exits; the reader waits
# on the renamed name, which cannot appear early because rename(2) is atomic.
predump_start() {   # $1 = dep index
  local d="$1"
  [ "$d" -gt "$MAXDEP" ] && return 0
  [ -e "$DEPPFX.pre$d" ] || [ -e "$DEPPFX.pre$d.part" ] && return 0
# Claim the slot BEFORE forking. .part used to be created inside the subshell by msieve, so two
# calls for the same dep could both pass the guard; the loser's mv then found no .part and
# truncated a complete dump to 0 bytes, which reads as "not a square". Silent dep loss.
  : > "$DEPPFX.pre$d.part"
  (
    DUMP_AB_FILE="$DEPPFX.pre$d.part" OMP_NUM_THREADS=1 "$MDUMP" -nc3 "$d,$d" \
        -s "$S" -nf "$FBDUMP" -l "$LOG.dump$d" -v "$N" >/dev/null 2>&1 || true
    # rename only on completion; an empty .part means "dep d is not a square" -> leave a 0-byte
    # sentinel so the reader learns that instead of waiting forever.
    # The `[ -e ... ]` arm is load-bearing: NEVER truncate a pre$d that already exists. Even with
    # the claim above, any future path that races here must degrade to a no-op, not to data loss.
    mv -f "$DEPPFX.pre$d.part" "$DEPPFX.pre$d" 2>/dev/null \
      || [ -e "$DEPPFX.pre$d" ] || : > "$DEPPFX.pre$d"
  ) &
}
# STORAGE. $WD is the validator's 10 GiB /tmp tmpfs. Live dumps at the widest point are
# PREDUMP+SQRT_WIN plus sqrt_one's own copies: ~18 files, ~2.0 GB at the defaults. Headroom is
# fine here -- the tmpfs high-water mark is during the pre-dedup warm-up, long before sqrt.
# Raising PREDUMP past SQRT_WIN buys nothing: the window consumes only SQRT_WIN dumps at a time.
for k in $(seq 1 "$PREDUMP"); do predump_start "$k"; done

# ---------------------------------------------------------------------------------------------
# CONCURRENT SQRT WINDOW. Dependencies are independent and one CADO sqrt does not saturate 24
# cores, so a small window converts draw luck into a fixed cost: the answer arrives when the FIRST
# successful dep finishes, not when every failure ahead of it has been retired in order.
# Serial cost is set by luck -- measured 165 s / 260 s / 443 s on the same matrix depending on
# which dep hit. Expected value is the wrong statistic against a hard deadline; the 443 s draw
# landed on the one run whose margin was 14 s. SQRT_WIN=1 restores strictly-serial behaviour.
#
# A historical 'all 16 deps not a square' report was the SEPARATE missing-m.dat.lp bug on the
# cache-HIT path, not a fault in this window -- msieve_dump -nc3 needs the large-prime file.
SQRT_WIN="${CADO_SQRT_WIN:-6}"
# Warn, never abort: a stray env var must not kill a 4 h run at its last stage. The staggered
# path still gives the right answer, just ~25 s per un-prefetched dep slower.
if { [ "$PREDUMP" -gt 0 ] && [ "$PREDUMP" -lt "$SQRT_WIN" ]; } 2>/dev/null; then
  echo "[cado-sqrt] WARNING: CADO_PREDUMP=$PREDUMP < CADO_SQRT_WIN=$SQRT_WIN --" \
       "deps $((PREDUMP+1))..$SQRT_WIN will each block ~22-27s on their dump (Configuration.md 5.4)."
fi
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
          -s "$S" -nf "$FBDUMP" -l "$LOG.dump" -v "$N" >/dev/null 2>&1 || true
    fi
    predump_start "$((d+PREDUMP))"           # keep the look-ahead window full
  else
    DUMP_AB_FILE="$DEPPFX.000" OMP_NUM_THREADS=1 "$MDUMP" -nc3 "$d,$d" \
        -s "$S" -nf "$FBDUMP" -l "$LOG.dump" -v "$N" >/dev/null 2>&1 || true
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
