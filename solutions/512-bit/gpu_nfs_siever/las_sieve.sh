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

# las_sieve.sh -- run CADO's `las` as a concurrent CPU siever alongside gpu_loop.
#
# The sieve is GPU-bound, so ~20 of the 24 cores would otherwise idle for the whole run. las on
# those cores adds real relations; taking them costs the GPU well under 1 %.
# Note its RAW rate overstates its value: ~75 % of what it finds the GPU re-finds anyway from the
# relation's other large prime, so its unique contribution is roughly a quarter of its output.
#
# ONE HARD CONSTRAINT: las MUST NOT write into gpu_loop's relation file. gpu_loop uses buffered
# stdio, so two writers interleave partial buffers and corrupt relations with nothing in any log
# to say so. las gets its OWN file; the caller concatenates. Do not 'simplify' this.
#
# DISJOINT q RANGES: las takes [QMIN, QSPLIT) and gpu_loop starts at QSPLIT, so neither re-finds
# the other's special-q. las gets the LOW end, where its fixed per-special-q cost amortises best.
#
# usage: las_sieve.sh <poly.cado> <qmin> <qsplit> <outfile> [logfile]
set -u
POLY="${1:?usage: las_sieve.sh <poly.cado> <qmin> <qsplit> <out> [log]}"
QMIN="${2:?}"; QSPLIT="${3:?}"; OUT="${4:?}"; LOG="${5:-/dev/null}"
SIEVER=$(cd "$(dirname "$0")" && pwd)
CADOBIN="${CADOBIN:-$(cd "$SIEVER/../cado-filter" 2>/dev/null && pwd || echo /nonexistent)}"

LAS="$CADOBIN/las"; MAKEFB="$CADOBIN/makefb"
# Class B degrade (Pipeline.md §12): the run is still correct GPU-only, just ~9 % slower. Say so
# loudly -- a silent skip here is a 1,400 s regression with nothing in the log to explain it.
if [ ! -x "$LAS" ] || [ ! -x "$MAKEFB" ]; then
  echo "  [las] WARNING: $LAS / $MAKEFB missing -- CPU siever disabled, GPU-only (~9% slower)" >&2
  exit 0
fi
[ -s "$POLY" ] || { echo "  [las] WARNING: no polynomial at $POLY -- CPU siever disabled" >&2; exit 0; }
[ "$QSPLIT" -gt "$QMIN" ] 2>/dev/null || { echo "  [las] WARNING: empty range [$QMIN,$QSPLIT) -- disabled" >&2; exit 0; }

# Must match build.sh, or las and gpu_loop are not sieving the same problem and their relations
# cannot be pooled. -A 26 = 2^26 cells = gpu_loop's 8192x8192 region; -sqside 1 = algebraic
# special-q (rational special-q measured -46 % at c154, FINDINGS §5e).
LAS_LIM="${LAS_LIM:-40000000}"
LAS_LPB="${LAS_LPB:-29}"
LAS_MFB="${LAS_MFB:-58}"
LAS_A="${LAS_A:-26}"          # -A 28 measured 61.1 rel/sq but only 317 rel/s; 26 is las's optimum
LAS_T="${LAS_T:-20}"          # gpu_loop runs at SPEC_SIEVE_OMP=4, so 20 cores are free
LAS_PIN="${LAS_PIN:-4-23}"

FB1="${LAS_FB1:-${TMPDIR:-/tmp}/las.fb1.$$}"
# makefb output is ~35 MB and lands on the 10 GiB tmpfs; it is deleted on exit.
trap 'rm -f "$FB1"' EXIT

echo "  [las] makefb side 1 lim=$LAS_LIM $(date +%T)"
if ! taskset -c "$LAS_PIN" "$MAKEFB" -poly "$POLY" -side 1 -lim "$LAS_LIM" -maxbits 14 \
        -t "$LAS_T" -out "$FB1" >>"$LOG" 2>&1; then
  echo "  [las] WARNING: makefb failed -- CPU siever disabled, GPU-only" >&2
  exit 0
fi

echo "  [las] sieving q=[$QMIN,$QSPLIT) on cores $LAS_PIN with -t $LAS_T -> $OUT  $(date +%T)"
# --foreground is NOT used here: this script is backgrounded by factor_msv.sh, which kills the
# whole process group on exit, and `las` is a single process with no timeout wrapper.
taskset -c "$LAS_PIN" "$LAS" -poly "$POLY" -fb1 "$FB1" \
    -lim0 "$LAS_LIM" -lim1 "$LAS_LIM" -lpb0 "$LAS_LPB" -lpb1 "$LAS_LPB" \
    -mfb0 "$LAS_MFB" -mfb1 "$LAS_MFB" -A "$LAS_A" \
    -q0 "$QMIN" -q1 "$QSPLIT" -sqside 1 -t "$LAS_T" -out "$OUT" >>"$LOG" 2>&1
rc=$?
# las writes its summary to the TAIL OF THE RELATION FILE, not to stdout or the log.
n=$(grep -cv '^#' "$OUT" 2>/dev/null || echo 0)
echo "  [las] done rc=$rc, $n relations  $(date +%T)"
[ "$rc" = 0 ] || echo "  [las] WARNING: las exited $rc -- keeping the $n relations it did produce" >&2
exit 0
