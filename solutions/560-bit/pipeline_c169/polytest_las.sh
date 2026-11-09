#!/usr/bin/env bash
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

# polytest_las.sh -- test-sieve the TOPN best root-optimized polynomials with las or glas and pick by
# MODELLED sieve time (polysel160.py pick), the las/glas counterpart of the 530 polytest160.sh (cuda-sieve).
#
# Why: MurphyE mis-ranks real yield by 0-16% (RSA-240/DLP-240/RSA-260 all test-sieved; the 530 team measured
# -2.9..-6% sieve time from re-ranking the top 5 at c160). The pick only moves away from the MurphyE poly
# for a modelled gain >= TS_MIN_GAIN. NEVER fatal: any failure keeps the MurphyE pick. Exit 0 always.
#
# usage: polytest_las.sh <work> <N> <ropt.out> <poly.cado (in/out)> <siever: las|glas> <sieverbin> <makefb> <HERE>
#                        RLIM ALIM LPB0 LPB1 MFB0 MFB1 LAMBDA0 LAMBDA1 NCURVES0 NCURVES1 A QMIN THREADS
# env: TS_K (5) TS_W (q-width per window, 3000) TS_MIN_GAIN (0.01) TS_MAX_SECS (600 CPU / 300 GPU)
#      TS_Q1/TS_Q2 (window starts; default 1.4 x QMIN and 3.2 x QMIN) TS_RELS (relation target for the
#      model, default 155e6) TS_QMIN (model start q, default QMIN)
set -u
WORK=$1; N=$2; ROPT=$3; POLY=$4; SIEVER=$5; SBIN=$6; MAKEFB=$7; HERE=$8
RLIM=$9; ALIM=${10}; LPB0=${11}; LPB1=${12}; MFB0=${13}; MFB1=${14}; LAM0=${15}; LAM1=${16}; NC0=${17}; NC1=${18}; A=${19}; QMIN=${20}; THREADS=${21}
TS_K=${TS_K:-5}; TS_W=${TS_W:-3000}; TS_MIN_GAIN=${TS_MIN_GAIN:-0.01}
TS_MAX_SECS=${TS_MAX_SECS:-$([ "$SIEVER" = glas ] && echo 300 || echo 600)}
TS_Q1=${TS_Q1:-$(( QMIN * 14 / 10 ))}; TS_Q2=${TS_Q2:-$(( QMIN * 32 / 10 ))}
TS_RELS=${TS_RELS:-155e6}; TS_QMIN=${TS_QMIN:-$QMIN}
TS="$WORK/ts"; mkdir -p "$TS" || exit 0
T0=$(date +%s); left(){ echo $(( TS_MAX_SECS - ($(date +%s) - T0) )); }
python3 "$HERE/polysel160.py" topn "$ROPT" "$N" "$TS_K" "$TS/c" > "$TS/topn.txt" 2>&1 || { echo "  [polytest] topn failed: $(tail -1 "$TS/topn.txt") -- keeping the MurphyE pick"; exit 0; }
cmp -s <(grep -E '^(c[0-5]|Y[01]|skew):' "$TS/c0.cado") <(grep -E '^(c[0-5]|Y[01]|skew):' "$POLY") \
  || { echo "  [polytest] c0 differs from the MurphyE pick -- keeping the MurphyE pick"; exit 0; }
EXTRA=(); [ "$SIEVER" = glas ] && EXTRA=(-gpu)
: > "$TS/results"; n=0
while read -r f e_lbl E; do
  b=${f%.cado}; name=$(basename "$b")
  [ "$(left)" -ge 10 ] || { echo "  [polytest] time budget ${TS_MAX_SECS}s reached after $n candidates"; break; }
  "$MAKEFB" -poly "$f" -side 1 -lim "$ALIM" -maxbits "$(( (A + 1) / 2 ))" -t "$THREADS" -out "$b.fb1" > "$b.makefb.log" 2>&1 \
    || { echo "  [polytest] $name: makefb failed"; continue; }
  for q in "$TS_Q1" "$TS_Q2"; do
    [ "$(left)" -ge 10 ] || { echo "  [polytest] $name: time budget reached before q=$q"; break; }
    t0=$(date +%s.%N)
    "$SBIN" -poly "$f" -fb1 "$b.fb1" -lim0 "$RLIM" -lim1 "$ALIM" -lpb0 "$LPB0" -lpb1 "$LPB1" -mfb0 "$MFB0" -mfb1 "$MFB1" \
        -lambda0 "$LAM0" -lambda1 "$LAM1" -ncurves0 "$NC0" -ncurves1 "$NC1" -A "$A" -q0 "$q" -q1 "$(( q + TS_W ))" -sqside 1 \
        -t "$THREADS" ${EXTRA[@]+"${EXTRA[@]}"} -out "$b.q$q.rel" > "$b.q$q.log" 2>&1
    S=$(grep -a '^#' "$b.q$q.rel" 2>/dev/null)
    nq=$(echo "$S" | grep -aoE "\(([0-9]+) special-q's\)" | grep -oE '[0-9]+' | paste -sd+ - | bc 2>/dev/null)
    rels=$(grep -avc '^#' "$b.q$q.rel" 2>/dev/null || echo 0)
    el=$(echo "$S" | grep -aoE 'Total elapsed time [0-9.]+' | grep -oE '[0-9.]+$' | head -1)
    [ -n "${nq:-}" ] && [ "${nq:-0}" -gt 0 ] && [ -n "${el:-}" ] || { echo "  [polytest] $name q=$q: no las summary ($(tail -1 "$b.q$q.log" | cut -c1-80))"; rm -f "$b.q$q.rel"; continue; }
    rq=$(python3 -c "print(f'{$rels/$nq:.3f}')"); ms=$(python3 -c "print(f'{1000*$el/$nq:.3f}')")
    # columns polysel160.py pick expects: name E q width nq rel_per_q ms_per_q skipped
    echo "$name $E $q $TS_W $nq $rq $ms 0" >> "$TS/results"
    rm -f "$b.q$q.rel"
  done
  rm -f "$b.fb1"; n=$((n+1))
done < "$TS/topn.txt"
python3 "$HERE/polysel160.py" pick "$TS/results" c0 "$TS_MIN_GAIN" "$TS_RELS" "$TS_QMIN" > "$TS/pick.txt" 2>&1 \
  || { echo "  [polytest] pick failed: $(tail -1 "$TS/pick.txt") -- keeping the MurphyE pick"; exit 0; }
sed 's/^/  [polytest]/' "$TS/pick.txt"
W=$(grep -a '^PICK' "$TS/pick.txt" | awk '{print $2}')
if [ -n "$W" ] && [ "$W" != c0 ] && [ -s "$TS/$W.cado" ]; then
  cp -f "$TS/$W.cado" "$POLY.ts" && mv -f "$POLY.ts" "$POLY" && echo "  [polytest] switched to $W ($(grep -E '^c5:' "$POLY")) in $(( $(date +%s) - T0 ))s"
else
  echo "  [polytest] keeping the MurphyE pick (c0) in $(( $(date +%s) - T0 ))s"
fi
exit 0
