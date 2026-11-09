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

# polytest160.sh -- GPU test-sieve of the ropt shortlist, pick by MODELLED sieve time (polysel160.py pick).
#
# MurphyE mis-ranks real yield by 0-16% (RSA-240/DLP-240/RSA-260 all test-sieved; measured here on key 3: the
# MurphyE-top of a 5.3k-candidate pool sieves 2.9-6% slower than its #2). The GPU is idle until this stage
# ends, so the cost (~TS_K x 2 x ~13 s) is on the critical path: the pick only moves away from the MurphyE
# poly for a modelled gain >= TS_MIN_GAIN. Doubles as a norm-width check (a candidate with skipped
# special-q is excluded; the MurphyE pick showing skips is reported LOUDLY).
# NEVER fatal: any failure keeps the MurphyE pick already in <poly.cado>. Exit 0 always.
#
# usage: polytest160.sh <work> <N> <ropt.out> <poly.cado (in/out)> <CS> <HERE> RLIM ALIM LPB MFB0 MFB1 LOGI JJ QMIN -- <bench args...>
set -u
WORK=$1; N=$2; ROPT=$3; POLY=$4; CS=$5; HERE=$6; RLIM=$7; ALIM=$8; LPB=$9; MFB0=${10}; MFB1=${11}; LOGI=${12}; JJ=${13}; QMIN=${14}
shift 14; [ "${1:-}" = "--" ] && shift
TS_K=${TS_K:-5}; TS_W=${TS_W:-6000}; TS_MIN_GAIN=${TS_MIN_GAIN:-0.01}; TS_MAX_SECS=${TS_MAX_SECS:-240}
TS_Q1=${TS_Q1:-$(( QMIN * 14 / 10 ))}; TS_Q2=${TS_Q2:-$(( QMIN * 32 / 10 ))}
TS="$WORK/ts"; mkdir -p "$TS" || exit 0
T0=$(date +%s)
python3 "$HERE/polysel160.py" topn "$ROPT" "$N" "$TS_K" "$TS/c" > "$TS/topn.txt" 2>&1 || { echo "  [polytest] topn failed: $(tail -1 "$TS/topn.txt") -- keeping the MurphyE pick"; exit 0; }
cmp -s <(grep -E '^(c[0-5]|Y[01]|skew):' "$TS/c0.cado") <(grep -E '^(c[0-5]|Y[01]|skew):' "$POLY") \
  || { echo "  [polytest] c0 differs from the MurphyE pick -- keeping the MurphyE pick"; exit 0; }
: > "$TS/results"
n=0
while read -r f e_lbl E; do
  b=${f%.cado}; name=$(basename "$b")
  [ $(( $(date +%s) - T0 )) -lt "$TS_MAX_SECS" ] || { echo "  [polytest] time budget ${TS_MAX_SECS}s reached after $n candidates"; break; }
  python3 "$HERE/polysel160.py" job "$f" "$b.job" "$RLIM" "$ALIM" "$LPB" "$LPB" "$MFB0" "$MFB1" || continue
  # HARD budget: every GPU step runs under timeout(remaining), so a hung fbgen/bench cannot stall the run
  rem=$(( TS_MAX_SECS - ($(date +%s) - T0) )); [ "$rem" -ge 5 ] || { echo "  [polytest] time budget ${TS_MAX_SECS}s reached after $n candidates"; break; }
  timeout -k 5 "$rem" "$CS/fbgen_gpu" --poly "$b.job" --lim "$ALIM" --maxbits "$LOGI" --out "$b.fb" > "$b.fbgen.log" 2>&1 || { echo "  [polytest] $name: fbgen failed (rc $?)"; continue; }
  for q in "$TS_Q1" "$TS_Q2"; do
    rem=$(( TS_MAX_SECS - ($(date +%s) - T0) )); [ "$rem" -ge 5 ] || { echo "  [polytest] $name: time budget reached before q=$q"; break; }
    timeout -k 5 "$rem" "$CS/bench" --pipeline --cofactor --fill-concurrent --poly "$b.job" --fb1 "$b.fb" --logI "$LOGI" --J "$JJ" --sq-side 1 \
        --qrange "$q:$(( q + TS_W - 1 ))" "$@" --relations "$b.q$q.rel" > "$b.q$q.log" 2>&1
    nq=$(grep -aoE 'band of [0-9]+ special-q' "$b.q$q.log" | grep -oE '[0-9]+' | head -1)
    rq=$(grep -aE 'ALL RELATIONS/q' "$b.q$q.log" | awk '{print $NF}')
    # GPU ms per special-q WITHOUT the in-loop cofactorisation flushes: production runs --candidates (the CPU splits
    # them), and the flush count per band is quantized (1 vs 2 flushes = +-1 ms/q between candidates: an artefact)
    ms=$(awk '/^ *wall clock per q  /{w=$(NF-1)} /cofactorisation, in-loop flushes/{f=$(NF-1)} END{if(w!="") printf "%.3f", w-(f==""?0:f)}' "$b.q$q.log")
    sk=$(grep -ac 'skipped for norm width' "$b.q$q.log")
    [ -n "$nq" ] && [ -n "$rq" ] && [ -n "$ms" ] || { echo "  [polytest] $name q=$q: no band summary ($(tail -1 "$b.q$q.log" | cut -c1-80))"; continue; }
    echo "$name $E $q $TS_W $nq $rq $ms $sk" >> "$TS/results"
    [ "$name" = c0 ] && [ "$sk" != 0 ] && echo "  [polytest] WARNING: the MurphyE pick SKIPS special-q for norm width at q=$q (bench BN_LIMBS too narrow for it)"
    rm -f "$b.q$q.rel" "$b.q$q.rel".part "$b.q$q.rel".part.ckpt "$b.q$q.rel".lock
  done
  rm -f "$b.fb"; n=$((n+1))
done < "$TS/topn.txt"
python3 "$HERE/polysel160.py" pick "$TS/results" c0 "$TS_MIN_GAIN" > "$TS/pick.txt" 2>&1 || { echo "  [polytest] pick failed: $(tail -1 "$TS/pick.txt") -- keeping the MurphyE pick"; exit 0; }
sed 's/^/  [polytest]/' "$TS/pick.txt"
W=$(grep -a '^PICK' "$TS/pick.txt" | awk '{print $2}')
if [ -n "$W" ] && [ "$W" != c0 ] && [ -s "$TS/$W.cado" ]; then
  cp -f "$TS/$W.cado" "$POLY.ts" && mv -f "$POLY.ts" "$POLY" && echo "  [polytest] switched to $W ($(grep -E '^c5:' "$POLY")) in $(( $(date +%s) - T0 ))s"
else
  echo "  [polytest] keeping the MurphyE pick (c0) in $(( $(date +%s) - T0 ))s"
fi
exit 0
