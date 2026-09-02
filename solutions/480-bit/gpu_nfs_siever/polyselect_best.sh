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

set -u
N="$1"; WORK="${2:-/tmp/polysel}"
SIEVER=$(cd "$(dirname "$0")" && pwd)
MSV="${MSV_DIR:-$(cd "$SIEVER/../msieve-gpu" 2>/dev/null && pwd || echo /root/msieve-gpu)}"

MIN_EVALUE="${MIN_EVALUE:-8e-12}"
FINE_HI="${FINE_HI:-32000}"
FINE_W="${FINE_W:-2000}"
COARSE_HI="${COARSE_HI:-200000}"
COARSE_N="${COARSE_N:-0}"
SLICE_TMO="${SLICE_TMO:-600}"
POLY_TOPK="${POLY_TOPK:-3}"
PROBE_WINS="${PROBE_WINS:-2400000:2415000 30000000:30015000 80000000:80015000}"

rm -rf "$WORK"; mkdir -p "$WORK"; cd "$MSV"
echo "### polyselect_best: parallel wide-range search  $(date +%T)"

idx=0
launch() {
  local lo=$1 hi=$2 d; d=$(printf "%s/inst%02d" "$WORK" "$idx"); mkdir -p "$d"
  timeout "$SLICE_TMO" ./msieve -g 0 -np "min_evalue=$MIN_EVALUE $lo,$hi" \
     -s "$d/p.dat" -nf "$d/msieve.fb" -l "$d/p.log" "$N" >"$d/out.txt" 2>&1 &
  idx=$((idx+1))
}
lo=1; while [ "$lo" -lt "$FINE_HI" ]; do hi=$((lo+FINE_W)); [ "$hi" -gt "$FINE_HI" ] && hi=$FINE_HI; launch "$lo" "$hi"; lo=$hi; done
if [ "$COARSE_N" -gt 0 ]; then
  cw=$(( (COARSE_HI-FINE_HI)/COARSE_N )); lo=$FINE_HI
  for j in $(seq 1 "$COARSE_N"); do hi=$((lo+cw)); [ "$j" = "$COARSE_N" ] && hi=$COARSE_HI; launch "$lo" "$hi"; lo=$hi; done
fi
echo "### launched $idx slices, waiting"; wait
echo "### search done $(date +%T); pooling + top-$POLY_TOPK by E"

python3 "$SIEVER/pick_topk.py" "$N" "$WORK" "$POLY_TOPK" "$WORK"/inst*/p.dat.p

echo "### yield bake-off $(date +%T)"
best=-1; bestf=""
for k in $(seq 0 $((POLY_TOPK-1))); do
  f="$WORK/cand$k.cado"; [ -f "$f" ] || continue
  tot=0; np=0
  for win in $PROBE_WINS; do
    qa=${win%:*}; qb=${win#*:}
    pr=$(cd "$SIEVER" && timeout 150 ./gpu_loop "$f" "$qa" "$qb" /dev/null 0 2>/dev/null \
         | grep -oE 'avg [0-9.]+ rel/lattice' | grep -oE '[0-9.]+' | head -1)
    [ -z "$pr" ] && pr=0; tot=$(awk -v t="$tot" -v r="$pr" 'BEGIN{print t+r}'); np=$((np+1))
  done
  r=$(awk -v t="$tot" -v n="$np" 'BEGIN{printf "%.1f",(n>0)?t/n:0}')
  c5=$(awk '/^c5/{print $2}' "$f"); echo "  cand$k (c5=$c5): $r rel/lat"
  awk -v a="$r" -v b="$best" 'BEGIN{exit !(a>b)}' && { best="$r"; bestf="$f"; }
done
[ -z "$bestf" ] && bestf="$WORK/cand0.cado"
cp "$bestf" "$WORK/poly.cado"
echo "### bake-off winner: $(basename "$bestf") at $best rel/lat -> $WORK/poly.cado"

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
