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
N="$1"; S="$2"; FB="$3"; LOG="$4"; NT="${5:-24}"
HERE=$(cd "$(dirname "$0")" && pwd)
MDUMP="${MDUMP:-$HERE/msieve_dump}"
CSQRT="${CSQRT:-$HERE/cado_sqrt}"
PY="${PYTHON:-python3}"
MAXDEP="${CADO_MAXDEP:-16}"
WD=$(dirname "$S"); POLY="$WD/poly_cado.txt"; DEPPFX="$WD/cadodep"

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

for d in $(seq 1 "$MAXDEP"); do
  rm -f "$DEPPFX.000"
  DUMP_AB_FILE="$DEPPFX.000" OMP_NUM_THREADS=1 "$MDUMP" -nc3 "$d,$d" \
      -s "$S" -nf "$FB" -l "$LOG.dump" -v "$N" >/dev/null 2>&1 || true
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
    Q=$("$PY" -c "print($N//$F)")
    { echo "factor: $F"; echo "factor: $Q"; } >> "$LOG"
    echo "[cado-sqrt] FACTOR on dep $d: p=$F"
    exit 0
  fi
  echo "[cado-sqrt] dep $d: trivial gcd, next"
done
echo "[cado-sqrt] no factor found in deps 1-$MAXDEP"
exit 1
