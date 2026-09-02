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

set -e
T_START=$(date +%s)
N="$1"; WORK="${2:-/tmp/fk}"
SIEVER=$(cd "$(dirname "$0")" && pwd)
MSV="${MSV_DIR:-$(cd "$SIEVER/../msieve-gpu" 2>/dev/null && pwd || echo /root/msieve-gpu)}"
POLYSELECT_SECS="${POLYSELECT_SECS:-540}"
ADMAX="${ADMAX:-200000}"
QMIN=2400000; QMAX="${QMAX:-220000000}"; REL_TARGET="${REL_TARGET:-96350000}"
GNFS_WALL="${GNFS_WALL:-14400}"; DOWNSTREAM_RESERVE="${DOWNSTREAM_RESERVE:-1920}"
rm -rf "$WORK"; mkdir -p "$WORK"
CACHE=""
if [ -n "${GNFS_CACHE:-}" ]; then
  NHASH=$(printf '%s' "$N" | md5sum | cut -c1-12)
  CACHE="$GNFS_CACHE/$NHASH"; mkdir -p "$CACHE"
  echo "### STAGE CACHE on: $CACHE  (N hash $NHASH) — completed stages are reused"
fi

echo "### [1/4] msieve GPU polyselect (parallel wide-range search) + yield bake-off $(date +%T)"
if [ -n "$CACHE" ] && [ -s "$CACHE/poly.cado" ] && [ -s "$CACHE/c.fb" ]; then
  echo "  [cache HIT] poly.cado + c.fb — skipping polyselect"; cp "$CACHE/poly.cado" "$CACHE/c.fb" "$WORK/"
else
  COARSE_N=0 bash "$SIEVER/polyselect_best.sh" "$N" "$WORK"
  if [ -n "$CACHE" ]; then cp "$WORK/poly.cado" "$WORK/c.fb" "$CACHE/" 2>/dev/null && echo "  [cache SAVE] poly.cado + c.fb" || true; fi
fi

NOW=$(date +%s); ELAPSED=$(( NOW - T_START ))
SIEVE_SECS=$(( GNFS_WALL - ELAPSED - DOWNSTREAM_RESERVE ))
[ "$SIEVE_SECS" -lt 600 ] && SIEVE_SECS=600
echo "### [2/4] GPU lattice sieve -> $REL_TARGET relations (q<=$QMAX, wall<=${SIEVE_SECS}s, reserve ${DOWNSTREAM_RESERVE}s) $(date +%T)"
RELPID=""
if [ -n "$CACHE" ] && [ -s "$CACHE/rels.txt" ]; then
  echo "  [cache HIT] rels.txt — skipping sieve"; cp "$CACHE/rels.txt" "$WORK/rels.txt"
else
  python3 "$SIEVER/relhold.py" "$WORK/rels.txt" & RELPID=$!
  for i in $(seq 1 100); do [ -L "$WORK/rels.txt" ] && break; sleep 0.1; done
  cd "$SIEVER"
  for attempt in 1 2 3; do
    GPULOOP_MAX_SECS="$SIEVE_SECS" ./gpu_loop "$WORK/poly.cado" $QMIN $QMAX "$WORK/rels.txt" $REL_TARGET && rc=0 || rc=$?
    nrel=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
    [ "$rc" = 0 ] && [ "$nrel" -gt 1000 ] && break
    echo "  sieve attempt $attempt: rc=$rc nrel=$nrel — retrying after GPU re-probe"
    python3 -c "import cupy; cupy.zeros(1); cupy.cuda.runtime.deviceSynchronize()" >/dev/null 2>&1 || sleep 5
  done
  if [ -n "$CACHE" ]; then cp "$WORK/rels.txt" "$CACHE/rels.txt" 2>/dev/null && echo "  [cache SAVE] rels.txt ($(wc -l < "$CACHE/rels.txt") relations)" || true; fi
fi
echo "  relations: $(wc -l < "$WORK/rels.txt")"

echo "### [3/4] memfd-backed msieve downstream (filter + in-RAM Lanczos + sqrt) $(date +%T)"
cd "$MSV"
GNFS_CACHE_DS="$CACHE" ./msv_launcher "$WORK/rels.txt" "$WORK/c.fb" "$WORK/ds" ./msvrun_gpu.sh "$N" "$WORK/ds"
[ -n "$RELPID" ] && kill "$RELPID" 2>/dev/null || true
if [ -f "$WORK/ds/msieve.log" ]; then
  grep -aE 'unique relations|keeping .* ideals|begin with|reduce to|filtering wants|matrix is' "$WORK/ds/msieve.log" | tail -6 | sed 's/^/  [filter] /'
fi

echo "### [4/4] FACTORS $(date +%T)"
python3 - "$N" "$WORK/ds/msieve.log" <<'PY'
import sys,re
N=int(sys.argv[1])
fs=set()
for m in re.findall(r'factor:\s*([0-9]+)', open(sys.argv[2]).read()):
    f=int(m)
    if 1<f<N and N%f==0: fs.add(f)
if fs:
    p=min(fs); q=N//p
    print(f"p={p}")
    print(f"q={q}")
    print(f"FACTORED: p*q==N {p*q==N}")
else:
    print("NO FACTOR (insufficient relations / matrix)")
PY
