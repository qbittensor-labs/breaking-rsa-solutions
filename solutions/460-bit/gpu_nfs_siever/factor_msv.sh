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

# ============================================================================
# Per-key 460-bit (c140-size) factorizer for the validator sandbox
# (1 GiB writable /tmp, 85 GiB RAM, 24 CPU, GPU, --network none, 4h).
#   N  ->  msieve GPU polyselect  ->  gpu_loop GPU sieve  ->  memfd-backed msieve
#          (filter + in-RAM Lanczos + sqrt)  ->  p, q
# All multi-GB intermediates live in anonymous RAM (memfd); /tmp stays ~100 MB.
# usage: factor_msv.sh <N> [workdir]
# ============================================================================
set -e
T_START=$(date +%s)
N="$1"; WORK="${2:-/tmp/fk}"
SIEVER=$(cd "$(dirname "$0")" && pwd)
# msieve toolchain dir (msieve + cub/sort_engine.so + msv_launcher + msvrun.sh).
# default: sibling 'msieve-gpu' next to the siever dir, else env MSV_DIR.
MSV="${MSV_DIR:-$(cd "$SIEVER/../msieve-gpu" 2>/dev/null && pwd || echo /root/msieve-gpu)}"
POLYSELECT_SECS="${POLYSELECT_SECS:-1080}"      # GPU polyselect budget (~18 min)
ADMAX="${ADMAX:-200000}"
# QMAX is intentionally LARGE: the sieve stops at REL_TARGET (good keys, early) or at the wall-time
# budget (low-yield keys), NOT at the q-range. A small QMAX used to make low-yield keys quit with too
# few relations while budget remained -> doomed downstream. GNFS_WALL = total seconds the GNFS may use
# (the caller's remaining wall budget); DOWNSTREAM_RESERVE is held back for filter+LA+sqrt.
# REL_TARGET=60M with lpb=2^29 (gpu_loop default): lowering the large-prime bound to 2^29 shrinks the
# matrix's unique-ideal bar ~5x (~60M->~12M ideals), so a weak-poly key (e.g. seed-684 @122 rel/lat that
# FAILED at lpb=2^30/85M) now builds a solvable matrix from far fewer relations. At 2^29 the per-lattice
# yield is ~37% lower, so keys wall-cap ~56M raw within budget regardless of this target; 60M is a safe
# cap. VERIFIED: seed-684 (weak) rescued AND seed-99 (good) still factors, both ~3h44m (<4h).
QMIN=2400000; QMAX="${QMAX:-100000000}"; REL_TARGET="${REL_TARGET:-60000000}"
GNFS_WALL="${GNFS_WALL:-14400}"; DOWNSTREAM_RESERVE="${DOWNSTREAM_RESERVE:-1800}"   # ~30min for filter+LA+sqrt (seed-99 took 19min)
rm -rf "$WORK"; mkdir -p "$WORK"

echo "### [1/4] msieve GPU polynomial selection (~$((POLYSELECT_SECS/60)) min) + yield bake-off $(date +%T)"
cd "$MSV"
rm -f msieve.fb 2>/dev/null || true   # clear a stale default fb (dev-box re-runs); harmless on read-only /app (fresh container has none)
# min_evalue lowers the save threshold so even an unlucky run banks plenty of candidates to choose among.
timeout "$POLYSELECT_SECS" ./msieve -g 0 -np "min_evalue=1.5e-11 1,$ADMAX" -s "$WORK/p.dat" -l "$WORK/p.log" "$N" >/dev/null 2>&1 || true
# YIELD BAKE-OFF: emit the top-K candidates by msieve-E, sieve-test each, keep the best measured yield.
# PREDICTIVE PROBE: sieve yield falls ~3x across the production q-range (measured ~174 rel/lat @ q=2.4M
# -> ~55 @ q=95M), so a low-q-only probe over-predicts the full-range mean by ~40% (170->122 in the
# field) and can mis-rank candidates. Probe each candidate at 3 q-points spanning the range and average,
# so the score tracks the real wide-sweep mean (-> correct ranking + honest telemetry for the no-go gate).
POLY_TOPK="${POLY_TOPK:-4}"
PROBE_WINS="${PROBE_WINS:-2400000:2415000 30000000:30015000 80000000:80015000}"
python3 "$SIEVER/pick_topk.py" "$N" "$WORK" "$POLY_TOPK" "$WORK/p.dat.p" || true   # -> $WORK/cand0..cand{K-1}.cado
best=-1; bestf=""
for k in $(seq 0 $((POLY_TOPK-1))); do
  f="$WORK/cand$k.cado"; [ -f "$f" ] || continue
  tot=0; np=0
  for win in $PROBE_WINS; do
    qa=${win%:*}; qb=${win#*:}
    pr=$(cd "$SIEVER" && timeout 120 ./gpu_loop "$f" $qa $qb /dev/null 0 2>/dev/null | grep -oE 'avg [0-9.]+ rel/lattice' | grep -oE '[0-9.]+' | head -1)
    [ -z "$pr" ] && pr=0
    tot=$(awk -v t="$tot" -v r="$pr" 'BEGIN{print t+r}'); np=$((np+1))
  done
  r=$(awk -v t="$tot" -v n="$np" 'BEGIN{printf "%.1f", (n>0)?t/n:0}')
  echo "  cand$k: $r rel/lat (mean over $np q-points)"
  awk -v a="$r" -v b="$best" 'BEGIN{exit !(a>b)}' && { best="$r"; bestf="$f"; }
done
echo "  bake-off mean yield (winner): $best rel/lat"
[ -z "$bestf" ] && bestf="$WORK/cand0.cado"   # fallback: msieve max-E candidate
cp "$bestf" "$WORK/poly.cado"
echo "  bake-off winner: $(basename "$bestf") at $best rel/lat"
# emit msieve .fb (postprocessing) from the chosen poly
python3 - "$N" "$WORK/poly.cado" "$WORK/c.fb" <<'PY'
import sys,re
N=sys.argv[1]; cado=sys.argv[2]; fb=sys.argv[3]
p={}
for line in open(cado):
    mm=re.match(r'(skew|c\d|Y\d):\s*(\S+)', line.strip())
    if mm: p[mm.group(1)]=mm.group(2)
with open(fb,'w') as f:
    f.write(f"N {N}\nSKEW {p['skew']}\nR0 {p['Y0']}\nR1 {p['Y1']}\n")
    for i in range(6): f.write(f"A{i} {p['c'+str(i)]}\n")
print(f"  c.fb written: skew={p['skew']} c5={p['c5']}")
PY

# Wall-time budget for the sieve = remaining GNFS budget minus a downstream reserve. The sieve uses
# this whole window for a low-yield key (gathering as many relations as possible) instead of quitting
# early at a fixed QMAX with budget to spare (the seed-42 failure mode).
NOW=$(date +%s); ELAPSED=$(( NOW - T_START ))
SIEVE_SECS=$(( GNFS_WALL - ELAPSED - DOWNSTREAM_RESERVE ))
[ "$SIEVE_SECS" -lt 600 ] && SIEVE_SECS=600   # floor so a late start still attempts a sieve
echo "### [2/4] GPU lattice sieve -> $REL_TARGET relations (q<=$QMAX, wall<=${SIEVE_SECS}s, reserve ${DOWNSTREAM_RESERVE}s) $(date +%T)"
# Bug-1 fix: the full relation set is ~9 GB but the validator's /tmp is a 1 GiB tmpfs, which would
# silently truncate rels.txt. Back rels.txt with an anonymous memfd (RAM) via relhold.py so the
# sieve output never touches /tmp.  (downstream .lp/.mat already use the msv_launcher memfd broker.)
python3 "$SIEVER/relhold.py" "$WORK/rels.txt" & RELPID=$!
for i in $(seq 1 100); do [ -L "$WORK/rels.txt" ] && break; sleep 0.1; done
cd "$SIEVER"
# Retry on a transient GPU-init miss (cudaErrorNoDevice / 0 relations) — belt-and-suspenders behind the
# breaking_rsa.py keep-alive, for the de-init window before the holder warms. Capped so it can't eat the wall.
for attempt in 1 2 3; do
  GPULOOP_MAX_SECS="$SIEVE_SECS" ./gpu_loop "$WORK/poly.cado" $QMIN $QMAX "$WORK/rels.txt" $REL_TARGET && rc=0 || rc=$?
  nrel=$(wc -l < "$WORK/rels.txt" 2>/dev/null || echo 0)
  [ "$rc" = 0 ] && [ "$nrel" -gt 1000 ] && break
  echo "  sieve attempt $attempt: rc=$rc nrel=$nrel — retrying after GPU re-probe"
  python3 -c "import cupy; cupy.zeros(1); cupy.cuda.runtime.deviceSynchronize()" >/dev/null 2>&1 || sleep 5
done
echo "  relations: $(wc -l < "$WORK/rels.txt")"

echo "### [3/4] memfd-backed msieve downstream (filter + in-RAM Lanczos + sqrt) $(date +%T)"
cd "$MSV"
./msv_launcher "$WORK/rels.txt" "$WORK/c.fb" "$WORK/ds" ./msvrun_gpu.sh "$N" "$WORK/ds"
kill "$RELPID" 2>/dev/null || true   # release the rels memfd
# Surface the filter outcome (unique relations vs ideals / excess / shortfall) so a failure is
# diagnosable from the container log instead of silently discarded.
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
    print(f"p={p}")     # line-start p=/q= for the orchestrator to parse
    print(f"q={q}")
    print(f"FACTORED: p*q==N {p*q==N}")
else:
    print("NO FACTOR (insufficient relations / matrix)")
PY
