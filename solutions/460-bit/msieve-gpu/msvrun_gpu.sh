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

# GPU-accelerated msieve downstream, run under msv_launcher (big files in memfd).
# Flow:  -nc1 filter  ->  -nc2 build matrix (kill before CPU Lanczos)  ->  GPU block-Lanczos  ->  -nc3 sqrt.
# Falls back to stock CPU `msieve -nc` if the GPU path fails for any reason.
# args: <N> <workdir>   (workdir has m.dat[memfd], m.fb; msieve writes m.dat.{cyc,mat[memfd],dep})
set -u
N="$1"; WORK="$2"
HERE=$(cd "$(dirname "$0")" && pwd); M="$HERE/msieve"; cd "$HERE"   # cwd has cub/sort_engine.so
NT="${NT:-24}"; S="$WORK/m.dat"; FB="$WORK/m.fb"; LOG="$WORK/msieve.log"
PY="${PYTHON:-python3}"

gpu_downstream() {
  echo "[gpu-la] 1/4 filter (msieve -nc1) $(date +%T)"
  "$M" -nc1 -t "$NT" -s "$S" -nf "$FB" -l "$LOG" -v "$N" 2>&1 | grep -aiE 'relations|matrix|merge|cycle|error' | tail -4
  echo "[gpu-la] 2/4 build matrix (msieve -nc2), stop before CPU Lanczos $(date +%T)"
  : > "$LOG.nc2"
  "$M" -nc2 -t "$NT" -s "$S" -nf "$FB" -l "$LOG.nc2" -v "$N" >/dev/null 2>&1 &
  local MP=$!
  local i
  for i in $(seq 1 1200); do
    grep -qa 'commencing Lanczos iteration' "$LOG.nc2" 2>/dev/null && break
    kill -0 "$MP" 2>/dev/null || break
    sleep 2
  done
  kill -9 "$MP" 2>/dev/null; wait "$MP" 2>/dev/null; sleep 1
  [ -e "$S.mat" ] || { echo "[gpu-la] no matrix produced"; return 1; }
  echo "[gpu-la] 3/4 GPU block-Lanczos $(date +%T)"
  "$PY" "$HERE/gpu_la.py" "$S.mat" "$S.dep" || return 1
  [ -s "$S.dep" ] || return 1
  echo "[gpu-la] 4/4 parallel sqrt (deps 1-4 concurrent, first factor wins) $(date +%T)"
  local NPAR="${SQRT_PAR:-4}" tpd=$(( NT/4>0 ? NT/4 : 1 )) d pids=() found=""
  for d in $(seq 1 "$NPAR"); do
    "$M" -nc3 "$d,$d" -t "$tpd" -s "$S" -nf "$FB" -l "$LOG.d$d" -v "$N" >/dev/null 2>&1 &
    pids+=($!)
  done
  for i in $(seq 1 600); do
    for d in $(seq 1 "$NPAR"); do grep -qaiE 'factor:' "$LOG.d$d" 2>/dev/null && { found=$d; break; }; done
    [ -n "$found" ] && break
    local alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 2
  done
  for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
  # if first NPAR deps gave nothing (rare), try the rest sequentially
  if [ -z "$found" ]; then
    "$M" -nc3 "$((NPAR+1)),64" -t "$NT" -s "$S" -nf "$FB" -l "$LOG.rest" -v "$N" >/dev/null 2>&1 || true
    grep -qaiE 'factor:' "$LOG.rest" 2>/dev/null && { cat "$LOG.rest" >> "$LOG"; return 0; }
  else
    cat "$LOG.d$found" >> "$LOG"; return 0
  fi
  return 1
}

if gpu_downstream; then
  echo "[gpu-la] GPU linear algebra path succeeded $(date +%T)"
else
  echo "[gpu-la] GPU path failed/incomplete -> CPU fallback (msieve -nc) $(date +%T)"
  "$M" -nc -t "$NT" -s "$S" -nf "$FB" -l "$LOG" -v "$N" 2>&1 \
     | grep -aivE 'sse|avx' | grep -aiE 'unique|matrix is|Lanczos|dependencies|factor:|prp|GCD|error' || true
fi
echo "=== FACTORS ==="; grep -aiE 'factor:|prp[0-9]' "$LOG" 2>/dev/null | grep -aivE 'sse|avx' | tail -4
