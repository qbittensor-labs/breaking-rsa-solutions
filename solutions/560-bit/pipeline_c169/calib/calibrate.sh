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

# calibrate.sh -- CALIBRATION SUBMISSION body (RSA560_CALIBRATE=1 in breaking_rsa.py): measure THIS host's speed on three
# frozen kernels inside the validator's container, predict S4rung's wall here, and hand the bucket code back to
# breaking_rsa.py (last line "CALIB_WALL=... CALIB_CODE=..."). Nothing here depends on the challenge N. ~4 min on our card.
#   1. CPU:  makefb side 1, lim 450M, -t 20 on key 1's e9c0 polynomial         (gpu/calib.sh reference: GCP g4 10.2 s)
#   2. GPU:  glas on q [12.5M, 12.6M), key 1's polynomial, lim 100M/150M, lpb 32/33, mfb 63/65, lambda 2.1, A 29, 4 slots,
#            GLAS_ADJUST=3 RHITS=0 COPRIME=3 PREFETCH=1 XFUSE=0, -t 16            (reference: GCP g4 71.2 s, 2,180,113 rels)
#   3. LA:   gpu_la_fast.py on a synthetic CADO matrix of CAL_LA_N ideals (default 3,000,000: out of the 128 MB L2 like
#            production), production SpMV env, LA_MAXIT=CAL_LA_IT (default 600) -> ms/it  (reference: CAL_REF_LA_MS, set
#            from this image's gate run on GCP; unset = LA ratio 1)
# Output: key=value lines + the CALIB_ line (calib_code.py). Every failure still yields a line (code sieve_lo).
set -u
A=${APP_DIR:-/app}; CF=${CF_DIR:-$A/cado-filter}; GLAS=${GLAS:-$A/glas/glas-gpu}; MG=${MSV_DIR:-$A/msieve-gpu}
HERE=$(cd "$(dirname "$0")" && pwd); W=${CAL_WORK:-/tmp/calib}; mkdir -p "$W"
P1=$HERE/key1.poly.cado; PE=$HERE/key1.e9c0.poly.cado
say(){ echo "[calib $(date -u +%T)] $*"; }
gpu_s=0; cpu_s=0; la_ms=0; rels=0
say "host: $(nproc) cpus, $(nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv,noheader 2>/dev/null | head -1)"
# 1. CPU
t0=$(date +%s.%N)
if timeout 600 "$CF/makefb" -poly "$PE" -side 1 -lim 450000000 -maxbits 15 -t 20 -out "$W/fb_e9.450" > "$W/makefb450.log" 2>&1; then
  cpu_s=$(python3 -c "print(f'{$(date +%s.%N)-$t0:.1f}')"); else say "makefb (CPU step) failed rc=$?"; fi
rm -f "$W"/fb_e9.450*; say "cpu_s=$cpu_s"
# 2. GPU (factor base first, untimed)
if timeout 600 "$CF/makefb" -poly "$P1" -side 1 -lim 150000000 -maxbits 15 -t 20 -out "$W/fb_c1.150" > "$W/makefb150.log" 2>&1; then
  t0=$(date +%s.%N)
  env GLAS_GPU_STATS=1 GLAS_ADJUST=3 GLAS_RHITS=0 GLAS_FILL_COPRIME=3 GLAS_REGION_PREFETCH=1 GLAS_XFUSE=0 GLAS_GPU_SLOTS=4 \
    timeout 900 "$GLAS" -poly "$P1" -fb1 "$W/fb_c1.150" -lim0 100000000 -lim1 150000000 -lpb0 32 -lpb1 33 -mfb0 63 -mfb1 65 \
    -lambda0 2.1 -lambda1 2.1 -ncurves0 17 -ncurves1 17 -A 29 -q0 12500000 -q1 12600000 -sqside 1 -t 16 -gpu \
    -out "$W/r.rels" > "$W/glas.log" 2>&1; rc=$?
  if [ "$rc" = 0 ]; then gpu_s=$(python3 -c "print(f'{$(date +%s.%N)-$t0:.1f}')"); rels=$(grep -vc '^#' "$W/r.rels"); else say "glas failed rc=$rc: $(tail -2 "$W/glas.log" | tr '\n' ' ' | cut -c1-200)"; fi
else say "makefb (factor base for glas) failed"; fi
rm -f "$W/r.rels" "$W"/fb_c1.150*; say "gpu_s=$gpu_s gpu_rels=$rels (reference 2,180,112/113)"
# known-answer check: a different relation count means a different binary or settings -> the timing is not comparable
case "$rels" in 2180112|2180113) ;; *) say "glas relation count $rels is not the reference count: GPU timing rejected"; gpu_s=0 ;; esac
# 3. LA micro-benchmark (8M ideals = ~830 MB of matrix: HBM-bound like production, well past the 128 MB L2; ~36 s to build)
N=${CAL_LA_N:-8000000}; IT=${CAL_LA_IT:-600}
if python3 - "$MG" "$W" "$N" > "$W/synth.log" 2>&1 <<'PY'
import sys; mg, w, n = sys.argv[1], sys.argv[2], int(sys.argv[3]); sys.path.insert(0, mg); import cado_mat
path, nr = cado_mat.synth_cado(w + "/la", n, excess=64, kmin=10, kmax=40, seed=20261023); print(path, nr)
PY
then
  MAT=$(awk '{print $1}' "$W/synth.log" | tail -1); NR=$(awk '{print $2}' "$W/synth.log" | tail -1)
  env LA_SPMV=seg LA_BLOCK=256 LA_SEG_G2=1 LA_SEG_HOTT=1 LA_DEFER_ROWS=64 LA_SEG_HINT=2 LA_CADO_NROWS="$NR" LA_MAXIT="$IT" \
    LA_LOG_EVERY=1000000 OPENBLAS_NUM_THREADS=1 timeout 900 python3 "$MG/gpu_la_fast.py" "$MAT" "$W/ker" > "$W/la.log" 2>&1
  la_ms=$(grep -a "LA_MAXIT=.*ms/it" "$W/la.log" | tail -1 | sed -n 's/.* = \([0-9.]*\) ms\/it.*/\1/p'); la_ms=${la_ms:-0}
  [ "$la_ms" = 0 ] && say "LA bench produced no ms/it: $(grep -a 'Error\|FATAL\|Traceback' "$W/la.log" | head -1 | cut -c1-160)"
else say "synthetic matrix failed: $(tail -1 "$W/synth.log" | cut -c1-160)"; fi
rm -rf "$W"/la* "$W/ker"*; say "la_ms=$la_ms (N $N, $IT its)"
python3 "$HERE/calib_code.py" "$gpu_s" "$cpu_s" "$la_ms"
