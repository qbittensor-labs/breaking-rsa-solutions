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

# GPU-accelerated msieve downstream, run under msv_launcher (big files in memfd).
# Flow:  -nc1 filter  ->  -nc2 build matrix (kill before CPU Lanczos)  ->  GPU block-Lanczos  ->  -nc3 sqrt.
# Falls back to stock CPU `msieve -nc` if the GPU path fails for any reason.
# args: <N> <workdir>   (workdir has m.dat[memfd], m.fb; msieve writes m.dat.{cyc,mat[memfd],dep})
set -u
N="$1"; WORK="$2"
HERE=$(cd "$(dirname "$0")" && pwd); M="$HERE/msieve_la"; cd "$HERE"   # msieve_la = QCB_SIZE=32 build (square-defect fix); plain 'msieve' is the GPU-polyselect build
NT="${NT:-24}"; S="$WORK/m.dat"; FB="$WORK/m.fb"; LOG="$WORK/msieve.log"
# CADO-filter detection. refilter/cado_probe.sh leaves m.dat.purged beside m.dat.cyc;
# msieve then needs "cado_filter=1" so -nc2 maps purge line numbers to relation numbers
# (gnfs/relation.c:738) instead of reading the cycles as relation indices. Self-detecting
# on the file rather than an env var, so the msieve path is untouched when it is absent.
# NB: msieve_la must be built with MAX_COL_IDEALS >= 16384 or -nc2 dies with
# "error: overflow merging ideals" and leaves a TRUNCATED matrix -- see common.h.
NC2_ARGS=""
if [ -s "$S.purged" ]; then
  NC2_ARGS="cado_filter=1"
  echo "[gpu-la] CADO-filtered cycles detected -> -nc2 $NC2_ARGS"
fi
PY="${PYTHON:-python3}"
# The GPU-LA path is a PYTHON script, so it needs an interpreter that can import cupy -- and the
# one on PATH is not necessarily that interpreter. On this box /venv/main (activated by
# run_test.sh) has NO cupy while /usr/bin/python3 does, so a run launched through the venv would
# have had gpu_la.py die on `import cupy`, silently fall through to the CPU `msieve -nc` fallback
# below, and burn hours. Pick an interpreter that actually works, and if none does say so LOUDLY
# rather than letting the fallback quietly eat the wall-time budget.
if ! "$PY" -c 'import cupy' >/dev/null 2>&1; then
  for _cand in /usr/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$_cand" >/dev/null 2>&1 && "$_cand" -c 'import cupy' >/dev/null 2>&1; then
      echo "[gpu-la] NOTE: '$PY' cannot import cupy; using '$_cand' for the GPU linear algebra"
      PY="$_cand"; break
    fi
  done
fi
"$PY" -c 'import cupy' >/dev/null 2>&1 || \
  echo "[gpu-la] WARNING: no python with cupy found -- GPU linear algebra WILL fail and this run will fall back to CPU msieve -nc (hours). Set PYTHON=<interpreter with cupy>."

# ---- MATRIX SANITY (2026-08-23) -------------------------------------------------------------
# -nc2 is deliberately SIGKILLed the moment it reaches "commencing Lanczos iteration", and the
# poll loop above ALSO gives up after 1200 x 2s. If that cap ever expires (a much larger matrix
# than the ones measured -- 111s on the audit box, 235s on the Xeon), the kill lands MID-WRITE.
# The old test was `[ -e "$S.mat" ]`, which is TRUE for a zero-byte or half-written file:
# gpu_la.py then parses garbage, and the BEST case is that it exits non-zero and the run falls
# back to CPU `msieve -nc` -- hours, i.e. the wall. Require the file to be non-empty AND to carry
# a self-consistent header (nrows/num_dense/ncols, three uint32 -- gpu_la.py:parse_mat), so a
# truncated matrix is named HERE instead of being diagnosed three stages later.
# Pure shell (od) on purpose: this must not depend on the python interpreter it is guarding.
_mat_ok() {
  local m="$1" _sz _nrows _ndense _ncols
  [ -s "$m" ] || { echo "[gpu-la] no matrix produced ($m missing or empty)"; return 1; }
  _sz=$(stat -Lc %s "$m" 2>/dev/null || echo 0)
  set -- $(od -An -tu4 -N12 -v "$m" 2>/dev/null)
  _nrows="${1:-0}"; _ndense="${2:-0}"; _ncols="${3:-0}"
  if [ "${_ncols:-0}" -le 0 ] || [ "${_nrows:-0}" -le 0 ] || [ "${_ndense:-0}" -ge 1048576 ]; then
    echo "[gpu-la] matrix at $m has a bad header (nrows=$_nrows dense=$_ndense ncols=$_ncols)" \
         "-- TRUNCATED or malformed, refusing it"; return 1
  fi
  if [ "$_sz" -lt $(( (3 + _ncols) * 4 )) ]; then
    echo "[gpu-la] matrix at $m is TRUNCATED: $_sz bytes for $_ncols columns" \
         "(needs at least $(( (3 + _ncols) * 4 )))"; return 1
  fi
  return 0
}

# ---- ADVISORY CLOCK FOR THE RECOVERY CASCADE (2026-08-24) -----------------------------------
# Seconds of validator wall left, from breaking_rsa.py's GNFS_WALL_AT (absolute epoch). A branch
# uses this to answer "can this alternative still FINISH?" -- entering a recovery path that cannot
# complete is not recovery, it just guarantees the loss with the wall spent. When the variable is
# absent (dev harness, direct invocation) this reports a large number, so no branch is ever
# BLOCKED by a missing clock -- the cascade degrades to its old, unguarded behaviour.
_wall_left() {
  local at="${GNFS_WALL_AT:-}"
  case "$at" in ''|*[!0-9]*) echo 999999; return;; esac
  echo $(( at - $(date +%s) ))
}

# ---- -nc2 WITH THE LANCZOS WATCHDOG ---------------------------------------------------------
# Factored out so both the checkpoint path and the full-filter path run byte-identical logic, and
# so the checkpoint path can RETRY through the full path when its matrix is rejected.
_nc2_build() {
  local extra="$1" mp i
  : > "$LOG.nc2"
  "$M" -nc2 -t "$NT" -s "$S" -nf "$FB" -l "$LOG.nc2" -v "$N" $extra >/dev/null 2>&1 &
  mp=$!
  for i in $(seq 1 1200); do
    grep -qa 'commencing Lanczos iteration' "$LOG.nc2" 2>/dev/null && break
    kill -0 "$mp" 2>/dev/null || break
    sleep 2
  done
  if [ "$i" -ge 1200 ]; then
    echo "[gpu-la] WARNING: -nc2 did not reach Lanczos within 2400s -- killing it. Any" \
         " matrix it left is TRUNCATED and will be rejected below."
  fi
  kill -9 "$mp" 2>/dev/null; wait "$mp" 2>/dev/null; sleep 1
  cat "$LOG.nc2" >> "$LOG" 2>/dev/null || true
  _mat_ok "$S.mat"
}

# ---- FULL FILTER: msieve's own -nc1, then -nc2 ----------------------------------------------
# The fallback arm of the matrix cascade. Reached either normally (no checkpoint cycles) or after
# a checkpoint matrix was rejected. Costs ~8-10 min, which is why it is gated on the wall: it is a
# genuine alternative, not a placeholder.
_build_matrix_full() {
  # A CADO checkpoint that just failed leaves purge-NUMBERED cycles and a purge map behind. msieve's
  # own -nc1 emits RELATION-indexed cycles, so both must go, and cado_filter=1 must NOT be passed --
  # mixing the two is precisely the "silently builds garbage" case.
  rm -f "$S.cyc" "$S.purged"
  echo "[gpu-la] 1/4 filter (msieve -nc1) $(date +%T)"
  "$M" -nc1 -t "$NT" -s "$S" -nf "$FB" -l "$LOG" -v "$N" 2>&1 | grep -aiE 'relations|matrix|merge|cycle|error' | tail -4
  echo "[gpu-la] 2/4 build matrix (msieve -nc2), stop before CPU Lanczos $(date +%T)"
  _nc2_build "" || return 1
  return 0
}

gpu_downstream() {
  # Optional matrix cache (dev only; set GNFS_CACHE_DS to a dir). On HIT, reuse the cached matrix
  # and skip the filter (-nc1/-nc2) so you can iterate on LA/sqrt without re-filtering. m.dat (the
  # relations) and m.fb are re-provided each run by msv_launcher, so only m.dat.mat + m.dat.cyc cache.
  if [ -n "${GNFS_CACHE_DS:-}" ] && [ -s "${GNFS_CACHE_DS}/m.dat.mat" ] && [ -s "${GNFS_CACHE_DS}/m.dat.cyc" ]; then
    echo "[gpu-la] [cache HIT] matrix — skipping filter (-nc1/-nc2) $(date +%T)"
    cp "${GNFS_CACHE_DS}/m.dat.mat" "$S.mat"; cp "${GNFS_CACHE_DS}/m.dat.cyc" "$S.cyc"
  elif [ "${SKIP_NC1:-0}" = 1 ] && [ -s "$S.cyc" ]; then
    # factor_msv.sh's adaptive checkpoint ladder already ran -nc1 to decide whether the relation
    # set was sufficient, and its cycle file (m.dat.cyc) is exactly what -nc2 consumes. Re-running
    # -nc1 here would repeat ~8 min of duplicate/singleton/clique work for an identical result.
    echo "[gpu-la] 1/4 filter (msieve -nc1) SKIPPED — reusing checkpoint cycles $(date +%T)"
    echo "[gpu-la] 2/4 build matrix (msieve -nc2), stop before CPU Lanczos $(date +%T)"
    if ! _nc2_build "$NC2_ARGS"; then
      # ---- CASCADE, DO NOT DEAD-END (2026-08-24) --------------------------------------------
      # The checkpoint matrix was rejected. Reusing CADO's cycles is only an OPTIMISATION (it saves
      # the ~8-10 min of a second -nc1); the relations themselves are unaffected and msieve can
      # still filter them itself. Returning 1 here used to drop the whole run to the CPU `msieve
      # -nc` fallback -- ~11x slower, msieve's weaker filter, and at 3h+ elapsed a guaranteed loss
      # that then reported a misleading "wants 1,000,000 more relations". Measured live 2026-08-24:
      # a single unparsable free relation crashed -nc2 and cost the entire run that way.
      # Retry through the full filter instead, but ONLY if it can still finish.
      local _left; _left=$(_wall_left)
      if [ "$_left" -lt "${NC1_RETRY_MIN_SECS:-900}" ]; then
        echo "[gpu-la] checkpoint matrix rejected and only ${_left}s of wall left --" \
             "a full -nc1 needs ~${NC1_RETRY_MIN_SECS:-900}s and cannot finish. Refusing to start it." >&2
        return 1
      fi
      echo "[gpu-la] checkpoint matrix REJECTED -> retrying with msieve's own full filter" \
           "(-nc1 + -nc2), ${_left}s of wall left $(date +%T)" >&2
      NC2_ARGS=""
      _build_matrix_full || return 1
    fi
    if [ -n "${GNFS_CACHE_DS:-}" ]; then cp "$S.mat" "${GNFS_CACHE_DS}/m.dat.mat" 2>/dev/null && cp "$S.cyc" "${GNFS_CACHE_DS}/m.dat.cyc" 2>/dev/null && echo "[gpu-la] [cache SAVE] matrix"; fi
  else
    _build_matrix_full || return 1
    if [ -n "${GNFS_CACHE_DS:-}" ]; then cp "$S.mat" "${GNFS_CACHE_DS}/m.dat.mat" 2>/dev/null && cp "$S.cyc" "${GNFS_CACHE_DS}/m.dat.cyc" 2>/dev/null && echo "[gpu-la] [cache SAVE] matrix"; fi
  fi
  # dependency cache (dev only): reuse a cached GPU-LA dependency to skip block-Lanczos and go
  # straight to sqrt -- lets you iterate on sqrt alone. Inert unless GNFS_CACHE_DS holds m.dat.dep.
  if [ -n "${GNFS_CACHE_DS:-}" ] && [ -s "${GNFS_CACHE_DS}/m.dat.dep" ]; then
    echo "[gpu-la] [cache HIT] dependency — skipping GPU block-Lanczos $(date +%T)"
    cp "${GNFS_CACHE_DS}/m.dat.dep" "$S.dep"
  else
    echo "[gpu-la] 3/4 GPU block-Lanczos $(date +%T)"
    # GPU-local CPU pinning (2026-08-12, see gpu_nfs_siever/gpu_affinity.py). Block-Lanczos drives
    # CuPy from ONE host thread and its host-side cost is the matrix upload (~400M nnz), so a
    # remote-socket placement is paid on the transfer and on nothing else -- narrowing the mask
    # here costs no parallelism at all.
    # !! DEFAULT FLIPPED TO OFF, 2026-08-13 -- see the long note in gpu_nfs_siever/factor_msv.sh. !!
    # Short version: the validator sets --cpus (a CFS quota), NOT --cpuset-cpus, so the premise the
    # pinning was built on does not exist, and on a multi-NUMA host this ALWAYS fires rather than
    # being the claimed no-op. The "one host thread doing an upload" justification for pinning THIS
    # stage is also wrong: parse_mat() in gpu_la.py runs a ~4.3M-iteration Python loop and builds
    # ~8-9 GB of host arrays (cumsum over 400M int64, a 400M gather, np.repeat of 400M) BEFORE CuPy
    # is touched -- the largest host-side allocation in the pipeline, first-touched onto one node.
    # GPU_NUMA_PIN=1 re-enables; the validator passes no -e, so this default is the setting.
    _AFF="$HERE/../gpu_nfs_siever/gpu_affinity.py"
    if [ "${GPU_NUMA_PIN:-0}" != "0" ] && [ -r "$_AFF" ]; then
      "$PY" "$_AFF" -- "$PY" "$HERE/gpu_la.py" "$S.mat" "$S.dep" || return 1
    else
      "$PY" "$HERE/gpu_la.py" "$S.mat" "$S.dep" || return 1
    fi
    [ -s "$S.dep" ] || return 1
    if [ -n "${GNFS_CACHE_DS:-}" ]; then cp "$S.dep" "${GNFS_CACHE_DS}/m.dat.dep" 2>/dev/null && echo "[gpu-la] [cache SAVE] dependency"; fi
  fi
  # 4/4 square root. Preferred: CADO multi-core (OpenMP, all cores) sqrt -- dumps each
  # dependency's (a,b) via patched msieve, runs CADO's parallel sqrt (~8x faster than
  # msieve's serial nc3). Falls back to the original parallel msieve -nc3 if anything is
  # missing or fails, so the pipeline stays robust.
  echo "[gpu-la] 4/4 multi-core CADO sqrt (all CPU) $(date +%T)"
  if [ -x "$HERE/cado_sqrt" ] && [ -x "$HERE/msieve_dump" ] && [ -f "$HERE/cado_sqrt.sh" ]; then
    if MDUMP="$HERE/msieve_dump" CSQRT="$HERE/cado_sqrt" PYTHON="$PY" \
         bash "$HERE/cado_sqrt.sh" "$N" "$S" "$FB" "$LOG" "$NT"; then
      return 0
    fi
    echo "[gpu-la] CADO sqrt failed -> msieve -nc3 fallback $(date +%T)"
  else
    echo "[gpu-la] CADO sqrt binaries absent -> msieve -nc3 fallback $(date +%T)"
  fi
  echo "[gpu-la] 4/4 (fallback) parallel msieve sqrt (deps 1-4 concurrent) $(date +%T)"
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
  # ---- THE LAST ARM OF THE CASCADE, AND IT MUST BE ABLE TO FINISH (2026-08-24) ---------------
  # CPU `msieve -nc` redoes filter + block-Lanczos + sqrt on CPU, ~11x slower than the GPU path.
  # It is NOT a "no GPU" path -- every trigger above (missing/truncated matrix, gpu_la.py failure,
  # empty .dep, no factor in any dep) fires on a perfectly healthy GPU, and one of them did on
  # 2026-08-24. Entered at 3h+ elapsed it cannot possibly complete: it burns the remaining wall and
  # then reports msieve's clamped "wants 1,000,000 more relations", which reads as a relation
  # shortfall and hides the real tooling failure. Only start it when it can actually win.
  _left=$(_wall_left)
  if [ "$_left" -ge "${CPU_NC_MIN_SECS:-7200}" ]; then
    echo "[gpu-la] GPU path failed/incomplete -> CPU fallback (msieve -nc), ${_left}s of wall left $(date +%T)"
    "$M" -nc -t "$NT" -s "$S" -nf "$FB" -l "$LOG" -v "$N" 2>&1 \
       | grep -aivE 'sse|avx' | grep -aiE 'unique|matrix is|Lanczos|dependencies|factor:|prp|GCD|error' || true
  else
    echo "[gpu-la] GPU path failed/incomplete. NOT starting the CPU fallback: it needs" \
         "~${CPU_NC_MIN_SECS:-7200}s and only ${_left}s of wall remain, so it would spend the rest" \
         "of the run and still fail -- while reporting a RELATION SHORTFALL that is not the real" \
         "cause. This is a TOOLING failure in the GPU downstream; see the [gpu-la] lines above." >&2
  fi
fi
echo "=== FACTORS ==="; grep -aiE 'factor:|prp[0-9]' "$LOG" 2>/dev/null | grep -aivE 'sse|avx' | tail -4
