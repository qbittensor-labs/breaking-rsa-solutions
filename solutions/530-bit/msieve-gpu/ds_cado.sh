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

# =============================================================================================
# ds_cado.sh -- CADO-NATIVE downstream: replay -> gpu_la (CADO matrix) -> characters -> sqrt.
# Replaces: replay --for_msieve + cado2msieve2 + msieve -nc2 (+ parse of the .mat) + msieve_dump.
# =============================================================================================
# usage: ds_cado.sh <filterwork> <poly.cado> <N> <outdir>
#   <filterwork>  a cado_filter160.sh work dir holding purged.gz + history (MERGE_ONLY=1 is enough;
#                 the raw relation chunks are NOT needed -- purged.gz carries every (a,b)).
#   <poly.cado>   the CADO poly the relations were filtered with (the same one sqrt must use).
#   <outdir>      small outputs: index, kernels, dep files, logs.  Writes "factor: <p>" lines to
#                 <outdir>/msieve.log (the name factor_c160.sh greps) and prints p=/q= on stdout.
# env:
#   SPARSE   path of the merged matrix, MUST end in .sparse.bin (default <outdir>/c.sparse.bin).
#            At c160 it is ~5 GB (4*(N+W)): point it at a memfd link (memhold.py name c.sparse.bin);
#            the small .rw.bin/.cw.bin sidecars are created next to it as regular files.
#   SKIP_REPLAY=1  reuse an existing SPARSE + <outdir>/c.index.gz (e.g. written by the filter probe)
#   NT (threads, default nproc)  LPB0 LPB1 (default 31)  NCHAR (algebraic characters, default 50 = cado-nfs default; costs only ~rank(obstruction)~5 deps)
#   LA_TRIES (default 2: a second Lanczos with another seed if the first yields no usable deps)
#   SQRT_WIN (concurrent sqrt processes, default 6)
#   MAXDEP (default 16): only this many dependencies get an (a,b) file. sqrt -ab writes EVERY
#            dependency it is given in one pass, ~2.4 MB/dep at c111 -> ~145 MB/dep at c160; all
#            ~59 would be ~8.6 GB on the 10 GiB /tmp. The kernel is masked to the first MAXDEP.  PYTHON (interpreter with cupy)
#   REPLAY CHARS CSQRT GPU_LA   binaries/scripts (defaults: next to this script / ../cado-filter)
#   KEEP_SPARSE=1  do not free SPARSE after the LA has loaded it
#   KEEP_HISTORY=1 do not delete <filterwork>/history after replay (default: deleted -- nothing reads
#            it after replay, and it is ~1 GB of the 10 GiB /tmp at c160)
# stdout: every stage line AND gpu_la's progress lines ("iter ...", LA done, VERIFY) are streamed
#   live (the validator captures container stdout on a wall kill; a buffered log would be lost).
# exit: 0 factor found, 1 tooling failure (named), 2 no factor from any dependency
set -u
FW=$1; POLY=$2; N=$3; OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
pick(){ for c in "$@"; do [ -x "$c" ] && { echo "$c"; return; }; done; echo "$1"; }
REPLAY=${REPLAY:-$(pick "$HERE/replay" "$HERE/../cado-filter/replay")}
CHARS=${CHARS:-$(pick "$HERE/characters")}
CSQRT=${CSQRT:-$(pick "$HERE/sqrt" "$HERE/cado_sqrt")}
GPU_LA=${GPU_LA:-$HERE/gpu_la.py}
PY=${PYTHON:-python3}
# The GPU LA is a PYTHON script: it needs an interpreter that can import cupy, and the one on PATH is
# not necessarily it (same discovery as msvrun_gpu.sh -- a venv python without cupy would otherwise
# die on `import cupy` and look like an LA failure). Pick one that works; say so LOUDLY if none does.
if ! "$PY" -c 'import cupy' >/dev/null 2>&1; then
  for _cand in /usr/bin/python3 /usr/local/bin/python3 python3; do
    if command -v "$_cand" >/dev/null 2>&1 && "$_cand" -c 'import cupy' >/dev/null 2>&1; then
      echo "[ds-cado] NOTE: '$PY' cannot import cupy; using '$_cand' for the GPU linear algebra"
      PY="$_cand"; break
    fi
  done
fi
"$PY" -c 'import cupy' >/dev/null 2>&1 || \
  echo "[ds-cado] WARNING: no python with cupy found -- the GPU linear algebra WILL fail. Set PYTHON=<interpreter with cupy>."
NT=${NT:-$(nproc)}; LPB0=${LPB0:-31}; LPB1=${LPB1:-31}; NCHAR=${NCHAR:-50}
LA_TRIES=${LA_TRIES:-2}; SQRT_WIN=${SQRT_WIN:-3}; MAXDEP=${MAXDEP:-16}   # 3 deps x NT/3 threads (8 at NT=24): measured faster than 6 x 2
unset CADO_CHARS_VERIFY CADO_CHARS_STOCK   # set only per-invocation, never inherited
# LA defaults (2026-09-27): 128-bit block Lanczos + segmented hot-ideal SpMV = 0.50x the LA time of the
# 64-bit production path on c160 (measured). OpenBLAS threads MUST be 1: the 128x128 host GF(2) ops
# otherwise fight the host load (measured 1.2 -> 8-14 ms/it CPU part).
export LA_BLOCK=${LA_BLOCK:-128} LA_SPMV=${LA_SPMV:-seg} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
mkdir -p "$OUT" || { echo "[ds-cado] FAILED: cannot create $OUT"; exit 1; }
SPARSE=${SPARSE:-$OUT/c.sparse.bin}
IDX="$OUT/c.index.gz"; KW="$OUT/c.ker.W"; KER="$OUT/c.kernel"; DEPP="$OUT/c.dep.gz"; LOG="$OUT/msieve.log"
T0=$(date +%s); el(){ echo "[$(( $(date +%s)-T0 ))s]"; }
say(){ echo "[ds-cado] $(el) $*"; }
fail(){ echo "[ds-cado] $(el) FAILED: $*" >&2; echo "[ds-cado] VERDICT FAIL $*"; exit 1; }
case "$SPARSE" in *.sparse.bin) ;; *) fail "SPARSE=$SPARSE must end in .sparse.bin (replay derives the sidecar names from it)";; esac
SBASE=${SPARSE%.bin}; SRW="$SBASE.rw.bin"; SCW="$SBASE.cw.bin"
for b in "$REPLAY" "$CHARS" "$CSQRT"; do [ -x "$b" ] || fail "missing or non-executable binary $b"; done
[ -r "$GPU_LA" ] || fail "missing $GPU_LA"
[ -r "$(dirname "$GPU_LA")/cado_mat.py" ] || fail "missing $(dirname "$GPU_LA")/cado_mat.py (gpu_la.py imports it for .sparse.bin input)"
[ -s "$FW/purged.gz" ] || fail "no $FW/purged.gz"
[ -s "$POLY" ] || fail "no poly $POLY"
: >> "$LOG"

# ---- 1. replay: history -> merged matrix (CADO binary) + index -----------------------------
# -skip 0: every column stays in sparse.bin, so there is NO dense.bin/heavyblock. gpu_la packs the
# ND heaviest columns (replay sorts columns by decreasing weight) as its dense rows instead.
if [ "${SKIP_REPLAY:-0}" = 1 ] && [ -s "$SPARSE" ] && [ -s "$IDX" ] && [ -s "$SRW" ]; then
  say "replay SKIPPED (reusing $SPARSE)"
else
  [ -s "$FW/history" ] || [ -s "$FW/history.gz" ] || fail "no $FW/history"
  HIS="$FW/history"; [ -s "$HIS" ] || HIS="$FW/history.gz"
  rm -f "$SRW" "$SCW" "$IDX"
  [ -L "$SPARSE" ] || rm -f "$SPARSE"          # never unlink a memfd link; replay writes through it
  "$REPLAY" -purged "$FW/purged.gz" -his "$HIS" -out "$SPARSE" -index "$IDX" -skip 0 \
      > "$OUT/replay.log" 2>&1 || fail "replay rc=$? (see $OUT/replay.log)"
  grep -qaiE 'No space left|ENOSPC|write error' "$OUT/replay.log" && fail "replay out of space"
  say "replay: $(grep -a 'Sparse submatrix' "$OUT/replay.log" | tail -1) $(grep -a 'Weight of the sparse' "$OUT/replay.log" | tail -1 | sed 's/# //')"
  # the merge history has no reader after replay (the LA retries below reuse SPARSE): free its /tmp
  [ "${KEEP_HISTORY:-0}" = 1 ] || rm -f "$FW/history" "$FW/history.gz"
fi
NR=$(grep -a 'Sparse submatrix' "$OUT/replay.log" 2>/dev/null | tail -1 | grep -oE 'nrows=[0-9]+' | cut -d= -f2)
NRW=$(( $(stat -Lc %s "$SRW" 2>/dev/null || echo 0) / 4 ))
NIX=$(gzip -dc "$IDX" 2>/dev/null | head -1)
[ -n "${NR:-}" ] || NR=$NRW
[ "${NR:-0}" -gt 0 ] && [ "$NR" = "${NIX:-x}" ] \
  || fail "matrix/index disagree: replay nrows=$NR index=${NIX:-?} (stale files?)"
# rw.bin carries one weight per row. It can be SHORTER than the matrix only through replay's
# empty-row (NULL) branch, which writes the row's 0 header to sparse.bin but nothing to rw.bin. Never
# silent: cado_mat.py then walks the matrix, checks it against the index's nrows (LA_CADO_NROWS) and
# drops EMPTY rows from the LA (their kernel bits are written as 0). Longer is always an error.
if [ "$NRW" != "$NR" ]; then
  [ "$NRW" -lt "$NR" ] || fail "rw.bin has $NRW rows but replay/index say $NR (stale files?)"
  say "WARNING: rw.bin has $NRW rows but the matrix has $NR -- $((NR-NRW)) empty merged row(s); gpu_la will walk the matrix and drop them"
fi

# ---- 2. GPU block Lanczos on the CADO matrix -> 64 kernel vectors (flat CADO ker format) ------
# ---- 3. characters: combine them with quadratic characters into true dependencies ------------
NDEP=0; try=0
while [ "$try" -lt "$LA_TRIES" ]; do
  try=$((try+1)); seed=$(( 5 + 2*(try-1) ))
  rm -f "$KW" "$KER"
  say "LA try $try (LA_SEED=$seed): $PY $(basename "$GPU_LA") $(basename "$SPARSE")"
  # stream the progress lines live (full log kept in gpu_la.<try>.log); rc of the python, not of the pipe
  # try 1 = the configured fast path (default 128-bit seg); later tries = the 64-bit production SpMV, so
  # a fault specific to the fast path cannot eat both attempts.
  if [ "$try" -eq 1 ]; then LAB=$LA_BLOCK; LAS=$LA_SPMV; else LAB=64; LAS=prod; fi
  say "  LA_BLOCK=$LAB LA_SPMV=$LAS"
  LA_BLOCK=$LAB LA_SPMV=$LAS LA_SEED=$seed LA_CADO_NROWS=$NR "$PY" "$GPU_LA" "$SPARSE" "$KW" 2>&1 | tee "$OUT/gpu_la.$try.log" \
    | grep --line-buffered -aE 'cado_mat|matrix |CSR|seg|iter |LA done|VERIFY|expanded|wrote|WARNING|FATAL|Error|Traceback' \
    | sed -u 's/^/    /'
  larc=${PIPESTATUS[0]}
  if [ "$larc" != 0 ] || [ ! -s "$KW" ]; then say "gpu_la rc=$larc -- no kernel"; continue; fi
  [ "$(stat -c %s "$KW")" = $(( NR * 8 )) ] || fail "kernel file $KW is $(stat -c %s "$KW") bytes, want $((NR*8))"
  "$CHARS" -poly "$POLY" -purged "$FW/purged.gz" -index "$IDX" -ker "$KW" -out "$KER" \
      -lpb0 "$LPB0" -lpb1 "$LPB1" -nchar "$NCHAR" -t "$NT" > "$OUT/characters.$try.log" 2>&1; crc=$?
  if [ "$crc" != 0 ] && [ -s "$KW" ]; then
    # The parallel characters (vendor/cado-characters-parallel.patch) fails LOUDLY on damaged input and has
    # its own abort points; before burning a whole second LA (~2,500 s at c160), retry the STOCK code path
    # (CADO_CHARS_STOCK=1, ~300 s) on the SAME kernel. Byte-identical output when both succeed.
    say "WARNING: parallel characters rc=$crc ($(grep -aE 'Error|error' "$OUT/characters.$try.log" | head -1)) -- retrying the stock path"
    CADO_CHARS_STOCK=1 "$CHARS" -poly "$POLY" -purged "$FW/purged.gz" -index "$IDX" -ker "$KW" -out "$KER" \
        -lpb0 "$LPB0" -lpb1 "$LPB1" -nchar "$NCHAR" -t "$NT" > "$OUT/characters.$try.stock.log" 2>&1; crc=$?
    [ "$crc" = 0 ] && cp -f "$OUT/characters.$try.stock.log" "$OUT/characters.$try.log"
  fi
  DIM=$(grep -aoE 'dim of ker = [0-9]+' "$OUT/characters.$try.log" | grep -oE '[0-9]+$')
  NDEP=$(grep -aoE 'Wrote [0-9]+ non-zero dependencies' "$OUT/characters.$try.log" | grep -oE '[0-9]+' | head -1)
  say "characters rc=$crc: $(grep -aE 'reduced to dimension' "$OUT/characters.$try.log" | tail -1) | dim of ker=${DIM:-?} | deps=${NDEP:-0}"
  [ "$crc" = 0 ] && [ -s "$KER" ] && [ "${NDEP:-0}" -gt 0 ] && break
  NDEP=0
done
[ "${NDEP:-0}" -gt 0 ] || fail "no dependency survived characters after $LA_TRIES LA tries (see $OUT/gpu_la.*.log, characters.*.log)"
if [ "${KEEP_SPARSE:-0}" != 1 ]; then
  if [ -L "$SPARSE" ]; then : > "$SPARSE"; else rm -f "$SPARSE"; fi      # 4*(N+W) bytes back
fi

# ---- 4. sqrt -ab: ONE pass over purged.gz writes every dependency's (a,b) file ----------------
rm -f "$OUT"/c.dep.[0-9]*.gz "$OUT"/c.dep.side*
# MAXDEP dependencies for sqrt (it reads only the first 64-bit word per row). NOT the first MAXDEP basis
# vectors: characters emits an echelon-style basis, and whether a dependency gives x = +-y (trivial gcd) is
# a GF(2)-LINEAR function of it, which can sit on few basis vectors -- measured: key 2 lost basis deps 0-3
# (run 1) and 0-5 (run 2), key 1 deps 0-1, each costing a whole ~230 s sqrt window. Every GF(2) combination
# of dependencies is a dependency, so each output dependency is a RANDOM combination (fixed seed) of ALL
# non-zero ones: it factors with probability exactly 1/2, independently. SQRT_MIX=0 = the old first-MAXDEP.
"$PY" - "$KER" "$KER.ab" "$NR" "$MAXDEP" "${SQRT_MIX:-1}" "${SQRT_MIX_SEED:-20260927}" <<'PYEOF' || fail "cannot build the MAXDEP sqrt dependencies"
import sys, numpy as np
src, dst, nr, mx, mix, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1", int(sys.argv[6])
k = np.fromfile(src, dtype="<u8")
assert k.size % nr == 0, "kernel size"
w0 = np.ascontiguousarray(k.reshape(nr, -1)[:, 0])
nz = int(np.bitwise_or.reduce(w0)); bits = [b for b in range(64) if (nz >> b) & 1]
if not mix or len(bits) <= 1:
    keep = 0
    for b in bits[:mx]: keep |= 1 << b
    (w0 & np.uint64(keep)).tofile(dst)
    print("kept %d of %d non-zero dependencies" % (min(mx, len(bits)), len(bits)))
    sys.exit(0)
rng = np.random.default_rng(seed); m = min(mx, len(bits)); masks = []; basis = []   # GF(2) row-echelon for independence
def reduce(v):
    for bv in basis: v = min(v, v ^ bv)
    return v
while len(masks) < m:
    v = 0
    for b in bits:
        if rng.integers(2): v |= 1 << b
    r = reduce(v)
    if r == 0: continue                         # zero or dependent on the ones already chosen: draw again
    basis.append(r); basis.sort(reverse=True); masks.append(v)
out = np.zeros(nr, dtype=np.uint64)
for j, v in enumerate(masks):
    par = (np.bitwise_count(w0 & np.uint64(v)) & np.uint8(1)).astype(np.uint64)
    out |= par << np.uint64(j)
nzo = int(np.bitwise_or.reduce(out))
assert nzo == (1 << m) - 1, "a mixed dependency came out empty"
out.tofile(dst)
print("mixed %d random GF(2) combinations of %d non-zero dependencies (seed %d)" % (m, len(bits), seed))
PYEOF
"$CSQRT" -ab -poly "$POLY" -prefix "$DEPP" -purged "$FW/purged.gz" -index "$IDX" -ker "$KER.ab" \
    > "$OUT/sqrt_ab.log" 2>&1 || fail "sqrt -ab rc=$? (see $OUT/sqrt_ab.log)"
ND=$(ls "$OUT"/c.dep.[0-9][0-9][0-9].gz 2>/dev/null | wc -l)
[ "$ND" -gt 0 ] || fail "sqrt -ab wrote no dependency files"
say "sqrt -ab: $ND dependency files ($(grep -a 'pairs' "$OUT/sqrt_ab.log" | head -1))"

# ---- 5. concurrent sqrt window, first non-trivial factor wins ---------------------------------
TPD=$(( NT / SQRT_WIN > 0 ? NT / SQRT_WIN : 1 ))
sqrt_one(){   # $1 dep ; writes the factor (if any) to $OUT/win.$1.factor
  local d=$1 dn; dn=$(printf %03d "$d")
  # MEASURED at c160 (18M-pair dependency, same machine): a .gz dependency forces CADO sqrt into
  # single-thread I/O ("cannot seek in dependency file") -- 398 s at OMP=2; the SAME dependency
  # uncompressed runs 256 s at OMP=4 and 184 s at OMP=8. So inflate each dependency (~313 MB) into
  # its own scratch dir just before its sqrt, and delete it after (<= SQRT_WIN x ~313 MB on /tmp).
  local SD="$OUT/sq.$dn"; rm -rf "$SD"; mkdir -p "$SD"
  if gzip -dc "${DEPP%.gz}.$dn.gz" > "$SD/c.dep.$dn" 2>/dev/null; then
    OMP_NUM_THREADS=$TPD "$CSQRT" -poly "$POLY" -prefix "$SD/c.dep" -dep "$d" -side0 -side1 -gcd \
        > "$OUT/sqrt.$d.out" 2> "$OUT/sqrt.$d.err"
  else   # cannot inflate (e.g. /tmp full): fall back to the compressed file, slower but correct
    rm -rf "$SD"
    OMP_NUM_THREADS=$TPD "$CSQRT" -poly "$POLY" -prefix "$DEPP" -dep "$d" -side0 -side1 -gcd \
        > "$OUT/sqrt.$d.out" 2> "$OUT/sqrt.$d.err"
  fi
  rm -rf "$SD"
  "$PY" - "$N" "$OUT/sqrt.$d.out" > "$OUT/win.$d.factor.part" <<'PYEOF'
import sys,re
N=int(sys.argv[1])
for line in open(sys.argv[2]):
    s=line.strip()
    if re.fullmatch(r"[0-9]+", s):
        f=int(s)
        if 1<f<N and N%f==0:
            print(f); break
PYEOF
  [ -s "$OUT/win.$d.factor.part" ] || rm -f "$OUT/c.dep.$(printf %03d "$d").gz" "$OUT"/c.dep.side*."$(printf %03d "$d")".gz
  mv -f "$OUT/win.$d.factor.part" "$OUT/win.$d.factor"      # atomic: -s never sees a torn line
}
d=0; F=""; FD=""
rm -f "$OUT"/win.*.factor*
while [ "$d" -lt "$ND" ] && [ -z "$F" ]; do
  pids=(); deps=()
  for k in $(seq 0 $((SQRT_WIN-1))); do
    dd=$((d+k)); [ "$dd" -ge "$ND" ] && break
    sqrt_one "$dd" & pids+=($!); deps+=("$dd")
  done
  say "sqrt deps ${deps[*]} (OMP=$TPD each)"
  while :; do
    for dd in "${deps[@]}"; do [ -s "$OUT/win.$dd.factor" ] && { F=$(cat "$OUT/win.$dd.factor"); FD=$dd; break; }; done
    [ -n "$F" ] && break
    alive=0; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && { alive=1; break; }; done
    [ "$alive" = 0 ] && { for dd in "${deps[@]}"; do [ -s "$OUT/win.$dd.factor" ] && { F=$(cat "$OUT/win.$dd.factor"); FD=$dd; }; done; break; }
    sleep 1
  done
  for p in "${pids[@]}"; do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done; wait 2>/dev/null
  d=$((d+SQRT_WIN))
done
if [ -n "$F" ]; then
  Q=$("$PY" -c "print($N//$F)")
  { echo "factor: $F"; echo "factor: $Q"; } >> "$LOG"
  say "FACTOR on dep $FD"
  echo "p=$F"; echo "q=$Q"
  exit 0
fi
say "no factor from $ND dependencies"
exit 2
