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
#   SKIP_REPLAY=1  reuse an existing SPARSE + <outdir>/c.index.gz (factor_c169.sh SPEC_MERGE=1: a speculatively prepared
#            matrix). REQUIRES SKIP_EPOCH = the content of <SPARSE minus .bin>.epoch (poly / probe / purged fingerprint /
#            TD, written before that matrix was published READY); a missing or different epoch replays from <filterwork>/history.
#   DS_STOP_AFTER_REPLAY=1 (measurement only) exit 0 once the matrix is in place, i.e. at the LA start
#   SC_T_STOP  epoch seconds of the stop verdict (factor_c169.sh): the LA start logs "stop->LA start <s>"
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
#   --- critical-path options (2026-10-20; measured by gpu/s23_cp.sh; every default = the previous behaviour) ---
#   DS_PURGED=<file>  purged relations to read (replay, characters, sqrt -ab, abx) instead of <filterwork>/purged.gz,
#            e.g. a decompressed copy (same content: the readers skip their gzip pipe/inflate)
#   SQRT_ABX=1  the (a,b) extraction without the post-characters pass over purged.gz: `abx prep` (ABX, default
#            ./abx; build: gcc -O2 -pthread -o abx abx.c) parses purged + index in the BACKGROUND during the LA into
#            ABX_FILE (default <outdir>/c.abx, ~1.4 GB at c169: point it off a small /tmp), then after characters
#            `abx deps` writes PLAIN c.dep.<k> files (byte-identical to sqrt -ab's) in seconds and the sqrt reads
#            them in place (no per-dependency inflate). Any abx failure falls back to `sqrt -ab`. ABX_GZ=1: abx writes
#            c.dep.<k>.gz through gzip -1 instead (10 plain c169 dependencies are ~5.2 GB: on the validator's 10 GiB /tmp
#            use ABX_GZ=1); the sqrt then inflates each one as before.
#   SQRT_POOL=1 sliding window: a new dependency starts as soon as one ends without a factor (default: windows of
#            SQRT_WIN that end together). SQRT_TPD threads per sqrt (default NT/SQRT_WIN). SQRT_STAGGER=<s>: at least
#            <s> seconds between two launches (pool mode; spreads the CPU-bound level 00 against the GPU multiplies).
# stdout: every stage line AND gpu_la's progress lines ("iter ...", LA done, VERIFY) are streamed
#   live (the validator captures container stdout on a wall kill; a buffered log would be lost).
# exit: 0 factor found, 1 tooling failure (named), 2 no factor from any dependency
#   --- LA start-up (2026-10-23, gpu/s65_lastart.sh) ---
#   LA_PARSE=cpu (default) | gpu | check: cado_mat.py's matrix loader (gpu: GPU parse, bit-identical arrays, ~40 s ->
#            a few s; check: both + MATCH/MISMATCH, numpy result used). Passed through to gpu_la in the environment.
#   LA_WARM=1 (default): `gpu_la_fast.py --warm` (the production LA on a tiny synthetic matrix: fills the CuPy kernel
#            cache, cold start +17 s) runs in the background beside replay -- unless LA_WARM_PID names one already
#            started (factor_c169.sh starts it when the sieve stops; its log: LA_WARM_LOG). Before LA try 1 we wait up to
#            LA_WARM_WAIT s (default 60) for it, then kill it. Exact: it only writes CuPy cache files.
#   `ds_cado.sh --la-warm`: run only that warm-up (foreground) with this script's LA environment and interpreter.
set -u
LA_WARM_ONLY=0; if [ "${1:-}" = --la-warm ]; then LA_WARM_ONLY=1; set -- - - - -; fi
FW=$1; POLY=$2; N=$3; OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
pick(){ for c in "$@"; do [ -x "$c" ] && { echo "$c"; return; }; done; echo "$1"; }
REPLAY=${REPLAY:-$(pick "$HERE/replay" "$HERE/../cado-filter/replay")}
CHARS=${CHARS:-$(pick "$HERE/characters")}
CSQRT=${CSQRT:-$(pick "$HERE/sqrt" "$HERE/cado_sqrt")}
# GPU combine (gpu_la_fast.py: gpu_la.py with combineV on the GPU) by default: selftest at the key-1b size IDENTICAL,
# GPU 3.61 s vs CPU 270.69 s (sandbox 2, 2026-10-20 22:00 MDT); any GPU-side exception falls back to the CPU combine
GPU_LA=${GPU_LA:-$HERE/gpu_la_fast.py}; [ -r "$GPU_LA" ] || GPU_LA=$HERE/gpu_la.py
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
# MAXDEP 10 (was 16): each dependency factors with p = 1/2 independently (random GF(2) mixes below), so
# 10 leave a 2^-10 miss chance and save ~6 x ~190 MB of /tmp at c169 (docs/rsa560_plan.md W4 /tmp plan)
LA_TRIES=${LA_TRIES:-2}; SQRT_WIN=${SQRT_WIN:-3}; MAXDEP=${MAXDEP:-10}   # 3 deps x NT/3 threads (8 at NT=24): measured faster than 6 x 2
unset CADO_CHARS_VERIFY CADO_CHARS_STOCK   # set only per-invocation, never inherited
# LA defaults (2026-09-27): 128-bit block Lanczos + segmented hot-ideal SpMV = 0.50x the LA time of the
# 64-bit production path on c160 (measured). OpenBLAS threads MUST be 1: the 128x128 host GF(2) ops
# otherwise fight the host load (measured 1.2 -> 8-14 ms/it CPU part).
# LA_BLOCK=256 (W6, 2026-10-17): half the iterations at ~1.12x the SpMV pass (measured on the RTX PRO 6000); try 2 stays 64/prod.
export LA_BLOCK=${LA_BLOCK:-128} LA_SPMV=${LA_SPMV:-seg} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
if [ "$LA_WARM_ONLY" = 1 ]; then
  case "$GPU_LA" in *gpu_la_fast.py) exec "$PY" "$GPU_LA" --warm;; esac
  echo "[ds-cado] --la-warm: $GPU_LA has no --warm mode -- nothing to do"; exit 0
fi
mkdir -p "$OUT" || { echo "[ds-cado] FAILED: cannot create $OUT"; exit 1; }
SPARSE=${SPARSE:-$OUT/c.sparse.bin}
# REPLAY_INDEX_PLAIN=1 (default 0 = c.index.gz, unchanged; docs/research/agents_2026-10-26/downstream.md #3): replay writes
# its index PLAIN. The .gz index goes through ONE gzip process on replay's critical path: "index write start 18.4s / done
# 31.4s" = 13.0 s of a 50 s replay on key 28 (data/rsa560/key28f_gcp/tmp_gnfs_key28f_ds_replay.txt), the serial tail arm
# B's docker stats show at 150-300 % CPU. Plain is ~1.5 GB (160M entries) in <outdir>. Every reader picks its decoder by
# suffix (abx.c rd_open; CADO fopen_maybe_compressed in characters / sqrt -ab). Not under SKIP_REPLAY=1 (a prepared
# matrix's index is c.index.gz, spec_job.sh / factor_c169.sh SPEC_USE).
IDX="$OUT/c.index.gz"; [ "${REPLAY_INDEX_PLAIN:-0}" = 1 ] && [ "${SKIP_REPLAY:-0}" != 1 ] && IDX="$OUT/c.index"
KW="$OUT/c.ker.W"; KER="$OUT/c.kernel"; DEPP="$OUT/c.dep.gz"; LOG="$OUT/msieve.log"
T0=$(date +%s); el(){ echo "[$(( $(date +%s)-T0 ))s]"; }
say(){ echo "[ds-cado] $(el) $*"; }
fail(){ echo "[ds-cado] $(el) FAILED: $*" >&2; echo "[ds-cado] VERDICT FAIL $*"; exit 1; }
case "$SPARSE" in *.sparse.bin) ;; *) fail "SPARSE=$SPARSE must end in .sparse.bin (replay derives the sidecar names from it)";; esac
SBASE=${SPARSE%.bin}; SRW="$SBASE.rw.bin"; SCW="$SBASE.cw.bin"
for b in "$REPLAY" "$CHARS" "$CSQRT"; do [ -x "$b" ] || fail "missing or non-executable binary $b"; done
[ -r "$GPU_LA" ] || fail "missing $GPU_LA"
[ -r "$(dirname "$GPU_LA")/cado_mat.py" ] || fail "missing $(dirname "$GPU_LA")/cado_mat.py (gpu_la.py imports it for .sparse.bin input)"
PURGED=${DS_PURGED:-$FW/purged.gz}
[ -s "$PURGED" ] || fail "no $PURGED"
[ -s "$POLY" ] || fail "no poly $POLY"
: >> "$LOG"

# LA_WARM (see the header): compile the LA's kernels now, while replay keeps the CPUs busy and the GPU is idle
LWPID=${LA_WARM_PID:-}; LWLOG=${LA_WARM_LOG:-$OUT/la_warm.log}
if [ -z "$LWPID" ] && [ "${LA_WARM:-1}" = 1 ] && [ "${LA:-gpu}" != bwc ] && [ "${SKIP_REPLAY:-0}" != 1 ]; then
  case "$GPU_LA" in *gpu_la_fast.py)
    LWLOG=$OUT/la_warm.log; ( exec nice -n 5 "$PY" "$GPU_LA" --warm ) > "$LWLOG" 2>&1 & LWPID=$!
    say "LA kernel-cache warm-up started beside replay (pid $LWPID -> $LWLOG)";; esac
fi

# ---- 1. replay: history -> merged matrix (CADO binary) + index -----------------------------
# -skip 0: every column stays in sparse.bin, so there is NO dense.bin/heavyblock. gpu_la packs the
# ND heaviest columns (replay sorts columns by decreasing weight) as its dense rows instead.
SKOK=0
if [ "${SKIP_REPLAY:-0}" = 1 ]; then
  if [ -z "${SKIP_EPOCH:-}" ]; then say "WARNING: SKIP_REPLAY=1 without SKIP_EPOCH -- replaying (an existing file is not proof)"
  elif [ "$(cat "$SBASE.epoch" 2>/dev/null)" != "$SKIP_EPOCH" ]; then say "WARNING: SKIP_REPLAY=1 but $SBASE.epoch != SKIP_EPOCH ($SKIP_EPOCH) -- replaying"
  elif [ -s "$SPARSE" ] && [ -s "$IDX" ] && [ -s "$SRW" ]; then SKOK=1
  else say "WARNING: SKIP_REPLAY=1 but $SPARSE / $IDX / $SRW incomplete -- replaying"; fi
fi
if [ "$SKOK" = 1 ]; then
  say "replay SKIPPED (reusing $SPARSE, epoch $SKIP_EPOCH)"
else
  [ -s "$FW/history" ] || [ -s "$FW/history.gz" ] || fail "no $FW/history"
  HIS="$FW/history"; [ -s "$HIS" ] || HIS="$FW/history.gz"
  rm -f "$SRW" "$SCW" "$IDX"
  [ -L "$SPARSE" ] || rm -f "$SPARSE"          # never unlink a memfd link; replay writes through it
  # -t: the row additions run in batches of disjoint merges under OpenMP; replay's default is 1 thread (s42, key1bds
  # TD 130: 293.4 s at -t 1 vs 138.9 s at -t 20, sparse.bin and index md5-identical)
  "$REPLAY" -purged "$PURGED" -his "$HIS" -out "$SPARSE" -index "$IDX" -skip 0 -t "${REPLAY_T:-$NT}" \
      > "$OUT/replay.log" 2>&1 || fail "replay rc=$? (see $OUT/replay.log)"
  grep -qaiE 'No space left|ENOSPC|write error' "$OUT/replay.log" && fail "replay out of space"
  say "replay: $(grep -a 'Sparse submatrix' "$OUT/replay.log" | tail -1) $(grep -a 'Weight of the sparse' "$OUT/replay.log" | tail -1 | sed 's/# //')"
  # the merge history has no reader after replay (the LA retries below reuse SPARSE): free its /tmp
  # (a memfd link -- factor_c169.sh MEMFD_PURGED=1 -- is truncated first: rm alone would leave its pages in memhold)
  if [ "${KEEP_HISTORY:-0}" != 1 ]; then for _h in "$FW/history" "$FW/history.gz"; do [ -L "$_h" ] && : > "$_h"; rm -f "$_h"; done; fi
fi
NR=$(grep -a 'Sparse submatrix' "$OUT/replay.log" 2>/dev/null | tail -1 | grep -oE 'nrows=[0-9]+' | cut -d= -f2)
NRW=$(( $(stat -Lc %s "$SRW" 2>/dev/null || echo 0) / 4 ))
NIX=$(gzip -dcf "$IDX" 2>/dev/null | head -1)   # -f: a plain index (REPLAY_INDEX_PLAIN=1) passes through unchanged
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

if [ "${DS_STOP_AFTER_REPLAY:-0}" = 1 ]; then
  say "DS_STOP_AFTER_REPLAY=1: matrix ready ($NR rows)$([ -n "${SC_T_STOP:-}" ] && echo "; stop->LA start $(( $(date +%s) - SC_T_STOP ))s")"; exit 0; fi
# SQRT_ABX=1: the (a,b) side of `sqrt -ab` needs only purged + index, both final now -- parse them while the GPU runs the LA
ABX=${ABX:-$HERE/abx}; ABXF=${ABX_FILE:-$OUT/c.abx}; ABXPID=""
# default on since sandbox 1 (2026-10-21): sqrt -ab 110 s -> abx prep 77 s beside the LA + abx deps 10 s, 10/10 files IDENTICAL
if [ "${SQRT_ABX:-1}" = 1 ]; then
  if [ -x "$ABX" ]; then
    rm -f "$ABXF" "$ABXF.tmp"; ( exec nice -n 5 "$ABX" prep "$PURGED" "$IDX" "$ABXF" > "$OUT/abx_prep.log" 2>&1 ) & ABXPID=$!
    trap '[ -n "$ABXPID" ] && kill "$ABXPID" 2>/dev/null' EXIT
    say "abx prep started beside the LA (pid $ABXPID -> $ABXF)"
  else say "WARNING: SQRT_ABX=1 but $ABX is missing -- sqrt -ab after characters"; fi
fi

# CHARS_PLAIN=1 (default): characters reads purged PLAIN (s36, key1f: 65.0 s vs 124.0 s from gzip, identical kernel);
# the gunzip (~61 s) runs in the background while the GPU runs the LA (the CPU is idle then), into CHARS_PLAIN_FILE
# (default /dev/shm, ~7.4 GB at c169). Falls back to $PURGED if it is not gzip or the gunzip failed.
CPURGED=$PURGED; GZPID=""; PLAINF=${CHARS_PLAIN_FILE:-/dev/shm/ds_purged.$$}
if [ "${CHARS_PLAIN:-1}" = 1 ] && [ "${PURGED%.gz}" != "$PURGED" ]; then
  ( exec nice -n 5 gzip -dc "$PURGED" > "$PLAINF" ) & GZPID=$!
  trap '[ -n "$ABXPID" ] && kill "$ABXPID" 2>/dev/null; [ -n "$GZPID" ] && kill "$GZPID" 2>/dev/null; rm -f "$PLAINF"' EXIT
  say "plain purged for characters: gunzip started beside the LA (pid $GZPID -> $PLAINF)"
fi

# CHARS_PRE=1 (default 0 until gpu/s70_chars.sh validates it on a c169 downstream): the kernel-independent half of
# `characters` -- the small character matrix (index rows x nchar bits; depends on poly/purged/index/lpb/nchar only) --
# is built BESIDE the LA (CADO_CHARS_SCMAT_OUT, gpu/chars_pre_patch.py) into CHARS_PRE_FILE (default <outdir>/c.scmat,
# nrows*8 B, ~130 MB at c169), so after the LA `characters` only loads it (CADO_CHARS_SCMAT_IN) and runs the stock
# kernel reduction/elimination/output: byte-identical c.kernel (laptop c115: IDENTICAL, combine 0.09 s vs 2-3 s full).
# Any failure (unpatched binary, precompute rc!=0, load mismatch) falls back to the full `characters` below.
CPPID=""; SCF=${CHARS_PRE_FILE:-$OUT/c.scmat}
if [ "${CHARS_PRE:-1}" = 1 ]; then   # default 1 since s70 (sb57 c100): combine byte-identical to full characters
  if grep -q CADO_CHARS_SCMAT_IN "$CHARS" 2>/dev/null; then
    rm -f "$SCF" "$SCF.tmp"
    ( export CADO_CHARS_SCMAT_OUT="$SCF"; exec nice -n 5 "$CHARS" -poly "$POLY" -purged "$PURGED" -index "$IDX" -out /dev/null \
        -lpb0 "$LPB0" -lpb1 "$LPB1" -nchar "$NCHAR" -t "${CHARS_PRE_T:-$NT}" > "$OUT/chars_pre.log" 2>&1 ) & CPPID=$!
    trap '[ -n "$ABXPID" ] && kill "$ABXPID" 2>/dev/null; [ -n "$GZPID" ] && kill "$GZPID" 2>/dev/null; [ -n "$CPPID" ] && kill "$CPPID" 2>/dev/null; rm -f "$PLAINF"' EXIT
    say "CHARS_PRE: character-matrix precompute started beside the LA (pid $CPPID -> $SCF)"
  else say "WARNING: CHARS_PRE=1 but $CHARS lacks gpu/chars_pre_patch.py -- full characters after the LA"; fi
fi
CPRC=""   # precompute rc, once reaped

# ---- 2. GPU block Lanczos on the CADO matrix -> 64 kernel vectors (flat CADO ker format) ------
# ---- 3. characters: combine them with quadratic characters into true dependencies ------------
if [ -n "$LWPID" ]; then   # never two GPU jobs at once: the warm-up finishes (or dies) before the LA starts
  _w=0; while kill -0 "$LWPID" 2>/dev/null && [ "$_w" -lt "${LA_WARM_WAIT:-60}" ]; do sleep 1; _w=$((_w+1)); done
  if kill -0 "$LWPID" 2>/dev/null; then
    kill "$LWPID" 2>/dev/null; sleep 1; say "WARNING: LA warm-up still running after ${_w}s -- killed (the LA compiles what is missing)"
  else say "LA warm-up finished (waited ${_w}s): $(grep -a 'la-warm\] done' "$LWLOG" 2>/dev/null | tail -1 | cut -c1-120)"; fi
  LWPID=""
fi
NDEP=0; try=0
while [ "$try" -lt "$LA_TRIES" ]; do
  try=$((try+1)); seed=$(( 5 + 2*(try-1) ))
  rm -f "$KW" "$KER"
  say "LA try $try (LA_SEED=$seed): $PY $(basename "$GPU_LA") $(basename "$SPARSE")$([ "$try" = 1 ] && [ -n "${SC_T_STOP:-}" ] && echo "; stop->LA start $(( $(date +%s) - SC_T_STOP ))s")"
  # stream the progress lines live (full log kept in gpu_la.<try>.log); rc of the python, not of the pipe
  # every try runs the configured fast path (256-bit seg) with a new seed, unless the previous try crashed
  # (no kernel): then the 64-bit production SpMV, so a fault specific to the fast path cannot eat both attempts.
  # (Before 2026-10-22 try 2 always took the 64-bit path: ~2x the LA time for a no-deps retry -- LA review.)
  # the 64-bit crash fallback cannot defer dense rows (gpu_la.py: LA_DEFER_ROWS needs LA_BLOCK=256 + LA_SPMV=seg, else exit 3:
  # 2026-10-25 submission dry run, the fallback died at once), so it runs LA_DEFER_ROWS=0 -- the pre-10-24 64/prod path
  if [ "${LACRASH:-0}" = 0 ]; then LAB=$LA_BLOCK; LAS=$LA_SPMV; LADR=${LA_DEFER_ROWS:-0}; else LAB=64; LAS=prod; LADR=0; fi; LACRASH=0
  say "  LA_BLOCK=$LAB LA_SPMV=$LAS$([ "$LADR" != 0 ] && echo " LA_DEFER_ROWS=$LADR")"
  if [ "${LA:-gpu}" = bwc ]; then
    # LA=bwc (c169 fork, CPU-only validation path): CADO's block Wiedemann on the same replay matrix.
    # bwc.pl writes the kernel as <wdir>/W, nrows x 64 bits -- the flat format `characters -ker` reads,
    # byte-compatible with gpu_la's c.ker.W. BWC_DIR holds bwc.pl and the bwc binaries of a CADO build.
    BWC=${BWC_DIR:-$HERE/../bwc}; [ -r "$BWC/bwc.pl" ] || fail "LA=bwc but no $BWC/bwc.pl (set BWC_DIR)"
    rm -rf "$OUT/bwc"; mkdir -p "$OUT/bwc"
    THR=${BWC_THR:-$([ "$NT" -ge 4 ] && echo 2x2 || echo "1x$NT")}
    say "  bwc.pl :complete nullspace=left m=64 n=64 thr=$THR (seed $seed)"
    perl "$BWC/bwc.pl" :complete matrix="$SPARSE" wdir="$OUT/bwc" nullspace=left mpi=1x1 thr="$THR" m=64 n=64 \
        interval=1000 seed="$seed" > "$OUT/bwc.$try.log" 2>&1; larc=$?
    [ -s "$OUT/bwc/W" ] && cp -f "$OUT/bwc/W" "$KW"
    [ "$larc" = 0 ] || say "  bwc rc=$larc: $(grep -aiE 'error|fail|abort' "$OUT/bwc.$try.log" | tail -1 | cut -c1-160)"
  else
  LA_BLOCK=$LAB LA_SPMV=$LAS LA_DEFER_ROWS=$LADR LA_SEED=$seed LA_CADO_NROWS=$NR "$PY" "$GPU_LA" "$SPARSE" "$KW" 2>&1 | tee "$OUT/gpu_la.$try.log" \
    | grep --line-buffered -aE 'cado_mat|matrix |CSR|seg|iter |LA done|VERIFY|expanded|wrote|WARNING|FATAL|Error|Traceback' \
    | sed -u 's/^/    /'
  larc=${PIPESTATUS[0]}
  fi
  if [ "$larc" != 0 ] || [ ! -s "$KW" ]; then say "gpu_la rc=$larc -- no kernel"; LACRASH=1
    # a Lanczos breakdown ("not all columns used", seed-dependent; c100 smoke 10-23) is not a fast-path fault: next seed, same path
    grep -aqE "not all columns used|produced no usable dependency" "$OUT/gpu_la.$try.log" && LACRASH=0; continue; fi
  [ "$(stat -c %s "$KW")" = $(( NR * 8 )) ] || fail "kernel file $KW is $(stat -c %s "$KW") bytes, want $((NR*8))"
  if [ -n "$GZPID" ]; then
    if wait "$GZPID"; then CPURGED=$PLAINF; say "plain purged ready ($(stat -c %s "$PLAINF") B)"; else CPURGED=$PURGED; say "WARNING: gunzip of purged failed -- characters reads gzip"; fi
    GZPID=""
  fi
  crc=1; CPDONE=0
  if [ -n "$CPPID" ]; then
    _t0=$(date +%s); wait "$CPPID"; CPRC=$?; CPPID=""
    say "CHARS_PRE: precompute rc=$CPRC (waited $(( $(date +%s)-_t0 ))s after the LA): $(grep -a 'CHARS_PRE: wrote' "$OUT/chars_pre.log" | cut -c1-140)"
  fi
  if [ "$CPRC" = 0 ] && [ -s "$SCF" ]; then
    _t0=$(date +%s.%N)
    CADO_CHARS_SCMAT_IN="$SCF" "$CHARS" -poly "$POLY" -purged "$CPURGED" -index "$IDX" -ker "$KW" -out "$KER" \
        -lpb0 "$LPB0" -lpb1 "$LPB1" -nchar "$NCHAR" -t "$NT" > "$OUT/characters.$try.log" 2>&1; crc=$?
    say "CHARS_PRE: combine rc=$crc in $(awk -v a="$_t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')s"
    [ "$crc" = 0 ] && CPDONE=1
    [ "$crc" = 0 ] || say "WARNING: CHARS_PRE combine failed ($(grep -aE 'Error' "$OUT/characters.$try.log" | head -1 | cut -c1-160)) -- full characters"
  fi
  [ "$CPDONE" = 1 ] || { "$CHARS" -poly "$POLY" -purged "$CPURGED" -index "$IDX" -ker "$KW" -out "$KER" \
      -lpb0 "$LPB0" -lpb1 "$LPB1" -nchar "$NCHAR" -t "$NT" > "$OUT/characters.$try.log" 2>&1; crc=$?; }
  if [ "$crc" != 0 ] && [ -s "$KW" ]; then
    # The parallel characters (vendor/cado-characters-parallel.patch) fails LOUDLY on damaged input and has
    # its own abort points; before burning a whole second LA (~2,500 s at c160), retry the STOCK code path
    # (CADO_CHARS_STOCK=1, ~300 s) on the SAME kernel. Byte-identical output when both succeed.
    say "WARNING: parallel characters rc=$crc ($(grep -aE 'Error|error' "$OUT/characters.$try.log" | head -1)) -- retrying the stock path"
    CADO_CHARS_STOCK=1 "$CHARS" -poly "$POLY" -purged "$CPURGED" -index "$IDX" -ker "$KW" -out "$KER" \
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
[ "${CHARS_PRE:-1}" = 1 ] && [ "${KEEP_SCMAT:-0}" != 1 ] && rm -f "$SCF" "$SCF.tmp"   # nrows*8 B back
if [ "${KEEP_SPARSE:-0}" != 1 ]; then
  if [ -L "$SPARSE" ]; then : > "$SPARSE"; else rm -f "$SPARSE"; fi      # 4*(N+W) bytes back
fi

# ---- 4. sqrt -ab: ONE pass over purged.gz writes every dependency's (a,b) file ----------------
rm -f "$OUT"/c.dep.[0-9]*.gz "$OUT"/c.dep.side* "$OUT"/c.dep.[0-9][0-9][0-9]
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
    x = w0 & np.uint64(v)                      # parity of the popcount; XOR-fold (numpy < 2.0 has no bitwise_count)
    for sh in (32, 16, 8, 4, 2, 1): x = x ^ (x >> np.uint64(sh))
    par = x & np.uint64(1)
    out |= par << np.uint64(j)
nzo = int(np.bitwise_or.reduce(out))
assert nzo == (1 << m) - 1, "a mixed dependency came out empty"
out.tofile(dst)
print("mixed %d random GF(2) combinations of %d non-zero dependencies (seed %d)" % (m, len(bits), seed))
PYEOF
DEPPLAIN=0
if [ -n "$ABXPID" ]; then   # SQRT_ABX=1: the purged/index pass ran during the LA; only the XOR + writes are left
  wait "$ABXPID"; arc=$?; ABXPID=""; trap 'rm -f "$PLAINF"' EXIT
  _apfx="$OUT/c.dep"; [ "${ABX_GZ:-0}" = 1 ] && _apfx="$DEPP"
  if [ "$arc" = 0 ] && [ -s "$ABXF" ] && "$ABX" deps "$ABXF" "$KER.ab" "$_apfx" > "$OUT/sqrt_ab.log" 2>&1; then
    DEPPLAIN=1; [ "${ABX_GZ:-0}" = 1 ] && DEPPLAIN=2
    say "abx: $(cat "$OUT/abx_prep.log" | tail -1 | cut -c1-160); $(head -1 "$OUT/sqrt_ab.log")"
  else say "WARNING: abx failed (prep rc=$arc: $(tail -1 "$OUT/abx_prep.log" 2>/dev/null | cut -c1-160); deps: $(tail -1 "$OUT/sqrt_ab.log" 2>/dev/null | cut -c1-160)) -- sqrt -ab"
    rm -f "$OUT"/c.dep.[0-9][0-9][0-9] "$OUT"/c.dep.[0-9][0-9][0-9].gz; fi
  rm -f "$ABXF"
fi
if [ "$DEPPLAIN" = 2 ]; then DEPPLAIN=0; ND=$(ls "$OUT"/c.dep.[0-9][0-9][0-9].gz 2>/dev/null | wc -l)   # ABX_GZ: CADO-style .gz files
elif [ "$DEPPLAIN" = 0 ]; then
"$CSQRT" -ab -poly "$POLY" -prefix "$DEPP" -purged "$CPURGED" -index "$IDX" -ker "$KER.ab" \
    > "$OUT/sqrt_ab.log" 2>&1 || fail "sqrt -ab rc=$? (see $OUT/sqrt_ab.log)"
ND=$(ls "$OUT"/c.dep.[0-9][0-9][0-9].gz 2>/dev/null | wc -l)
else ND=$(ls "$OUT"/c.dep.[0-9][0-9][0-9] 2>/dev/null | wc -l); fi
[ "$ND" -gt 0 ] || fail "sqrt -ab wrote no dependency files"
say "sqrt -ab: $ND dependency files ($(grep -a 'pairs' "$OUT/sqrt_ab.log" | head -1))"

# ---- 5. concurrent sqrt window, first non-trivial factor wins ---------------------------------
TPD=${SQRT_TPD:-$(( NT / SQRT_WIN > 0 ? NT / SQRT_WIN : 1 ))}
sqrt_one(){   # $1 dep ; writes the factor (if any) to $OUT/win.$1.factor
  local d=$1 dn; dn=$(printf %03d "$d")
  # MEASURED at c160 (18M-pair dependency, same machine): a .gz dependency forces CADO sqrt into
  # single-thread I/O ("cannot seek in dependency file") -- 398 s at OMP=2; the SAME dependency
  # uncompressed runs 256 s at OMP=4 and 184 s at OMP=8. So inflate each dependency (~313 MB) into
  # its own scratch dir just before its sqrt, and delete it after (<= SQRT_WIN x ~313 MB on /tmp).
  local SD="$OUT/sq.$dn"; rm -rf "$SD"; mkdir -p "$SD"
  if [ "$DEPPLAIN" = 1 ]; then   # abx wrote plain, seekable files: sqrt reads them in place
    rm -rf "$SD"
    OMP_NUM_THREADS=$TPD "$CSQRT" -poly "$POLY" -prefix "$OUT/c.dep" -dep "$d" -side0 -side1 -gcd \
        > "$OUT/sqrt.$d.out" 2> "$OUT/sqrt.$d.err"
  elif gzip -dc "${DEPP%.gz}.$dn.gz" > "$SD/c.dep.$dn" 2>/dev/null; then
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
  [ -s "$OUT/win.$d.factor.part" ] || rm -f "$OUT/c.dep.$(printf %03d "$d").gz" "$OUT"/c.dep.side*."$(printf %03d "$d")".gz \
                                          "$OUT/c.dep.$(printf %03d "$d")" "$OUT"/c.dep.side*."$(printf %03d "$d")"
  mv -f "$OUT/win.$d.factor.part" "$OUT/win.$d.factor"      # atomic: -s never sees a torn line
}
d=0; F=""; FD=""
rm -f "$OUT"/win.*.factor*
if [ "${SQRT_POOL:-0}" = 1 ]; then
  # sliding window: keep SQRT_WIN running, start the next dependency as soon as one ends without a factor
  declare -A RUNP=(); nxt=0; tlast=0; STG=${SQRT_STAGGER:-0}
  while [ -z "$F" ]; do
    while [ "${#RUNP[@]}" -lt "$SQRT_WIN" ] && [ "$nxt" -lt "$ND" ] && [ $(( $(date +%s) - tlast )) -ge "$STG" ]; do
      sqrt_one "$nxt" & RUNP[$nxt]=$!; tlast=$(date +%s); say "sqrt dep $nxt started (OMP=$TPD, pool of $SQRT_WIN)"; nxt=$((nxt+1))
    done
    # an EMPTY pool with dependencies left means the stagger guard above refused the next start (a dependency ended within
    # SQRT_STAGGER s of the last start -- the c100 gate of S4a, 2026-10-25: deps take ~1 s there, so the loop broke out after
    # dep 0 and the whole c100 re-ran through the retry ladder): wait for the stagger instead of giving up
    [ "${#RUNP[@]}" -gt 0 ] || { [ "$nxt" -lt "$ND" ] || break; }
    sleep 1
    for dd in "${!RUNP[@]}"; do
      [ -s "$OUT/win.$dd.factor" ] && { F=$(cat "$OUT/win.$dd.factor"); FD=$dd; break; }
      if ! kill -0 "${RUNP[$dd]}" 2>/dev/null; then
        wait "${RUNP[$dd]}" 2>/dev/null
        [ -s "$OUT/win.$dd.factor" ] && { F=$(cat "$OUT/win.$dd.factor"); FD=$dd; break; }
        say "sqrt dep $dd: no factor"; unset "RUNP[$dd]"
      fi
    done
  done
  for p in "${RUNP[@]}"; do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done; wait 2>/dev/null
  d=$ND
fi
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
