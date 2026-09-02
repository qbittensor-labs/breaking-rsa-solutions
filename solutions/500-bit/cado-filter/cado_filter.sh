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

# =============================================================================
# cado_filter.sh — CADO-NFS filtering as a drop-in replacement for `msieve -nc1`
# =============================================================================
# WHY THIS EXISTS (measured 2026-08-04, key03 of the yield survey)
#
# msieve's filter is the binding constraint on hard keys, not the sieve. On a key
# whose budget sieve produced 65,496,292 relations, msieve `-nc1`:
#     E_core 245,809 >= target_excess 211,895  (a genuine excess SURPLUS)
#     found 326,800 cycles, need 2,263,595  ->  "wants 1,000,000 more relations"
# i.e. it had enough relations and still could not build a matrix. Every msieve
# filter knob is INERT in this build -- `filter_lpbound` (2^28/2^27/2^26),
# `target_density` (70/110) and `max_weight` all produce BYTE-IDENTICAL output,
# joining the already-documented `filter_maxrels`. There is no tuning left.
#
# CADO's filter builds a matrix from THE SAME relations, and does it faster:
#
#     stage        msieve            CADO
#     filter       572s -> FAILED    353s -> MATRIX
#     matrix -nc2  373s              309s
#     GPU Lanczos  692s              663s
#     sqrt         161s              ~150s
#     downstream   1,798s            1,475s      (-323s)
#
# The two agree on the relation set to within ONE relation (40,719,216 vs
# 40,719,217 unique), so this is purely a clique-removal/merge difference --
# plus CADO emitting 1.9x more free relations (235,258 vs 122,480), which is
# excess msieve leaves on the table.
#
# VERIFIED END-TO-END on key03 (44.81 rel/lat, the hardest of 8 surveyed keys,
# which msieve declared 1M relations short): p*q == N, total 3.806h, 700s inside
# the 4h wall.
#
# ---------------------------------------------------------------------------
# TWO CONSTRAINTS THAT WILL BITE IF REDISCOVERED
#
# 1. msieve's CADO interop is pinned to a 2013-era format ("[should work with
#    CADO-NFS revision aca5658]", README.msieve). Modern CADO's purged file is
#    "# nrows ncolsmax ncols" + "a,b:ideals" with a,b in HEX; msieve wants line 1
#    = a COUNT and each later line to START with a relation number. strtoul on
#    "# 17744909..." returns 0, so msieve silently reads zero cycles. `cado2msieve`
#    bridges this by emitting relations in PURGE ORDER with an identity index map.
#
# 2. msieve's MAX_COL_IDEALS is 1000 (include/common.h) and CADO relation-sets
#    exceed it -> "error: overflow merging ideals" while building the initial
#    matrix, leaving a TRUNCATED .mat that then dies in gpu_la.py as
#    "zero-size array to reduction operation CUPY_CUB_MAX". Lowering
#    target_density is NOT a sufficient fix: 170 AND 100 both overflow at the
#    stock limit. msieve_la must be rebuilt with MAX_COL_IDEALS=16384
#    (ideal_t is 12 bytes; two stack arrays -> ~393 KB, fine on an 8 MB stack).
#    msieve_la contains no CUDA, so this rebuild is toolkit-independent.
#
# ---------------------------------------------------------------------------
# DISK -- the validator's /tmp is a 10 GB TMPFS and breaking_rsa.py puts the
# workdir there (breaking_rsa.py:165, TMPDIR default /tmp). MEASURED PEAK OF THIS
# SCRIPT: 7.53 GB on a 68.5M-relation dump, i.e. 2.47 GB of headroom. Naively
# written it peaked at ~12.1 GB and would have hit ENOSPC ~3.3h into a run. Three
# things keep it down, all load-bearing:
#   * compressed intermediates (-outfmt .gz)
#   * dup1 + renumber.gz deleted the moment purge is done reading them
#   * purged.gz STREAMED into cado2msieve rather than decompressed to disk (~1.9 GB)
# NOTE dup2 appears NOT to preserve compression when it rewrites its input, which
# is why the peak is 7.53 GB and not the ~3 GB the compressed split alone implies.
# Fits as-is; revisit if the relation count grows much beyond 70M.
#
# usage: cado_filter.sh <rels> <poly.cado> <N> <workdir> <out_prefix>
#   env: LPB (default 29)  TD (target_density, default 100)  NT (threads, 24)
#        CADO_BIN (default alongside this script)
#
# output (all consumed by `msieve_la -nc2 "cado_filter=1" -s <out_prefix>`):
#   <out_prefix>          relations in msieve format, in purge order
#   <out_prefix>.purged   identity index map
#   <out_prefix>.cyc      cycle file from `replay --for_msieve`
# =============================================================================
set -u

RELS="$1"; POLY="$2"; NUM="$3"; WORK="$4"; OUT="$5"
HERE=$(cd "$(dirname "$0")" && pwd)
BIN="${CADO_BIN:-$HERE}"
LPB="${LPB:-29}"; TD="${TD:-100}"; NT="${NT:-24}"; KEEP="${KEEP:-160}"

for t in freerel dup1 dup2 purge merge replay cado2msieve; do
  [ -x "$BIN/$t" ] || { echo "[cado-filter] FATAL: $BIN/$t missing or not executable" >&2; exit 1; }
done

mkdir -p "$WORK" || exit 1
echo "[cado-filter] start $(date +%T)  lpb=$LPB td=$TD nt=$NT"

# ---- storage accounting helpers (used by the HEADROOM GUARD in step 2) ------
# $RELS is normally an anonymous memfd exposed as /proc/<pid>/fd/N, and $STRIPPED is a symlink
# TO that symlink. `stat -L` follows both and reports the real byte count; verified against
# relhold.py's memfd rather than assumed, because `readlink -f` notoriously does NOT work here
# (it yields the literal string "/memfd:rels.txt (deleted)" -- see the note in step 2).
_kb_size(){ local _s; _s=$(stat -Lc %s "$1" 2>/dev/null) || return 1; echo $(( _s / 1024 )); }
_kb_free(){ df -Pk "$1" 2>/dev/null | awk 'NR==2{print $4}'; }
_mount_of(){ df -P "$1" 2>/dev/null | awk 'NR==2{print $6}'; }
# ENOSPC is the likeliest failure on a 10 GB tmpfs and CADO reports it only inside its own log,
# so a bare "dup1 FAILED" sends the reader hunting for a CADO defect instead of a full disk.
# Name it explicitly when it happens.
_diskerr(){   # $1 = log file, $2 = stage name
  grep -qaiE 'No space left on device|ENOSPC|write error|disk full' "$1" 2>/dev/null || return 0
  echo "[cado-filter] $2: OUT OF SPACE on $(_mount_of "$WORK") --" \
       "$(( $(_kb_free "$WORK" 2>/dev/null || echo 0) / 1024 )) MB free against a" \
       "$(( ${RELKB:-0} / 1024 )) MB relation file. This is a storage limit, NOT a CADO defect." >&2
}

# --- 1. renumber table + free relations -------------------------------------
# CADO generates its OWN free relations; msieve's (appended in place by -nc1)
# must NOT be present in the input -- see step 2.
#
# CACHE (2026-08-09): freerel's output depends ONLY on (poly, lpb) -- not on a single relation.
# It costs 11.07s measured on this box, and it ran inside EVERY probe: once for the speculative
# probe, again for each re-probe, again for any ladder probe. The polynomial is final the moment
# polyselect ends (~3.6h before the first probe), so factor_msv.sh precomputes it into
# CADO_FREEREL_CACHE while the sieve runs and we symlink it here.
# Keyed on poly content + lpb so a different key or a changed lpb can never reuse a stale table
# -- silently filtering against the wrong renumber table would corrupt the matrix, not just slow
# it down. Symlink (not copy): renumber.gz is ~385 MB and every consumer only reads it.
FRC="${CADO_FREEREL_CACHE:-}"
FRKEY=""
if [ -n "$FRC" ]; then FRKEY=$(cat "$POLY" 2>/dev/null | md5sum | cut -c1-16)-lpb$LPB; fi
if [ -n "$FRC" ] && [ -s "$FRC/$FRKEY/renumber.gz" ] && [ -s "$FRC/$FRKEY/freerel.gz" ] \
   && [ -s "$FRC/$FRKEY/freerel.log" ]; then
  ln -sf "$FRC/$FRKEY/freerel.gz"  "$WORK/freerel.gz"
  ln -sf "$FRC/$FRKEY/renumber.gz" "$WORK/renumber.gz"
  cp -f  "$FRC/$FRKEY/freerel.log" "$WORK/freerel.log"
  echo "[cado-filter] freerel CACHE HIT ($FRKEY) -- skipped 11s $(date +%T)"
else
  "$BIN/freerel" -poly "$POLY" -lpb0 "$LPB" -lpb1 "$LPB" \
      -out "$WORK/freerel.gz" -renumber "$WORK/renumber.gz" -t "$NT" \
      > "$WORK/freerel.log" 2>&1 || { echo "[cado-filter] freerel FAILED" >&2; exit 1; }
  # Populate the cache for the NEXT probe in this run (re-probe loop / ladder), atomically:
  # build in a temp dir and rename, so a probe killed mid-write can never leave a half table
  # that a later probe would symlink and filter against.
  if [ -n "$FRC" ]; then
    _t="$FRC/.tmp.$$"; rm -rf "$_t"; mkdir -p "$_t" 2>/dev/null || true
    if cp -f "$WORK/freerel.gz" "$WORK/renumber.gz" "$WORK/freerel.log" "$_t/" 2>/dev/null; then
      rm -rf "$FRC/$FRKEY"; mv -f "$_t" "$FRC/$FRKEY" 2>/dev/null || rm -rf "$_t"
    else rm -rf "$_t"; fi
  fi
fi
NPRIMES=$(grep -aoE 'nprimes=[0-9]+' "$WORK/freerel.log" | tail -1 | cut -d= -f2)
[ -n "$NPRIMES" ] || { echo "[cado-filter] could not read nprimes" >&2; exit 1; }
echo "[cado-filter] renumber: nprimes=$NPRIMES  $(date +%T)"

# --- 1b. EXCLUDE FREE RELATIONS FOR PRIMES DIVIDING THE LEADING COEFFICIENT (2026-08-24) ----
# msieve's own relation parser reconstructs a CADO free relation ("p,0:") itself -- it takes an
# unconditional early return for b==0 and never reads anything after the colon (gnfs/relation.c
# nfs_read_relation), re-deriving the algebraic roots via poly_get_zeros(). That reconstruction
# rejects any p dividing the leading coefficient (c5): `high_coeff==0` -> `return -4`, because such
# a p needs a PROJECTIVE root (a point at infinity) msieve's reconstruction has no representation
# for. CADO's own freerel correctly emits 6 ideals for that p (5 finite + 1 at infinity, verified
# against a live freerel.gz); msieve can only ever produce 5, so the parse is unconditionally
# fatal -- "error: relation N corrupt", and msieve_la calls exit(-1) INSTANTLY, on the very first
# -nc2 that reads a cycle referencing it. Reproduced live in a cold e2e (2026-08-24): p=11 | c5=
# 34320 crashed -nc2 the first time SKIP_NC1's checkpoint cycles referenced it, discarding an
# already-successful CADO MATRIX verdict and falling back to msieve's own filter, which then came
# up 1,000,000 relations short on the SAME relations. Since polyselect deliberately searches a
# smooth/composite c5 zone, virtually every winning polynomial has small prime factors here --
# this is not a rare edge case, it is the common case for any probe that reaches MATRIX.
# Fix: never hand purge a free relation msieve cannot reconstruct in the first place. freerel.gz
# is consumed by purge as a plain relation file (the invocation below never passes -renumber), so
# dropping lines here is local and safe: nothing in dup1/dup2/purge/merge/replay/cado2msieve needs
# renumber.gz touched, and the excluded ideals simply never appear in any relation or cycle.
# NOTE: this covers the PROVEN failure mode (p | c5). A prime dividing the polynomial's
# discriminant (a ramified/repeated root) could in principle hit the same `num_roots != degree`
# check with high_coeff != 0; that has not been observed and is not covered here.
C5=$(grep -aoE '^c5:[[:space:]]*-?[0-9]+' "$POLY" 2>/dev/null | tail -1 | grep -oE -- '-?[0-9]+$')
if [ -n "${C5:-}" ] && [ "$C5" != "0" ] && [ -s "$WORK/freerel.gz" ]; then
  _n=${C5#-}; _p=2; BADHEX=""
  while [ $(( _p * _p )) -le "$_n" ]; do
    while [ $(( _n % _p )) -eq 0 ]; do
      BADHEX="$BADHEX $(printf '%x' "$_p")"; _n=$(( _n / _p ))
    done
    _p=$(( _p + 1 ))
  done
  [ "$_n" -gt 1 ] && BADHEX="$BADHEX $(printf '%x' "$_n")"
  BADHEX=$(printf '%s\n' $BADHEX | sort -u)
  if [ -n "$BADHEX" ]; then
    PATTERN=$(printf '%s\n' $BADHEX | sed 's/^/^/; s/$/,0:/' | paste -sd'|')
    _frbefore=$(gzip -dc "$WORK/freerel.gz" 2>/dev/null | wc -l)
    if gzip -dc "$WORK/freerel.gz" 2>/dev/null | grep -avE "$PATTERN" | gzip -c > "$WORK/freerel.gz.filtered" \
        && [ -s "$WORK/freerel.gz.filtered" ]; then
      mv -f "$WORK/freerel.gz.filtered" "$WORK/freerel.gz"
      _frafter=$(gzip -dc "$WORK/freerel.gz" 2>/dev/null | wc -l)
      echo "[cado-filter] excluded $(( _frbefore - _frafter )) free relation(s) unreconstructable" \
           "by msieve (p | c5=$C5, hex $(printf '%s' "$BADHEX" | tr ' ' ',')) $(date +%T)"
    else
      rm -f "$WORK/freerel.gz.filtered"
      echo "[cado-filter] WARNING: bad-prime free-relation filter produced no output --" \
           "leaving freerel.gz unfiltered (p | c5=$C5 may still crash -nc2)" >&2
    fi
  fi
fi

# --- 2. strip msieve free relations, then split ------------------------------
# `msieve -nc1` MUTATES ITS INPUT IN PLACE, appending free relations as "p,0:"
# (Pipeline.md §9). If a probe ran on this dump those lines are present, and CADO
# must not see them (it makes its own). Note the trailing colon -- the pattern in
# README.msieve ('^[0-9]+,0$') does NOT match this format and silently strips zero.
# !! DO NOT `readlink -f` $RELS !!  On the real validator path rels.txt is an
# ANONYMOUS MEMFD exposed as /proc/<pid>/fd/N. That magic link is openable and
# stat-able, but `readlink -f` DEREFERENCES it to the literal string
# "/memfd:rels.txt (deleted)", which is not a path. Symlinking to that yields a
# DANGLING link: `head` reads nothing, the compression probe below silently
# concludes "unsupported", and dup1 dies with SIGABRT. Measured 2026-08-05 in the
# first full breaking_rsa.py run -- invisible in every /dev/shm test, where rels.txt
# is an ordinary file. Same defect class as run_filter.sh's abspath note.
STRIPPED="$WORK/rels_nofree"
RELKB=$(_kb_size "$RELS" 2>/dev/null || echo 0)
if grep -qaE '^[0-9]+,0:' "$RELS" 2>/dev/null; then
  # !! THIS BRANCH WRITES A FULL SECOND COPY OF THE RELATION FILE ONTO $WORK !!
  # On the common path $STRIPPED is a symlink and costs nothing. It only becomes a real ~8 GB
  # file when msieve -nc1 has already appended its own free relations in place (it mutates its
  # input -- Pipeline.md 9), which the REL_CKPT ladder can reach: a CADO probe returning any rc
  # other than 0 or 2 falls back to msieve -nc1 on $WORK/rels.txt, and the NEXT ladder probe then
  # lands here. Against a 10 GB tmpfs already peaking at 8.7 GB that is a certain ENOSPC, so it
  # must be checked BEFORE the write rather than discovered as a truncated file.
  # !! BUDGET FOR THE COPY *AND* THE SPLIT THAT FOLLOWS IT (2026-08-23). !!
  # The old test only asked whether the ~8 GB copy itself would fit (1.05x RELKB). On a 10 GiB
  # tmpfs it therefore PASSED, wrote the full copy, and then died 70 lines later at the headroom
  # guard -- which needs a FURTHER RELKB free for the dup1 split. Net effect: ~2 min and 8 GB of
  # tmpfs churn to reach a refusal that was arithmetically certain before the first byte was
  # written. Charge both halves up front so the refusal is instant and states the real reason.
  # This is reachable in production: a CADO probe returning any rc other than 0 or 2 falls back to
  # msieve -nc1, which mutates rels.txt in place by appending its free relations, and the NEXT
  # ladder probe then lands in this branch.
  _fkb=$(_kb_free "$WORK")
  _need_strip=$(( RELKB + RELKB / 20 + RELKB ))    # copy (+5% slack) + the split that follows
  if [ -n "${_fkb:-}" ] && [ "${RELKB:-0}" -gt 0 ] && [ "$_fkb" -lt "$_need_strip" ]; then
    echo "[cado-filter] FATAL: a stripped copy plus its split needs ~$(( _need_strip / 1024 )) MB" \
         "over a $(( RELKB / 1024 )) MB relation file, but only $(( _fkb / 1024 )) MB is free on" \
         "$(_mount_of "$WORK") -- refusing before writing the copy" >&2
    exit 3
  fi
  echo "[cado-filter] stripping msieve free relations"
  grep -avE '^[0-9]+,0:' "$RELS" > "$STRIPPED" || {
    echo "[cado-filter] FATAL: could not write $STRIPPED (out of space on $(_mount_of "$WORK")?)" >&2
    exit 3; }
else
  ln -sf "$RELS" "$STRIPPED"     # $RELS is already absolute+openable (cado_probe.sh abspath)
fi
[ -r "$STRIPPED" ] && [ -s "$STRIPPED" ] || {
  echo "[cado-filter] FATAL: relation input $STRIPPED is unreadable/empty (dangling memfd symlink?)" >&2
  exit 3; }

# Pre-create the slice dirs and DROP -mkdir: with -outfmt, CADO opens the output
# through a shell pipe ("gzip -c --fast > f") BEFORE it creates the directory, so
# the first slice dies with "Directory nonexistent" and rc=141. Pre-creating side-
# steps that ordering bug entirely.
NSLICES=4
rm -rf "$WORK/dup1"; for _i in $(seq 0 $((NSLICES-1))); do mkdir -p "$WORK/dup1/$_i"; done
# freerel.gz is ALREADY renumbered (hex a, ideal indices) and must NOT go through
# dup1 -- dup1 parses a as decimal and dies on the hex digits. It joins at purge.
# -outfmt .gz keeps the working set ~3.5x smaller. THE VALIDATOR /tmp IS A 10 GB
# TMPFS and the uncompressed split is ~8.6 GB on its own -- see the DISK section.
# NB: stock CADO REJECTS -outfmt (is_supported_compression_format compares char*
# POINTERS, so no user string ever matches); cado-local.sh documents the strcmp fix.
# Probe on a 100-line sample (cheap, and never touches the real split).
DUP1FMT=""; RELEXT=""
head -100 "$STRIPPED" > "$WORK/.fmtchk.in" 2>/dev/null
mkdir -p "$WORK/.fmtchk/0"
if "$BIN/dup1" -out "$WORK/.fmtchk" -prefix t -n 0 -outfmt .gz "$WORK/.fmtchk.in" \
     >/dev/null 2>&1 && [ -s "$WORK/.fmtchk/0/t.0000.gz" ]; then
  DUP1FMT="-outfmt .gz"; RELEXT=".gz"
  echo "[cado-filter] compressed intermediates ON (~3.5x less /tmp)"
else
  echo "[cado-filter] compressed intermediates OFF -- needs ~8.6 GB of /tmp." >&2
  echo "[cado-filter]   Stock CADO rejects -outfmt: is_supported_compression_format()" >&2
  echo "[cado-filter]   compares char* POINTERS, so no user string ever matches." >&2
  echo "[cado-filter]   Rebuild with the strcmp fix (see cado-local.sh)." >&2
fi
rm -rf "$WORK/.fmtchk" "$WORK/.fmtchk.in"

# ---- /tmp HEADROOM GUARD -- BOTH PATHS (2026-08-20) ------------------------
# This check used to sit INSIDE the `compression OFF` branch above, so on the compressed path --
# the one every validator run actually takes -- nothing checked free space at all. The comment
# there asserted it "would also catch a shrunken tmpfs on the compressed path"; as written it
# could never execute there. Hoisted out and made to cover both.
#
# It is not academic. Peak $WORK on one key08 run each: EPYC 6.15 GB, Xeon 8.69 GB, against a
# 10 GB tmpfs. Those two disagree because they are peaks of PERIODIC SAMPLES taken by different
# harnesses -- a sampler missing the maximum is the ordinary explanation, not a contradiction --
# so plan against the larger. An ENOSPC part-way through dup1/dup2/purge surfaces ~3.3h into a
# run as a generic tooling failure; breaking_rsa.py then holds for external termination, so the
# validator records WallTimeFailure with nothing anywhere saying storage was the cause. That
# misattribution is the real defect. Refusing here costs one msieve fallback and states why.
#
# SIZED FROM THE ACTUAL RELATION FILE, not a constant, so it keeps holding when the relation
# count changes (a harder key, or a larger lpb). Calibrated on the Xeon run, where $WORK peaked
# at 8.49 GiB over an 8.01 GiB input with ~0.5 GB of renumber/freerel already present at this
# point -- i.e. almost exactly 1.00x the input still to be written:
#     compressed    refuse below 1.00x   warn below 1.15x
#     uncompressed  refuse below 2.00x   warn below 2.15x   (the split alone adds ~1x on top)
# Those thresholds leave the known-good Xeon run (1.19x headroom) silent, warn at ~5% more
# relations, and refuse at ~19% more. CADO_MIN_FREE_KB still overrides both with a flat floor.
# The cache check must run BEFORE the headroom guard: the two paths do NOT need the same space.
# ---- INCREMENTAL dup1/dup2 (2026-08-21) ------------------------------------
# If cado_predup.sh has already deduplicated a PREFIX of this relation file --
# work done under the running sieve, on cores that were idle -- then dup1 only
# has to split the DELTA, and dup2 only has to renumber the delta: it loads the
# prefix slices as "already renumbered" at 618k rel/s instead of re-renumbering
# them at 93k rel/s. That is half the probe's cost, and the probe is on the
# critical path (the sieve running under it is discarded the moment it returns
# MATRIX). See cado_predup.sh for the measurement and the correctness argument.
#
# Every condition below is a REFUSAL to use the cache, falling back to the
# one-shot path. A wrongly-matched cache would filter against relations that are
# not in this file, which corrupts the matrix rather than merely slowing it down,
# so this is deliberately paranoid and deliberately cheap to fail.
PRE="${CADO_PREDEDUP_DIR:-}"
PRE_OK=0
if [ -n "$PRE" ] && [ -f "$PRE/READY" ] && [ -s "$PRE/meta" ]; then
  NBYTES=0; NRELS=0; TAILMD5=""; SLICES=0; UNIQ=0
  . "$PRE/meta"
  _cursz=$(stat -Lc %s "$STRIPPED" 2>/dev/null || echo 0)
  if [ ! -L "$STRIPPED" ]; then
    # $STRIPPED is a real file only when msieve free relations had to be removed,
    # which renumbers every byte offset and voids the prefix.
    echo "[cado-filter] pre-dedup cache SKIPPED: relation file was stripped in place" >&2
  elif [ "${SLICES:-0}" != "$NSLICES" ] || [ "${NBYTES:-0}" -le 0 ]; then
    echo "[cado-filter] pre-dedup cache SKIPPED: unusable meta" >&2
  elif [ "$_cursz" -le "${NBYTES:-0}" ]; then
    echo "[cado-filter] pre-dedup cache SKIPPED: prefix is not shorter than this snapshot" >&2
  else
    # Re-read the last 4 KB of the prefix FROM THIS SNAPSHOT and compare against
    # the fingerprint the cache recorded. dd seeks (the input is a memfd), so this
    # costs one 4 KB read, not a pass over 8 GB.
    _tm=$(dd if="$STRIPPED" bs=4096 skip=$(( NBYTES - 4096 )) count=4096 \
             iflag=skip_bytes,count_bytes 2>/dev/null | md5sum | cut -d' ' -f1)
    if [ "$_tm" != "$TAILMD5" ]; then
      echo "[cado-filter] pre-dedup cache SKIPPED: prefix fingerprint mismatch" >&2
    else
      _allsl=1
      for i in 0 1 2 3; do [ -s "$PRE/s$i/rel.0000$RELEXT" ] || _allsl=0; done
      if [ "$_allsl" = 1 ]; then PRE_OK=1
      else echo "[cado-filter] pre-dedup cache SKIPPED: missing slice files" >&2; fi
    fi
  fi
fi


_freekb=$(_kb_free "$WORK")
# ---- PATH-AWARE BUDGET (2026-08-23) -------------------------------------------------------
# !! THE GUARD USED TO CHARGE THE ONE-SHOT BUDGET TO THE INCREMENTAL PATH. THAT WAS THE BUG. !!
# It ran BEFORE the cache check above and always demanded ~1.00x RELKB -- the cost of splitting
# EVERY relation. On the incremental path dup1 only ever writes the DELTA, so that requirement is
# roughly 4 GB too large. Measured cold e2e 2026-08-23: it refused at 7,694 MB free against a
# 7,842 MB "requirement" while the work it was about to do needed ~5,100 MB. exit 3 -> msieve
# fallback -> the CADO filter never ran, on a run that had ample room.
#
# The two paths, decomposed against the Xeon calibration (peak 8.49 GiB over an 8.01 GiB input):
#     purge/merge/replay   ~0.50x RELKB   paid on BOTH paths -- same unique relations either way
#     one-shot split       ~0.52x RELKB   dup1 writes gz slices for EVERY relation   (= ~1.02x)
#     incremental          ~1.5x DELTA    delta.rels (raw, transient) + its gz split + its dup2
#                                         output; the prefix slices are ALREADY WRITTEN
# So the incremental path's peak $WORK is LOWER than the one-shot path's, not higher: the 2,172 MB
# of prefix slices REPLACE a 4,280 MB full split. Pre-dedup is storage-POSITIVE. The guard only
# ever saw it as negative because it measured free space after the cache was written and then
# compared it against work the incremental path does not do.
if [ -n "${_freekb:-}" ] && [ "${RELKB:-0}" -gt 0 ]; then
  _budget() {
    if [ "$PRE_OK" = 1 ]; then
      _deltakb=$(( (${_cursz:-0} - ${NBYTES:-0}) / 1024 ))
      [ "$_deltakb" -lt 0 ] && _deltakb=0
      _needkb=$(( _deltakb * 3 / 2 + RELKB / 2 ))
      _lbl="incremental (delta $(( _deltakb / 1024 )) MB + shared purge/merge)"
    elif [ -n "$DUP1FMT" ]; then
      _needkb=$(( RELKB )); _lbl="compressed one-shot"
    else
      _needkb=$(( RELKB * 2 )); _lbl="uncompressed one-shot"
    fi
    _warnkb=$(( _needkb * 115 / 100 ))
    if [ -n "${CADO_MIN_FREE_KB:-}" ]; then _needkb="$CADO_MIN_FREE_KB"; _warnkb="$CADO_MIN_FREE_KB"; fi
  }
  _budget
  # ---- LAST-RESORT SELF-HEAL: THE CACHE IS EXPENDABLE, THE FILTER IS NOT ------------------
  # If even the (much smaller) incremental budget does not fit, the cache itself is the only
  # thing in $WORK we are allowed to delete. Dropping it costs ~128 s of dup1/dup2 and keeps the
  # filter; keeping it costs the filter entirely, and losing the CADO filter is what makes hard
  # keys unsolvable (Pipeline.md §11). Note this can only ever HELP: it frees the slices AND
  # switches the budget back to the one-shot rule, which is what we would then actually run.
  if [ "$_freekb" -lt "$_needkb" ] && [ "$PRE_OK" = 1 ] && [ -n "${PRE:-}" ] && [ -d "$PRE" ]; then
    _pdkb=$(du -sk "$PRE" 2>/dev/null | cut -f1)
    echo "[cado-filter] incremental budget short by $(( (_needkb - _freekb) / 1024 )) MB --" \
         "dropping the ${_pdkb:-?} KB pre-dedup cache and taking the one-shot path" >&2
    rm -rf "${PRE:?}"
    PRE=""; PRE_OK=0; CADO_PREDEDUP_DIR=""
    _freekb=$(_kb_free "$WORK"); _budget
  fi
  if [ "$_freekb" -lt "$_needkb" ]; then
    echo "[cado-filter] FATAL: $_lbl intermediates need ~$(( _needkb / 1024 )) MB over a" \
         "$(( RELKB / 1024 )) MB relation file, but only $(( _freekb / 1024 )) MB is free on" \
         "$(_mount_of "$WORK") -- refusing to start" >&2
    exit 3
  elif [ "$_freekb" -lt "$_warnkb" ]; then
    echo "[cado-filter] WARNING: only $(( _freekb / 1024 )) MB free on $(_mount_of "$WORK") for" \
         "$_lbl intermediates over a $(( RELKB / 1024 )) MB relation file. This fits, but a key" \
         "needing more relations will not -- see the HEADROOM GUARD note." >&2
  fi
  echo "[cado-filter] storage: $_lbl -- need ~$(( _needkb / 1024 )) MB, free $(( _freekb / 1024 )) MB"
fi

if [ "$PRE_OK" = 1 ]; then
  echo "[cado-filter] INCREMENTAL: reusing $NRELS pre-deduplicated relations ($UNIQ unique)"
  # The delta is a byte range of an append-only file, so tail -c seeks straight
  # to it. It is written out, split, and deleted immediately -- at its peak it is
  # smaller than the full dup1 split it replaces, so /tmp is not worse off.
  rm -rf "$WORK/dup1raw"; for i in 0 1 2 3; do mkdir -p "$WORK/dup1raw/$i"; done
  tail -c +$(( NBYTES + 1 )) "$STRIPPED" > "$WORK/delta.rels" || {
    echo "[cado-filter] FATAL: could not write the delta (out of space on $(_mount_of "$WORK")?)" >&2
    exit 3; }
  "$BIN/dup1" -out "$WORK/dup1raw" -prefix del -n 2 $DUP1FMT "$WORK/delta.rels" \
      > "$WORK/dup1.log" 2>&1 || { _diskerr "$WORK/dup1.log" dup1
                                   echo "[cado-filter] dup1 (delta) FAILED" >&2; exit 1; }
  rm -f "$WORK/delta.rels"
  echo "[cado-filter] dup1 (delta only) done $(date +%T)"

  # --- 3. dedup: prefix slice (already renumbered) + delta slice, per slice ---
  # -outdir keeps the output off the prefix cache, so the cache stays valid for
  # the next probe if this one comes back SHORT. dup2 does not re-emit an
  # already-renumbered input, so the only file written is the delta's.
  pids=""
  for i in 0 1 2 3; do
    np=$(grep -aoE "slice $i received [0-9]+" "$PRE/dup1.log" | grep -oE '[0-9]+$')
    nd=$(grep -aoE "slice $i received [0-9]+" "$WORK/dup1.log" | grep -oE '[0-9]+$')
    "$BIN/dup2" -poly "$POLY" -renumber "$WORK/renumber.gz" \
        -nrels $(( ${np:-20000000} + ${nd:-1000000} )) -outdir "$WORK/dup1/$i" \
        "$PRE/s$i/rel.0000$RELEXT" "$WORK/dup1raw/$i/del.0000$RELEXT" \
        > "$WORK/dup2_$i.log" 2>&1 &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p" || { for _l in "$WORK"/dup2_*.log; do _diskerr "$_l" dup2; done
                                      echo "[cado-filter] dup2 FAILED" >&2; exit 1; }; done
  rm -rf "$WORK/dup1raw"
  for i in 0 1 2 3; do
    [ -s "$WORK/dup1/$i/del.0000$RELEXT" ] || { echo "[cado-filter] dup2 produced no delta slice $i" >&2
                                                exit 1; }
  done
  # ---- ORDER IS LOAD-BEARING ------------------------------------------------
  # purge numbers relations by the order it reads them, and its singleton/clique
  # cascade is order-sensitive, so a different order yields a different (still
  # valid, but not comparable) matrix. The one-shot path feeds purge slice 0,1,2,3
  # where each slice holds its prefix relations followed by its delta relations,
  # in original file order. Listing the two halves of each slice ADJACENTLY -- not
  # all four prefixes then all four deltas -- reproduces that stream exactly.
  # Verified rather than reasoned: concatenating <prefix slice i> and <delta slice
  # i> is BYTE-IDENTICAL to the one-shot path's slice i, for all four slices.
  PURGE_IN=""
  for i in 0 1 2 3; do
    PURGE_IN="$PURGE_IN $PRE/s$i/rel.0000$RELEXT $WORK/dup1/$i/del.0000$RELEXT"
  done
else
  "$BIN/dup1" -out "$WORK/dup1" -prefix rel -n 2 $DUP1FMT "$STRIPPED" \
      > "$WORK/dup1.log" 2>&1 || { _diskerr "$WORK/dup1.log" dup1; echo "[cado-filter] dup1 FAILED" >&2; exit 1; }
  echo "[cado-filter] dup1 done $(date +%T)"

  # --- 3. dedup per slice (duplicates hash to the same slice) ----------------
  pids=""
  for i in 0 1 2 3; do
    n=$(grep -aoE "slice $i received [0-9]+" "$WORK/dup1.log" | grep -oE '[0-9]+$')
    "$BIN/dup2" -poly "$POLY" -renumber "$WORK/renumber.gz" -nrels "${n:-20000000}" \
        "$WORK/dup1/$i/rel.0000$RELEXT" > "$WORK/dup2_$i.log" 2>&1 &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p" || { for _l in "$WORK"/dup2_*.log; do _diskerr "$_l" dup2; done
                                      echo "[cado-filter] dup2 FAILED" >&2; exit 1; }; done
  PURGE_IN=""
  for i in 0 1 2 3; do PURGE_IN="$PURGE_IN $WORK/dup1/$i/rel.0000$RELEXT"; done
fi
# Shell arithmetic, NOT `bc`: the runtime image installs only python3/bash/coreutils/
# findutils (Dockerfile), so bc is absent and this would have failed in the container
# while working fine in dev.
UNIQ=0
for _n in $(grep -h 'At the end:.*remaining relations' "$WORK"/dup2_*.log 2>/dev/null \
            | grep -oE '[0-9]+'); do UNIQ=$((UNIQ+_n)); done
echo "[cado-filter] dup2 done: $UNIQ unique  $(date +%T)"

# --- 4. purge (singleton + clique removal) ----------------------------------
# $PURGE_IN is set by step 2/3 above: four slice files on the one-shot path, or
# four prefix-cache slices plus four delta slices on the incremental one.
"$BIN/purge" -out "$WORK/purged.gz" -col-max-index "$NPRIMES" -keep "$KEEP" -t "$NT" \
    $PURGE_IN "$WORK/freerel.gz" \
    > "$WORK/purge.log" 2>&1
# !! THE SHORT TEST MUST BE purged.gz, NOT A LOG GREP (fixed 2026-08-06) !!
# This guard used to be `grep -aq 'nrows=' purge.log`, which NEVER FIRES. purge prints an
# `nrows=... ncols=... excess=...` line for EVERY singleton-removal iteration, so the pattern
# matches just as happily on a run that found nothing:
#     Sing. rem.:   iter 004: nrows=68 ncols=26000 excess=-25932
#     number of rows < number of columns + keep
#     Final values: nrows=68 ncols=26000 excess=-25932
# On that run purge writes NO purged.gz, so the guard passed, `merge` was started on a file that
# does not exist, and merge died with SIGABRT ("purged.gz: No such file or directory"). That
# reached the caller as `merge FAILED` -> exit 1 -> cado_probe.sh maps any rc != 2 to exit 3
# "TOOLING FAILURE", so A GENUINE RELATION SHORTFALL WAS REPORTED AS BROKEN TOOLING.
#
# Consequences on a real key, all of them expensive:
#   * run_probe() falls back to a full `msieve -nc1` (~570s) on EVERY probe -- and the speculative
#     probe at 97% of REL_TARGET is DESIGNED to come back short, as is every ladder step below the
#     crossover. Spec probe + 4 ladder steps = ~45 min of wall burned on a filter that CADO exists
#     to replace precisely because it cannot build a matrix on hard keys.
#   * the ladder's rc=2 branch (factor_msv.sh, "CADO ran and said SHORT -- a real verdict") became
#     DEAD CODE: CADO could not return 2 through this path.
#   * the 2026-08-05 fix for "tooling failure reported as SHORT" inverted this into "SHORT reported
#     as tooling failure" -- the old blanket `exit 2` was accidentally right for this case.
# The robust test is purge's PRODUCT: no purged.gz means no matrix, whatever the log says.
if [ ! -s "$WORK/purged.gz" ] || grep -aq 'number of rows < number of columns' "$WORK/purge.log"; then
  # !! DISTINGUISH ENOSPC FROM A REAL SHORTFALL BEFORE EXITING 2 !!  rc=2 is a LOAD-BEARING
  # verdict: factor_msv.sh's run_probe treats it as "CADO ran and said SHORT" and deliberately
  # does NOT retry on msieve, so the ladder just keeps sieving. A purge that died on a full
  # tmpfs also leaves no purged.gz, and would therefore be laundered into "this key needs more
  # relations" -- the sieve would run on to REL_TARGET chasing relations that were never the
  # problem, and the run would end at the wall with storage never mentioned. Exit 3 instead so
  # the caller degrades to the msieve backend and the log names the real cause.
  if grep -qaiE 'No space left on device|ENOSPC|write error|disk full' "$WORK/purge.log" 2>/dev/null; then
    _diskerr "$WORK/purge.log" purge
    echo "[cado-filter] purge died out of space -- this is NOT a relation shortfall" >&2
    exit 3
  fi
  echo "[cado-filter] purge produced no matrix -- excess-limited, MORE RELATIONS NEEDED" >&2
  grep -aoE 'nrows=[0-9]+ ncols=[0-9]+ excess=-?[0-9]+' "$WORK/purge.log" | tail -1 >&2
  tail -3 "$WORK/purge.log" >&2
  exit 2
fi
# anchor on the real summary line -- a plain 'nrows=' also matches "weight*nrows="
grep -aoE 'nrows=[0-9]+ ncols=[0-9]+ excess=[0-9-]+' "$WORK/purge.log" | tail -1 | sed 's/^/[cado-filter] purge: /'
# DISK: purge has consumed the split relations and the renumber table; nothing
# downstream reads either. On the validator's 10 GB /tmp this is the difference
# between fitting and ENOSPC mid-filter, ~3.3h into a run.
# On the incremental path the prefix cache is purge's input too, and this is the
# MATRIX path -- there is no next probe -- so it is freed here with everything
# else. A SHORT verdict exits above without reaching this line, which is exactly
# what keeps the cache alive for the re-probe.
rm -rf "$WORK/dup1" "$WORK/renumber.gz" "$WORK/freerel.gz"
if [ "${PRE_OK:-0}" = 1 ]; then rm -rf "$PRE"; fi

# --- 5. merge ---------------------------------------------------------------
# TD is a CEILING set by msieve's MAX_COL_IDEALS, not a free knob -- see header.
"$BIN/merge" -out "$WORK/history" -mat "$WORK/purged.gz" -target_density "$TD" -t "$NT" \
    > "$WORK/merge.log" 2>&1 || { _diskerr "$WORK/merge.log" merge; echo "[cado-filter] merge FAILED" >&2; exit 1; }
grep -a 'Final matrix' "$WORK/merge.log" | sed 's/^/[cado-filter] /'

# --- 6. cycles in msieve format --------------------------------------------
# -skip 0 is MANDATORY with --for_msieve (replay refuses otherwise).
"$BIN/replay" --for_msieve -skip 0 -purged "$WORK/purged.gz" -his "$WORK/history" \
    -out "$OUT.cyc" > "$WORK/replay.log" 2>&1 || { _diskerr "$WORK/replay.log" replay; echo "[cado-filter] replay FAILED" >&2; exit 1; }

# --- 7. bridge CADO's purged format to what msieve expects ------------------
# Stream it: decompressing to disk would cost ~1.9 GB of /tmp for no reason. The
# entry count comes from purge.log so cado2msieve needs only ONE sequential pass.
# ANCHOR on the final summary line and take the LAST one. A loose 'nrows=' + head -1
# grabs purge's PRE-singleton-removal count (42,421,830 vs the real 14,210,778 on key01)
# and the map header would then claim 3x the entries the file holds.
NPUR=$(grep -aoE 'nrows=[0-9]+ ncols=[0-9]+ excess=[0-9-]+' "$WORK/purge.log" \
       | tail -1 | grep -oE 'nrows=[0-9]+' | cut -d= -f2)
[ -n "$NPUR" ] || { echo "[cado-filter] could not read purge nrows" >&2; exit 1; }
"$BIN/cado2msieve" "$STRIPPED" <(gzip -dc "$WORK/purged.gz") "$OUT" "$NPUR" 2>&1 | sed 's/^/[cado-filter] /'
rc=${PIPESTATUS[0]}
# Self-check: the map header MUST equal the number of entries actually written, or
# msieve reads past the end. Cheap, and catches any future miscount at build time
# rather than 3.5h into a validator run.
if [ "$rc" = 0 ] && [ -s "$OUT.purged" ]; then
  _hdr=$(head -1 "$OUT.purged"); _cnt=$(( $(wc -l < "$OUT.purged") - 1 ))
  [ "$_hdr" = "$_cnt" ] || { echo "[cado-filter] FATAL: purge map header $_hdr != $_cnt entries" >&2; rc=1; }
fi
[ "$rc" = 0 ] || { echo "[cado-filter] cado2msieve FAILED (rc=$rc)" >&2; exit 1; }

echo "[cado-filter] done $(date +%T) -> $OUT{,.purged,.cyc}"
