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

# cado_filter.sh -- CADO filtering: freerel -> dup1 -> dup2 -> purge -> merge -> replay,
# emitting cycles in msieve format.
#
# CADO replaces msieve's filter because msieve fails to build a matrix from relation sets that
# are demonstrably sufficient, and every msieve filter knob is inert on this build. CADO builds
# one from the SAME relations, faster, and emits ~2x more free relations -- excess msieve leaves
# on the table.
#
# TWO CONSTRAINTS THAT WILL BITE IF REDISCOVERED:
#  1. msieve's CADO interop is pinned to a 2013-era format. Modern purged files start with a '#'
#     header that msieve's strtoul reads as 0, so it silently sees ZERO cycles. cado2msieve
#     bridges this by emitting relations in PURGE ORDER with an identity index map.
#  2. msieve's stock MAX_COL_IDEALS=1000 overflows on CADO relation-sets, leaving a truncated
#     matrix that dies later as a zero-size array. msieve_la must be rebuilt with 16384.
#     Lowering target_density is NOT a sufficient workaround.
#
# DISK: the validator's /tmp is a 10 GB tmpfs and the workdir lives there. Peak here is ~7.5 GB,
# kept down by compressed intermediates, deleting dup1/renumber the moment purge stops reading
# them, and streaming purged.gz into cado2msieve rather than decompressing it to disk.
#
# usage: cado_filter.sh <rels> <poly.cado> <N> <workdir> <out_prefix>
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

# --- 1b. Exclude free relations for primes dividing the leading coefficient -------------------
# msieve cannot reconstruct those, so leaving them in makes the cycle file reference ideals the
# matrix build will never see.
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
# This branch writes a full second copy of the relation file onto $WORK -- on a 10 GB tmpfs that
# is the difference between fitting and ENOSPC mid-filter, hours into a run.
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

# ---- /tmp headroom guard ---------------------------------------------------------------------
# Refuse up front rather than half-measure: a filter that runs out of disk produces a SHORT
# verdict indistinguishable from a real relation shortfall, and the caller then sieves longer.
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
# The budget must be PATH-AWARE. Charging the one-shot cost to the incremental path refused a
# filter that would have fit, and the refusal surfaced as a tooling failure -> msieve fallback,
# so the CADO filter never ran at all. The pre-dedup warm-up REDUCES peak workdir usage.
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
# The SHORT test must be purged.gz itself, not a log grep: purge's wording changes between
# versions, and a grep that stops matching turns a real shortfall into a silent success.
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
