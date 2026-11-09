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
# cado_incdup.sh -- INCREMENTAL dup1+dup2 for ONE sieve chunk (lever B, RSA-530 c160).
# =============================================================================================
# Today every probe re-runs dup1+dup2 over ALL raw chunks (~6-9 min at 150-200M relations). This
# script moves that work OFF the critical path: as soon as chunk k lands in its memfd, it
#   1. dup1-splits chunk k into the 16 scratch slices $MEM/d1/<i>/rel.0000.gz -- the same 16-way
#      split as the one-shot `dup1 -n 4` (slice = top 4 bits of a hash of (a,b), so a relation and
#      all its duplicates land in the same slice whichever chunk they come from);
#   2. dedups them with dup2 in INC_GROUPS groups of slices (one dup2 per group: the c160
#      renumber table costs 15-33 s to load per dup2 process, 16 loads/chunk was 3/4 of the time).
#      Each dup2 gets, as "already renumbered" inputs, the KEY SHADOWS ("a,b:" lines, 1 colon) of the
#      committed parts of chunks 1..k-1 for its slices -- dup2 only hashes (a,b) of renumbered
#      inputs, so a key-only line is equivalent (verified: byte-identical output) and ~4x cheaper
#      to read than the full part -- and the new slices as its only "new" inputs. Output: the
#      deduplicated, renumbered chunk-k part of slice i at $MEM/dd/<kkk>/s<ii>.gz (IMMUTABLE memfd;
#      dup2 names outputs <outdir>/<input basename>, hence the per-slice s<ii>.gz input links).
#   3. writes the key shadow $MEM/kk/<kkk>/s<ii> of each new part (its line count re-verifies the
#      part end to end), then commits chunk k by atomically writing $CACHE/chunk_<k>.done.
# A probe then only needs purge (+merge/replay) over the dd parts of chunks 1..K listed SLICE-MAJOR
# (slice 0: chunks 1..K, slice 1: chunks 1..K, ...): dup1/dup2 preserve input order, so that stream
# is byte-identical to the one-shot dup2 slices (test_equiv.sh demonstrates it on real relations).
#
# Why immutable per-chunk parts instead of one growing file per slice: purge reads its inputs TWICE
# (pass1 + pass2) and a probe runs while chunk k+1 is processed. Nothing a probe reads is ever
# written again, so no locking is needed; a failed/killed step leaves no .done marker and re-running
# it rewrites only chunk k's own parts.
#
# Bugs of the shipped c155 incremental path (cado_predup.sh + cado_filter.sh PRE_OK) avoided here:
#  * a delta slice with ZERO relations is classified by dup2 as "already renumbered"
#    (check_whether_file_is_renumbered returns true on an empty file), so dup2 wrote no output and
#    the old path died "dup2 produced no delta slice" = tooling failure on a small delta. Here an
#    empty slice never reaches dup2; we write a valid empty gzip member for it.
#  * dup2 silently under-counts a truncated/unreadable old input; here every group is checked:
#    "Total so far: nrels=" must equal sum(old unique) + sum(new raw), each new file's nrels= must
#    equal its dup1 count, and each part's key-shadow line count must equal its rem=.
#  * one prefix cache rebuilt from scratch by head -n (byte-offset identity, stripped-file hazards)
#    -> here every chunk is incremental and keyed by chunk number only.
#  * dup2 -outdir always differs from its input dir (patched write-through requires it).
#
# usage: cado_incdup.sh <cache_dir> <mem> <poly.cado> <k> <chunk_file>
#   <mem> must hold memhold links d1/<i>/rel.0000.gz (i=0..15), dd/<kkk>/s<ii>.gz and kk/<kkk>/s<ii>
#   (kkk = 001..MAXCHUNKS, ii = 00..15). <cache_dir> is small (logs, .done files, input links).
#   env: CADO_BIN (binaries)  FRC (freerel cache with renumber.gz, REQUIRED)
#        INC_RENUMBER (renumber table to load; default $FRC/renumber.gz. A plain decompressed copy
#          in a memfd loads ~2x faster but costs ~4.3 GB RAM at c160)
#        INC_GROUPS (dup2 processes per chunk, 1|2|4|8|16, default 4)
#        INC_PAR (concurrent dup2, default 2; each dup2 ~= 4-5 threads: 3 root threads hard-coded)
#        INC_NICE (default 15)
#        INC_PAR_FILE (optional): a file holding the CURRENT allowed dup2 concurrency; re-read before
#          each group launch (every 2 s while throttled), so the caller can raise/lower it mid-chunk
#          to keep sieve host + probe + incdup threads <= NCPU. Unset: INC_PAR, exactly as before.
# exit: 0 committed   1 failure (chunk k NOT committed)
# stdout: "[incdup] COMMIT k=<k> raw=<n> rawcum=<n> uniqcum=<n> t=<s>s"
set -u
CACHE=$1; MEM=$2; POLY=$3; K=$4; CHUNK=$5
HERE=$(cd "$(dirname "$0")" && pwd); BIN=${CADO_BIN:-$HERE}
# SLICE_FMT (.gz default; the orchestrator passes .zst when zstd is available): dup1/dup2 slice + part suffix
SL=${SLICE_FMT:-.gz}; SLRE="\\$SL"
FRC=${FRC:?FRC (freerel cache dir) required}; RENUM=${INC_RENUMBER:-$FRC/renumber$SL}
INC_GROUPS=${INC_GROUPS:-4}; INC_PAR=${INC_PAR:-2}; NI=${INC_NICE:-15}
T0=$(date +%s)
say(){ echo "[incdup] [$(( $(date +%s)-T0 ))s] k=$K $*"; }
fail(){ echo "[incdup] [$(( $(date +%s)-T0 ))s] k=$K FAILED: $*" >&2; echo "[incdup] FAIL k=$K $*"; exit 1; }
dd_(){ printf '%s/dd/%03d/s%02d'"$SL" "$MEM" "$1" "$2"; }   # committed part (gz, renumbered) of chunk $1 slice $2
kk_(){ printf '%s/kk/%03d/s%02d' "$MEM" "$1" "$2"; }      # its key shadow ("a,b:" lines, plain)

case "$K" in ''|*[!0-9]*|0) fail "bad chunk number '$K'";; esac
case "$INC_GROUPS" in 1|2|4|8|16) ;; *) fail "INC_GROUPS must divide 16";; esac
GS=$(( 16 / INC_GROUPS ))
for b in dup1 dup2; do [ -x "$BIN/$b" ] || fail "missing binary $BIN/$b"; done
[ -s "$RENUM" ] && [ -s "$FRC/renumber$SL" ] && grep -q c5filter=done "$FRC/freerel.log" 2>/dev/null || fail "renumber cache $FRC not ready"
[ -r "$CHUNK" ] || fail "chunk file $CHUNK unreadable"
mkdir -p "$CACHE/in" || fail "cannot create $CACHE"
CACHE=$(cd "$CACHE" && pwd) && MEM=$(cd "$MEM" && pwd) || fail "cannot resolve $CACHE / $MEM"   # links below must be absolute
[ "$K" = 1 ] || [ -s "$CACHE/chunk_$((K-1)).done" ] || fail "chunk $((K-1)) not committed (steps must run in order)"
[ -e "$CACHE/chunk_$((K+1)).done" ] && fail "chunk $((K+1)) already committed -- refusing to rewrite chunk $K under it"
for i in $(seq 0 15); do
  [ -L "$MEM/d1/$i/rel.0000$SL" ] || fail "memfd slot d1/$i missing"
  [ -L "$(dd_ "$K" "$i")" ] && [ -L "$(kk_ "$K" "$i")" ] || fail "memfd slot dd|kk/$K/$i missing (MAXCHUNKS too small?)"
done
rm -f "$CACHE/chunk_$K.done" "$CACHE/chunk_$K.done.tmp" "$CACHE"/dup2_${K}_*.log

# previous cumulative per-slice unique counts and totals
declare -a OLD NEW CUM REM; RAWCUM0=0
for i in $(seq 0 15); do OLD[$i]=0; done
if [ "$K" -gt 1 ]; then
  P="$CACHE/chunk_$((K-1)).done"
  for i in $(seq 0 15); do OLD[$i]=$(grep -oE "^u$i=[0-9]+" "$P" | cut -d= -f2); [ -n "${OLD[$i]}" ] || fail "bad $P (u$i)"; done
  RAWCUM0=$(grep -oE '^rawcum=[0-9]+' "$P" | cut -d= -f2); [ -n "$RAWCUM0" ] || fail "bad $P (rawcum)"
fi

# ---- 1. dup1: split this chunk into the 16 scratch slices --------------------------------------
# THREADS: -outfmt .gz forks 16 `gzip -c --fast` writers at once (measured on a 954 MB c160 chunk:
# 14.9 s user CPU, ~3 cores for a few seconds). The caller's INC_TPG budget counts dup2 groups only;
# this burst precedes them, fits the same CPU share, and runs at nice 15 (see factor_c160.sh).
for i in $(seq 0 15); do : > "$MEM/d1/$i/rel.0000$SL"; : > "$(dd_ "$K" "$i")"; : > "$(kk_ "$K" "$i")"; done
nice -n "$NI" "$BIN/dup1" -out "$MEM/d1" -prefix rel -n 4 -outfmt "$SL" "$CHUNK" > "$CACHE/dup1_$K.log" 2>&1 \
  || fail "dup1 (see $CACHE/dup1_$K.log)"
RAW=0
for i in $(seq 0 15); do
  [ -L "$MEM/d1/$i/rel.0000$SL" ] || fail "dup1 replaced memfd link d1/$i (unpatched CADO?)"
  [ -e "$MEM/d1/$i/rel.0001$SL" ] && fail "dup1 wrote a second file for slice $i"
  NEW[$i]=$(grep -aoE "slice $i received [0-9]+" "$CACHE/dup1_$K.log" | grep -oE '[0-9]+$')
  [ -n "${NEW[$i]}" ] || fail "dup1 log has no count for slice $i"; RAW=$((RAW+NEW[$i]))
  ln -sfn "$MEM/d1/$i/rel.0000$SL" "$CACHE/in/s$(printf %02d "$i")$SL" || fail "input link s$i"
done
say "dup1: $RAW relations"

# ---- 2. one dup2 per group of slices, against the key shadows of chunks 1..k-1 -----------------
run_group(){ local g=$1 i j olds=() news=() lg="$CACHE/dup2_${K}_g$1.log" so=0 sn=0
  for i in $(seq $((g*GS)) $((g*GS+GS-1))); do
    for j in $(seq 1 $((K-1))); do olds+=("$(kk_ "$j" "$i")"); done
    so=$((so+OLD[$i]))
    if [ "${NEW[$i]}" = 0 ]; then { [ "$SL" = .zst ] && zstd -q < /dev/null || gzip -c < /dev/null; } > "$(dd_ "$K" "$i")" || return 1   # never hand dup2 an empty new file
    else news+=("$CACHE/in/s$(printf %02d "$i")$SL"); sn=$((sn+NEW[$i])); fi
  done
  [ "${#news[@]}" -gt 0 ] || { echo "NO-NEW-RELATIONS" > "$lg"; return 0; }
  nice -n "$NI" "$BIN/dup2" -poly "$POLY" -renumber "$RENUM" -nrels $(( (so + sn) * 14 / 10 + 1000 )) \
      -outdir "$MEM/dd/$(printf %03d "$K")" "${olds[@]}" "${news[@]}" > "$lg" 2>&1 || return 1
  # ---- verify the group, then write the key shadows of its new parts ----
  local tot nold
  tot=$(grep -aoE 'Total so far: nrels=[0-9]+' "$lg" | tail -1 | grep -oE '[0-9]+$')
  [ "${tot:-x}" = $((so + sn)) ] || { echo "VERIFY: dup2 read ${tot:-?} rels, expected $so old + $sn new (old part truncated?)" >> "$lg"; return 1; }
  nold=$(grep -aoE 'Reading [0-9]+ files already renumbered' "$lg" | grep -oE '[0-9]+' | head -1)
  [ "${nold:-x}" = "${#olds[@]}" ] || { echo "VERIFY: ${nold:-?} files classified renumbered, expected ${#olds[@]}" >> "$lg"; return 1; }
  grep -aq 'hash table is full' "$lg" && { echo "VERIFY: hash table full" >> "$lg"; return 1; }
  for i in $(seq $((g*GS)) $((g*GS+GS-1))); do
    [ "${NEW[$i]}" = 0 ] && continue
    local s nr rem nl; s=$(printf %02d "$i")
    nr=$(grep -aoE "^s$s$SLRE: nrels=[0-9]+" "$lg" | grep -oE '[0-9]+$')
    rem=$(grep -aoE "^s$s$SLRE: nrels=[0-9]+ dup=[0-9]+ \([0-9.]+%\) rem=[0-9]+" "$lg" | grep -oE '[0-9]+$')
    [ "${nr:-x}" = "${NEW[$i]}" ] && [ -n "$rem" ] || { echo "VERIFY: slice $i read ${nr:-?} of ${NEW[$i]}" >> "$lg"; return 1; }
    [ -L "$(dd_ "$K" "$i")" ] || { echo "VERIFY: dup2 replaced memfd link $(dd_ "$K" "$i")" >> "$lg"; return 1; }
    # decode with the SLICE format (a gzip -dc here silently yields 0 lines on .zst parts: sed's rc masked it and every
    # incdup step failed VERIFY -> one-shot probes for the whole run). zstd needs -f: every dd part is a memfd symlink.
    { if [ "$SL" = .zst ]; then zstd -dcf "$(dd_ "$K" "$i")"; else gzip -dc "$(dd_ "$K" "$i")"; fi; } | sed 's/:.*/:/' > "$(kk_ "$K" "$i")"
    local ps="${PIPESTATUS[*]}"; [ "$ps" = "0 0" ] || { echo "VERIFY: decoding part $i failed (rc $ps)" >> "$lg"; return 1; }
    nl=$(wc -l < "$(kk_ "$K" "$i")")
    [ "$nl" = "$rem" ] || { echo "VERIFY: part $i has $nl lines, dup2 rem=$rem" >> "$lg"; return 1; }
    echo "SLICE $i rem=$rem" >> "$lg"
    : > "$MEM/d1/$i/rel.0000$SL"
  done
}
cur_par(){ local p=""
  [ -n "${INC_PAR_FILE:-}" ] && [ -s "$INC_PAR_FILE" ] && p=$(head -c 16 "$INC_PAR_FILE" 2>/dev/null | tr -dc 0-9)
  [ -n "$p" ] && [ "$p" -ge 1 ] 2>/dev/null && echo "$p" || echo "$INC_PAR"; }
pids=()
for g in $(seq 0 $((INC_GROUPS-1))); do
  run_group "$g" & pids[$g]=$!
  if [ -n "${INC_PAR_FILE:-}" ]; then
    while [ "$(jobs -rp | wc -l)" -ge "$(cur_par)" ]; do sleep 2; done
  else
    while [ "$(jobs -rp | wc -l)" -ge "$INC_PAR" ]; do wait -n || true; done
  fi
done
fails=0; for g in $(seq 0 $((INC_GROUPS-1))); do wait "${pids[$g]}" || { fails=$((fails+1)); say "dup2 group $g FAILED: $(grep -a VERIFY "$CACHE/dup2_${K}_g$g.log" | tail -1)"; }; done
[ "$fails" = 0 ] || fail "dup2 ($fails groups; see $CACHE/dup2_${K}_g*.log)"
rm -f "$CACHE"/in/s*"$SL"

# ---- 3. commit ----------------------------------------------------------------------------------
UNIQ=0
for i in $(seq 0 15); do
  if [ "${NEW[$i]}" = 0 ]; then REM[$i]=0
  else REM[$i]=$(grep -ahoE "^SLICE $i rem=[0-9]+" "$CACHE"/dup2_${K}_g*.log | grep -oE '[0-9]+$')
       [ -n "${REM[$i]}" ] || fail "slice $i has no verified count"; fi
  CUM[$i]=$(( OLD[$i] + REM[$i] )); UNIQ=$((UNIQ + CUM[$i]))
done
{ echo "k=$K"; echo "raw=$RAW"; echo "rawcum=$((RAWCUM0 + RAW))"; echo "uniqcum=$UNIQ"
  for i in $(seq 0 15); do echo "u$i=${CUM[$i]}"; done
  echo "chunk=$CHUNK"; echo "t=$(( $(date +%s)-T0 ))"; } > "$CACHE/chunk_$K.done.tmp" \
  && mv -f "$CACHE/chunk_$K.done.tmp" "$CACHE/chunk_$K.done" || fail "cannot write commit marker"
echo "[incdup] COMMIT k=$K raw=$RAW rawcum=$((RAWCUM0 + RAW)) uniqcum=$UNIQ t=$(( $(date +%s)-T0 ))s"
exit 0
