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

# =============================================================================
# cado_filter160.sh -- CADO-NFS filter for c160-scale runs, with EVERY multi-GB
# intermediate held in RAM (memfds from memhold.py) instead of the 10 GiB /tmp.
# =============================================================================
# Sizes at c160 (lpb 31/31, ~150-190M raw relations), measured 2026-09-26:
#   raw relations ~131 B/rel (~21 GB)     -> memfd chunks  (caller's pool)
#   dup1 slices   ~70 B/rel gz (~11 GB)   -> memfd pool $MEM/d1/<i>/rel.0000.gz
#   dup2 slices   ~62 B/rel gz (~8 GB)    -> memfd pool $MEM/d2/<i>/rel.0000.gz
#   renumber.gz   ~1.5 GB, purged.gz ~1-2 GB, history, cyc -> /tmp (fits)
# This REQUIRES the patched CADO binaries (vendor/cado-symlink-writethrough.patch): stock
# dup1/dup2 write a temp file and rename() it over the symlink, which silently puts the data back
# on /tmp (and at c160 overflows it). A post-dup check below makes that failure LOUD.
#
# 16 slices (dup1 -n 4) ALWAYS: dup2 reads at most 2^25 relations per file (cado-dup2 saturation);
# 190M/16 = 11.9M per slice, well under 33.5M.
#
# usage: cado_filter160.sh <work> <mem> <poly.cado> <N> <out_prefix> <chunk files...>
#   env: MERGE_ONLY=1 (stop after merge: report matrix N/W)  LPB0 LPB1 (31) NT (threads) DUP2_PAR (concurrent dup2, RAM-bound: ~2.5 GB each) TD KEEP
#        FRC (freerel cache dir; freerel depends only on poly+lpb)  PURGE_ONLY=1 (probe mode)
#        INC_CACHE=<dir> INC_K=<K>: INCREMENTAL probe (lever B). Skip dup1/dup2 entirely and purge the
#          per-chunk deduplicated parts cado_incdup.sh committed for chunks 1..K ($MEM/dd/<kkk>/s<ii>.gz),
#          listed SLICE-MAJOR so purge sees exactly the one-shot slice stream. The positional chunk
#          files must then be exactly chunks 1..K (cado2msieve2 still streams the raw chunks).
#          The dd parts are NOT freed on SHORT (the cache outlives a SHORT probe). Once purge has
#          SUCCEEDED (a MATRIX-bound probe) nothing reads them again -- merge/replay read purged.gz and
#          cado2msieve2 the raw chunks -- so they are truncated right there (~8 GB at c160) and
#          <work>/inc_consumed is written: the caller must then treat the incremental cache as spent
#          (stop incdup, free the key shadows; any later probe is one-shot).
# exit: 0 MATRIX (<out>, <out>.purged, <out>.cyc written)   2 SHORT (not enough relations)
#       1 tooling failure (named)
# stdout carries one machine-readable verdict line:
#   [filter160] VERDICT <MATRIX|SHORT> raw=<n> unique=<n> sing_excess=<n> sing_ncols=<n> ...
set -u
WORK=$1; MEM=$2; POLY=$3; N=$4; OUT=$5; shift 5
HERE=$(cd "$(dirname "$0")" && pwd); BIN=${CADO_BIN:-$HERE}
LPB0=${LPB0:-31}; LPB1=${LPB1:-31}; NT=${NT:-16}; DUP2_PAR=${DUP2_PAR:-6}; TD=${TD:-100}; KEEP=${KEEP:-160}
# SLICE_FMT (.gz default; .zst from the orchestrator when zstd exists): dup1/dup2 slices, incremental parts, renumber.
# purged.gz and freerel.gz stay gzip (characters' zlib reader, the c5 filter). Measured: zstd decompresses 8x faster.
SL=${SLICE_FMT:-.gz}
FRC=${FRC:-$WORK/freerel}; T0=$(date +%s)
el(){ echo "[$(( $(date +%s)-T0 ))s]"; }
say(){ echo "[filter160] $(el) $*"; }
fail(){ echo "[filter160] $(el) FAILED: $*" >&2; echo "[filter160] VERDICT FAIL $*"; exit 1; }
mkdir -p "$WORK" "$FRC" || fail "cannot create $WORK"
for b in freerel dup1 dup2 purge merge replay cado2msieve2; do [ -x "$BIN/$b" ] || fail "missing binary $BIN/$b"; done
[ "$#" -ge 1 ] || fail "no relation files given"
INC_CACHE=${INC_CACHE:-}; INC_K=${INC_K:-}
if [ -n "$INC_CACHE" ]; then
  case "$INC_K" in ''|*[!0-9]*|0) fail "INC_CACHE set but INC_K='$INC_K' is not a chunk count";; esac
  [ "$#" = "$INC_K" ] || fail "incremental probe: $# chunk files given but INC_K=$INC_K"
  for k in $(seq 1 "$INC_K"); do [ -s "$INC_CACHE/chunk_$k.done" ] || fail "incremental cache: chunk $k not committed"; done
else
  for i in $(seq 0 15); do
    [ -L "$MEM/d1/$i/rel.0000$SL" ] && [ -L "$MEM/d2/$i/rel.0000$SL" ] || fail "memfd pool slot $i missing under $MEM (memhold not running?)"
  done
fi

# ---- 1. freerel + renumber (cached: depends only on poly + lpb) ----------------------------
if [ ! -s "$FRC/renumber$SL" ] || [ ! -s "$FRC/freerel.gz" ] || ! grep -q "nprimes=" "$FRC/freerel.log" 2>/dev/null \
   || ! grep -q "c5filter=done" "$FRC/freerel.log" 2>/dev/null; then
  "$BIN/freerel" -poly "$POLY" -lpb0 "$LPB0" -lpb1 "$LPB1" -out "$FRC/freerel.gz" -renumber "$FRC/renumber$SL" \
      -t "$NT" > "$FRC/freerel.log" 2>&1 || fail "freerel"
  # Drop free relations for p | c5: msieve cannot reconstruct them and -nc2 crashes (see cado_filter.sh).
  C5=$(grep -aoE '^c5:[[:space:]]*-?[0-9]+' "$POLY" | tail -1 | grep -oE -- '-?[0-9]+$')
  if [ -n "${C5:-}" ] && [ "$C5" != 0 ]; then
    BAD=$(python3 -c "
n=abs(int('$C5'));p=2;s=set()
while p*p<=n:
    while n%p==0: s.add(p); n//=p
    p+=1
if n>1: s.add(n)
print('|'.join('^%x,0:'%q for q in sorted(s)))")
    if [ -n "$BAD" ]; then
      gzip -dc "$FRC/freerel.gz" | grep -avE "$BAD" | gzip -c > "$FRC/freerel.gz.f" && [ -s "$FRC/freerel.gz.f" ] \
        && mv -f "$FRC/freerel.gz.f" "$FRC/freerel.gz" || fail "free-relation c5 filter"
      say "free relations: dropped p | c5=$C5"
    fi
  fi
  echo "c5filter=done" >> "$FRC/freerel.log"
fi
NPRIMES=$(grep -aoE 'nprimes=[0-9]+' "$FRC/freerel.log" | tail -1 | cut -d= -f2)
[ -n "$NPRIMES" ] || fail "could not read nprimes"
say "renumber ready nprimes=$NPRIMES"

if [ -n "$INC_CACHE" ]; then
# ---- 2+3 (INCREMENTAL): dedup already done chunk by chunk ------------------------------------
  D="$INC_CACHE/chunk_$INC_K.done"
  RAW=$(grep -oE '^rawcum=[0-9]+' "$D" | cut -d= -f2); UNIQ=$(grep -oE '^uniqcum=[0-9]+' "$D" | cut -d= -f2)
  [ -n "$RAW" ] && [ -n "$UNIQ" ] || fail "unreadable $D"
  IN=""
  for i in $(seq 0 15); do for k in $(seq 1 "$INC_K"); do
    f=$(printf '%s/dd/%03d/s%02d'"$SL" "$MEM" "$k" "$i")
    [ -L "$f" ] && [ -s "$f" ] || fail "incremental part $f missing or empty"
    IN="$IN $f"
  done; done
  say "incremental: $UNIQ unique of $RAW (chunks 1..$INC_K, $((16*INC_K)) parts, no dup1/dup2)"
else
# ---- 2. dup1: split into 16 memfd slices ---------------------------------------------------
for i in $(seq 0 15); do : > "$MEM/d1/$i/rel.0000$SL"; : > "$MEM/d2/$i/rel.0000$SL"; done
"$BIN/dup1" -out "$MEM/d1" -prefix rel -n 4 -outfmt "$SL" "$@" > "$WORK/dup1.log" 2>&1 || fail "dup1 (see $WORK/dup1.log)"
# LOUD check: every slice must still be a symlink (a stock CADO would have renamed a regular file
# over it -- data on /tmp, not RAM) and no extra rel.0001 files may exist (they would be ignored).
for i in $(seq 0 15); do
  [ -L "$MEM/d1/$i/rel.0000$SL" ] || fail "dup1 replaced memfd link for slice $i (unpatched CADO?)"
  [ -e "$MEM/d1/$i/rel.0001$SL" ] && fail "dup1 wrote a second file for slice $i (would be dropped)"
done
RAW=0; for n in $(grep -aoE 'slice [0-9]+ received [0-9]+' "$WORK/dup1.log" | grep -oE '[0-9]+$'); do RAW=$((RAW+n)); done
say "dup1 done: $RAW relations into 16 memfd slices"

# ---- 3. dup2 per slice (dedup + renumber), RAM-bounded concurrency --------------------------
run_dup2(){ local i=$1 n
  n=$(grep -aoE "slice $i received [0-9]+" "$WORK/dup1.log" | grep -oE '[0-9]+$')
  "$BIN/dup2" -poly "$POLY" -renumber "$FRC/renumber$SL" -nrels $(( ${n:-20000000} * 14 / 10 + 1000 )) \
      -outdir "$MEM/d2/$i" "$MEM/d1/$i/rel.0000$SL" > "$WORK/dup2_$i.log" 2>&1 || return 1
  : > "$MEM/d1/$i/rel.0000$SL"        # free the slice's RAM as soon as it is consumed
}
pids=(); fails=0
for i in $(seq 0 15); do
  run_dup2 "$i" & pids[$i]=$!
  while [ "$(jobs -rp | wc -l)" -ge "$DUP2_PAR" ]; do wait -n || true; done
done
# waiting on a specific pid returns its saved status even if `wait -n` already reaped it
for i in $(seq 0 15); do wait "${pids[$i]}" || { fails=$((fails+1)); say "dup2 slice $i FAILED"; }; done
[ "$fails" = 0 ] || fail "dup2 ($fails slices; see $WORK/dup2_*.log)"
for i in $(seq 0 15); do grep -aq 'At the end:.*remaining relations' "$WORK/dup2_$i.log" || fail "dup2 slice $i has no completion line"; done
for i in $(seq 0 15); do [ -L "$MEM/d2/$i/rel.0000$SL" ] || fail "dup2 replaced memfd link for slice $i (unpatched CADO?)"; done
UNIQ=0; for n in $(grep -h 'At the end:.*remaining relations' "$WORK"/dup2_*.log | grep -oE '[0-9]+'); do UNIQ=$((UNIQ+n)); done
say "dup2 done: $UNIQ unique of $RAW"

IN=""; for i in $(seq 0 15); do IN="$IN $MEM/d2/$i/rel.0000$SL"; done
fi

# ---- 4. purge ---------------------------------------------------------------------------------
rm -f "$WORK/purged.gz" "$WORK"/purged.gz.tmp.*
# INPUT-COMPLETENESS CHECK (below): purge reads its inputs through `gzip -dc` and ACCEPTS a
# truncated/emptied input with rc=0 (only gzip's "unexpected end of file" on stderr shows it --
# measured on c111: 5,783,874 of 7,600,684 relations read, a smaller matrix, rc=0). Every pass must
# read exactly UNIQ (dup2 / incremental unique) + the free-relation count. The free-relation count is
# cached next to freerel.gz, keyed by its size (a rebuilt freerel.gz invalidates it).
FRSZ=$(stat -c%s "$FRC/freerel.gz" 2>/dev/null) || fail "cannot stat $FRC/freerel.gz"
FRN=""; [ -s "$FRC/freerel.count" ] && [ "$(cut -d' ' -f1 "$FRC/freerel.count")" = "$FRSZ" ] && FRN=$(cut -d' ' -f2 "$FRC/freerel.count")
if [ -z "$FRN" ]; then
  FRN=$(set -o pipefail; gzip -dc "$FRC/freerel.gz" | grep -avc '^#') || [ "${FRN:-x}" = 0 ] || fail "cannot count free relations in $FRC/freerel.gz"
  echo "$FRSZ $FRN" > "$FRC/freerel.count.tmp.$$" && mv -f "$FRC/freerel.count.tmp.$$" "$FRC/freerel.count"
fi
# PURGE_COLMIN_DIV=D (default 40 = cado-nfs cadofactor's own default; 0 = off): pass -col-min-index NPRIMES/D (the
# smallest ideals are never singletons/cliques, so purge does not track them). Measured (key 1, 22 chunks, NT 16):
# purge 357 -> 335 s wall, peak RSS 17.7 -> 7.9 GB, same MATRIX verdict (sing_excess 3,301,197 vs 3,301,191) and
# matrix (N 9.666M vs 9.661M). Purge still COUNTS those columns, so excess/-keep stay conservative.
PCM=(); [ "${PURGE_COLMIN_DIV:-40}" -gt 0 ] 2>/dev/null && PCM=(-col-min-index "$(( NPRIMES / ${PURGE_COLMIN_DIV:-40} ))")
"$BIN/purge" -out "$WORK/purged.gz" -col-max-index "$NPRIMES" "${PCM[@]}" -keep "$KEEP" -t "$NT" $IN "$FRC/freerel.gz" \
    > "$WORK/purge.log" 2>&1; PRC=$?
rm -f "$WORK"/purged.gz.tmp.*
grep -qaiE 'No space left|ENOSPC|write error' "$WORK/purge.log" && fail "purge out of space"
SHORTMSG=$(grep -acE 'number of rows < number of columns|excess / ncols' "$WORK/purge.log")
case "$PRC" in
  0|2) ;;
  *) fail "purge rc=$PRC (killed/crashed -- OOM? see $WORK/purge.log)";;
esac
# gzip AND zstd (SLICE_FMT=.zst) decoder errors; the read-count check below is the backstop either way
PERR='^gzip:|unexpected end of file|invalid compressed data|crc error|^zstd:|premature end|Read error|Decoding error'
grep -qaE "$PERR" "$WORK/purge.log" \
  && fail "purge input unreadable (truncated/freed part?): $(grep -aE "$PERR" "$WORK/purge.log" | head -1)"
EXPREAD=$(( UNIQ + FRN ))
READS=$(grep -aoE '^# Done: Read [0-9]+ relations' "$WORK/purge.log" | grep -oE '[0-9]+')
[ -n "$READS" ] || fail "purge log has no '# Done: Read' line (see $WORK/purge.log)"
for r in $READS; do
  [ "$r" = "$EXPREAD" ] || fail "purge read $r relations, expected $EXPREAD = $UNIQ unique + $FRN free (input truncated or freed mid-purge?)"
done
# the TRUE margin: excess after singleton removal, BEFORE clique removal ("Step 0" block only;
# from "Step 1" on, clique removal trims the excess down to -keep and says nothing about margin)
SING=$(awk '/^Step [1-9]/{exit} /^Sing\. rem\./' "$WORK/purge.log" | grep -aoE 'nrows=[0-9]+ ncols=[0-9]+ excess=-?[0-9]+' | tail -1)
SX=$(echo "$SING" | grep -oE 'excess=-?[0-9]+' | cut -d= -f2); SC=$(echo "$SING" | grep -oE 'ncols=[0-9]+' | cut -d= -f2)
# purge prints "Final values:" and "nrows=... ncols=... excess=..." on SEPARATE lines (purge.cpp:321);
# the per-iteration lines carry a "Sing. rem.:" prefix, so an anchored match picks the final one.
FIN=$(grep -aoE '^nrows=[0-9]+ ncols=[0-9]+ excess=-?[0-9]+' "$WORK/purge.log" | tail -1)
if [ "$PRC" = 2 ] || [ "$SHORTMSG" -gt 0 ] || [ ! -s "$WORK/purged.gz" ]; then
  [ "$PRC" = 0 ] && [ "$SHORTMSG" = 0 ] && fail "purge rc=0 but no purged.gz"
  echo "[filter160] VERDICT SHORT raw=$RAW unique=$UNIQ sing_excess=${SX:-?} sing_ncols=${SC:-?} final='${FIN:-}' t=$(( $(date +%s)-T0 ))s"
  exit 2
fi
[ -n "$FIN" ] || fail "could not parse purge Final values (see $WORK/purge.log)"
say "purge: singleton-stage $SING | final $FIN"
if [ "${PURGE_ONLY:-0}" = 1 ]; then
  echo "[filter160] VERDICT MATRIX raw=$RAW unique=$UNIQ sing_excess=${SX:-?} sing_ncols=${SC:-?} final='${FIN:-}' t=$(( $(date +%s)-T0 ))s purge-only"
  exit 0
fi
[ -n "$INC_CACHE" ] || for i in $(seq 0 15); do : > "$MEM/d2/$i/rel.0000$SL"; done      # purge consumed them
if [ -n "$INC_CACHE" ]; then      # ... and the incremental parts likewise (see INC_CACHE above)
  for k in $(seq 1 "$INC_K"); do for i in $(seq 0 15); do : > "$(printf '%s/dd/%03d/s%02d'"$SL" "$MEM" "$k" "$i")"; done; done
  echo "k=$INC_K" > "$WORK/inc_consumed"
  say "incremental parts of chunks 1..$INC_K freed (purge consumed them)"
fi

# ---- 5. merge, replay, cado2msieve --------------------------------------------------------------
"$BIN/merge" -out "$WORK/history" -mat "$WORK/purged.gz" -target_density "$TD" -t "$NT" > "$WORK/merge.log" 2>&1 || fail "merge"
say "merge: $(grep -a 'Final matrix' "$WORK/merge.log" | tail -1)"
if [ "${MERGE_ONLY:-0}" = 1 ]; then      # measurement mode: matrix size after merge, no replay/cado2msieve
  echo "[filter160] VERDICT MATRIX raw=$RAW unique=$UNIQ sing_excess=${SX:-?} sing_ncols=${SC:-?} final='${FIN:-}' t=$(( $(date +%s)-T0 ))s merge-only $(grep -a 'Final matrix' "$WORK/merge.log" | tail -1)"
  exit 0
fi
"$BIN/replay" --for_msieve -skip 0 -purged "$WORK/purged.gz" -his "$WORK/history" -out "$OUT.cyc" > "$WORK/replay.log" 2>&1 || fail "replay"
NPUR=$(echo "$FIN" | grep -oE 'nrows=[0-9]+' | cut -d= -f2); [ -n "$NPUR" ] || fail "could not read purge nrows"
# cado2msieve2: single streaming pass over the raw chunk files, keeps only purged relations
# (pipe/memfd-safe; the old cado2msieve fseek()ed into ONE seekable file).
"$BIN/cado2msieve2" <(gzip -dc "$WORK/purged.gz") "$OUT" "$NPUR" "$@" > "$WORK/c2m.log" 2>&1 \
    || fail "cado2msieve (see $WORK/c2m.log)"
if [ -s "$OUT.purged" ]; then
  _hdr=$(head -1 "$OUT.purged"); _cnt=$(( $(wc -l < "$OUT.purged") - 1 ))
  [ "$_hdr" = "$_cnt" ] || fail "purge map header $_hdr != $_cnt entries"
else fail "no $OUT.purged"; fi
echo "[filter160] VERDICT MATRIX raw=$RAW unique=$UNIQ sing_excess=${SX:-?} sing_ncols=${SC:-?} final='${FIN:-}' t=$(( $(date +%s)-T0 ))s $(grep -a 'Final matrix' "$WORK/merge.log" | tail -1)"
exit 0
