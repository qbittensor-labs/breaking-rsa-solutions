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
# cado_predup.sh — pre-deduplicate a PREFIX of the relation file, ahead of time
# =============================================================================
#
# WHY THIS EXISTS.  The filter probe is on the critical path: while it runs, the
# sieve keeps going and every relation it produces is DISCARDED the moment the
# probe returns MATRIX. Measured breakdown of a 261 s MATRIX probe (key07 e2e):
#     freerel 11 + strip/dup1 50 + dup2 82 + purge 41 + merge/replay 77
# so 132 s of it -- half -- is dup1+dup2 re-reading all 8.5 GB of relations that
# were ALREADY read by every previous probe. On a Xeon 6952P the same probe is
# 347 s (Pipeline.md §16), which is where the wall-time margin actually goes.
#
# Relations are append-only, and CADO is built for this: `dup2` classifies each
# input file as "already renumbered" or "new", loads the former straight into its
# hash table and only re-emits the latter. Measured on this box, key08 poly:
#     already-renumbered input   618,451 rel/s
#     new (raw dup1) input        93,035 rel/s     -- 6.65x slower
# So a probe that only has to renumber the DELTA does dup1+dup2 in a fraction of
# the time, and the prefix half of the work moves off the critical path entirely
# -- onto the ~21 of 24 cores that Pipeline.md §14 measured sitting idle under
# the GPU-bound sieve.
#
# CORRECTNESS.  dup1's slice assignment is a deterministic function of the
# relation, so a relation lands in the same slice whether it arrives in the
# prefix or the delta, and a cross-boundary duplicate is therefore always seen by
# the same dup2 invocation. Verified end-to-end rather than argued: prefix+delta
# through this path produced a deduplicated relation set BYTE-IDENTICAL to the
# one-shot dup1+dup2 baseline (368,364 relations, md5 9bd22c32...), including the
# 617 cross-boundary duplicates only the joint pass can catch.
#
# usage:  cado_predup.sh <rels> <poly.cado> <cachedir> <nrels>
#   env:  LPB (29)  NT (12)  CADO_BIN  CADO_FREEREL_CACHE
#
# output: <cachedir>/s<i>/rel.0000.gz   deduplicated prefix slices (i = 0..3)
#         <cachedir>/meta               NBYTES / NRELS / TAILMD5 / SLICES
#         <cachedir>/READY              written last, so a killed run is not
#                                       mistaken for a complete cache
# =============================================================================
set -u

RELS="$1"; POLY="$2"; CACHE="$3"; WANT="$4"
HERE=$(cd "$(dirname "$0")" && pwd)
BIN="${CADO_BIN:-$HERE}"
LPB="${LPB:-29}"; NT="${NT:-12}"
NSLICES=4

for t in freerel dup1 dup2; do
  [ -x "$BIN/$t" ] || { echo "[predup] FATAL: $BIN/$t missing" >&2; exit 1; }
done

rm -rf "$CACHE"; mkdir -p "$CACHE" || exit 1
echo "[predup] start $(date +%T)  prefix=$WANT relations  nt=$NT"

# --- renumber table ---------------------------------------------------------
# Same cache factor_msv.sh precomputes during the sieve, keyed on poly+lpb, so
# this normally costs nothing. Building it here would be an 11 s duplicate.
FRC="${CADO_FREEREL_CACHE:-}"
if [ -n "$FRC" ]; then FRKEY=$(md5sum < "$POLY" | cut -c1-16)-lpb$LPB; fi
if [ -n "$FRC" ] && [ -s "$FRC/$FRKEY/renumber.gz" ]; then
  ln -sf "$FRC/$FRKEY/renumber.gz" "$CACHE/renumber.gz"
else
  echo "[predup] no freerel cache -- building the renumber table here" >&2
  "$BIN/freerel" -poly "$POLY" -lpb0 "$LPB" -lpb1 "$LPB" \
      -out "$CACHE/freerel.gz" -renumber "$CACHE/renumber.gz" -t "$NT" \
      > "$CACHE/freerel.log" 2>&1 || { echo "[predup] freerel FAILED" >&2; exit 1; }
fi

# --- snapshot the prefix into RAM -------------------------------------------
# head -n (not -c) so the cut always lands on a line boundary; the byte length is
# read back afterwards and is what the delta is taken from. The snapshot is an
# anonymous memfd exactly like the probe's -- it must NOT be written to the
# validator's 10 GB /tmp, which already peaks at 8.7 GB. It is freed the moment
# dup1 has consumed it, ~45 s later, so it never overlaps the probe's own.
SNAP="$CACHE/pre.rels"
python3 "$HERE/../gpu_nfs_siever/relhold.py" "$SNAP" >/dev/null 2>&1 & SNAPPID=$!
for _i in $(seq 1 100); do { [ -L "$SNAP" ] && [ -w "$SNAP" ]; } && break; sleep 0.1; done
{ [ -L "$SNAP" ] && [ -w "$SNAP" ]; } || {
  echo "[predup] snapshot memfd never appeared -- aborting" >&2
  kill "$SNAPPID" 2>/dev/null; exit 1; }
_pd_cleanup(){ kill "$SNAPPID" 2>/dev/null || true; rm -f "$SNAP" 2>/dev/null || true; }
trap _pd_cleanup EXIT INT TERM

head -n "$WANT" "$RELS" > "$SNAP" 2>/dev/null || {
  echo "[predup] snapshot write FAILED -- aborting" >&2; exit 1; }
NBYTES=$(stat -Lc %s "$SNAP" 2>/dev/null || echo 0)
NRELS=$(wc -l < "$SNAP")
[ "$NBYTES" -gt 0 ] && [ "$NRELS" -gt 0 ] || { echo "[predup] empty snapshot" >&2; exit 1; }

# A relation set with msieve free relations in it ("p,0:") is stripped by
# cado_filter.sh into a SEPARATE file, which destroys the byte-offset identity
# the delta depends on. Refuse rather than hand back a cache that silently
# mismatches: the caller just falls back to the one-shot path.
if grep -qaE '^[0-9]+,0:' "$SNAP" 2>/dev/null; then
  echo "[predup] relation set carries msieve free relations -- prefix caching disabled" >&2
  exit 1
fi
# Fingerprint the last 4 KB of the prefix. The consumer re-reads the same bytes
# from ITS relation file and compares, so a cache built from a different file (or
# a file that was rewritten rather than appended to) can never be used by mistake.
TAILMD5=$(tail -c 4096 "$SNAP" | md5sum | cut -d' ' -f1)

# --- dup1: split the prefix into slices -------------------------------------
# Pre-create BOTH names: dup1 writes into <out>/<slice-number>/ and, with
# -outfmt, opens that file through a shell pipe BEFORE creating the directory --
# the same ordering bug cado_filter.sh pre-creates around. s<i> is where the
# finished slices are moved to, so a half-written cache can never look complete.
for _i in $(seq 0 $((NSLICES-1))); do mkdir -p "$CACHE/$_i" "$CACHE/s$_i"; done
"$BIN/dup1" -out "$CACHE" -prefix rel -n 2 -outfmt .gz "$SNAP" \
    > "$CACHE/dup1.log" 2>&1 || { echo "[predup] dup1 FAILED" >&2; exit 1; }
# dup1 writes into <out>/<slice>/, i.e. 0..3, not s0..s3 -- normalise.
for _i in $(seq 0 $((NSLICES-1))); do
  [ -s "$CACHE/$_i/rel.0000.gz" ] || { echo "[predup] dup1 slice $_i missing" >&2; exit 1; }
  mv -f "$CACHE/$_i/rel.0000.gz" "$CACHE/s$_i/rel.0000.gz"
done
rm -rf $(for _i in $(seq 0 $((NSLICES-1))); do echo "$CACHE/$_i"; done)
_pd_cleanup; trap - EXIT      # free the 8 GB prefix memfd NOW, before dup2
echo "[predup] dup1 done $(date +%T)"

# --- dup2: deduplicate each slice in place ----------------------------------
pids=""
for i in $(seq 0 $((NSLICES-1))); do
  n=$(grep -aoE "slice $i received [0-9]+" "$CACHE/dup1.log" | grep -oE '[0-9]+$')
  "$BIN/dup2" -poly "$POLY" -renumber "$CACHE/renumber.gz" -nrels "${n:-20000000}" \
      "$CACHE/s$i/rel.0000.gz" > "$CACHE/dup2_$i.log" 2>&1 &
  pids="$pids $!"
done
for p in $pids; do wait "$p" || { echo "[predup] dup2 FAILED" >&2; exit 1; }; done
UNIQ=0
for _n in $(grep -h 'At the end:.*remaining relations' "$CACHE"/dup2_*.log 2>/dev/null \
            | grep -oE '[0-9]+'); do UNIQ=$((UNIQ+_n)); done

printf 'NBYTES=%s\nNRELS=%s\nTAILMD5=%s\nSLICES=%s\nUNIQ=%s\n' \
    "$NBYTES" "$NRELS" "$TAILMD5" "$NSLICES" "$UNIQ" > "$CACHE/meta"
: > "$CACHE/READY"
echo "[predup] done $(date +%T): $NRELS raw ($NBYTES bytes) -> $UNIQ unique in $NSLICES slices"
