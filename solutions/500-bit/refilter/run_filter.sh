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

# Run ONLY msieve's -nc1 filter on a (possibly lpb-capped) relation set and report the
# excess = unique_relations - unique_ideals. No memfd broker needed on this dev box (100GB disk);
# msieve reads the CADO relation dump directly as its .dat (msv_launcher just RAM-loads the same bytes).
#
# usage: run_filter.sh <rels_file> <fb_file> <N> <workdir> [target_density]
set -u
RELS="$1"; FB="$2"; N="$3"; WORK="$4"; TD="${5:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
MSV="${MSV_DIR:-$(cd "$HERE/../msieve-gpu" && pwd)}"
M="$MSV/msieve_la"          # QCB_SIZE=32 build used in production filtering
NT="${NT:-24}"

mkdir -p "$WORK"
# msieve -nc1 runs after `cd "$MSV"`, so ALL paths passed to it must be ABSOLUTE
# (PROGRESS.md gotcha #1) or the relative log/dat paths resolve against $MSV and fail.
WORK=$(cd "$WORK" && pwd)
# abspath: make a path ABSOLUTE and OPENABLE. Do NOT use bare `readlink -f` here.
#
# factor_msv.sh backs rels.txt with an ANONYMOUS MEMFD (relhold.py), exposed as
#     rels.txt -> /proc/<holder_pid>/fd/N
# That /proc link is a magic link: it is fully stat-able and openable (`wc -l` on it
# works, and a symlink chain through it resolves fine). But `readlink -f` DEREFERENCES
# it to the literal string "/memfd:rels.txt (deleted)", which is not a path at all.
# msieve then dies with `error: cannot open '<work>/m.dat'`.
#
# Measured 2026-07-28: this broke EVERY adaptive checkpoint on the no-cache path -- i.e.
# the real validator path -- and was invisible in development because GNFS_CACHE copies
# rels.txt to a real file first, at which point `readlink -f` is harmless. The failure
# was silent and unrecoverable: run_filter.sh exited non-zero, factor_msv.sh died on an
# unguarded `set -e` assignment, and the orphaned relhold.py held breaking_rsa.py's
# stdout pipe open so the solver hung forever emitting no result payload.
abspath() {
  local p="$1" r
  case "$p" in /*) ;; *) p="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")";; esac
  r=$(readlink -f "$p" 2>/dev/null)
  # Prefer the canonical form ONLY when it really exists (ordinary files -- unchanged
  # behaviour). Otherwise keep the absolute symlink path, which the kernel resolves.
  if [ -n "$r" ] && [ -e "$r" ]; then printf '%s\n' "$r"; else printf '%s\n' "$p"; fi
}
RELS=$(abspath "$RELS")
FB=$(abspath "$FB")
# msieve derives sibling names (m.dat.cyc/.mat) from -s; give it a dedicated dat path.
ln -sf "$RELS" "$WORK/m.dat"
cp -f "$FB" "$WORK/m.fb"
LOG="$WORK/msieve.log"; : > "$LOG"

echo "### filter (-nc1) on $(basename "$RELS") -> $WORK  $(date +%T)"
echo "    rels lines: $(wc -l < "$RELS")"
cd "$MSV"   # cub/sort_engine.so resolves relative to CWD
if [ -n "$TD" ]; then
  "$M" -nc1 -t "$NT" -s "$WORK/m.dat" -nf "$WORK/m.fb" -l "$LOG" -v "$N" "target_density=$TD"
else
  "$M" -nc1 -t "$NT" -s "$WORK/m.dat" -nf "$WORK/m.fb" -l "$LOG" -v "$N"
fi

echo "### filter result:"
grep -aE 'relations and|unique relations|unique ideals|begin with|reduce to|excess|filtering wants|found .* cycles|keeping' "$LOG" | sed 's/^/  [filter] /'
