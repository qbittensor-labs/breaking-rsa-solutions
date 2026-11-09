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
# cado_probe.sh — CADO filtering behind run_filter.sh's EXACT contract
# =============================================================================
# Drop-in for `refilter/run_filter.sh` at factor_msv.sh's two probe sites. Same
# call shape, same success signal, so the checkpoint ladder and the speculative
# probe need no restructuring:
#
#   usage:  cado_probe.sh <rels> <fb> <N> <workdir>
#   success: writes <workdir>/m.dat.cyc   (ladder/probe test exactly this)
#   short  : exit 2, no m.dat.cyc         (relations genuinely insufficient)
#
# It ADDITIONALLY writes the two files the CADO downstream needs, which msieve's
# -nc1 never produced:
#   <workdir>/m.dat.rels    relations in PURGE ORDER (msieve-parsable)
#   <workdir>/m.dat.purged  identity index map
# factor_msv.sh hands m.dat.rels to the downstream as DS_RELS and copies
# m.dat.purged next to the cycles; msvrun_gpu.sh then self-detects the CADO case.
#
# WHY: msieve's filter is the binding constraint on hard keys -- it fails to build
# a matrix from relation sets that are demonstrably sufficient (E_core ABOVE
# target_excess), and every one of its filter knobs is inert on this build.
# See cado-filter/Pipeline.md §11 for the measurements.
#
# The POLYNOMIAL is required (CADO needs .cado, not msieve's .fb). It is taken
# from $POLY, else poly.cado beside the .fb -- which is where polyselect_best.sh
# puts it and where factor_msv.sh's $WORK/c.fb lives.
# =============================================================================
set -u
RELS="$1"; FB="$2"; N="$3"; WORK="$4"
HERE=$(cd "$(dirname "$0")" && pwd)
CF="${CADO_DIR:-$(cd "$HERE/../cado-filter" 2>/dev/null && pwd || echo /app/cado-filter)}"
NT="${NT:-24}"

POLY="${POLY:-$(dirname "$FB")/poly.cado}"
[ -s "$POLY" ] || { echo "[cado-probe] no polynomial at $POLY -- cannot run CADO filter" >&2; exit 3; }
[ -x "$CF/cado_filter.sh" ] || { echo "[cado-probe] $CF/cado_filter.sh missing" >&2; exit 3; }

# Same abspath contract as run_filter.sh: rels.txt may be a memfd symlink
# (/proc/<pid>/fd/N) and `readlink -f` would resolve it to the literal string
# "/memfd:rels.txt (deleted)", which is not a path. See run_filter.sh's note.
abspath() {
  local p="$1" r
  case "$p" in /*) ;; *) p="$(cd "$(dirname "$p")" && pwd)/$(basename "$p")";; esac
  r=$(readlink -f "$p" 2>/dev/null)
  if [ -n "$r" ] && [ -e "$r" ]; then printf '%s\n' "$r"; else printf '%s\n' "$p"; fi
}
RELS=$(abspath "$RELS"); POLY=$(abspath "$POLY")
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)

echo "### cado filter probe on $(basename "$RELS") -> $WORK  $(date +%T)"
NT="$NT" TD="${TD:-100}" LPB="${LPB:-29}" KEEP="${KEEP:-160}" \
  bash "$CF/cado_filter.sh" "$RELS" "$POLY" "$N" "$WORK/cf" "$WORK/m.dat.rels"
rc=$?

if [ "$rc" = 2 ] && [ ! -s "$WORK/m.dat.rels.cyc" ]; then
  # ONLY rc=2 means purge genuinely found no matrix (excess-limited). The caller
  # treats this as a real shortfall and does NOT retry on msieve.
  echo "### cado filter: SHORT -- no matrix at this relation count"
  rm -rf "$WORK/cf"
  exit 2
fi
if [ "$rc" != 0 ] || [ ! -s "$WORK/m.dat.rels.cyc" ]; then
  # ANY other failure is TOOLING, not a shortfall. Reporting it as SHORT (which this
  # did until 2026-08-05) makes factor_msv.sh skip the msieve fallback and treat a
  # crashed filter as "needs more relations" -- so a broken CADO silently costs the
  # run its filter entirely. Exit 3 so run_probe() falls back.
  echo "### cado filter: TOOLING FAILURE (rc=$rc) -- falling back to msieve" >&2
  # Keep the logs: the previous code rm -rf'd them here, so the ONE artifact needed
  # to diagnose a validator-only failure was destroyed at the moment it was produced.
  [ -d "$WORK/cf" ] && { mkdir -p "$WORK/cf-failed"; cp -f "$WORK/cf"/*.log "$WORK/cf-failed/" 2>/dev/null; }
  rm -rf "$WORK/cf"
  exit 3
fi

# Present the cycle file under the name the ladder tests for.
# !! THE ORDER OF THESE TWO mv's IS LOAD-BEARING (2026-08-24). !!
# m.dat.cyc is the ladder's SUCCESS SIGNAL and the purge map is what tells msvrun_gpu.sh to pass
# `cado_filter=1`; without the map msieve reads the cycle file as RELATION INDICES and silently
# builds a garbage matrix (no error at any layer -- see factor_msv.sh's guard). This probe races a
# running sieve and is killed on MATRIX, so a kill CAN land between these two renames. Publish the
# map FIRST: then a kill leaves purged-without-cyc, which reads as "probe not finished" and is
# handled safely, instead of cyc-without-purged, which reads as success and produces garbage.
mv -f "$WORK/m.dat.rels.purged" "$WORK/m.dat.purged"
mv -f "$WORK/m.dat.rels.cyc"    "$WORK/m.dat.cyc"
rm -rf "$WORK/cf"

echo "### cado filter: MATRIX"
echo "  [filter] cycles: $(wc -l < "$WORK/m.dat.cyc" 2>/dev/null || echo '?') bytes $(stat -c%s "$WORK/m.dat.cyc")"
echo "  [filter] purge-ordered relations: $(wc -l < "$WORK/m.dat.rels")"
exit 0
