#!/bin/bash
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

# ds_index_plain.sh (local, no GPU/CADO) -- known-answer test of the REPLAY_INDEX_PLAIN knob in ds_cado.sh
# (docs/research/agents_2026-10-26/downstream.md #3): the index name is c.index.gz unless REPLAY_INDEX_PLAIN=1 (and never
# plain under SKIP_REPLAY=1), and the row-count read (`gzip -dcf | head -1`) returns the same first line from a plain and
# a gzip index. The readers' suffix dispatch is checked where the code is in this repo (abx.c); CADO's characters/sqrt use
# fopen_maybe_compressed (suffix-based) -- confirmed by the c100 gate, not here. Exit 1 on any failure.
set -u
D=$(cd "$(dirname "$0")" && pwd); DS=$D/../msieve-gpu/ds_cado.sh; W=$(mktemp -d)
fails=0; ok(){ echo "  ok: $*"; }; bad(){ echo "  FAIL: $*"; fails=$((fails+1)); }
bash -n "$DS" && ok "bash -n ds_cado.sh" || bad "ds_cado.sh does not parse"
IDXLINE=$(grep -E '^IDX="\$OUT/c.index.gz"; \[ "\$\{REPLAY_INDEX_PLAIN:-0\}" = 1 \]' "$DS"); [ -n "$IDXLINE" ] && ok "IDX knob line present" || bad "IDX knob line not found in ds_cado.sh"
NIXLINE=$(grep -E '^NIX=\$\(gzip -dcf "\$IDX"' "$DS" | sed 's/ *#.*//'); [ -n "$NIXLINE" ] && ok "NIX line uses gzip -dcf" || bad "NIX line not found / not -dcf"
OUT=$W
name(){ ( export REPLAY_INDEX_PLAIN="${1:-}" SKIP_REPLAY="${2:-}"; [ -n "$REPLAY_INDEX_PLAIN" ] || unset REPLAY_INDEX_PLAIN; [ -n "$SKIP_REPLAY" ] || unset SKIP_REPLAY; eval "$IDXLINE"; basename "$IDX" ); }
[ "$(name)" = c.index.gz ] && ok "default: c.index.gz" || bad "default name $(name)"
[ "$(name 0)" = c.index.gz ] && ok "REPLAY_INDEX_PLAIN=0: c.index.gz" || bad "=0 name $(name 0)"
[ "$(name 1)" = c.index ] && ok "REPLAY_INDEX_PLAIN=1: c.index" || bad "=1 name $(name 1)"
[ "$(name 1 1)" = c.index.gz ] && ok "REPLAY_INDEX_PLAIN=1 + SKIP_REPLAY=1: c.index.gz (prepared matrix keeps its name)" || bad "=1 skip name $(name 1 1)"
# the row-count read on both formats: a 3-row index whose first line is the row count (as CADO replay writes it)
printf '3\n2 0 1\n1 2\n3 0 1 2\n' > "$W/c.index"; gzip -c "$W/c.index" > "$W/c.index.gz"
for f in c.index c.index.gz; do IDX=$W/$f; eval "$NIXLINE"; [ "$NIX" = 3 ] && ok "first line of $f = 3" || bad "first line of $f = '$NIX'"; done
# the old expression (gzip -dc without -f) fails on a plain index: the reason for -f
IDX=$W/c.index; old=$(gzip -dc "$IDX" 2>/dev/null | head -1); [ "$old" = 3 ] && bad "gzip -dc read a plain file (unexpected gzip)" || ok "gzip -dc (old) yields '' on a plain index: -f is needed"
# abx.c dispatches on the .gz suffix (popen gzip -dc) and opens anything else with fopen
A=$D/../msieve-gpu/abx.c
grep -q 'gzip -dc' "$A" && grep -qE 'fopen\(name, *"rb"\)' "$A" && ok "abx.c reader: .gz -> gzip pipe, else fopen" || bad "abx.c reader dispatch not as expected"
rm -rf "$W"
[ "$fails" = 0 ] && { echo "ALL OK"; exit 0; } || { echo "$fails FAILURE(S)"; exit 1; }
