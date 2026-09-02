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

# msieve postprocessing with big files in RAM (memfd symlinks). Run under msv_launcher.
# args: <N> <workdir>
set -e
N="$1"; WORK="$2"
HERE=$(cd "$(dirname "$0")" && pwd)   # msieve toolchain dir (has msieve + cub/sort_engine.so)
M="$HERE/msieve"
cd "$HERE"                            # cub/sort_engine.so is resolved relative to CWD
NT="${NT:-24}"
# disk monitor: count ONLY real files in the workdir (symlinks->memfds don't count)
( p=0; while [ -d "$WORK" ]; do s=$(find "$WORK" -type f -printf '%s\n' 2>/dev/null|awk '{x+=$1}END{print int(x/1048576)}');
  [ -n "$s" ]&&[ "$s" -gt "$p" ]&&{ p=$s; echo "[/tmp-realfiles] ${p}MB $(date +%T)"; }; sleep 8; done ) &
MON=$!
echo "[msieve] -nc (filter + in-RAM Lanczos + sqrt), big files in memfd $(date +%T)"
"$M" -nc -t "$NT" -s "$WORK/m.dat" -nf "$WORK/m.fb" -l "$WORK/msieve.log" -v "$N" 2>&1 \
   | grep -aivE 'sse|avx' | grep -aiE 'unique|matrix is|Lanczos|dependencies|factor:|prp|GCD|error' || true
kill $MON 2>/dev/null || true
echo "[msieve] done $(date +%T)"
echo "=== FACTORS ==="; grep -aiE 'factor:|prp[0-9]' "$WORK/msieve.log" 2>/dev/null | grep -aivE 'sse|avx' | tail -4
