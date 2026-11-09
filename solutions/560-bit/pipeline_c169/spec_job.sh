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

# spec_job.sh -- one speculative matrix preparation (SPEC_MERGE=1, pipeline_c169/spec_merge.sh): merge at TD, then replay
# -skip 0, on an immutable purge-only snapshot. Run under setsid (own process group: spec_kill / SIGSTOP take the tree).
# usage: spec_job.sh <outdir> <purged> <rows> <epoch> <TD> <NT> <nice> <cpus|-> <cado bin dir>
# outdir gets: T0, merge.log, history, MERGED (merge secs), replay.log, c.sparse.bin (+ .rw/.cw), c.index.gz,
# c.sparse.epoch, and LAST of all READY (written as READY.tmp, then renamed). Exit 0 only with READY.
set -u
E=$1 PUR=$2 ROWS=$3 EP=$4 TD=$5 NT=$6 NI=$7 CPUS=$8 CF=$9
cd "$E" || exit 1
TM=(); [ -x /usr/bin/time ] && TM=(/usr/bin/time -f %M -o)   # peak RSS (kB) of merge / replay -> READY (calibrates SPEC_B_PER_ROW)
run(){ local o=$1; shift; [ ${#TM[@]} -gt 0 ] && set -- "${TM[@]}" "$o" "$@"; if [ "$CPUS" != - ]; then taskset -c "$CPUS" nice -n "$NI" "$@"; else nice -n "$NI" "$@"; fi; }
t0=$(date +%s); echo "$t0" > T0
run merge.rss "$CF/merge" -out history -mat "$PUR" -target_density "$TD" -t "$NT" > merge.log 2>&1 || exit 2
grep -aq 'Final matrix' merge.log && [ -s history ] || exit 3
t1=$(date +%s); echo "$(( t1-t0 ))" > MERGED
run replay.rss "$CF/replay" -purged "$PUR" -his history -out "$E/c.sparse.bin" -index "$E/c.index.gz" -skip 0 -t "$NT" > replay.log 2>&1 || exit 4
grep -qaiE 'No space left|ENOSPC|write error' replay.log && exit 5
t2=$(date +%s); nr=$(grep -a 'Sparse submatrix' replay.log | tail -1 | grep -oE 'nrows=[0-9]+' | cut -d= -f2)
[ -n "$nr" ] && [ "$nr" = "$(gzip -dc c.index.gz 2>/dev/null | head -1)" ] && [ -s c.sparse.rw.bin ] || exit 6
fm=$(grep -a 'Final matrix' merge.log | tail -1)
n=$(echo "$fm" | grep -oE ' N=[0-9]+' | head -1 | cut -d= -f2); w=$(echo "$fm" | grep -oE ' W=[0-9]+' | cut -d= -f2)
[ -n "$n" ] && [ -n "$w" ] || exit 7
echo "$EP" > c.sparse.epoch
printf "epoch=%s\nrows=%s\nN=%s\nW=%s\nt_merge=%s\nt_replay=%s\nnt=%s\nrss_merge_kb=%s\nrss_replay_kb=%s\n" "$EP" "$ROWS" "$n" "$w" "$(( t1-t0 ))" "$(( t2-t1 ))" "$NT" "$(tail -1 merge.rss 2>/dev/null)" "$(tail -1 replay.rss 2>/dev/null)" > READY.tmp \
  && mv -f READY.tmp READY || exit 8
exit 0
