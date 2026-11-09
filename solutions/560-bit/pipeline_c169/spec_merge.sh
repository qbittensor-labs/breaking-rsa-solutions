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

# spec_merge.sh -- sourced by factor_c169.sh when SPEC_MERGE=1 (STOP_RULE=cost, DS_MODE=cado). Codex review proposal 4:
# prepare a stop-ready matrix on spare CPUs while the GPU still sieves.
#
#  * after a purge-only MATRIX probe that the cost rule sieves on from (sc_after_probe -> continue), the kept snapshot
#    (SC_KEEP: probe<i>/purged) is merged at TD and replayed in the background: ONE speculative preparation at a time
#    (a newer probe does not preempt a running one; when it ends, the newest kept snapshot is next), nice SPEC_NICE,
#    SPEC_NT threads pinned to the last SPEC_NT physical cores of the actual cpuset (sched_getaffinity + sysfs
#    thread_siblings_list; GLAS keeps the first ones), started only if MemAvailable leaves SPEC_MEM_FLOOR_GB after its
#    estimated footprint (SPEC_B_PER_ROW x purged rows: merge/replay RSS + sparse.bin/history/index files), killed if
#    MemAvailable falls below SPEC_MEM_KILL_GB. Outputs go to SPEC_DIR/<probe> (history, c.sparse.bin + .rw/.cw sidecars,
#    c.index.gz, merge.log, replay.log); never into the probe dir and never into the downstream's memfd.
#  * epoch = poly md5 + probe id + purged fingerprint (size, inode, mtime, first/last MiB) + TD + replay -skip 0. It is
#    written to c.sparse.epoch BEFORE READY; READY is published by rename (READY.tmp -> READY). At use, the purged
#    fingerprint is re-taken and must match; ds_cado.sh SKIP_REPLAY=1 additionally requires SKIP_EPOCH == c.sparse.epoch.
#  * reference count: a snapshot whose purged file a preparation reads (or whose READY matrix is held) carries a pin
#    file probe<i>/.pin.<tag>; sc_rm_purged (stop_cost.sh) only marks a pinned snapshot .orphan, and the last unpin
#    deletes it. The READY matrix of an older snapshot is dropped when a newer one becomes READY.
#  * at a stop (stop_probe / stop_final from sc_after_probe or sc_after_land) spec_choose pauses the preparation
#    (SIGSTOP), asks `stop_cost.py spec` for C_ready / C_inprog / C_new (remaining wall only: a finished merge/replay is
#    not charged again) and either uses the READY matrix (no final incdup/purge, no merge, no replay: MATRIX_DIR = its
#    probe dir, SPEC_USE = its epoch dir, the downstream runs ds_cado.sh SKIP_REPLAY=1 SKIP_EPOCH=...), finishes the
#    in-progress one at nice 0 and uses it, or kills it and proceeds exactly as SPEC_MERGE=0.
#  * SPEC_PAUSE_PROBE=1 (default 0): SIGSTOP the preparation while a matrix-era probe runs (the probe may be the one
#    that stops the sieve); the final incdup + purge burst never overlaps a preparation (it is killed or used first).
SPEC_NT=${SPEC_NT:-6}; SPEC_RESERVE=${SPEC_RESERVE:-6}; SPEC_NICE=${SPEC_NICE:-10}; SPEC_DIR=${SPEC_DIR:-$WORK/spec}
SPEC_B_PER_ROW=${SPEC_B_PER_ROW:-250}; SPEC_MEM_FLOOR_GB=${SPEC_MEM_FLOOR_GB:-12}; SPEC_MEM_KILL_GB=${SPEC_MEM_KILL_GB:-4}
SPEC_PAUSE_PROBE=${SPEC_PAUSE_PROBE:-0}; SPEC_MAXFAIL=${SPEC_MAXFAIL:-2}
# remaining-time model of a preparation in flight (us per purged row at SPEC_NT, beside the sieve); replaced by the
# run's own measurement once one preparation has finished. Guesses from merge -t 20 108 s / replay -t 20 70 s (s66).
SPEC_MERGE_US=${SPEC_MERGE_US:-3.6}; SPEC_REPLAY_US_NT=${SPEC_REPLAY_US_NT:-2.6}
SPEC_PID=""; SPEC_RUN=""; SPEC_READY=""; SPEC_USE=""; SPEC_FAILS=0; SPEC_PAUSED=0; SPEC_LOG="$WORK/spec.log"
mkdir -p "$SPEC_DIR"
spec_say(){ say "spec-merge: $*"; echo "$(el) $*" >> "$SPEC_LOG"; }
SPEC_CPUS=$(python3 - "$SPEC_NT" "$SPEC_RESERVE" <<'PY' 2>/dev/null
import os, sys
k, res = int(sys.argv[1]), int(sys.argv[2])
aff = sorted(os.sched_getaffinity(0)); seen = set(); phys = []
for c in aff:
    try: sib = open('/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list' % c).read().strip()
    except OSError: sib = str(c)
    if sib in seen: continue
    seen.add(sib); phys.append(c)
pool = phys if len(phys) >= k + res else aff
print(','.join(map(str, pool[-k:])) if len(pool) >= k + res else '')
PY
)
command -v taskset > /dev/null || SPEC_CPUS=""
command -v setsid > /dev/null || die "SPEC_MERGE=1 needs setsid (a preparation runs in its own process group)"
SPEC_POLY=$(md5sum "$WORK/poly.cado" 2>/dev/null | cut -c1-12)
spec_say "on: NT $SPEC_NT nice $SPEC_NICE cpus ${SPEC_CPUS:-unpinned} (affinity $(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null) CPUs, reserve $SPEC_RESERVE), dir $SPEC_DIR, mem floor ${SPEC_MEM_FLOOR_GB} GB"
spec_fp(){ { stat -Lc '%s %i %Y' "$1"; head -c 1048576 "$1"; tail -c 1048576 "$1"; } 2>/dev/null | md5sum | cut -c1-16; }
spec_memavail_gb(){ awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo; }
spec_rows(){ grep -a 'VERDICT MATRIX' "$1/filter.out" 2>/dev/null | grep -a purge-only | tail -1 | grep -oE "final='nrows=[0-9]+" | grep -oE '[0-9]+$'; }
spec_pin(){ : > "$1/.pin.$2"; }
spec_unpin(){   # $1 probe dir $2 tag: drop the pin; the last one deletes an orphaned snapshot
  rm -f "$1/.pin.$2"; ls "$1"/.pin.* > /dev/null 2>&1 && return 0
  [ -e "$1/.orphan" ] && [ "$1" != "${SC_KEEP:-}" ] && [ "$1" != "${MATRIX_DIR:-}" ] && { rmf "$1"/purged* "$1/.orphan"; spec_say "unpinned $1: orphaned snapshot deleted"; }
  return 0; }
spec_sig(){   # $1 = signal: the preparation's process group (fallback: its process tree)
  kill -"$1" -- -"$SPEC_PID" 2>/dev/null || kill -"$1" "$SPEC_PID" $(sc_kids "$SPEC_PID") 2>/dev/null; }
spec_kill(){   # $1 = why: kill the running preparation's process group, drop its outputs and pin
  [ -n "$SPEC_PID" ] || return 0
  spec_sig CONT; spec_sig TERM; wait "$SPEC_PID" 2>/dev/null
  spec_say "preparation of $(basename "$SPEC_RUN") killed ($1)"
  rm -rf "$SPEC_DIR/$(basename "$SPEC_RUN")"; spec_unpin "$SPEC_RUN" spec; SPEC_PID=""; SPEC_RUN=""; SPEC_PAUSED=0; return 0; }
spec_pause(){ [ -n "$SPEC_PID" ] && [ "$SPEC_PAUSED" = 0 ] && spec_sig STOP && { SPEC_PAUSED=1; echo "$(el) paused" >> "$SPEC_LOG"; }; return 0; }
spec_resume(){ [ -n "$SPEC_PID" ] && [ "$SPEC_PAUSED" = 1 ] && { spec_sig CONT; SPEC_PAUSED=0; echo "$(el) resumed" >> "$SPEC_LOG"; }; return 0; }
spec_epoch(){ echo "poly=$SPEC_POLY probe=$(basename "$1") purged=$(spec_fp "$1/$PG") td=$TD skip=0"; }
spec_start(){   # $1 = probe dir (a kept purge-only MATRIX snapshot)
  local d=$1 e r need av ep; e="$SPEC_DIR/$(basename "$d")"; r=$(spec_rows "$d")
  [ -n "$r" ] && [ -s "$d/$PG" ] || return 0
  need=$(( r * SPEC_B_PER_ROW / 1073741824 + 1 )); av=$(spec_memavail_gb)
  if [ "${av:-0}" -lt $(( need + SPEC_MEM_FLOOR_GB )) ]; then spec_say "not started on $(basename "$d"): MemAvailable $av GB < ~$need GB + floor $SPEC_MEM_FLOOR_GB GB"; return 0; fi
  ep=$(spec_epoch "$d"); rm -rf "$e"; mkdir -p "$e"; echo "$ep" > "$e/EPOCH"; spec_pin "$d" spec
  # setsid: no job control in this script, so the background child is not a group leader and setsid(1) does not fork --
  # $! is the session / process-group id of the whole preparation
  setsid bash "$HERE/spec_job.sh" "$e" "$d/$PG" "$r" "$ep" "$TD" "$SPEC_NT" "$SPEC_NICE" "${SPEC_CPUS:--}" "$CF" > "$e/job.log" 2>&1 < /dev/null &
  SPEC_PID=$!
  SPEC_RUN=$d; SPEC_PAUSED=0
  spec_say "preparation started on $(basename "$d") (raw $(grep -a "VERDICT MATRIX" "$d/filter.out" | tail -1 | grep -oE " raw=[0-9]+" | cut -d= -f2), $r purged rows, need ~$need GB, MemAvailable $av GB): $ep"; }
spec_ready_kv(){ grep -a "^$2=" "$1/READY" 2>/dev/null | head -1 | cut -d= -f2-; }
spec_poll(){   # never blocks: memory guard, reap a finished preparation, start the next one on the newest kept snapshot
  [ "$SPEC_MERGE" = 1 ] || return 0
  if [ -n "$SPEC_PID" ]; then
    if kill -0 "$SPEC_PID" 2>/dev/null; then
      [ "$(spec_memavail_gb)" -lt "$SPEC_MEM_KILL_GB" ] && spec_kill "memory guard: MemAvailable < $SPEC_MEM_KILL_GB GB"
      return 0; fi
    local rc=0 e; wait "$SPEC_PID" || rc=$?; e="$SPEC_DIR/$(basename "$SPEC_RUN")"
    if [ "$rc" = 0 ] && [ -s "$e/READY" ]; then
      if [ -n "$SPEC_READY" ] && [ "$SPEC_READY" != "$e" ]; then rm -rf "$SPEC_READY"; spec_unpin "$WORK/$(basename "$SPEC_READY")" ready; fi
      spec_pin "$SPEC_RUN" ready; spec_unpin "$SPEC_RUN" spec; SPEC_READY=$e
      local r m p; r=$(spec_ready_kv "$e" rows); m=$(spec_ready_kv "$e" t_merge); p=$(spec_ready_kv "$e" t_replay)
      [ "${r:-0}" -gt 0 ] && { SPEC_MERGE_US=$(awk -v t="$m" -v r="$r" 'BEGIN{printf "%.3f", t*1e6/r}'); SPEC_REPLAY_US_NT=$(awk -v t="$p" -v r="$r" 'BEGIN{printf "%.3f", t*1e6/r}'); }
      spec_say "READY $(basename "$e"): N $(spec_ready_kv "$e" N) W $(spec_ready_kv "$e" W) merge ${m}s replay ${p}s at NT $SPEC_NT"
    else
      SPEC_FAILS=$((SPEC_FAILS+1)); spec_say "WARNING: preparation of $(basename "$SPEC_RUN") failed (rc=$rc, #$SPEC_FAILS): $(tail -n 2 "$e/merge.log" "$e/replay.log" 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
      rm -rf "$e"; spec_unpin "$SPEC_RUN" spec
      [ "$SPEC_FAILS" -ge "$SPEC_MAXFAIL" ] && { SPEC_MERGE=0; spec_say "disabled after $SPEC_FAILS failures"; }
    fi
    SPEC_PID=""; SPEC_RUN=""; SPEC_PAUSED=0
  fi
  [ "$SPEC_MERGE" = 1 ] && [ -z "$MATRIX_DIR" ] && [ "$SC_FINAL" = 0 ] && [ -n "$SC_KEEP" ] && [ -s "$SC_KEEP/$PG" ] || return 0
  [ "$SPEC_READY" = "$SPEC_DIR/$(basename "$SC_KEEP")" ] && return 0
  spec_start "$SC_KEEP"; }
spec_probe_hook(){   # $1 = launch|reap (matrix-era probes only): SPEC_PAUSE_PROBE
  [ "$SPEC_MERGE" = 1 ] && [ "$SPEC_PAUSE_PROBE" = 1 ] || return 0
  case "$1" in launch) [ "$TP_SEEN" = 1 ] && spec_pause;; reap) spec_resume;; esac; }
spec_end(){   # the stop is decided and SPEC_USE is not a preparation: free everything speculative
  [ -n "$SPEC_PID" ] && spec_kill "${1:-stop}"
  if [ -n "$SPEC_READY" ] && [ "$SPEC_READY" != "$SPEC_USE" ]; then rm -rf "$SPEC_READY"; spec_unpin "$WORK/$(basename "$SPEC_READY")" ready; SPEC_READY=""; fi
  return 0; }
# spec_choose <stop_probe|stop_final> <landed raw>: 0 = use a prepared matrix (MATRIX_DIR, SPEC_USE set; SC_FINAL=0,
# TP_MERGE=0), 1 = proceed as SPEC_MERGE=0 (the preparation is killed)
spec_choose(){
  [ "$SPEC_MERGE" = 1 ] || { spec_end "SPEC_MERGE off"; return 1; }
  spec_poll; spec_pause
  local rd=(- - -) ip=(- -) out act el0 tm e rows
  if [ -n "$SPEC_READY" ] && [ -s "$SPEC_READY/READY" ]; then
    rd=("$(spec_ready_kv "$SPEC_READY" rows)" "$(spec_ready_kv "$SPEC_READY" N)" "$(spec_ready_kv "$SPEC_READY" W)"); fi
  if [ -n "$SPEC_PID" ]; then
    e="$SPEC_DIR/$(basename "$SPEC_RUN")"; rows=$(spec_rows "$SPEC_RUN"); el0=$(( $(date +%s) - $(cat "$e/T0" 2>/dev/null || date +%s) ))
    if [ -s "$e/MERGED" ]; then tm=$(cat "$e/MERGED")
      ip=("$rows" "$(awk -v r="$rows" -v u="$SPEC_REPLAY_US_NT" -v el="$el0" -v tm="$tm" 'BEGIN{x=r*u/1e6-(el-tm); printf "%.0f", (x<5?5:x)}')")
    else ip=("$rows" "$(awk -v r="$rows" -v um="$SPEC_MERGE_US" -v ur="$SPEC_REPLAY_US_NT" -v el="$el0" 'BEGIN{x=r*um/1e6-el; printf "%.0f", (x<10?10:x)+r*ur/1e6}')"); fi
    [ -n "$rows" ] || ip=(- -)
  fi
  out=$(env STOP_NFAC="$STOP_NFAC" STOP_C_NNZ_PS="$STOP_C_NNZ_PS" STOP_DENSE_NS="$STOP_DENSE_NS" STOP_NNZ_ROW="$STOP_NNZ_ROW" \
        STOP_POST_US="$STOP_POST_US" STOP_LAMBDA="$STOP_LAMBDA" SPEC_REPLAY_US="${SPEC_REPLAY_US:-}" SPEC_MARGIN_S="${SPEC_MARGIN_S:-}" \
        STOP_LA_SCALE="${STOP_LA_SCALE:-1}" \
        python3 "$HERE/stop_cost.py" spec "$SC_PTS" "$2" "$1" "${rd[@]}" "${ip[@]}" 2>>"$SPEC_LOG") || out="new reason=stop_cost.py_failed"
  act=${out%% *}
  spec_say "at $1 ($2 raw): $out [ready $(basename "${SPEC_READY:-none}"), in flight $(basename "${SPEC_RUN:-none}")]"
  if [ "$act" = use_inprog ] && [ -n "$SPEC_PID" ]; then   # finish it at full priority (the sieve is being stopped)
    spec_resume; renice -n 0 -g "$SPEC_PID" > /dev/null 2>&1 || true; local w0; w0=$(date +%s)
    while kill -0 "$SPEC_PID" 2>/dev/null; do sleep 1; done; spec_poll
    if [ "$SPEC_READY" = "$e" ]; then act=use_ready; spec_say "in-flight preparation finished after $(( $(date +%s)-w0 ))s of waiting"
    else act=new; spec_say "WARNING: in-flight preparation did not become READY -- proceeding as SPEC_MERGE=0"; fi
  fi
  if [ "$act" = use_ready ] && [ -n "$SPEC_READY" ]; then
    local pd ep; pd="$WORK/$(basename "$SPEC_READY")"; ep=$(spec_epoch "$pd")
    if [ "$ep" != "$(cat "$SPEC_READY/c.sparse.epoch" 2>/dev/null)" ] || [ "$ep" != "$(spec_ready_kv "$SPEC_READY" epoch)" ]; then
      spec_say "WARNING: epoch mismatch for $(basename "$SPEC_READY") ($ep vs $(cat "$SPEC_READY/c.sparse.epoch" 2>/dev/null)) -- not used"
    else
      SPEC_USE=$SPEC_READY; [ -n "$SPEC_PID" ] && spec_kill "a prepared matrix is used"
      sc_probe_kill; MATRIX_DIR=$pd; TP_MERGE=0; SC_FINAL=0; SC_KEEP=$pd
      cp -f "$SPEC_USE/merge.log" "$pd/merge.log" 2>/dev/null
      echo "[filter160] VERDICT MATRIX raw=$(grep -a 'VERDICT MATRIX' "$pd/filter.out" | grep -a purge-only | tail -1 | grep -oE ' raw=[0-9]+' | cut -d= -f2) speculative (epoch $ep, NT=$SPEC_NT, merge $(spec_ready_kv "$SPEC_USE" t_merge)s replay $(spec_ready_kv "$SPEC_USE" t_replay)s) $(grep -a 'Final matrix' "$SPEC_USE/merge.log" | tail -1)" >> "$pd/filter.out"
      spec_say "USING the prepared matrix of $(basename "$pd"): no final handoff, merge or replay (stop -> LA start)"; return 0
    fi
  fi
  spec_end "the stop proceeds without a prepared matrix"; return 1; }
