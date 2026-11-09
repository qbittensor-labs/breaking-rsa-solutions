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

# stop_cost.sh -- sourced by factor_c169.sh (STOP_RULE=cost). The model is stop_cost.py; this file holds the
# orchestrator side: matrix points, the measured sieve rate, the decision, killing a probe, and the final handoff.
#
# Flow (two-phase purge-only probes, DS_MODE=cado):
#  * before the first matrix the excess predictor aims at STOP_PRED_EXCESS (default 1M: just past the crossover).
#  * each MATRIX probe (purge-only) -> a point (raw, purged rows, excess, t) in $SC_PTS; its purged file is KEPT
#    (sc_keep: the stop matrix if the rule later says "stop_probe"; the previous kept one is deleted). Probe cadence
#    after the first matrix: the next committed chunk (STOP_PROBE_EVERY chunks); a probe only launches when none runs,
#    so a probe longer than a chunk interval simply skips commits.
#  * the rule is evaluated (a) when a matrix probe is reaped, (b) STOP_PREDICTIVE=1: at every chunk landing BEFORE
#    the next chunk is started, from the model alone (no probe needed at the stop point: key1g lost 259 s = 97 s
#    incdup lag + 162 s probe between chunk 60 landing and the stop verdict, with the GPU sieving relations nobody used).
#  * stop_probe -> the kept probe's purged file is merged at once (TP_MERGE, MERGE_NT threads).
#    stop_final -> the sieve (and its in-flight chunk) is killed, a running probe is killed, incdup commits every
#    landed chunk with all its groups at nice 0 (STOP_FINAL_GROUPS), and one purge-only pass on chunks 1..LANDED runs
#    at STOP_FINAL_NT (default NCPU) -> merge. Any failure there falls back to the kept probe's matrix.
#  * STOP_MIN_EXCESS stays a backstop: a probe at >= that excess stops as in the excess rule.
SC_PTS="$WORK/stopcost.tsv"; SC_LOG="$WORK/stopcost.log"; SC_N=0; SC_KEEP=""; SC_FINAL=0; SC_ACT=""; SC_STEP=""; SC_RW=0
declare -a SC_RATES=(); STOP_FINAL_GROUPS=${STOP_FINAL_GROUPS:-$INC_GROUPS}
sc_say(){ say "stop rule (cost): $*"; echo "$(el) $*" >> "$SC_LOG"; }
sc_rate(){   # rel/s for the cost of the next step: STOP_RATE, else the median of the last STOP_RATE_N GPU chunks
  # Prints ONLY the number: sc_decide captures it with $(...), so anything else printed here rides into stop_cost.py's
  # rate argument. The empty-SC_RATES warning used to be printed from inside this capture (and SC_RW=1 was set in the
  # subshell, so it repeated): in REPLAY mode (RELS_IN, no GPU chunks -> SC_RATES empty) every decision then failed with
  # "stop_cost.py failed -- sieving on" (key 29 container replay 2026-10-25: 6 of 6 decisions, no rate warning in the log)
  # and the replay gate never exercised the stop rule / final handoff. Live runs have SC_RATES by the time SC_N >= 2, so
  # the production path is unchanged; the warning is now sc_rate_warn, called by sc_decide AFTER the capture.
  if [ -n "${STOP_RATE:-}" ]; then echo "$STOP_RATE"; return 0; fi
  local n=${#SC_RATES[@]}
  [ "$n" -eq 0 ] && { echo "$STOP_RATE_DEFAULT"; return 0; }
  local k=$(( n < STOP_RATE_N ? n : STOP_RATE_N ))
  printf '%s\n' "${SC_RATES[@]: -$k}" | sort -n | awk '{a[NR]=$1} END{print (NR%2 ? a[(NR+1)/2] : int((a[NR/2]+a[NR/2+1])/2))}'; }
sc_rate_warn(){   # once per run, outside any command substitution (see sc_rate)
  [ -z "${STOP_RATE:-}" ] && [ "${#SC_RATES[@]}" -eq 0 ] && [ "$SC_RW" = 0 ] \
    && { SC_RW=1; sc_say "WARNING: no measured chunk rate yet -- using STOP_RATE_DEFAULT ${STOP_RATE_DEFAULT} rel/s"; }; return 0; }
sc_record(){   # $1 = purge-only MATRIX probe dir -> one point
  local v raw sx rows t
  v=$(grep -a 'VERDICT MATRIX' "$1/filter.out" 2>/dev/null | grep -a 'purge-only' | tail -1)
  raw=$(echo "$v" | grep -oE ' raw=[0-9]+' | cut -d= -f2); sx=$(echo "$v" | grep -oE 'sing_excess=-?[0-9]+' | cut -d= -f2)
  rows=$(echo "$v" | grep -oE "final='nrows=[0-9]+" | grep -oE '[0-9]+$'); t=$(echo "$v" | grep -oE ' t=[0-9]+s' | tr -dc 0-9)
  [ -n "$raw" ] && [ -n "$rows" ] || { sc_say "WARNING: could not parse $1/filter.out -- no point"; return 1; }
  printf '%s\t%s\t%s\t%s\t%s\n' "$raw" "$rows" "${sx:-0}" "$1" "${t:-0}" >> "$SC_PTS"; SC_N=$((SC_N+1))
  sc_say "point $SC_N: raw $raw purged rows $rows (N ~ $(awk -v r="$rows" -v f="$STOP_NFAC" 'BEGIN{printf "%.2fM", r*f/1e6}')) excess ${sx:-?} t ${t:-?}s"; }
sc_rm_purged(){   # $1 = probe dir: drop its purged file -- unless SPEC_MERGE pins it (spec_merge.sh: a preparation reads
  # it or its prepared matrix is held); then it is only marked .orphan and the last unpin deletes it
  if ls "$1"/.pin.* > /dev/null 2>&1; then : > "$1/.orphan"; else rmf "$1"/purged*; fi; }   # rmf: factor_c169.sh (truncates a memfd link first)
sc_keep(){   # $1 = probe dir whose purged file is now the stop matrix; the previous one's is dropped
  [ -n "$SC_KEEP" ] && [ "$SC_KEEP" != "$1" ] && [ "$SC_KEEP" != "${MATRIX_DIR:-}" ] && sc_rm_purged "$SC_KEEP"
  SC_KEEP=$1; }
sc_decide(){   # $1 = probe|land  $2 = landed raw -> SC_ACT continue|stop_probe|stop_final
  local r out; SC_ACT=continue; r=$(sc_rate); sc_rate_warn; [ -n "$r" ] || return 0
  out=$(env STOP_NFAC="$STOP_NFAC" STOP_C_NNZ_PS="$STOP_C_NNZ_PS" STOP_DENSE_NS="$STOP_DENSE_NS" STOP_NNZ_ROW="$STOP_NNZ_ROW" \
        STOP_POST_US="$STOP_POST_US" STOP_LAMBDA="$STOP_LAMBDA" STOP_MARGIN="$STOP_MARGIN" STOP_PROBE_TAX="$STOP_PROBE_TAX" \
        STOP_SHAPE="${STOP_SHAPE:-0}" STOP_KAPPA="${STOP_KAPPA:-9.9}" STOP_SX="${STOP_SX:-0.22}" STOP_SHAPE_A="${STOP_SHAPE_A:-0.102}" \
        STOP_SHAPE_B="${STOP_SHAPE_B:-0.44}" STOP_MINPTS="${STOP_MINPTS:-2}" \
        STOP_LA_SCALE="${STOP_LA_SCALE:-1}" \
        python3 "$HERE/stop_cost.py" decide "$SC_PTS" "$r" "$2" "${SC_STEP:-$CHUNK_RELS_LATE}" 2>>"$SC_LOG") \
    || { sc_say "WARNING: stop_cost.py failed -- sieving on"; return 0; }
  SC_ACT=${out%% *}; case "$SC_ACT" in continue|stop_probe|stop_final) ;; *) SC_ACT=continue;; esac
  # a stop needs a kept matrix to fall back on (stop_probe merges it; stop_final falls back to it)
  if [ "$SC_ACT" != continue ] && ! { [ -n "$SC_KEEP" ] && [ -s "$SC_KEEP/$PG" ]; }; then SC_ACT=continue; out="continue (no kept matrix) $out"; fi
  [ "$SC_ACT" != continue ] && [ -z "${SC_T_STOP:-}" ] && SC_T_STOP=$(date +%s)   # stop verdict -> LA start (ds_cado.sh)
  sc_say "at $1: $out"; }
sc_kids(){ local c; for c in $(pgrep -P "$1" 2>/dev/null); do sc_kids "$c"; echo "$c"; done; }
sc_probe_kill(){   # kill a running (non-final) probe and everything under it; its partial outputs are dropped
  [ -n "${PROBEPID:-}" ] || return 0
  local k; k=$(sc_kids "$PROBEPID"); kill -TERM "$PROBEPID" $k 2>/dev/null; wait "$PROBEPID" 2>/dev/null
  sc_say "probe $PROBE_I (chunks 1..$PROBE_N) killed: superseded by the stop"
  [ "$WORK/probe$PROBE_I" = "$SC_KEEP" ] || { sc_rm_purged "$WORK/probe$PROBE_I"; rmf "$WORK/probe$PROBE_I"/history*; }
  PROBEPID=""; PROBE_RUN_NT=0
  [ "${INC_FREE_PENDING:-0}" = 1 ] && { INC_FREE_PENDING=0; inc_free; }; return 0; }
sc_fallback(){ # $1 = why: merge the kept probe's matrix
  [ -n "$SC_KEEP" ] && [ -s "$SC_KEEP/$PG" ] || die "stop rule: final handoff failed ($1) and no kept matrix"
  sc_say "WARNING: final handoff: $1 -- using the kept matrix $SC_KEEP"; MATRIX_DIR=$SC_KEEP; TP_MERGE=1; }
sc_delta_base(){   # DELTA_PURGE: newest finished singleton-only base below INC_STEP (waits <= DELTA_WAIT s for a running one)
  local s b="" bk=0 k t=0 p
  while :; do
    p=""; for s in "$WORK"/probe*/sing; do [ -d "$s" ] && [ ! -e "$s/sing.done" ] && [ ! -e "$s/sing.fail" ] && p=$s; done
    [ -n "$p" ] && [ "$t" -lt "$DELTA_WAIT" ] || break; sleep 2; t=$((t+2))
  done
  if [ -n "$p" ]; then k=$(cat "$p/sing.pid" 2>/dev/null); [ -n "$k" ] && kill -TERM $(sc_kids "$k") "$k" 2>/dev/null
    sc_say "delta purge: base $p still running after ${DELTA_WAIT}s -- killed" >&2; rm -rf "$p"; fi
  for s in "$WORK"/probe*/sing; do [ -s "$s/sing.done" ] || continue
    k=$(grep -oE '^inc_k=[0-9]+' "$s/sing.done" | cut -d= -f2)
    [ -n "$k" ] && [ "$k" -lt "$INC_STEP" ] && [ "$k" -gt "$bk" ] && { bk=$k; b=$s; }; done
  [ -n "$b" ] && sc_say "delta purge: base $b (incdup steps 1..$bk, $(grep -oE '^rows=[0-9]+' "$b/sing.done" | cut -d= -f2) rows, built in $(grep -oE '^t=[0-9]+' "$b/sing.done" | cut -d= -f2)s; waited ${t}s)"  >&2
  echo "$b"; }
sc_final(){    # the sieve is stopped: commit every landed chunk with all cores, purge chunks 1..LANDED at STOP_FINAL_NT
  local t0 t1 d irc k K; t0=$(date +%s); sc_probe_kill
  [ "$INC_BROKEN" = 0 ] || { sc_fallback "incremental cache unavailable (a one-shot dup would take minutes)"; return 0; }
  SIEVE_HOST_T=0; PROBE_RUN_NT=0; NEXT_PROBE=0   # the GPU sieve is gone: incdup may take every core
  if [ -n "${INCPID:-}" ]; then   # an incdup step of the last landed chunk(s) is running at nice 15 / par 1: widen it
    inc_set_par "$STOP_FINAL_GROUPS"; renice -n 0 -g "$INCPID" > /dev/null 2>&1 || true
    while kill -0 "$INCPID" 2>/dev/null; do sleep 1; done; inc_reap
  fi
  while [ "$INC_BROKEN" = 0 ] && [ "$INC_DONE" -lt "$LANDED" ]; do   # synchronous catch-up (replay's path), full width
    k=$((INC_STEP+1)); INC_FROM=$((INC_DONE+1)); INC_TO=$LANDED
    CADO_BIN="$CF" FRC="$WORK/freerel" INC_RENUMBER="$(inc_renum)" INC_PAR="$STOP_FINAL_GROUPS" INC_GROUPS="$STOP_FINAL_GROUPS" INC_NICE=0 \
      python3 -c "$INC_PYL" "$$" timeout -k 10 "$INC_TIMEOUT" bash "$CF/cado_incdup.sh" "$INC_CACHE" "$MEM" \
      "$WORK/poly.cado" "$k" $(for _c in $(seq "$INC_FROM" "$INC_TO"); do printf '%s ' "$MEM/$(printf c%03d "$_c")"; done) >> "$INC_CACHE.log" 2>&1 & INCPID=$!
    echo "$INCPID" > "$WORK/inc.pgid"; wait "$INCPID"; irc=$?; INCPID=""
    if [ "$irc" = 0 ] && [ -s "$INC_CACHE/chunk_$k.done" ]; then INC_STEP=$k; STEP_END[$k]=$INC_TO; INC_DONE=$INC_TO
    else INC_BROKEN=1; sc_fallback "final incdup step $k failed (rc=$irc)"; return 0; fi
  done
  [ "$INC_BROKEN" = 0 ] || { sc_fallback "incdup failed during the handoff"; return 0; }
  t1=$(date +%s); K=$LANDED; frwait; PROBE_I=$((PROBE_I+1)); d="$WORK/probe$PROBE_I"; mkdir -p "$d"; pp_link "$d" "$PROBE_I"
  CH=(); for k in $(seq 1 "$K"); do CH+=("$MEM/$(printf c%03d $k)"); done
  : > "$MEM/rels_out"; rm -f "$MEM/rels_out.cyc" "$MEM/rels_out.purged"
  local db="" dtag=""; [ "${DPS:-0}" = 1 ] && db=$(sc_delta_base)
  if [ -n "$db" ]; then   # DELTA_PURGE: singleton-only base + the steps after it; any failure -> the full purge below
    env INC_CACHE="$INC_CACHE" INC_K="$INC_STEP" DELTA_BASE="$db" DELTA_SING=0 MERGE_ONLY=1 PURGE_ONLY=1 LPB0=$LPB0 LPB1=$LPB1 NT="$STOP_FINAL_NT" \
      DUP2_PAR=$DUP2_PAR TD=$TD FRC="$WORK/freerel" bash "$CF/cado_filter160.sh" "$d" "$MEM" "$WORK/poly.cado" "$N" "$MEM/rels_out" "${CH[@]}" > "$d/filter.out" 2>&1; irc=$?
    if [ "$irc" = 0 ] && [ -n "$(tp_excess "$d")" ] && [ -s "$d/$PG" ]; then dtag=" [delta purge: base $db]"
    else sc_say "WARNING: delta purge on $db rc=$irc ($(grep -a 'VERDICT\|FAILED' "$d/filter.out" | tail -1)) -- full purge"; db=""; mv -f "$d/filter.out" "$d/filter.delta.out"; fi
  fi
  [ -n "$db" ] || { env INC_CACHE="$INC_CACHE" INC_K="$INC_STEP" DELTA_SING=0 MERGE_ONLY=1 PURGE_ONLY=1 LPB0=$LPB0 LPB1=$LPB1 NT="$STOP_FINAL_NT" DUP2_PAR=$DUP2_PAR TD=$TD \
    FRC="$WORK/freerel" bash "$CF/cado_filter160.sh" "$d" "$MEM" "$WORK/poly.cado" "$N" "$MEM/rels_out" "${CH[@]}" > "$d/filter.out" 2>&1; irc=$?; }
  if [ "$irc" = 0 ] && [ -n "$(tp_excess "$d")" ] && [ -s "$d/$PG" ]; then
    sc_say "final handoff on chunks 1..$K (${HAVE_AT[$K]} raw): incdup $(( t1-t0 ))s + purge NT=$STOP_FINAL_NT $(( $(date +%s)-t1 ))s: $(grep -a VERDICT "$d/filter.out" | tail -1 | sed 's/.*VERDICT //')$dtag"
    sc_record "$d" || true; [ -n "$SC_KEEP" ] && [ "$SC_KEEP" != "$d" ] && sc_rm_purged "$SC_KEEP"
    SC_KEEP=$d; MATRIX_DIR=$d; TP_MERGE=1
  else sc_fallback "final purge rc=$irc: $(grep -a 'VERDICT\|FAILED' "$d/filter.out" | tail -1)"; fi
  sc_say "stop->merge handoff $(( $(date +%s)-t0 ))s"; return 0; }
