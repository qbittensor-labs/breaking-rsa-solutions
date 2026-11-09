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

# stoprule_rate_capture.sh (local, no GPU/CADO) -- known-answer test of stop_cost.sh sc_rate / sc_decide after the capture
# fix (docs/research/agents_2026-10-26/downstream.md #2): $(sc_rate) must be a bare number in every mode, sc_decide must
# reach a verdict with SC_RATES EMPTY (replay mode: the key 29 container replay printed "stop_cost.py failed" at all 6
# decisions because the rate warning rode into the captured rate), and the live path must be unchanged: with arm B's last
# four chunk rates the decision line must carry rate=30266, the value arm B logged at [10942s] (data/rsa560/audit_s2/
# s2b_armB/h_full_stdout.txt). Points = arm B's measured probes 4 and 5. Exit 1 on any failure.
set -u
D=$(cd "$(dirname "$0")" && pwd); HERE=$D/..; WORK=$(mktemp -d)/w; rm -rf "$WORK"; mkdir -p "$WORK"
T0=$(date +%s); el(){ echo "[$(( $(date +%s)-T0 ))s]"; }; say(){ echo "$(el) $*"; }; die(){ echo "FAIL: $*"; exit 1; }
inc_set_par(){ :; }; inc_reap(){ :; }; inc_free(){ :; }; frwait(){ :; }; inc_renum(){ :; }
NCPU=20; PG=purged; CHUNK_RELS_LATE=4000000; INC_GROUPS=2; MATRIX_DIR=""; TP_MERGE=0; PROBEPID=""; PROBE_I=0; PROBE_N=0
# the STOP_* defaults exactly as factor_c169.sh resolves them (the block from STOP_RULE= to STOP_FINAL_GROUPS=)
eval "$(sed -n '/^STOP_RULE=\${STOP_RULE:-/,/^STOP_FINAL_GROUPS=/p' "$HERE/factor_c169.sh")"; STOP_RULE=cost; NCPU=20
[ -n "${STOP_NFAC:-}" ] && [ -n "${STOP_RATE_DEFAULT:-}" ] && [ -n "${STOP_RATE_N:-}" ] || die "STOP_* block not extracted from factor_c169.sh"
SC_N=0; SC_FINAL=0; SC_KEEP=""; . "$HERE/stop_cost.sh"
type sc_rate sc_rate_warn sc_decide > /dev/null || die "functions not loaded (sc_rate_warn missing = unpatched stop_cost.sh)"
# two measured purge-only points (arm B probes 4 and 5): raw rows excess dir secs
printf '370719899\t87080116\t238800\t%s/probe4\t64\n374694380\t81581603\t1055452\t%s/probe5\t66\n' "$WORK" "$WORK" > "$SC_PTS"
mkdir -p "$WORK/probe5"; echo x > "$WORK/probe5/purged"; SC_KEEP=$WORK/probe5; SC_N=2; SC_STEP=3950681
num(){ [[ "$1" =~ ^[0-9]+$ ]]; }
fails=0; ok(){ echo "  ok: $*"; }; bad(){ echo "  FAIL: $*"; fails=$((fails+1)); }

# ---- case A: replay mode (SC_RATES empty, STOP_RATE unset) ----------------------------------------------------------
unset STOP_RATE; SC_RATES=(); SC_RW=0; : > "$SC_LOG"
r=$(sc_rate); num "$r" && [ "$r" = "$STOP_RATE_DEFAULT" ] && ok "A: \$(sc_rate) = $r (bare STOP_RATE_DEFAULT)" || bad "A: \$(sc_rate) = '$r'"
sc_decide land 378645061
case "$SC_ACT" in continue|stop_probe|stop_final) ok "A: sc_decide verdict $SC_ACT";; *) bad "A: SC_ACT='$SC_ACT'";; esac
grep -q "stop_cost.py failed" "$SC_LOG" && bad "A: stop_cost.py failed (the pre-fix symptom)" || ok "A: no 'stop_cost.py failed'"
grep -q "rate=$STOP_RATE_DEFAULT" "$SC_LOG" && ok "A: decision line carries rate=$STOP_RATE_DEFAULT" || bad "A: rate missing: $(tail -1 "$SC_LOG")"
sc_decide land 378645061; sc_decide land 382580748
w=$(grep -c "no measured chunk rate" "$SC_LOG"); [ "$w" = 1 ] && ok "A: rate warning printed exactly once over 3 decisions" || bad "A: warning count $w"
[ "$SC_RW" = 1 ] && ok "A: SC_RW latched in the parent shell" || bad "A: SC_RW=$SC_RW"

# ---- case B: live path, arm B's last four landed chunk rates (60..63) -> median 30266 as logged at [10942s] ---------
unset STOP_RATE; SC_RATES=(31492 31050 29482 29153); SC_RW=0; : > "$SC_LOG"
r=$(sc_rate); [ "$r" = 30266 ] && ok "B: median of the last 4 rates = 30266 (arm B [10942s] rate=30266)" || bad "B: sc_rate = '$r'"
sc_decide land 382580748
grep -q "rate=30266" "$SC_LOG" && ok "B: decision line carries rate=30266" || bad "B: $(tail -1 "$SC_LOG")"
grep -q "no measured chunk rate" "$SC_LOG" && bad "B: spurious rate warning on the live path" || ok "B: no warning on the live path"
case "$SC_ACT" in continue|stop_probe|stop_final) ok "B: verdict $SC_ACT";; *) bad "B: SC_ACT='$SC_ACT'";; esac

# ---- case C: STOP_RATE passthrough (replay harness / gates) -------------------------------------------------------
export STOP_RATE=31000; SC_RATES=(); SC_RW=0; : > "$SC_LOG"
r=$(sc_rate); [ "$r" = 31000 ] && ok "C: STOP_RATE passthrough" || bad "C: sc_rate = '$r'"
sc_decide land 378645061
grep -q "rate=31000" "$SC_LOG" && ok "C: decision line carries rate=31000" || bad "C: $(tail -1 "$SC_LOG")"
grep -q "no measured chunk rate" "$SC_LOG" && bad "C: warning with STOP_RATE set" || ok "C: no warning with STOP_RATE set"
unset STOP_RATE

# ---- the pre-fix behaviour, reproduced from the rsa560 tip's function text (documentation of the bug) ---------------
old=$(cd "$HERE/.." && git show rsa560:pipeline_c169/stop_cost.sh 2>/dev/null | awk '/^sc_rate\(\)\{/,/^  printf .*awk/')
if [ -n "$old" ]; then
  r=$(SC_RATES=(); SC_RW=0; eval "$old"; sc_rate 2>/dev/null); num "$r" && bad "old sc_rate unexpectedly clean ('$r')" \
    || ok "old sc_rate on rsa560 returns '$(echo "$r" | head -c 60)...' (not a number: the bug)"
else echo "  (rsa560 branch not readable here: pre-fix reproduction skipped)"; fi

rm -rf "$(dirname "$WORK")"
[ "$fails" = 0 ] && { echo "ALL OK"; exit 0; } || { echo "$fails FAILURE(S)"; exit 1; }
