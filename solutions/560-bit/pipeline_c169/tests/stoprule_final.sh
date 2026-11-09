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

# stoprule_final.sh (local, no GPU/CADO) -- sc_final dry run with stub incdup / filter / timeout: commits the landed chunks, purges at STOP_FINAL_NT, sets MATRIX_DIR;
# then the failure path (filter fails -> kept matrix)
set -u
D=$(cd "$(dirname "$0")" && pwd); HERE=$D/..; WORK=$(mktemp -d)/w; rm -rf "$WORK"; mkdir -p "$WORK/inc" "$WORK/mem" "$WORK/stub" "$WORK/bin"
T0=$(date +%s); el(){ echo "[$(( $(date +%s)-T0 ))s]"; }; say(){ echo "$(el) $*"; }; die(){ echo "FATAL: $*"; exit 1; }
cat > "$WORK/bin/timeout" <<'EOF'
#!/bin/bash
[ "$1" = -k ] && shift 2; shift; exec "$@"
EOF
cat > "$WORK/stub/cado_incdup.sh" <<'EOF'
#!/bin/bash
echo "stub incdup k=$4 par=$INC_PAR groups=$INC_GROUPS nice=$INC_NICE chunks=$(( $# - 4 ))"; echo ok > "$1/chunk_$4.done"; echo "[incdup] COMMIT k=$4 t=1s"
EOF
cat > "$WORK/stub/cado_filter160.sh" <<'EOF'
#!/bin/bash
[ -n "${FAILF:-}" ] && { echo "[filter160] VERDICT FAIL stub"; exit 1; }
echo "stub filter NT=$NT PURGE_ONLY=$PURGE_ONLY INC_K=$INC_K chunks=$(( $# - 5 ))" >&2
echo x > "$1/purged"; echo "[filter160] VERDICT MATRIX raw=313060597 unique=1 sing_excess=7000000 sing_ncols=1 final='nrows=58000000 ncols=1 excess=160' t=2s purge-only"
EOF
chmod +x "$WORK/bin/timeout" "$WORK"/stub/*; PATH=$WORK/bin:$PATH
CF=$WORK/stub; MEM=$WORK/mem; N=123; LPB0=31; LPB1=32; DUP2_PAR=4; TD=130; INC_TIMEOUT=60; INC_CACHE=$WORK/inc; NCPU=20
PG=purged; CHUNK_RELS_LATE=4000000; INC_GROUPS=2; MATRIX_DIR=""; TP_MERGE=0; PROBEPID=""; PROBE_I=3; PROBE_N=0; INCPID=""
INC_BROKEN=0; INC_FREED=0; INC_DONE=53; INC_STEP=20; LANDED=55; declare -a HAVE_AT; HAVE_AT[55]=313060597; declare -A STEP_END=()
inc_set_par(){ :; }; inc_reap(){ :; }; inc_free(){ :; }; frwait(){ :; }; inc_renum(){ :; }
tp_excess(){ grep -a 'VERDICT MATRIX' "$1/filter.out" 2>/dev/null | tail -1 | grep -a 'purge-only' | grep -oE 'sing_excess=-?[0-9]+' | cut -d= -f2; }
eval "$(sed -n '/^INC_PYL=/,/^os.execvp/p' "$HERE/factor_c169.sh")"
eval "$(sed -n '/^STOP_RULE=\${STOP_RULE:-excess}/,/^STOP_FINAL_GROUPS=/p' "$HERE/factor_c169.sh")"; STOP_RULE=cost; STOP_FINAL_NT=$NCPU
SC_N=0; SC_FINAL=0; SC_KEEP=""; . "$HERE/stop_cost.sh"
mkdir -p "$WORK/probe3"; echo x > "$WORK/probe3/purged"; SC_KEEP=$WORK/probe3
sc_final; echo "=> MATRIX_DIR=$MATRIX_DIR TP_MERGE=$TP_MERGE INC_DONE=$INC_DONE INC_STEP=$INC_STEP kept=$SC_KEEP probe3 purged left: $(ls $WORK/probe3)"
cat "$WORK/inc.log"
MATRIX_DIR=""; TP_MERGE=0; LANDED=56; HAVE_AT[56]=317000000; SC_KEEP=$WORK/probe4; export FAILF=1
sc_final; echo "=> (filter fails) MATRIX_DIR=$MATRIX_DIR TP_MERGE=$TP_MERGE"
