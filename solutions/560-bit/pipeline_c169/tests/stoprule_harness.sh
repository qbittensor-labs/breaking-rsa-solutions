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

# stoprule_harness.sh (local, no GPU/CADO; MODE=replay|live, PEVERY=n) -- dry-run harness: the REAL stop_cost.sh + sc_after_probe/sc_after_land (extracted from factor_c169.sh) driven by
# key1g's chunk landings (run_seg2_sb48) and purge-only matrix points (rows interpolated between the measured ones)
set -u
D=$(cd "$(dirname "$0")" && pwd); HERE=$D/..; WORK=$(mktemp -d)/w; rm -rf "$WORK"; mkdir -p "$WORK"
T0=$(date +%s); el(){ echo "[$(( $(date +%s)-T0 ))s]"; }; say(){ echo "$(el) $*"; }; die(){ echo "FATAL: $*"; exit 1; }
inc_set_par(){ :; }; inc_reap(){ :; }; inc_free(){ :; }; frwait(){ :; }; inc_renum(){ :; }
NCPU=20; PG=purged; CHUNK_RELS_LATE=4000000; INC_GROUPS=2; MATRIX_DIR=""; TP_MERGE=0; PROBEPID=""; PROBE_I=0; PROBE_N=0
eval "$(sed -n '/^STOP_RULE=\${STOP_RULE:-excess}/,/^STOP_FINAL_GROUPS=/p' "$HERE/factor_c169.sh")"; STOP_RULE=cost; NCPU=20
SC_N=0; SC_FINAL=0; SC_KEEP=""; . "$HERE/stop_cost.sh"
eval "$(awk '/^sc_after_probe\(\)\{/,/^  esac; return 1; }$/' "$HERE/factor_c169.sh")"
eval "$(awk '/^sc_after_land\(\)\{/,/^  esac; return 1; }$/' "$HERE/factor_c169.sh")"
type sc_after_probe sc_after_land sc_decide > /dev/null || die "functions not loaded"
# key1g live chunk landings: slot raw n dt (chunks 49..61)
LAND="49 289079116 3903978 208
50 293080773 4001657 212
51 297044276 3963503 209
52 300962821 3918545 209
53 305048818 4085997 216
54 309032091 3983273 215
55 313060597 4028506 234
56 316977720 3917123 211
57 320984847 4007127 214
58 324889825 3904978 210
59 328925408 4035583 218
60 332904632 3979224 218
61 336945450 4040818 241"
# measured purge-only matrix points (raw rows excess): s61 + live probe 4; others log-interpolated
rows_at(){ python3 - "$1" <<'PY'
import sys, math
P=[(300962821,64033076),(309032091,59650068),(316977720,56203479),(332904632,50845327)]
a=float(sys.argv[1])
for (a1,r1),(a2,r2) in zip(P,P[1:]):
    if a1<=a<=a2: print(int(math.exp(math.log(r1)+(math.log(r2)-math.log(r1))*(a-a1)/(a2-a1)))); break
else: print(int(P[-1][1]*math.exp(-0.0068*(a-P[-1][0])/1e6)) if a>P[-1][0] else 0)
PY
}
mkprobe(){ PROBE_I=$((PROBE_I+1)); d=$WORK/probe$PROBE_I; local r; mkdir -p "$d"; r=$(rows_at "$1")
  echo "[filter160] VERDICT MATRIX raw=$1 unique=0 sing_excess=$2 sing_ncols=0 final='nrows=$r ncols=$r excess=160' t=150s purge-only" > "$d/filter.out"
  echo x > "$d/purged"; }
MODE=${MODE:-replay}   # replay: probe each landed chunk at once; live: points lag one chunk (incdup + probe)
while read -r slot have n dt; do
  SC_RATES+=("$(( n / dt ))"); SC_STEP=$n; LANDED=$slot; eval "HAVE_AT[$slot]=$have"
  if sc_after_land "$have"; then echo "=> STOP at landing of slot $slot ($have raw): act=$SC_ACT final=$SC_FINAL matrix=${MATRIX_DIR:-<final purge>} kept=$SC_KEEP"; break; fi
  pa=$have; [ "$MODE" = live ] && pa=$(echo "$LAND" | awk -v s=$((slot-1)) '$1==s{print $2}')
  [ "$pa" -ge 300000000 ] || continue; [ $(( slot % ${PEVERY:-1} )) = 0 ] || continue
  mkprobe "$pa" 4000000; PROBE_AT=$pa
  if sc_after_probe "$d" "$have"; then echo "=> STOP at probe on $pa raw (landed $have): act=$SC_ACT final=$SC_FINAL matrix=${MATRIX_DIR:-<final purge>}"; break; fi
  echo "   kept=$SC_KEEP purged files: $(ls "$WORK"/probe*/purged 2>/dev/null | wc -l | tr -d ' ')"
done <<< "$LAND"
