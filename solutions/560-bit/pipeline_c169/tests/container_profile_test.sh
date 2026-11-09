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

# container_profile_test.sh (local, no GPU/CADO) -- unit test of the CONTAINER=1 regime switches in factor_c169.sh:
# RSA560_LPB0 (default-off, 2026-10-25) next to RSA560_LPB1 / RSA560_MFB1 / RSA560_LAMBDA1 / RSA560_ALIM, and the
# FIRST_PROBE_AUTO knob. The CONTAINER block and the LPB/FIRST_PROBE defaults are extracted from the real script and
# eval'd in a subshell per case. Run: bash pipeline_c169/tests/container_profile_test.sh
set -u
D=$(cd "$(dirname "$0")" && pwd); F=$D/../factor_c169.sh; fails=0
BLOCK=$(sed -n '/^CONTAINER=\${CONTAINER:-0}/,/^fi$/p' "$F")           # the validator profile block
LPBS=$(grep -m1 '^LPB0=\${LPB0:-31}' "$F")                              # LPB0/LPB1/MFB0/MFB1 defaults
LAMS=$(grep -m1 '^LAMBDA0=\${LAMBDA0:-2.1}' "$F")                       # LAMBDA/NCURVES defaults
FPA=$(sed -n '/^if \[ "\${FIRST_PROBE_AUTO:-0}" = 1 \]/,/^FIRST_PROBE=\${FIRST_PROBE:-270000000}/p' "$F")
[ -n "$BLOCK" ] && [ -n "$LPBS" ] && [ -n "$LAMS" ] && [ -n "$FPA" ] || { echo "FAIL: could not extract the blocks from $F"; exit 1; }
# run one case: env assignments in $1, expected "LPB0 LPB1 LAMBDA0 LAMBDA1 FIRST_PROBE" in $2 (or DIE)
run_case(){
  local out
  out=$(env -i PATH="$PATH" HOME=/tmp bash -c "$1; HERE=$D/..; MSV=/nonexistent; say(){ :; }; die(){ echo DIE; exit 1; };
    $BLOCK
    $LPBS
    $LAMS
    $FPA
    echo \"\$LPB0 \$LPB1 \$LAMBDA0 \$LAMBDA1 \$FIRST_PROBE\"" 2>/dev/null | tail -1)
  if [ "$out" = "$2" ]; then echo "ok   [$1] -> $out"; else echo "FAIL [$1] -> '$out' (expected '$2')"; fails=$((fails+1)); fi; }
run_case "CONTAINER=0"                                       "31 32 2.1 2.1 270000000"      # molab/sandbox: nothing changes
run_case "CONTAINER=1"                                       "31 32 2.1 2.1 270000000"      # container default = 31/32
run_case "CONTAINER=1 RSA560_LPB1=33"                        "31 33 2.1 2.0379 270000000"   # S1/S2B switch
run_case "CONTAINER=1 RSA560_LPB0=32"                        "32 32 2.0352 2.1 270000000"   # side-0 switch alone
run_case "CONTAINER=1 RSA560_LPB0=32 RSA560_LPB1=33"         "32 33 2.0352 2.0379 270000000" # the 32/33 candidate (key1m.env)
run_case "CONTAINER=1 RSA560_LPB0=31 RSA560_LPB1=33"         "31 33 2.1 2.0379 270000000"
run_case "CONTAINER=1 RSA560_LPB0=32 LPB0=31"                "31 32 2.1 2.1 270000000"      # explicit LPB0 wins (small-key profile)
run_case "CONTAINER=1 RSA560_LPB0=32 LAMBDA0=2.2"            "32 32 2.2 2.1 270000000"      # explicit LAMBDA0 wins
run_case "CONTAINER=0 RSA560_LPB0=32"                        "31 32 2.1 2.1 270000000"      # the switch is container-only
run_case "CONTAINER=1 RSA560_LPB0=30"                        "DIE"                           # out of range
run_case "CONTAINER=1 RSA560_LPB0=32 RSA560_LPB1=33 FIRST_PROBE_AUTO=1" "32 33 2.0352 2.0379 406000000"   # 0.92 x 441M
run_case "CONTAINER=1 RSA560_LPB1=33 FIRST_PROBE_AUTO=1"     "31 33 2.1 2.0379 337000000"   # 0.92 x 366M
run_case "CONTAINER=1 FIRST_PROBE_AUTO=1"                    "31 32 2.1 2.1 270000000"      # 31/32: the production default
run_case "CONTAINER=0 LPB0=32 LPB1=34 FIRST_PROBE_AUTO=1"    "32 34 2.1 2.1 473000000"      # bare pipeline, modelled regime
run_case "CONTAINER=0 LPB0=32 LPB1=34 FIRST_PROBE_AUTO=1 FIRST_PROBE=300000000" "32 34 2.1 2.1 300000000"  # explicit wins
run_case "CONTAINER=0 LPB0=32 LPB1=34 FIRST_PROBE_AUTO=1 FIRST_PROBE_FRAC=0.85" "32 34 2.1 2.1 437000000"
bash -n "$F" && echo "ok   bash -n $F" || { echo "FAIL bash -n"; fails=$((fails+1)); }
echo "$([ $fails = 0 ] && echo PASS || echo FAIL): $fails failures"; exit $(( fails > 0 ))
