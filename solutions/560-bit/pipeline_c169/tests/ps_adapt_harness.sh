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

# ps_adapt_harness.sh (local, no GPU/CADO) -- PS_ADAPT dry run: the REAL polyselect block of factor_c169.sh (extracted
# between "### [1/4] polyselect" and the poly.cado N check), the real ps_overlap.sh and polysel160.py, with stub gps1 /
# sopt / polyselect_ropt / timeout. The gps1 stub writes one batch every BATCH_S s; a candidate's c5 encodes its root
# MurphyE (c5 div 1000 = E in 1e-12 units, read back by the ropt stub), so each scenario fixes what the tail must pick.
# Scenarios (all must PASS):
#   ref    PS_ADAPT=0 (new script) == the committed HEAD script on the same stub: identical poly.cado and log (times stripped)
#   stop   PS_ADAPT=1, b3 E 1.217e-9 >= 1.15e-9 -> decision stop, poly.cado == ref's, gps1 killed before batch 4 matters
#   ext    PS_ADAPT=1, b3 1.090e-9 -> extend to 6, b5 holds 1.246e-9 -> final 1.246e-9, pick b6
#   worse  PS_ADAPT=1, b3 1.090e-9; batches 4-6 have better exp_E (push the b3 winner out of top-ROPT_K) but lower E
#          -> final pick loses -> the b3 polynomial is kept (poly.cado == the PS_ADAPT=0 poly)
#   cap    PS_ADAPT=1 extend with PS_MAX cutting the search at batch 4 -> final from 4 batches, wall < PS_MAX + tail
#   afail  PS_ADAPT=1, pass a's ropt fails -> decision fail -> production path (seq) on the current raw.p, no crash
#   memo   (on ext) pass b reuses pass a's ropt results: ropt miss = the batch 4-6 newcomers only (3 of 20; was 20 of 20
#          before memo v2), and ps_overlap.sh seq on the same final raw.p gives a byte-identical poly.cado
#   roll   PS_MEMO_ROLL=1 on WEAK: rolling passes on batches 4 and 5, poly.cado == ext's, pass b ropt miss 1 of 20
#   roll6  PS_MEMO_ROLL=1 vs 0 on WEAK6 (batch 6 adds nothing to the top 20): pass b re-runs no ropt at all; pass b
#          re-timed standalone on both runs' saved state: identical poly.cado, roll's wall shorter
#   rkill  PS_MEMO_ROLL=1 with a rolling ropt that hangs: killed at the stop (no stub left running), poly.cado == ext's
# The ropt stub prints what multithreaded CADO polyselect_ropt prints: command header, ALL input blocks first, outputs in
# completion (here: reverse) order, '# side 1 MurphyE(...)=', a '# Stat:' trailer; it sleeps FAKE_ROPT_S0 + FAKE_ROPT_S1
# per input (a call's wall is mostly fixed, as measured: s46 1 input 100 s, 20 inputs 115-135 s).
# usage: bash pipeline_c169/tests/ps_adapt_harness.sh   (BATCH_S=4 default; ~3 min)
set -u
D=$(cd "$(dirname "$0")" && pwd); HERE=$(cd "$D/.." && pwd); ROOT=$(mktemp -d); BATCH_S=${BATCH_S:-4}
STUB=$ROOT/stub; mkdir -p "$STUB/bin"; PASS=0; FAIL=0
ok(){ echo "PASS $*"; PASS=$((PASS+1)); }; bad(){ echo "FAIL $*"; FAIL=$((FAIL+1)); }
cat > "$STUB/bin/timeout" <<'EOF'
#!/bin/bash
[ "$1" = --foreground ] && shift; shift; exec "$@"
EOF
# gps1 stub: GPS1_PROFILE "Ecode:expE[,Ecode:expE...] ..." one word per batch (the batch's special candidates); every
# batch also has 100 fillers (Ecode 900, exp_E 44+). Writes each batch in one go, then sleeps until killed.
cat > "$STUB/gps1" <<'EOF'
#!/bin/bash
b=0
for word in $GPS1_PROFILE; do b=$((b+1)); sleep "$BATCH_S"
  python3 - "$b" "$word" <<'PY'
import sys
b = int(sys.argv[1]); spec = [w.split(':') for w in sys.argv[2].split(',')]
out = []
def cand(ecode, e, i):
    c5 = ecode * 1000 + b * 100 + i % 100
    co = dict(n=1234567, Y1=1000 + b, Y0=-(77 + 100 * i), c0=0, c1=-3 - b, c2=5, c3=7, c4=-9 - i, c5=c5)
    m = (-co['Y0'] * pow(co['Y1'], -1, co['n'])) % co['n']     # f(m) = 0 mod n, as for a real polynomial pair
    co['c0'] = -sum(co[f'c{k}'] * pow(m, k, co['n']) for k in range(1, 6)) % co['n']
    s = "# Raw polynomial:\n" + ''.join(f"# {k}: {co[k]}\n" for k in ('n', 'Y1', 'Y0', 'c5', 'c4', 'c3', 'c2', 'c1', 'c0'))
    s += "# Size-optimized polynomial:\n" + ''.join(f"{k}: {co[k]}\n" for k in ('n', 'Y1', 'Y0', 'c5', 'c4', 'c3', 'c2', 'c1', 'c0'))
    s += f"skew: {1000.0 + i}\n# lognorm 50.00, exp_E {e:.2f}, alpha -5.0 (proj -1.0), 1 real root\n\n"
    return s
for j, (ec, e) in enumerate(spec): out.append(cand(int(ec), float(e), j))
for i in range(100): out.append(cand(900, 44.0 + b + i / 100, 50 + i))
sys.stdout.write(''.join(out)); sys.stdout.flush()
PY
done
exec sleep 100000
EOF
# sopt stub: echo each input, return it unchanged as the size-optimized poly (exp_E 46: originals stay ahead in topk)
cat > "$STUB/sopt" <<'EOF'
#!/bin/bash
f=$2; python3 - "$f" <<'PY'
import sys, re
t = open(sys.argv[1]).read().strip().split('\n\n'); K = ('n', 'Y0', 'Y1', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5')
for i, b in enumerate(x for x in t if x.strip()):
    p = dict(re.findall(r'^(\w+): (-?\d+)$', b, re.M))
    print(f"### Input raw polynomial ({i}) ###"); [print(f"# {k}: {p[k]}") for k in K]
    print(f"### Size-optimized polynomial ({i}) ###"); [print(f"{k}: {p[k]}") for k in K]
    print("skew: 999.0\n# side 1 lognorm 51.00, exp_E 46.00\n")
PY
EOF
# ropt stub: MurphyE = (c5 div 1000) * 1e-12; FAKE_ROPT_FAIL_OV=1 fails any input under /ov/ (pass a / b);
# FAKE_ROPT_HANG_R=S sleeps S s more on a rolling pass's input (under /ov/R)
cat > "$STUB/polyselect_ropt" <<'EOF'
#!/bin/bash
f=$2; [ "${FAKE_ROPT_FAIL_OV:-0}" = 1 ] && case "$f" in */ov/*) exit 1;; esac
case "$f" in */ov/R*) [ "${FAKE_ROPT_HANG_R:-0}" != 0 ] && sleep "$FAKE_ROPT_HANG_R";; esac
python3 - "$0" "$@" <<'PY'
import sys, re, time, os
a = sys.argv[2:]; o = {a[i][1:]: a[i + 1] for i in range(0, len(a) - 1, 2)}
K = ('n', 'Y0', 'Y1', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5')
bl = [b for b in re.split(r'(?=# Size-optimized polynomial)', open(o['inputpolys']).read()) if b.startswith('# Size-optimized')]
time.sleep(float(os.environ.get('FAKE_ROPT_S0', '0.5')) + float(os.environ.get('FAKE_ROPT_S1', '0.05')) * len(bl))
print(f"# (stub) {sys.argv[1]} {' '.join(a)}\n# Info: Will use {o['t']} threads\n# {len(bl)} polynomial(s) read.\n")
P = [dict(re.findall(r'^(\w+): (-?[\d.]+)$', b, re.M)) for b in bl]
for i, p in enumerate(P):
    print(f"### input polynomial {i} ###"); [print(f"# {k}: {p[k]}") for k in K]
    print(f"# skew: {float(p['skew']):.3f}\n# # side 1 lognorm 52.00, exp_E 45.00, alpha -5.0 (proj -1.0), 5 real roots\n")
for i in reversed(range(len(P))):
    p = P[i]; E = (abs(int(p['c5'])) // 1000) * 1e-12
    print(f"### root-optimized polynomial {i} ###"); [print(f"{k}: {p[k]}") for k in K]
    print(f"skew: {float(p['skew']):.3f}\n# side 1 lognorm 52.00, E 45.00, alpha -5.0 (proj -1.0), 5 real roots")
    print(f"# side 1 MurphyE(Bf={float(o['Bf']):.3e},Bg={float(o['Bg']):.3e},area={float(o['area']):.3e})={E:.3e}")
    print(f"### Best MurphyE so far is {E:.3e}, av. exp_E 45.00, av. E 45.00\n")
print("# Average exp_E: 45.00, average E: 45.00\n# Stat: total phase took 0.10s\n# Stat: rootsieve took 0.10s")
PY
EOF
chmod +x "$STUB"/bin/* "$STUB"/gps1 "$STUB"/sopt "$STUB"/polyselect_ropt
export PATH=$STUB/bin:$PATH BATCH_S
extract(){ sed -n '/^echo "### \[1\/4\] polyselect/,/^grep -q "\^n: \$N/p' "$1" | sed '$d'; }
extract "$HERE/factor_c169.sh" > "$ROOT/block.new"
# reference = factor_c169.sh as it was just before PS_ADAPT was introduced (BASE_REV overrides; HEAD while uncommitted)
_intro=$(git -C "$HERE" log --format=%H -S'PS_ADAPT=${PS_ADAPT:-0}' -- factor_c169.sh 2>/dev/null | tail -1)
BASE_REV=${BASE_REV:-$([ -n "$_intro" ] && echo "$_intro^" || echo HEAD)}; echo "reference script: factor_c169.sh @ $BASE_REV"
git -C "$HERE" show "$BASE_REV:pipeline_c169/factor_c169.sh" > "$ROOT/factor.head" 2>/dev/null && extract "$ROOT/factor.head" > "$ROOT/block.head"
[ -s "$ROOT/block.new" ] || { echo "FATAL: polyselect block not found"; exit 1; }
# run NAME BLOCK [VAR=val ...] -> $ROOT/NAME/{log,poly.cado,wall}
run(){ local name=$1 blk=$2; shift 2; local W=$ROOT/$name/w; mkdir -p "$W"
  ( set -u; WORK=$W; N=1234567; CF=$STUB; GPS1=$STUB/gps1; SIEVER=glas; APP=$ROOT; NCPU=4; PS_NCPU=4; ALIM=450000000; RLIM=200000000
    PS_SECS=300; PS_MAX=900; PS_MIN_CANDS=200; ROPT_K=20; ROPT_MAX=420; ROPT_EFF=6; SOPT_EFF=9; SOPT_K=500; SOPT_MAX=120; SOPT_BIN=$STUB/sopt
    PS_P=1445000; PS_ADMIN=3360; PS_ADMAX=291000; PS_INCR=420; PS_NQ=19531; PS_BATCHES=3; HERE=$HERE
    for kv in "$@"; do eval "$kv"; export "${kv%%=*}"; done
    T0=$(date +%s); el(){ echo "[$(( $(date +%s)-T0 ))s]"; }; say(){ echo "$(el) $*"; }; die(){ echo "$(el) FATAL: $*"; exit 1; }
    export GPS1_PROFILE FAKE_ROPT_FAIL_OV=${FAKE_ROPT_FAIL_OV:-0}
    . "$blk"; echo "WALL $(( $(date +%s)-T0 ))" ) > "$ROOT/$name/log" 2>&1
  cp -f "$W/poly.cado" "$ROOT/$name/poly.cado" 2>/dev/null; pkill -f "$STUB/gps1" 2>/dev/null; return 0; }
adline(){ grep -a 'PS_ADAPT: b[0-9]* E=' "$ROOT/$1/log" | tail -1; }
strip(){ sed -E 's/\[[0-9]+s\]//g; s/ at [0-9]+s/ at Ts/; s/in [0-9]+s/in Ts/g; s/[0-9]{2}:[0-9]{2}:[0-9]{2}/HH:MM:SS/; /^WALL /d' "$1"; }
STRONG="1100:41.0 1150:40.9 1217:40.5 1300:40.0 1300:39.9 1300:39.8"   # b3 incumbent 1.217 (later better too: must NOT be used)
WEAK="1050:41.0 1080:40.9 1090:40.5 1100:41.5 1246:40.6 1200:40.7"     # b3 1.090, b5 has 1.246
WORSE="1050:41.0 1080:40.9 1090:40.5 1000:30.1,1000:30.2,1000:30.3,1000:30.4,1000:30.5,1000:30.6,1000:30.7,1000:30.8,1000:30.9,1000:31.0,1000:31.1,1000:31.2,1000:31.3,1000:31.4,1000:31.5,1000:31.6,1000:31.7,1000:31.8,1000:31.9,1000:32.0 1000:32.1 1000:32.2"

# ref: PS_ADAPT=0 new vs HEAD (byte-identical poly + log), strong and weak profiles
for prof in STRONG WEAK WORSE; do
  run "ref0_$prof" "$ROOT/block.new" "GPS1_PROFILE='${!prof}'" PS_ADAPT=0
  if [ -s "$ROOT/block.head" ]; then
    run "head_$prof" "$ROOT/block.head" "GPS1_PROFILE='${!prof}'"
    if cmp -s "$ROOT/ref0_$prof/poly.cado" "$ROOT/head_$prof/poly.cado" && diff <(strip "$ROOT/ref0_$prof/log") <(strip "$ROOT/head_$prof/log") > "$ROOT/ref_$prof.diff"; then
      ok "ref $prof: PS_ADAPT=0 == HEAD (poly.cado identical, log identical modulo times)"
    else bad "ref $prof: PS_ADAPT=0 differs from HEAD (see $ROOT/ref_$prof.diff)"; fi
  fi
done
E0(){ grep -a '^MurphyE' "$ROOT/$1/log" | tail -1 | awk '{print $2}'; }
echo "  PS_ADAPT=0 picks: STRONG $(E0 ref0_STRONG) WEAK $(E0 ref0_WEAK) WORSE $(E0 ref0_WORSE)"

run stop "$ROOT/block.new" "GPS1_PROFILE='$STRONG'" PS_ADAPT=1; L=$(adline stop); echo "  stop : $L"
case "$L" in *"E=1.2170e-09 -> decision stop"*"final E=1.2170e-09"*) cmp -s "$ROOT/stop/poly.cado" "$ROOT/ref0_STRONG/poly.cado" \
  && ok "stop: decision stop, poly.cado == PS_ADAPT=0" || bad "stop: poly differs from PS_ADAPT=0" ;; *) bad "stop: $L" ;; esac
grep -aq 'gps1 batch 4 complete' "$ROOT/stop/log" && echo "  (note: batch 4 landed during the stop tail -- ignored by design)"

run ext "$ROOT/block.new" "GPS1_PROFILE='$WEAK'" PS_ADAPT=1; L=$(adline ext); echo "  ext  : $L"
case "$L" in *"E=1.0900e-09 -> decision extend"*"6 batches, pick b6) final E=1.2460e-09"*) ok "ext: extend to 6, final 1.246e-9" ;; *) bad "ext: $L" ;; esac
grep -aq '^n: 1234567$' "$ROOT/ext/poly.cado" && grep -aq "^c5: 1246" "$ROOT/ext/poly.cado" && ok "ext: poly.cado is the 1.246 polynomial" || bad "ext: poly.cado"

run worse "$ROOT/block.new" "GPS1_PROFILE='$WORSE'" PS_ADAPT=1 ROPT_K=20; L=$(adline worse); echo "  worse: $L"
case "$L" in *"decision extend"*"pick b3) final E=1.0900e-09"*) cmp -s "$ROOT/worse/poly.cado" "$ROOT/ref0_WORSE/poly.cado" \
  && ok "worse: final tail lost, b3 polynomial kept (== PS_ADAPT=0)" || bad "worse: poly differs from PS_ADAPT=0" ;; *) bad "worse: $L" ;; esac

run cap "$ROOT/block.new" "GPS1_PROFILE='$WEAK'" PS_ADAPT=1 PS_MAX=$((BATCH_S*4+3)); L=$(adline cap); echo "  cap  : $L"
case "$L" in *"decision extend"*"4 batches"*"final E=1.1000e-09"*) ok "cap: PS_MAX stopped at batch 4 (final 1.100e-9)" ;; *) bad "cap: $L" ;; esac

run afail "$ROOT/block.new" "GPS1_PROFILE='$WEAK'" PS_ADAPT=1 FAKE_ROPT_FAIL_OV=1; L=$(adline afail); echo "  afail: $L"
case "$L" in *"decision fail"*) [ -s "$ROOT/afail/poly.cado" ] && ok "afail: pass a failure -> production path, poly.cado written" || bad "afail: no poly" ;; *) bad "afail: $L" ;; esac

run thr "$ROOT/block.new" "GPS1_PROFILE='$STRONG'" PS_ADAPT=1 PS_ADAPT_E=1.3e-9 PS_ADAPT_MAXB=5; L=$(adline thr); echo "  thr  : $L"
case "$L" in *"decision extend"*"5 batches, pick b5) final E=1.3000e-09"*) ok "thr: PS_ADAPT_E / PS_ADAPT_MAXB honoured" ;; *) bad "thr: $L" ;; esac

# ---- memo v2 + rolling memo (PS_MEMO_ROLL) -------------------------------------------------------------------------
rmiss(){ grep -ao 'ropt miss [0-9]* of [0-9]*' "$ROOT/$1/w/ps_overlap.b.log" | tail -1; }
psenv(){ env N=1234567 CF=$STUB NCPU=4 ALIM=450000000 RLIM=200000000 SOPT_EFF=9 SOPT_K=500 SOPT_MAX=120 SOPT_BIN=$STUB/sopt \
             ROPT_K=20 ROPT_EFF=6 ROPT_MAX=420 HERE=$HERE "$@"; }
# retime NAME: re-run pass b standalone on a copy of NAME's saved state (final raw.p = ov/raw.orig), best of 2 walls;
# prints the wall and checks the poly.cado it writes is byte-identical to the one the run wrote
retime(){ local R=$ROOT/$1.rt best=""; local i w
  for i in 1 2; do rm -rf "$R"; cp -R "$ROOT/$1/w" "$R"; cp -f "$R/ov/raw.orig" "$R/raw.p"; rm -f "$R/poly.cado"
    w=$(python3 -I -c 'import subprocess,sys,time; t=time.perf_counter(); r=subprocess.run(sys.argv[1:],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print("%.3f" % (time.perf_counter()-t) if r.returncode == 0 else "FAIL")' \
        env N=1234567 CF=$STUB NCPU=4 ALIM=450000000 RLIM=200000000 SOPT_EFF=9 SOPT_K=500 SOPT_MAX=120 SOPT_BIN=$STUB/sopt ROPT_K=20 ROPT_EFF=6 ROPT_MAX=420 \
        HERE=$HERE bash "$HERE/ps_overlap.sh" b "$R")
    [ "$w" = FAIL ] && { echo FAIL; return; }
    cmp -s "$R/poly.cado" "$ROOT/$1/poly.cado" || { echo DIFF; return; }
    best=$(awk -v a="$w" -v b="$best" 'BEGIN{print (b == "" || a + 0 < b + 0) ? a : b}'); done; echo "$best"; }

# memo: pass b on ext reuses pass a's ropt results, and equals the sequential tail on the same final raw.p
M=$(rmiss ext); echo "  memo : ext pass b: $M (before memo v2 every logged extend run printed 'ropt miss 20 of 20')"
[ "$M" = "ropt miss 3 of 20" ] && ok "memo: pass b re-ran only the 3 batch-4..6 newcomers (17 of 20 ropt results reused)" || bad "memo: $M"
mkdir -p "$ROOT/seqchk"; cp -f "$ROOT/ext/w/ov/raw.orig" "$ROOT/seqchk/raw.p"
psenv bash "$HERE/ps_overlap.sh" seq "$ROOT/seqchk" > "$ROOT/seqchk/log" 2>&1
cmp -s "$ROOT/seqchk/poly.cado" "$ROOT/ext/poly.cado" && ok "memo: pass b poly.cado == ps_overlap.sh seq on the same final raw.p (byte-identical)" \
  || bad "memo: pass b poly.cado differs from seq ($ROOT/seqchk)"

run roll "$ROOT/block.new" "GPS1_PROFILE='$WEAK'" PS_ADAPT=1 PS_MEMO_ROLL=1; L=$(adline roll); M=$(rmiss roll); echo "  roll : $L"
echo "  roll : $(grep -a 'PS_MEMO_ROLL' "$ROOT/roll/log" | sed 's/^/         /' | tr '\n' ' ')"; echo "  roll : pass b $M"
case "$L" in *"decision extend"*"6 batches, pick b6) final E=1.2460e-09"*) cmp -s "$ROOT/roll/poly.cado" "$ROOT/ext/poly.cado" \
  && ok "roll: poly.cado == ext (PS_MEMO_ROLL=0)" || bad "roll: poly differs from ext" ;; *) bad "roll: $L" ;; esac
grep -aq 'PS_MEMO_ROLL: batch-4 pass done' "$ROOT/roll/log" && grep -aq 'PS_MEMO_ROLL: batch-5 pass done' "$ROOT/roll/log" \
  && [ "$M" = "ropt miss 1 of 20" ] && ok "roll: passes on batches 4 and 5 ran; pass b re-ran only the batch-6 newcomer (19 of 20 reused)" || bad "roll: passes/misses ($M)"

WEAK6="1050:41.0 1080:40.9 1090:40.5 1100:41.5 1246:40.6 1200:45.5"    # as WEAK, batch 6's special is outside the top 20
run ext6 "$ROOT/block.new" "GPS1_PROFILE='$WEAK6'" PS_ADAPT=1
run roll6 "$ROOT/block.new" "GPS1_PROFILE='$WEAK6'" PS_ADAPT=1 PS_MEMO_ROLL=1
M0=$(rmiss ext6); M1=$(rmiss roll6); T0=$(retime ext6); T1=$(retime roll6)
echo "  roll6: pass b without roll: $M0, re-timed ${T0}s; with roll: $M1, re-timed ${T1}s"
cmp -s "$ROOT/roll6/poly.cado" "$ROOT/ext6/poly.cado" && [ "$M1" = "ropt miss 0 of 20" ] \
  && ok "roll6: poly.cado identical with/without roll; pass b re-ran no ropt ($M0 -> $M1)" || bad "roll6: $M0 / $M1"
case "$T0 $T1" in *FAIL*|*DIFF*) bad "roll6: re-timed pass b $T0 / $T1" ;;
  *) awk -v a="$T0" -v b="$T1" 'BEGIN{exit !(b + 0 < a + 0)}' && ok "roll6: pass b wall ${T0}s -> ${T1}s (re-timed, same poly.cado)" || bad "roll6: pass b not faster ($T0 -> $T1)" ;; esac

run rkill "$ROOT/block.new" "GPS1_PROFILE='$WEAK'" PS_ADAPT=1 PS_MEMO_ROLL=1 PS_MEMO_ROLL_WAIT=2 FAKE_ROPT_HANG_R=60; L=$(adline rkill)
echo "  rkill: $(grep -a 'PS_MEMO_ROLL' "$ROOT/rkill/log" | tr '\n' ' ')"; echo "  rkill: pass b $(rmiss rkill)"
sleep 1; _left=$(pgrep -f "$STUB/polyselect_ropt" | wc -l | tr -d ' ')
grep -aq 'PS_MEMO_ROLL: batch-4 pass killed' "$ROOT/rkill/log" && [ "$_left" = 0 ] && cmp -s "$ROOT/rkill/poly.cado" "$ROOT/ext/poly.cado" \
  && ok "rkill: hung rolling pass killed at the stop (no ropt left running), poly.cado == ext" || bad "rkill: left=$_left $L"
pbw(){ sed -n 's/.*\[ps_overlap b \([0-9]*\)s\] ropt.*/\1/p' "$ROOT/$1/w/ps_overlap.b.log" | tail -1; }
echo "pass-b walls in the runs (s, 1-s resolution): ext $(pbw ext) roll $(pbw roll) ext6 $(pbw ext6) roll6 $(pbw roll6)"

echo "walls (s, BATCH_S=$BATCH_S): $(for n in ref0_STRONG stop ref0_WEAK ext worse cap afail roll ext6 roll6 rkill; do printf '%s=%s ' $n "$(awk '/^WALL/{print $2}' "$ROOT/$n/log")"; done)"
echo "harness: $PASS passed, $FAIL failed  (work: $ROOT)"; [ "$FAIL" = 0 ]
