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

# las_cosieve_harness.sh (local, no GPU/CADO) -- known-answer test of the CPU co-sieve (LAS=1) beside the glas SESSION
# siever: the REAL factor_c169.sh sieve loop (band sizing, ses_band, las_poll/las_finish/las_take ride landing, the
# relation gates, las_band_guard, las_end, the two-phase stop) with mock binaries: glas-gpu and las are
# tests/mock_siever.py (exact synthetic relations for a tiny polynomial, the real las_ctl.py gates), makefb / freerel /
# cado_incdup.sh / cado_filter160.sh / merge / ds_cado.sh are stubs. Needs Linux (memfd_create, bash >= 4, GNU coreutils):
# anywhere else it re-runs itself inside `docker run python:3.12-slim` with pipeline_c169/ mounted read-only.
# Scenarios (SCENARIOS="..." selects; all must PASS):
#   retire   LAS_QMIN just above the first GPU bands: las lands >= 1 chunk riding on a GPU slot, then the GPU reaches its
#            band -> las RETIRED, the running chunk discarded, the GPU leaps over [LAS_QMIN, LAS_QDONE); FACTORED
#   ride     LAS_QMIN never reached: las lands chunk after chunk until MATRIX; LAS_NCPU=10 -> max confine budget 7
#   gatefail chunk 1 lists a wrong prime: the exact gate rejects it, las DISABLED, no las relation lands; FACTORED
#   crash    las exits 1 on chunk 2: chunk 1's relations landed, nothing else, las DISABLED; FACTORED
#   torn     chunk 1 published without a final newline: rejected, DISABLED; FACTORED
#   makefb   LAS_FB_REUSE=0: las builds its own factor base (mock makefb), then sieves; FACTORED
#   profile  CONTAINER=1 RSA560_CPU_LAS=1: the knob resolves LAS=1 A 29 q>=20M -t 19 budget 24 and glas -t 4; FACTORED
# Every landed slot (DEBUG_CHUNK_COPY) is checked line for line: GPU band lines first, then exactly the las chunks that
# rode along (sidecar copies), no '#' lines; GPU and las q-ranges are disjoint; the [las] summary counts match.
# usage: bash pipeline_c169/tests/las_cosieve_harness.sh            (~3-5 min; KAT_KEEP=1 keeps the work dir)
set -u
D=$(cd "$(dirname "$0")" && pwd); HERE=$(cd "$D/.." && pwd)
if [ "${1:-}" != --native ]; then
  if [ "$(uname -s)" != Linux ] || ! python3 -c 'import os; os.memfd_create' 2>/dev/null; then
    IMG=${KAT_IMAGE:-python:3.12-slim}; echo "[kat] not Linux/memfd: running inside docker $IMG"
    exec docker run --rm -v "$HERE:/src/pipeline_c169:ro" -e SCENARIOS="${SCENARIOS:-}" -e KAT_KEEP="${KAT_KEEP:-0}" \
         "$IMG" bash /src/pipeline_c169/tests/las_cosieve_harness.sh --native
  fi
fi
ROOT=$(mktemp -d /tmp/lascosieve.XXXXXX); S=$ROOT/stub; mkdir -p "$S/cf" "$S/msv" "$S/glas" "$S/bin"
PASS=0; FAIL=0; ok(){ echo "PASS $*"; PASS=$((PASS+1)); }; bad(){ echo "FAIL $*"; FAIL=$((FAIL+1)); }
# ---- the key: N = p*q (>= 30 digits for the driver's preflight) and a tiny polynomial the mocks can factor norms for
read -r P Q N < <(python3 -I - <<'PY'
def ip(n):
    if n < 2: return False
    for p in (2,3,5,7,11,13,17,19,23,29,31,37):
        if n % p == 0: return n == p
    d, s = n-1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in (2,3,5,7,11,13,17,19,23,29,31,37):
        x = pow(a, d, n)
        if x in (1, n-1): continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1: break
        else: return False
    return True
p = 10**15 + 1
while not ip(p): p += 2
q = p + 2
while not ip(q): q += 2
print(p, q, p*q)
PY
)
printf 'n: %s\nY0: -5\nY1: 3\nc0: 11\nc1: -7\nc2: 5\nc3: 1\nc4: -3\nc5: 2\nskew: 1.0\n' "$N" > "$ROOT/poly.cado"
# ---- stubs ---------------------------------------------------------------------------------------------------------
printf '#!/bin/bash\nexec python3 -I "%s/mock_siever.py" glas "$@"\n' "$D" > "$S/glas/glas-gpu"
printf '#!/bin/bash\nexec python3 -I "%s/mock_siever.py" las "$@"\n' "$D" > "$S/cf/las"
for b in polyselect polyselect_ropt dup1 dup2 purge replay cado2msieve2; do printf '#!/bin/sh\nexit 0\n' > "$S/cf/$b"; done
for b in characters sqrt; do printf '#!/bin/sh\nexit 0\n' > "$S/msv/$b"; done
: > "$S/msv/cado_mat.py"; : > "$S/msv/gpu_la.py"; printf '#!/bin/sh\nexit 0\n' > "$S/bin/nvidia-smi"
cat > "$S/cf/makefb" <<'EOF'
#!/bin/bash
out=""; while [ $# -gt 0 ]; do case "$1" in -out) out=$2; shift;; esac; shift; done
sleep 1; { echo "# mock fb1"; seq 1 2000; } > "$out"
EOF
cat > "$S/cf/freerel" <<'EOF'
#!/bin/bash
out=""; rn=""; while [ $# -gt 0 ]; do case "$1" in -out) out=$2; shift;; -renumber) rn=$2; shift;; esac; shift; done
sleep 2; python3 -I -c 'import gzip,sys; gzip.open(sys.argv[1],"wt").write("2,0:0\n3,0:1\n"); open(sys.argv[2],"wb").write(b"\0"*64)' "$out" "$rn"
echo "nprimes=2"
EOF
cat > "$S/cf/cado_incdup.sh" <<'EOF'
#!/bin/bash
# mock incdup: commits the chunk(s) after MOCK_INC_SECS; records the step's raw count for the mock probe (FREE_RAW=1 truncates
# the raw memfds after a commit, so an incremental probe cannot re-count them -- exactly as the real one only sees dd parts)
CACHE=$1; MEM=$2; POLY=$3; K=$4; shift 4
n=0; for f in "$@"; do c=$(grep -avc '^#' "$f"); n=$((n+${c:-0})); done
sleep "${MOCK_INC_SECS:-1}"; echo "$n" > "$CACHE/raw_$K"; echo "committed $*" > "$CACHE/chunk_$K.done"
echo "[incdup] COMMIT k=$K raw=$n rawcum=? uniqcum=? t=${MOCK_INC_SECS:-1}s"
EOF
cat > "$S/cf/cado_filter160.sh" <<'EOF'
#!/bin/bash
# mock probe: SHORT below MOCK_MATRIX_AT raw relations, else a purge-only MATRIX (writes the purged file the stop merges)
W=$1; MEM=$2; POLY=$3; N=$4; OUT=$5; shift 5
raw=0
if [ -n "${INC_CACHE:-}" ]; then for k in $(seq 1 "${INC_K:-0}"); do c=$(cat "$INC_CACHE/raw_$k" 2>/dev/null); raw=$((raw+${c:-0})); done
else for f in "$@"; do c=$(grep -avc '^#' "$f" 2>/dev/null); raw=$((raw+${c:-0})); done; fi
sleep "${MOCK_PROBE_SECS:-1}"
if [ "$raw" -ge "${MOCK_MATRIX_AT:-999999999}" ]; then
  seq 1 100 > "$W/${PURGED_NAME:-purged}"
  echo "[filter160] VERDICT MATRIX raw=$raw unique=$raw sing_excess=5 sing_ncols=100 final='nrows=100 ncols=95 excess=5' t=1s purge-only"; exit 0
fi
echo "[filter160] VERDICT SHORT raw=$raw unique=$raw sing_excess=-100 sing_ncols=100 final='nrows=10 ncols=100 excess=-100' t=1s"; exit 2
EOF
cat > "$S/cf/merge" <<'EOF'
#!/bin/bash
out=""; while [ $# -gt 0 ]; do case "$1" in -out) out=$2; shift;; esac; shift; done
seq 1 50 > "$out"; echo "Final matrix has N=95 nc=95 (5) W=1900"
EOF
cat > "$S/msv/ds_cado.sh" <<'EOF'
#!/bin/bash
[ "${1:-}" = --la-warm ] && { echo "[la-warm] mock"; exit 0; }
DS=$4; mkdir -p "$DS"; echo "[ds-cado] mock downstream: FACTOR on dep 0"
printf 'factor: %s\nfactor: %s\n' "$MOCK_P" "$MOCK_Q" > "$DS/msieve.log"; exit 0
EOF
chmod +x "$S"/glas/* "$S"/cf/* "$S"/msv/* "$S"/bin/*
# ---- one driver run ---------------------------------------------------------------------------------------------------
# small everything: 8-CPU budget, ~300-500-relation bands of 2 s, probes from 1,500 raw, MATRIX at 6,000 raw (~12 bands)
run_driver(){   # $1 = scenario; the rest = extra/override env (VAR=value)
  local name=$1; shift; local W=$ROOT/$name; mkdir -p "$W/mock" "$W/slots"
  env -i PATH="$S/bin:/usr/local/bin:/usr/bin:/bin" HOME="$ROOT" TMPDIR="$ROOT" \
      CF_DIR="$S/cf" MSV_DIR="$S/msv" GLAS="$S/glas/glas-gpu" SIEVER=glas GLAS_SESSION=1 POLY_IN="$ROOT/poly.cado" \
      NCPU=8 SIEVE_T=2 SIEVE_HOST_T=2 PROBE_NT=2 PROBE_LATE_NT=2 INC_TPG=1 INC_PAR=1 INC_PAR_PROBE=1 INC_GROUPS=1 \
      LPB0=50 LPB1=50 MFB0=100 MFB1=100 RLIM=1000 ALIM=1000 SIEVE_A=29 \
      QMIN=2200000 CHUNK_W0=20000 CHUNK_RELS=500 CHUNK_RELS_LATE=300 FIRST_PROBE=1500 PROBE_STEP=400 \
      STOP_RULE=excess STOP_MIN_EXCESS=1 PREDICT_PROBE=0 LATE_SECS=0 TD=10 \
      INCDUP=1 FREE_RAW=0 MEMFD_PURGED=0 RES_LOG=0 LA_WARM=0 DS_MODE=cado LA=gpu MAXCHUNKS=40 PP_SLOTS=4 \
      LAS=1 LAS_A=29 LAS_QMIN=2300000 LAS_W0=20000 LAS_WMIN=2000 LAS_CHUNK_SECS=3 LAS_NICE=0 LAS_MEM_MB=1 LAS_STALL=600 LAS_NCPU=8 \
      DEBUG_CHUNK_COPY="$W/slots" MOCK_DIR="$W/mock" MOCK_QSTEP=1000 MOCK_K=12 MOCK_BAND_SECS=2 MOCK_LAS_SECS=2 \
      MOCK_MATRIX_AT=6000 MOCK_P="$P" MOCK_Q="$Q" "$@" \
      bash "$HERE/factor_c169.sh" "$N" "$W/work" > "$W/driver.log" 2>&1; echo $? > "$W/rc"
}
# ---- the checks (python): driver log + sidecars + slot copies -> PASS/FAIL lines ------------------------------------------
check(){   # $1 = scenario dir, $2 = expectation keyword list
  python3 -I - "$1" "$2" <<'PY'
import os, re, sys
W, exp = sys.argv[1], sys.argv[2].split(); log = open(os.path.join(W, 'driver.log'), errors='replace').read()
rc = open(os.path.join(W, 'rc')).read().strip(); res = []
def chk(c, m): res.append(('PASS' if c else 'FAIL') + ' ' + m)
chk(rc == '0' and 'FACTORED: p*q==N True' in log, 'run FACTORED rc=%s' % rc)
chk('relation gate PASS' in log, 'GPU relation gate PASS')
# landed slots: "chunk I q[q0,q1) +N in Ts (R rel/s)[ + las NL] -> slot S"
slots = []
for m in re.finditer(r'chunk (\d+) q\[(\d+),(\d+)\) \+(\d+) in \d+s \(\d+ rel/s\)(?: \+ las (\d+))? -> slot (\d+)', log):
    slots.append(dict(i=int(m[1]), q0=int(m[2]), q1=int(m[3]), n=int(m[4]), nl=int(m[5] or 0), s=int(m[6])))
chk(len(slots) >= 3, '%d GPU chunks landed' % len(slots))
lasc = [dict(j=int(m[1]), q0=int(m[2]), q1=int(m[3]), n=int(m[4]))
        for m in re.finditer(r'\[las\] chunk (\d+) q\[(\d+),(\d+)\) \+(\d+) in', log)]
def rels(mode, ident):
    try: return open(os.path.join(W, 'mock', mode + '_rels', '%s.rels' % ident)).read().split('\n')[:-1]
    except OSError: return None
# slot contents: GPU band lines, then the las chunks that rode along (in finish order), nothing else
ride = [c for c in lasc if c['n'] > 0]; used = []; good = True; nland = 0; nlas = 0
for s in slots:
    f = os.path.join(W, 'slots', 'c%03d' % s['s'])
    try: lines = open(f).read().split('\n')[:-1]
    except OSError: good = False; res.append('FAIL slot copy %s missing' % f); continue
    g = rels('glas', str(s['i']))
    if g is None or lines[:s['n']] != g or len(lines) != s['n'] + s['nl'] or any(L.startswith('#') for L in lines):
        good = False; res.append('FAIL slot %d: %d lines vs GPU %s + las %d' % (s['s'], len(lines), None if g is None else len(g), s['nl']))
    rest = lines[s['n']:]; want = []; take = 0
    while take < s['nl'] and ride:
        c = ride.pop(0); used.append(c); r = rels('las', 'c%d' % c['j']); want += (r or []); take += c['n']
    if take != s['nl'] or rest != want:
        good = False; res.append('FAIL slot %d: las tail %d lines != expected %d (chunks %s)' % (s['s'], len(rest), len(want), [c['j'] for c in used]))
    if s['nl']: nland += 1; nlas += s['nl']
chk(good, 'slot contents = GPU band + riding las chunks, line for line (%d slots, %d with las)' % (len(slots), nland))
# q-disjointness: no landed GPU range overlaps a landed las range
ov = [(s['i'], c['j']) for s in slots for c in used if s['q0'] < c['q1'] and c['q0'] < s['q1']]
chk(not ov, 'GPU and las q-ranges disjoint (%d GPU bands, %d las chunks landed)' % (len(slots), len(used)))
m = re.search(r'\[las\] summary: (\d+) relations landed \((\d+) landings', log)
chk(m and int(m[1]) == nlas and int(m[2]) == nland, 'las summary %s == landed %d rels / %d landings' % (m and m.groups(), nlas, nland))
if 'lasland' in exp: chk(nland >= 1 and '[las] relation gate PASS' in log, 'las landed (%d relations, gate PASS)' % nlas)
if 'nolas' in exp: chk(nland == 0 and nlas == 0, 'no las relation landed')
if 'retire' in exp:
    mm = re.search(r'GPU leaps over the las-covered q-band \[(\d+),(\d+)\)', log)
    nxt = [s for s in slots if mm and s['q0'] >= int(mm[2])]
    chk('[las] RETIRED' in log and mm and nxt and not any(int(mm[1]) <= s['q0'] < int(mm[2]) for s in slots),
        'las RETIRED and the GPU leapt over %s' % (mm and mm.group(0)))
else: chk('RETIRED' not in log, 'las never retired')
if 'disabled' in exp: chk('CPU co-sieve DISABLED' in log, 'las fail-soft: DISABLED, run continued')
else: chk('DISABLED' not in log, 'las never disabled')
if 'crash1' in exp: chk([c['j'] for c in used] == [1], 'only las chunk 1 landed before the crash')
if 'gatefail' in exp: chk('relation gate FAILED on chunk 1' in log, 'gate rejected the bad prime')
if 'torn' in exp: chk('torn line' in log, 'torn output rejected')
if 'fbreuse' in exp: chk("[las] factor base: the sieve's fb1 reused" in log, 'fb1 reused (no second makefb)')
if 'makefb' in exp: chk('[las] makefb side 1' in log and '[las] factor base ready' in log, 'own makefb ran')
try: thr = open(os.path.join(W, 'work', 'las_throttle.log')).read()
except OSError: thr = ''
conf = [int(x) for x in re.findall(r'confine to (\d+) CPUs', thr)]
for e in exp:
    # the budget is LAS_NCPU-host-reserve minus whatever probe/incdup/freerel are live, so it varies; the test is only that
    # it exceeds the NCPU=8 default's ceiling (8-2-1=5), proving LAS_NCPU raised it -- accept [lo, hi] where hi=LAS_NCPU-3
    if e.startswith('maxconf<='): hi = int(e[9:]); chk(conf and hi - 1 <= max(conf) <= hi, 'max las budget %s in [%d,%d] (confines %s)' % (conf and max(conf), hi - 1, hi, sorted(set(conf))))
if 'profile' in exp:
    on = re.search(r'\[las\] CPU co-sieve ON: (.*)', log); on = on and on[1]
    chk(on and 'q from 20000000, A 29, -t 19' in on and 'budget 24' in on and 'siever -t 4' in on, 'profile line: %s' % on)
    try: ga = open(os.path.join(W, 'mock', 'glas_args.log')).read()
    except OSError: ga = ''
    chk(' -t 4 ' in ga, 'glas launched with -t 4 (RSA560_CPU_LAS_GT)')
print('\n'.join(res))
PY
}
report(){ while read -r L; do case "$L" in PASS*) ok "$1: ${L#PASS }";; FAIL*) bad "$1: ${L#FAIL }";; esac; done < <(check "$ROOT/$1" "$2"); }
SC=${SCENARIOS:-retire ride gatefail crash torn makefb profile}
for sc in $SC; do
  echo "== scenario $sc"; t0=$(date +%s)
  case "$sc" in
    retire)   run_driver retire; report retire "lasland retire fbreuse" ;;
    ride)     run_driver ride LAS_QMIN=9000000 LAS_NCPU=10; report ride "lasland fbreuse maxconf<=7" ;;
    gatefail) run_driver gatefail LAS_QMIN=9000000 MOCK_LAS_FAIL=badprime:1; report gatefail "nolas disabled gatefail" ;;
    crash)    run_driver crash LAS_QMIN=9000000 MOCK_LAS_FAIL=rc1:2; report crash "lasland disabled crash1" ;;
    torn)     run_driver torn LAS_QMIN=9000000 MOCK_LAS_FAIL=torn:1; report torn "nolas disabled torn" ;;
    makefb)   run_driver makefb LAS_QMIN=9000000 LAS_FB_REUSE=0; report makefb "lasland makefb" ;;
    profile)  run_driver profile CONTAINER=1 RSA560_CPU_LAS=1 LAS= LAS_A= LAS_QMIN= LAS_NCPU= SIEVE_T= SIEVE_HOST_T= RES_LOG=0 \
                  MAXCHUNKS=40 MOCK_MATRIX_AT=4000; report profile "lasland profile fbreuse" ;;
    *) bad "unknown scenario $sc" ;;
  esac
  grep -a 'WARNING\|FATAL' "$ROOT/$sc/driver.log" 2>/dev/null | grep -v 'co-sieve DISABLED\|injected' | head -5 | sed 's/^/   driver: /'
  echo "   ($(( $(date +%s)-t0 )) s; $(grep -ac ' -> slot ' "$ROOT/$sc/driver.log" 2>/dev/null) GPU chunks, $(grep -ac '\[las\] chunk' "$ROOT/$sc/driver.log" 2>/dev/null) las chunks)"
done
echo "== $PASS passed, $FAIL failed (work: $ROOT)"
if [ "${KAT_SHOW:-0}" = 1 ]; then   # evidence for a write-up: the las lines of each driver log + the throttle log
  for sc in $SC; do echo "==== $sc: driver log (las / guard / threads lines)"
    grep -a 'co-sieve ON\|\[las\]\|leaps over\|clipped to\|threads:\|### c169' "$ROOT/$sc/driver.log" 2>/dev/null | head -30
    echo "---- $sc: las_throttle.log (first 12)"; head -12 "$ROOT/$sc/work/las_throttle.log" 2>/dev/null; done
fi
[ "${KAT_KEEP:-0}" = 1 ] || rm -rf "$ROOT"
[ "$FAIL" = 0 ]
