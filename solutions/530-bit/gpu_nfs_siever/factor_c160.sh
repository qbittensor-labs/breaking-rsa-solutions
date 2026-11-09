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

# =============================================================================================
# factor_c160.sh -- GNFS for ~158-162 digit N (RSA-530) on 1x RTX PRO 6000 + 24 CPU, 14,400 s.
# =============================================================================================
# usage: factor_c160.sh <N> <workdir>        (prints "p=..." / "q=..." on success)
#
# Stage map (all parameters measured on the validator-class GPU, 2026-09-26; see Pipeline160.md):
#  [1/4] polyselect  CADO size-opt search (CPU, time-boxed) -> ropt top-K -> best MurphyE.
#                    NEVER pick by exp_E alone (4x yield loss measured at c160).
#  [2/4] factor base fbgen_gpu (GPU, ~1 s).
#  [3/4] sieve       cuda-sieve bench --pipeline --cofactor --fill-concurrent, lpb 31/31,
#                    mfb 61/62, logI 15, J 16384, algebraic special-q from QMIN, in explicit
#                    q-chunks [q, qb-1] (q advances by construction). Each chunk -> its own memfd.
#                    CADO filter probes (cado_filter160.sh) run concurrently on an immutable chunk
#                    list; the first MATRIX verdict stops the sieve.
#                    INCREMENTAL DEDUP (INCDUP=1, default): as each chunk lands, cado_incdup.sh
#                    (nice 15, watchdog INC_TIMEOUT) dup1+dup2s it against the committed parts of
#                    earlier chunks into immutable memfds $MEM/dd/<kkk>/s<ii>.gz (+ key shadows
#                    $MEM/kk/<kkk>/s<ii>); a probe then only runs purge(+merge) over chunks 1..K
#                    committed. Any incdup/INC-probe failure (or a watchdog timeout) permanently
#                    falls back to the one-shot dup1/dup2 probe. INCDUP=0 = the one-shot probes only.
#                    PROBE CADENCE: a prefix is never probed twice; one-shot probes launch only when
#                    a chunk LANDS, incremental probes when incdup COMMITS a chunk.
#                    THREADS: sieve host (SIEVE_HOST_T) + incdup (INC_TPG per dup2 group) + probe
#                    (NT + its gzip reader) <= NCPU while incdup is live (dup1's burst: nice 15).
#  [4/4] downstream  DS_MODE=cado (default): probes stop after merge (MERGE_ONLY=1) and ds_cado.sh
#                    runs CADO replay -> gpu_la.py GPU block Lanczos ON THE CADO MATRIX -> CADO
#                    characters -> CADO sqrt. No replay --for_msieve, cado2msieve2 or msieve -nc2.
#                    DS_MODE=msieve: the previous path, msv_launcher + msvrun_gpu.sh: msieve -nc2
#                    (cado_filter=1) -> gpu_la.py GPU block Lanczos -> CADO sqrt.
# Storage: /tmp is a 10 GiB tmpfs. Raw relations (~21 GB), dup1/dup2 slices (~11/~8 GB), the
# incremental parts (~8 + ~2.5 GB), the merged CADO matrix (~5 GB) and the purge-ordered relations
# all live in memfds (memhold.py); /tmp holds one staging chunk (<=1.3 GB), renumber.gz (~1.5 GB),
# purged.gz, merge history and the cycle file.
set -u
N="$1"; WORK="$2"
HERE=$(cd "$(dirname "$0")" && pwd); APP=$(cd "$HERE/.." && pwd)
CS="${CS_DIR:-$APP/cuda-sieve}"; CF="${CF_DIR:-$APP/cado-filter}"; MSV="${MSV_DIR:-$APP/msieve-gpu}"
T0=$(date +%s); el(){ echo "[$(( $(date +%s)-T0 ))s]"; }
say(){ echo "$(el) $*"; }
die(){ echo "$(el) FATAL: $*"; echo "NO FACTOR (fatal: $*)"; exit 1; }

# ---- parameters (env-overridable) -----------------------------------------------------------
NCPU=${NCPU:-$(python3 -c "
try:
    q,p=open('/sys/fs/cgroup/cpu.max').read().split()
    print(max(1,round(int(q)/int(p))) if q!='max' else __import__('os').cpu_count())
except Exception: print(__import__('os').cpu_count())")}
LPB=${LPB:-31}; MFB0=${MFB0:-61}; MFB1=${MFB1:-62}
RLIM=${RLIM:-100000000}; ALIM=${ALIM:-150000000}
LOGI=${LOGI:-15}; JJ=${JJ:-16384}; QMIN=${QMIN:-2200000}
CHUNK_RELS=${CHUNK_RELS:-8000000}; MAXCHUNKS=${MAXCHUNKS:-60}
# CHUNK_RELS_LATE: chunk size once `have` is within one chunk of FIRST_PROBE. Probes see whole committed
# chunks, so the MATRIX probe overshoots the crossover by ~half a chunk on average: 4M halves that
# (~140 s of sieve at c160). incdup/probe per-chunk overheads stay off the critical path.
CHUNK_RELS_LATE=${CHUNK_RELS_LATE:-4000000}
# ALLOW0/ALLOW1: bench survivor allowance in bits (rational/algebraic). bench's derived default is mfb + ~1.5;
# measured c160 q=5M: mfb - 0.5 = -1.9% ms/q for -0.34% relations. ALLOW_OFF=1 = bench's own default.
ALLOW0=${ALLOW0:-$(awk -v m="$MFB0" 'BEGIN{printf "%.1f", m-0.5}')}; ALLOW1=${ALLOW1:-$(awk -v m="$MFB1" 'BEGIN{printf "%.1f", m-0.5}')}
ALLOWARGS=(--allowance0 "$ALLOW0" --allowance "$ALLOW1"); [ "${ALLOW_OFF:-0}" = 1 ] && ALLOWARGS=()
# BKTHRESH: bucket-sieve primes >= this (bench default 2^logI = 32768). 65536 cuts the fill 19.8 -> 15.4 ms/q but
# doubles the per-q TD small-prime tables; with the patched bench building those while both sides sieve
# (vendor/cuda-sieve-fast-emit.patch, part 3) the net is 25.1 -> 24.1 ms/q (-4%), equivalent relation set.
BKTHRESH=${BKTHRESH:-65536}; ALLOWARGS+=(--bkthresh "$BKTHRESH")
# PROBE_STEP must be BELOW CHUNK_RELS_LATE: late chunks land at ~3.94M (bench stops at the q-width
# estimate), and a 4M step made every other one miss the threshold by ~50k (timed run 2: no probe on
# 164.32M because NEXT_PROBE was 164.37M) -- doubling the MATRIX detection granularity.
FIRST_PROBE=${FIRST_PROBE:-160000000}; PROBE_STEP=${PROBE_STEP:-3500000}
PROBE_NT=${PROBE_NT:-$(( NCPU > 6 ? NCPU - 4 : 2 ))}; DUP2_PAR=${DUP2_PAR:-6}; TD=${TD:-100}
PS_SECS=${PS_SECS:-480}; PS_MAX=${PS_MAX:-900}; PS_MIN_CANDS=${PS_MIN_CANDS:-200}
ROPT_K=${ROPT_K:-40}; ROPT_EFF=${ROPT_EFF:-10}; ROPT_MAX=${ROPT_MAX:-420}
# DS_MODE=cado (default): probes stop after merge (MERGE_ONLY=1); downstream = ds_cado.sh
# (CADO replay -> gpu_la on the CADO matrix -> CADO characters -> CADO sqrt). No cado2msieve2, no
# msieve -nc2. DS_MODE=msieve restores the previous path.
DS_MODE=${DS_MODE:-cado}
# INCDUP=1 (default): incremental dedup (cado_incdup.sh) + incremental probes. INCDUP=0: one-shot probes
# exactly as before. INC_PAR: concurrent dup2 groups while no probe runs; INC_PAR_PROBE: for a chunk a
# probe is waiting on (it crossed NEXT_PROBE) -- that incdup IS on the critical path. Both are capped
# by the thread budget: SIEVE_HOST_T (bench host threads, measured load ~1.0) + INC_TPG threads per
# dup2 group (3 hard-coded root threads + gzip/reader) + probe NT <= NCPU. An incremental probe is
# given NT = NCPU - SIEVE_HOST_T - INC_TPG - 1 (one dup2 group stays free for the next chunk's incdup;
# 1 for purge's gzip input reader). incdup's short dup1 burst is covered by nice 15 (see probe_launch).
# INC_TIMEOUT: watchdog per incdup step (a hang falls back to one-shot probes instead of stalling).
INCDUP=${INCDUP:-1}; INC_PAR=${INC_PAR:-2}; INC_NICE=${INC_NICE:-15}
INC_PAR_PROBE=${INC_PAR_PROBE:-4}; INC_GROUPS=${INC_GROUPS:-4}; INC_TIMEOUT=${INC_TIMEOUT:-1200}
# COFAC_CPU=1 (default): the GPU sieve does NOT cofactorise (bench without --cofactor). It writes its 2LP
# candidates (--candidates) and cofac_cpu (CADO facul ECM/P-1/P+1, rho fallback) splits them on COFAC_T
# host threads WHILE the band runs (follow mode: it tails the candidates .part), so a chunk lands seconds
# after its band ends. Measured c160 q=5M: 26.2 vs 28.2 ms/q (-7%), relation SET identical to --cofactor,
# CPU ~0.7 core. COFAC_CPU=0 = the GPU cofactorisation queue (previous production path).
COFAC_CPU=${COFAC_CPU:-1}; COFAC_T=${COFAC_T:-2}
SIEVE_HOST_T=${SIEVE_HOST_T:-2}; INC_TPG=${INC_TPG:-5}
FR_T=$(( NCPU > 8 ? 8 : NCPU ))     # background freerel threads (counted in the las budget while it runs)
# CPU CO-SIEVE (LAS=1, default): CADO las sieves algebraic special-q from LAS_QMIN UPWARD -- far above the
# GPU's band (2.2M..~8.5M), so few of its relations are GPU re-finds -- on the cores the pipeline leaves
# idle, in bounded q-chunks (las writes <out>.tmp.<pid> and renames only on completion: a killed las
# loses its chunk). Measured at c160 (lim 100M/150M lpb 31 mfb 61/62 A 28): 626 rel/s at -t 14 = +4.3%
# of the GPU, NO measurable GPU slowdown; RSS ~2.8 GB (-t 14) .. ~3.7 GB (-t 22); makefb 22 s.
#   THREADS: las is the LOWEST priority consumer. Its -t is fixed at launch (LAS_T), but every 3 s poll
#     confines all its threads to las_budget() = NCPU - SIEVE_HOST_T - LAS_RESERVE - (probe NT + 1)
#     - (incdup groups * INC_TPG) - (freerel) CPUs (sched_setaffinity via las_ctl.py; measured: -t 14
#     on 6 CPUs = 95% of -t 6), and SIGSTOPs it below LAS_MIN_CPUS. nice LAS_NICE on top.
#   LANDING (LAS_LAND): ride (default) = finished las outputs are appended to the NEXT GPU chunk's slot,
#     so the chunk pool, its numbering (HAVE_AT/LANDED/INC_DONE) and the number of incdup steps are
#     exactly the GPU-only ones -- an incdup step costs ~O(cumulative relations) (renumber-table loads +
#     every older key shadow), so a separate ~0.5M-relation las slot would cost nearly a full GPU chunk's
#     incdup time. slot = each finished las output becomes its own chunk slot as soon as it is validated.
#   FAIL-SOFT: any las failure (makefb, las rc, relation gate, stall/wall watchdog, memory headroom,
#     landing) disables it for the rest of the run with a loud WARNING -- the run continues GPU-only.
#   Stopped (and its partial chunk discarded) at MATRIX; retired if the GPU's q-range reaches LAS_QMIN
#     (the GPU then leaps over the las-covered band instead of re-sieving it).
LAS=${LAS:-1}; LAS_QMIN=${LAS_QMIN:-12000000}; LAS_A=${LAS_A:-28}; LAS_LAND=${LAS_LAND:-ride}
LAS_LIM0=${LAS_LIM0:-$RLIM}; LAS_LIM1=${LAS_LIM1:-$ALIM}; LAS_LPB=${LAS_LPB:-$LPB}; LAS_MFB0=${LAS_MFB0:-$MFB0}; LAS_MFB1=${LAS_MFB1:-$MFB1}
LAS_NICE=${LAS_NICE:-10}; LAS_RESERVE=${LAS_RESERVE:-1}; LAS_MIN_CPUS=${LAS_MIN_CPUS:-2}
LAS_T=${LAS_T:-$(( NCPU - SIEVE_HOST_T - LAS_RESERVE - ( COFAC_CPU == 1 ? COFAC_T : 0 ) ))}
# q-chunking: first chunk LAS_W0 wide (~6-7 min at c160, -t 14: 51 q/s measured at q=12M), then sized
# from the previous chunk's active q-rate for LAS_CHUNK_SECS (at most doubling per chunk).
LAS_W0=${LAS_W0:-20000}; LAS_WMIN=${LAS_WMIN:-2000}; LAS_CHUNK_SECS=${LAS_CHUNK_SECS:-900}
LAS_STALL=${LAS_STALL:-600}; LAS_CHUNK_MAX=${LAS_CHUNK_MAX:-$(( 6 * LAS_CHUNK_SECS ))}
LAS_MEM_MB=${LAS_MEM_MB:-6144}; LAS_SLOT_KEEP=${LAS_SLOT_KEEP:-8}
LAS_BAND_GUARD=${LAS_BAND_GUARD:-1}   # 0 = TEST ONLY (lets a mock sieve overlap las's band to exercise dedup)
# PS_NQ 19531 (was CADO's 156250): polyselect only releases an ad's matches when the ad FINISHES; at 156250 an ad
# takes ~640 s so a 480-720 s search releases few. Measured key 3 (720 s x 16 thr): 5,336 vs 2,316 candidates, best
# ropt pick -4.3% modelled sieve time (it also contains the -6.2% poly a 1,440 s stock search finds).
PS_P=${PS_P:-1445000}; PS_ADMIN=${PS_ADMIN:-3360}; PS_ADMAX=${PS_ADMAX:-291000}; PS_INCR=${PS_INCR:-420}; PS_NQ=${PS_NQ:-19531}

# incdup runs in its OWN session (see inc_start), so a SIGKILL of this script's process group
# (breaking_rsa's _kill_tree on its timeout/exception path) does not reach it. It is launched with
# PR_SET_PDEATHSIG=SIGTERM (timeout then TERMs its whole group), and as a second line of defence a
# previous attempt's incdup group recorded in $WORK/inc.pgid is SIGKILLed here, BEFORE rm -rf, so
# none of its by-path writes ($MEM/..., $CACHE/chunk_K.done) can land in this run's identical paths.
# The group is only killed if one of its members is recognisably ours (timeout/cado_incdup/dup1/dup2).
if [ -s "$WORK/inc.pgid" ]; then
  python3 - "$(tr -dc 0-9 < "$WORK/inc.pgid")" <<'PY' || true
import os, sys, signal, time
g = int(sys.argv[1] or 0)
def members():
    m = []
    for d in os.listdir('/proc'):
        if not d.isdigit(): continue
        try:
            s = open('/proc/%s/stat' % d).read(); f = s[s.rfind(')') + 2:].split()
            if int(f[2]) == g: m.append(open('/proc/%s/cmdline' % d, 'rb').read().replace(b'\0', b' '))
        except Exception: pass
    return m
if g > 1:
    m = members()
    if any(k in c for c in m for k in (b'cado_incdup', b'/dup1', b'/dup2', b'timeout')):
        print("killing a previous attempt's incdup group %d (%d processes)" % (g, len(m)), flush=True)
        try: os.killpg(g, signal.SIGKILL)
        except OSError: pass
        for _ in range(50):
            if not members(): break
            time.sleep(0.1)
PY
fi
rm -rf "$WORK"; mkdir -p "$WORK" || die "cannot create $WORK"
MEM="$WORK/mem"; STOP="$WORK/stop.sieve"
echo "### c160 GNFS: $(echo -n "$N" | wc -c) digits, $NCPU CPUs, lpb $LPB/$LPB mfb $MFB0/$MFB1 lim $RLIM/$ALIM logI $LOGI J $JJ qmin $QMIN allowance ${ALLOWARGS[*]:-default} cofac $([ "$COFAC_CPU" = 1 ] && echo "cpu x$COFAC_T" || echo gpu)"

# ---- preflight: fail fast and BY NAME -------------------------------------------------------
case "$N" in ''|*[!0-9]*) die "N is not a decimal integer";; esac
[ "${#N}" -ge 30 ] || die "N has only ${#N} digits"
for b in "$CS/bench" "$CS/fbgen_gpu" "$CF/polyselect" "$CF/polyselect_ropt" "$CF/cado_filter160.sh" \
         "$CF/freerel" "$CF/dup1" "$CF/dup2" "$CF/purge" "$CF/merge" "$CF/replay" "$CF/cado2msieve2" \
         "$MSV/msv_launcher" "$MSV/msvrun_gpu.sh" "$MSV/msieve_la" "$HERE/memhold.py" "$HERE/polysel160.py"; do
  [ -e "$b" ] || die "missing $b"
done
case "$DS_MODE" in cado|msieve) ;; *) die "DS_MODE must be cado or msieve (got '$DS_MODE')";; esac
case "$INCDUP" in 0|1) ;; *) die "INCDUP must be 0 or 1 (got '$INCDUP')";; esac
case "$LAS" in 0|1) ;; *) die "LAS must be 0 or 1 (got '$LAS')";; esac
case "$LAS_LAND" in ride|slot) ;; *) die "LAS_LAND must be ride or slot (got '$LAS_LAND')";; esac
if [ "$LAS" = 1 ]; then   # fail-SOFT: a missing co-siever costs ~4% of the sieve, never the run
  for b in "$CF/las" "$CF/makefb"; do [ -x "$b" ] || { say "WARNING: [las] $b missing -- CPU co-sieve DISABLED, GPU-only"; LAS=0; }; done
  [ -r "$HERE/las_ctl.py" ] || { say "WARNING: [las] $HERE/las_ctl.py missing -- CPU co-sieve DISABLED, GPU-only"; LAS=0; }
  [ "$LAS_T" -ge 1 ] 2>/dev/null || { say "WARNING: [las] LAS_T=$LAS_T < 1 -- CPU co-sieve DISABLED"; LAS=0; }
fi
case "$COFAC_CPU" in 0) ;; 1) [ -x "$CS/cofac_cpu" ] || die "missing or non-executable $CS/cofac_cpu (COFAC_CPU=1)";; *) die "COFAC_CPU must be 0 or 1 (got '$COFAC_CPU')";; esac
if [ "$DS_MODE" = cado ]; then
  for b in "$MSV/characters" "$MSV/sqrt" "$CF/replay"; do [ -x "$b" ] || die "missing or non-executable $b"; done
  for b in "$MSV/ds_cado.sh" "$MSV/cado_mat.py" "$MSV/gpu_la.py"; do [ -r "$b" ] || die "missing $b"; done
fi
if [ "$INCDUP" = 1 ]; then
  [ -r "$CF/cado_incdup.sh" ] || die "missing $CF/cado_incdup.sh"
  command -v timeout >/dev/null || die "timeout (coreutils) missing -- the incdup watchdog needs it"
fi
# memhold keeps EVERY memfd open in one process: c%03d x MAXCHUNKS + d1/d2 x 32 + rels_out + c.sparse.bin
# (+ dd/kk x 32*MAXCHUNKS with INCDUP=1), plus a few of its own. The validator's nofile is 4096:4096,
# so MAXCHUNKS >= 124 with INCDUP=1 would die at startup with EMFILE -- say so BY NAME, up front.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
_nfd=$(( MAXCHUNKS + 34 + 16 )); [ "$INCDUP" = 1 ] && _nfd=$(( _nfd + 32 * MAXCHUNKS ))
_nofile=$(ulimit -n); [ "$_nofile" = unlimited ] || [ "$_nfd" -le "$_nofile" ] \
  || die "MAXCHUNKS=$MAXCHUNKS needs ~$_nfd memfds in memhold but nofile is $_nofile (lower MAXCHUNKS or set INCDUP=0)"
command -v gzip >/dev/null || die "gzip missing (CADO .gz I/O pipes through it)"
# SLICE_FMT (.gz DEFAULT | .zst | auto = .zst when zstd exists): format of the dup1/dup2 slices, incremental parts and
# the renumber table; purged.gz/freerel.gz stay gzip. Exported to the filter scripts. .zst was MEASURED NOT faster
# (key 1, 22 chunks, one-shot NT 16: purge 437 s vs 335-401 s gzip, pass 1 192 vs 148-182 s): purge's reading is
# bound by its single parsing consumer (~0.75-1M relations/s), not by decompression. So gzip, the tested path.
SLICE_FMT=${SLICE_FMT:-.gz}
if [ "$SLICE_FMT" = auto ]; then command -v zstd >/dev/null && SLICE_FMT=.zst || { SLICE_FMT=.gz; say "WARNING: zstd not found -- gzip slices"; }; fi
case "$SLICE_FMT" in .gz|.zst) ;; *) die "SLICE_FMT must be .gz or .zst";; esac; export SLICE_FMT; SL=$SLICE_FMT
say "slice format $SLICE_FMT"
python3 -c 'import os; os.memfd_create("t")' 2>/dev/null || die "memfd_create unavailable"
nvidia-smi -L >/dev/null 2>&1 || python3 -c 'import cupy; cupy.cuda.runtime.getDeviceCount()' >/dev/null 2>&1 || die "no GPU visible"
_tmpfree=$(df -Pm "$WORK" | awk 'NR==2{print $4}'); say "workdir free: ${_tmpfree} MB"

inc_kill(){ [ -n "${INCPID:-}" ] && { kill -TERM -- -"$INCPID" 2>/dev/null; wait "$INCPID" 2>/dev/null; INCPID=""; INC_RUN_PAR=0; }; return 0; }
# incdup launcher (python, then exec): its OWN session, so inc_kill and the watchdog (timeout -k) take
# down timeout + cado_incdup.sh + dup1/dup2/gzip as ONE group -- and PR_SET_PDEATHSIG=SIGTERM, so a
# SIGKILL of THIS script (breaking_rsa's killpg, which skips the EXIT trap and cannot reach another
# session) still TERMs timeout, which forwards it to its whole group (KILL after -k 10). Verified with
# a group SIGKILL of the caller: nothing of the incdup tree survives. argv[1] = the expected parent
# ($$): if we were already orphaned before prctl took effect, exit 125 (-> a loud incdup failure).
INC_PYL='import os,sys,signal
os.setsid()
try:
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0: raise OSError(ctypes.get_errno(), "prctl")
except Exception as e:
    sys.stderr.write("[incdup-launcher] WARNING: PR_SET_PDEATHSIG unavailable (%s)\n" % e)
if os.getppid() != int(sys.argv[1]): os._exit(125)
os.execvp(sys.argv[2], sys.argv[2:])'
cleanup(){ [ -n "${BENCHPID:-}" ] && kill "$BENCHPID" 2>/dev/null; [ -n "${PROBEPID:-}" ] && kill "$PROBEPID" 2>/dev/null; inc_kill
           for _p in ${LASPID:-} ${LAS_FBPID:-}; do kill -CONT "$_p" 2>/dev/null; kill -KILL "$_p" 2>/dev/null; done
           [ -n "${COFPID:-}" ] && kill "$COFPID" 2>/dev/null
           [ -n "${PSPID:-}" ] && kill "$PSPID" 2>/dev/null; [ -n "${FRPID:-}" ] && kill "$FRPID" 2>/dev/null; [ -n "${MHPID:-}" ] && kill "$MHPID" 2>/dev/null; }
trap cleanup EXIT

# ---- memfd pool -----------------------------------------------------------------------------
# c.sparse.bin (DS_MODE=cado): the merged CADO matrix replay writes for gpu_la (~5 GB at c160).
# dd/<kkk>/s<ii>.gz + kk/<kkk>/s<ii> (INCDUP=1): incremental per-(chunk,slice) dedup parts and their
# key shadows (cado_incdup.sh). 32*MAXCHUNKS more memfds (1920 at 60; nofile 4096, preflight-checked:
# INCDUP=1 caps MAXCHUNKS at ~122) -- sparse, free until written.
python3 "$HERE/memhold.py" "$MEM" "c%03d:1:$MAXCHUNKS" "d1/%d/rel.0000$SL:0:15" "d2/%d/rel.0000$SL:0:15" rels_out \
    $([ "$DS_MODE" = cado ] && echo c.sparse.bin) \
    $([ "$INCDUP" = 1 ] && for _c in $(seq 1 "$MAXCHUNKS"); do printf 'dd/%03d/s%%02d'"$SL"':0:15 kk/%03d/s%%02d:0:15 ' "$_c" "$_c"; done) \
    > "$WORK/memhold.out" 2>&1 & MHPID=$!
for _i in $(seq 1 50); do grep -q READY "$WORK/memhold.out" 2>/dev/null && break; sleep 0.1; done
grep -q READY "$WORK/memhold.out" || die "memhold did not start: $(cat "$WORK/memhold.out")"

# =============================================================================================
echo "### [1/4] polyselect (CADO size-opt ${PS_SECS}s..${PS_MAX}s, ropt top-$ROPT_K effort $ROPT_EFF) $(date +%T)"
if [ -n "${POLY_IN:-}" ] && [ -s "$POLY_IN" ]; then
  python3 "$HERE/polysel160.py" norm "$POLY_IN" "$WORK/poly.cado" || die "cannot read supplied polynomial $POLY_IN"
  say "using supplied polynomial $POLY_IN"
else
  ( cd "$WORK" && exec "$CF/polyselect" -N "$N" -degree 5 -P "$PS_P" -admin "$PS_ADMIN" -admax "$PS_ADMAX" \
        -incr "$PS_INCR" -nq "$PS_NQ" -t "$NCPU" > "$WORK/raw.p" 2> "$WORK/ps.err" ) & PSPID=$!
  while :; do
    sleep 5; _el=$(( $(date +%s)-T0 )); _nc=$(grep -c '^# Size-optimized' "$WORK/raw.p" 2>/dev/null)
    kill -0 "$PSPID" 2>/dev/null || break
    [ "$_el" -ge "$PS_SECS" ] && [ "${_nc:-0}" -ge "$PS_MIN_CANDS" ] && break
    [ "$_el" -ge "$PS_MAX" ] && break
  done
  kill "$PSPID" 2>/dev/null; wait "$PSPID" 2>/dev/null; PSPID=""
  say "size-opt search stopped: $(grep -c '^# Size-optimized' "$WORK/raw.p") candidates"
  python3 "$HERE/polysel160.py" topk "$WORK/raw.p" "$ROPT_K" "$WORK/top.poly" || die "no size-optimized candidates"
  # --foreground: plain timeout setpgid()s itself out of our process group, so breaking_rsa's killpg
  # would leave ropt (-t NCPU) running; polyselect_ropt is a single process, nothing else to reap.
  ( cd "$WORK" && timeout --foreground "$ROPT_MAX" "$CF/polyselect_ropt" -inputpolys "$WORK/top.poly" -ropteffort "$ROPT_EFF" \
        -t "$NCPU" -area 2.4e16 -Bf "$ALIM" -Bg "$RLIM" > "$WORK/ropt.out" 2> "$WORK/ropt.err" )
  say "ropt rc=$? ($(grep -c 'MurphyE' "$WORK/ropt.out") MurphyE lines)"
  python3 "$HERE/polysel160.py" best "$WORK/ropt.out" "$N" "$WORK/poly.cado" || die "ropt produced no usable polynomial"
  # POLY_TS=1 (default): GPU test-sieve of the TS_K best root-optimized polys, pick by modelled sieve time
  # (polytest160.sh; never fatal, keeps the MurphyE pick unless a candidate models >= TS_MIN_GAIN faster).
  if [ "${POLY_TS:-1}" = 1 ]; then
    say "test-sieving the top-${TS_K:-5} root-optimized polynomials (GPU, ~2 min)"
    bash "$HERE/polytest160.sh" "$WORK" "$N" "$WORK/ropt.out" "$WORK/poly.cado" "$CS" "$HERE" \
        "$RLIM" "$ALIM" "$LPB" "$MFB0" "$MFB1" "$LOGI" "$JJ" "$QMIN" -- "${ALLOWARGS[@]}" 2>&1 | grep -a '\[polytest\]'
    rm -rf "$WORK/ts"
  fi
fi
grep -q "^n: $N\$" "$WORK/poly.cado" || die "poly.cado N mismatch"
python3 "$HERE/polysel160.py" job  "$WORK/poly.cado" "$WORK/c160.job" "$RLIM" "$ALIM" "$LPB" "$LPB" "$MFB0" "$MFB1"
python3 "$HERE/polysel160.py" msfb "$WORK/poly.cado" "$WORK/c.fb" "$RLIM" "$ALIM" "$LPB" "$LPB"
say "polynomial: $(grep -E '^(c5|skew):' "$WORK/poly.cado" | tr '\n' ' ')"

# freerel + renumber depend only on (poly, lpb): build them NOW, in the background, while the GPU
# sieves -- measured 141 s at 4 threads in the sandbox, which would otherwise sit inside the first
# probe on the critical path. cado_filter160.sh finds the cache via FRC and skips its own freerel.
mkdir -p "$WORK/freerel"
( nice -n 5 "$CF/freerel" -poly "$WORK/poly.cado" -lpb0 "$LPB" -lpb1 "$LPB" -out "$WORK/freerel/freerel.gz" \
     -renumber "$WORK/freerel/renumber$SL" -t "$FR_T" > "$WORK/freerel/freerel.build.log" 2>&1 \
  && python3 - "$WORK/poly.cado" "$WORK/freerel" <<'PY' \
  && echo "c5filter=done" >> "$WORK/freerel/freerel.build.log" \
  && mv -f "$WORK/freerel/freerel.build.log" "$WORK/freerel/freerel.log"
import re, sys, gzip, os
poly, d = sys.argv[1], sys.argv[2]
c5 = abs(int(re.search(r'^c5:\s*(-?\d+)', open(poly).read(), re.M).group(1)))
bad, n, p = set(), c5, 2
while p * p <= n:
    while n % p == 0: bad.add(p); n //= p
    p += 1
if n > 1: bad.add(n)
pre = tuple('%x,0:' % q for q in bad)
src, dst = os.path.join(d, 'freerel.gz'), os.path.join(d, 'freerel.gz.f')
with gzip.open(src, 'rt') as fi, gzip.open(dst, 'wt', compresslevel=1) as fo:
    for L in fi:
        if not (pre and L.startswith(pre)): fo.write(L)
os.replace(dst, src)
PY
) &
FRPID=$!

# =============================================================================================
echo "### [2/4] factor base (fbgen_gpu, alim $ALIM) $(date +%T)"
"$CS/fbgen_gpu" --poly "$WORK/c160.job" --lim "$ALIM" --maxbits "$LOGI" --out "$WORK/fb.roots1" > "$WORK/fbgen.log" 2>&1 \
    || die "fbgen_gpu failed: $(tail -2 "$WORK/fbgen.log")"
say "factor base $(stat -c%s "$WORK/fb.roots1") bytes"

# =============================================================================================
echo "### [3/4] GPU lattice sieve + concurrent CADO probes (first probe at $FIRST_PROBE raw) $(date +%T)"
have=0; q=$QMIN; i=0; relq=""; w=200000; NEXT_PROBE=$FIRST_PROBE; PROBEPID=""; PROBE_I=0; MATRIX_DIR=""; TOOLFAIL=0
# LANDED: chunks 1..LANDED are in their memfds.  LAST_PK: the longest prefix 1..K ever probed (a prefix
# is never probed twice).  INC_DONE: chunks 1..INC_DONE committed by incdup.  PROBE_RUN_NT / INC_RUN_PAR:
# threads of the running probe / dup2 groups allowed to the running incdup (the thread budget).
LANDED=0; LAST_PK=0; PROBE_RUN_NT=0; PROBE_INC=0; declare -a HAVE_AT; HAVE_AT[0]=0
INC_CACHE="$WORK/inc"; INC_PARF="$WORK/inc/par"; INC_DONE=0; INC_RUNK=0; INCPID=""; INC_RUN_PAR=0; INC_BROKEN=0; INC_FREED=0
# INC_FREE_PENDING: incdup failed while an INC probe was still purging the dd parts -- the truncation
# waits until that probe is reaped.  INC_WAIT_K: the commit at which an INC probe was held back once to
# wait for the just-landed chunk (probe_launch waits at most once).
INC_FREE_PENDING=0; INC_WAIT_K=0
[ "$INCDUP" = 1 ] && mkdir -p "$INC_CACHE" || INC_BROKEN=1
# matrix_ok <probe dir>: did this probe produce what the downstream consumes? (defined BEFORE the replay
# block, which uses it)
matrix_ok(){
  if [ "$DS_MODE" = cado ]; then
    grep -qa 'VERDICT MATRIX' "$1/filter.out" 2>/dev/null && [ -s "$1/purged.gz" ] && [ -s "$1/history" ]
  else [ -s "$MEM/rels_out.cyc" ]; fi
}
[ "$DS_MODE" = cado ] && PROBE_MO=1 || PROBE_MO=0
inc_free(){   # truncate every incremental part + key shadow (their RAM back); the cache is dead after this
  [ "$INCDUP" = 1 ] && [ "$INC_FREED" = 0 ] || return 0
  local c s f; for c in $(seq 1 "$MAXCHUNKS"); do for s in $(seq 0 15); do
    printf -v f '%s/dd/%03d/s%02d'"$SL" "$MEM" "$c" "$s"; : > "$f"; printf -v f '%s/kk/%03d/s%02d' "$MEM" "$c" "$s"; : > "$f"; done; done
  INC_FREED=1; INC_BROKEN=1; }
inc_par_budget(){   # $1 = wanted dup2 groups -> allowed (>=1) under SIEVE_HOST_T + probe NT + groups*INC_TPG <= NCPU
  local p=$(( (NCPU - SIEVE_HOST_T - PROBE_RUN_NT) / INC_TPG )); [ "$p" -ge 1 ] || p=1; [ "$p" -le "$1" ] || p=$1; echo "$p"; }
inc_set_par(){ INC_RUN_PAR=$1; echo "$1" > "$INC_PARF.tmp" && mv -f "$INC_PARF.tmp" "$INC_PARF"; }
frwait(){   # never fatal: a bad background cache is deleted and cado_filter160.sh rebuilds it (~2 min)
  [ -n "${FRPID:-}" ] || return 0
  local frc=0; wait "$FRPID" || frc=$?; FRPID=""
  if [ "$frc" = 0 ] && grep -q nprimes= "$WORK/freerel/freerel.log" 2>/dev/null && grep -q c5filter=done "$WORK/freerel/freerel.log"; then
    say "freerel cache ready"
  else
    say "WARNING: background freerel failed (rc=$frc: $(tail -1 "$WORK/freerel/freerel.build.log" 2>/dev/null)) -- discarding it; the filter will rebuild"
    rm -f "$WORK/freerel/freerel.gz" "$WORK/freerel/freerel.gz.f" "$WORK/freerel/renumber$SL" "$WORK/freerel/freerel.log"
  fi; }
# ---- REPLAY MODE (test harness only): RELS_IN="<file> [<file> ...]" feeds pre-sieved relation
# files (in sieve order) through the IDENTICAL probe + downstream path, one file per chunk slot,
# instead of running the GPU sieve. Probes then run back-to-back as if the chunks had just arrived.
if [ -n "${RELS_IN:-}" ]; then
  for f in $RELS_IN; do
    i=$((i+1)); [ "$i" -le "$MAXCHUNKS" ] || die "replay: more files than chunk slots"
    cat "$f" > "$MEM/$(printf c%03d $i)" || die "replay: cannot load $f"
    n=$(grep -avc '^#' "$f"); have=$(( have + n ))
    say "replay chunk $i <- $(basename "$f") +$n total $have"
    [ "$have" -ge "$NEXT_PROBE" ] || continue
    if [ "$INC_BROKEN" = 0 ]; then   # replay: catch the cache up synchronously (same watchdog as live)
      frwait; [ -s "$WORK/freerel/renumber$SL" ] || { inc_free; say "incdup disabled: no freerel cache"; }
      while [ "$INC_BROKEN" = 0 ] && [ "$INC_DONE" -lt "$i" ]; do
        k=$((INC_DONE+1))
        CADO_BIN="$CF" FRC="$WORK/freerel" INC_PAR=$INC_PAR INC_GROUPS=$INC_GROUPS INC_NICE=0 \
          nice -n "$INC_NICE" python3 -c "$INC_PYL" "$$" timeout -k 10 "$INC_TIMEOUT" bash "$CF/cado_incdup.sh" "$INC_CACHE" "$MEM" \
          "$WORK/poly.cado" "$k" "$MEM/$(printf c%03d $k)" >> "$INC_CACHE.log" 2>&1 & INCPID=$!
        echo "$INCPID" > "$WORK/inc.pgid"; wait "$INCPID"; irc=$?; INCPID=""
        if [ "$irc" = 0 ] && [ -s "$INC_CACHE/chunk_$k.done" ]; then INC_DONE=$k
        else inc_free; say "WARNING: incdup chunk $k failed (rc=$irc$([ "$irc" = 124 ] && echo ', watchdog timeout')) -- falling back to one-shot probes: $(grep -a FAIL "$INC_CACHE.log" | tail -1)"; fi
      done
    fi
    frwait; PROBE_I=$((PROBE_I+1)); PROBE_N=$i; PROBE_AT=$have; LAST_PK=$i; mkdir -p "$WORK/probe$PROBE_I"
    CH=(); for k in $(seq 1 "$i"); do CH+=("$MEM/$(printf c%03d $k)"); done
    : > "$MEM/rels_out"; rm -f "$MEM/rels_out.cyc" "$MEM/rels_out.purged"
    INCENV=(); [ "$INC_BROKEN" = 0 ] && INCENV=(INC_CACHE="$INC_CACHE" INC_K="$i")
    env "${INCENV[@]}" MERGE_ONLY=$PROBE_MO LPB0=$LPB LPB1=$LPB NT=$PROBE_NT DUP2_PAR=$DUP2_PAR TD=$TD FRC="$WORK/freerel" \
      bash "$CF/cado_filter160.sh" "$WORK/probe$PROBE_I" "$MEM" "$WORK/poly.cado" "$N" "$MEM/rels_out" "${CH[@]}" \
      > "$WORK/probe$PROBE_I/filter.out" 2>&1; prc=$?
    say "replay probe $PROBE_I ($have raw$([ "${#INCENV[@]}" -gt 0 ] && echo ', incremental cache')) rc=$prc: $(grep -a VERDICT "$WORK/probe$PROBE_I/filter.out" | tail -1 | sed 's/.*VERDICT //')"
    if [ "$prc" = 0 ] && matrix_ok "$WORK/probe$PROBE_I"; then MATRIX_DIR="$WORK/probe$PROBE_I"; break; fi
    [ "$prc" = 2 ] || die "replay probe tooling failure: $(grep -a FAILED "$WORK/probe$PROBE_I/filter.out" | tail -1)"
    NEXT_PROBE=$(( have + PROBE_STEP ))
  done
  [ -n "$MATRIX_DIR" ] || die "replay: all input relations used, no matrix"
  [ "$DS_MODE" = cado ] || rm -f "$MATRIX_DIR"/purged.gz* "$MATRIX_DIR"/history*
fi
probe_reap(){   # sets MATRIX_DIR on success; returns 0 if a probe finished (either verdict)
  [ -n "$PROBEPID" ] || return 1
  kill -0 "$PROBEPID" 2>/dev/null && return 1
  wait "$PROBEPID"; local rc=$?; PROBEPID=""
  local v; v=$(grep -a 'VERDICT' "$WORK/probe$PROBE_I/filter.out" | tail -1)
  say "probe $PROBE_I (chunks 1..$PROBE_N, $PROBE_AT raw) rc=$rc: ${v#*VERDICT }"
  # an incdup step failed while this (incremental) probe was reading the dd parts: free them only NOW
  if [ "$INC_FREE_PENDING" = 1 ]; then INC_FREE_PENDING=0; inc_free; say "incremental cache freed (deferred until probe $PROBE_I ended)"; fi
  PROBE_RUN_NT=0
  if [ "$rc" = 0 ] && matrix_ok "$WORK/probe$PROBE_I"; then MATRIX_DIR="$WORK/probe$PROBE_I"; return 0; fi
  rm -f "$WORK/probe$PROBE_I"/purged.gz* "$WORK/probe$PROBE_I"/history* "$MEM/rels_out.cyc" "$MEM/rels_out.purged"; : > "$MEM/rels_out"
  if [ "$rc" = 2 ]; then
    TOOLFAIL=0; NEXT_PROBE=$(( have > PROBE_AT + PROBE_STEP ? have : PROBE_AT + PROBE_STEP ))
    # INCREMENTAL probes only ever see the COMMITTED prefix: requiring `have` (everything landed by now)
    # would make them wait for incdup to catch up with the sieve, and when an incdup step takes longer
    # than a chunk that gap grows every round (measured with stubs: probes on 1..3, 1..5, 1..8, 1..13).
    # The next committed chunk past PROBE_AT + PROBE_STEP is enough; `have` stays the one-shot rule.
    [ "$PROBE_INC" = 1 ] && [ "$INC_BROKEN" = 0 ] && NEXT_PROBE=$(( PROBE_AT + PROBE_STEP ))
    # the probe's threads are free again: let a running incdup widen (re-read before each dup2 group)
    [ -n "$INCPID" ] && inc_set_par "$(inc_par_budget "$([ "${HAVE_AT[$INC_RUNK]}" -ge "$NEXT_PROBE" ] && echo "$INC_PAR_PROBE" || echo "$INC_PAR")")"
  elif [ "$PROBE_INC" = 1 ]; then       # the incremental path failed: fall back for good (one-shot at the next landing)
    inc_kill; inc_free; NEXT_PROBE=$have
    say "WARNING: incremental probe tooling failure (rc=$rc): $(grep -a 'FAILED' "$WORK/probe$PROBE_I/filter.out" | tail -1) -- incremental cache freed, one-shot probes from the next chunk on"
  else
    TOOLFAIL=$((TOOLFAIL+1))
    say "probe tooling failure #$TOOLFAIL (rc=$rc$([ "$rc" -ge 128 ] && echo ', killed -- OOM?')): $(grep -a 'FAILED' "$WORK/probe$PROBE_I/filter.out" | tail -1)"
    [ "$TOOLFAIL" -ge 2 ] && die "probe tooling failed twice in a row: $(grep -a 'FAILED' "$WORK/probe$PROBE_I/filter.out" | tail -1)"
    NEXT_PROBE=$(( have + PROBE_STEP ))
  fi
  return 0
}
inc_consumed(){   # the running INC probe's purge SUCCEEDED and freed the dd parts: stop incdup, free the rest NOW
  [ "$PROBE_INC" = 1 ] && [ "$INC_FREED" = 0 ] && [ -e "$WORK/probe$PROBE_I/inc_consumed" ] || return 0
  inc_kill; inc_free
  say "probe $PROBE_I purge consumed the incremental cache -- incdup stopped, key shadows freed (a later probe, if any, is one-shot)"; }
inc_reap(){   # never blocks: reap a finished incdup step
  [ -n "$INCPID" ] || return 0
  kill -0 "$INCPID" 2>/dev/null && return 0
  local rc=0; wait "$INCPID" || rc=$?; INCPID=""; INC_RUN_PAR=0
  [ "$INC_BROKEN" = 0 ] || return 0
  if [ "$rc" = 0 ] && [ -s "$INC_CACHE/chunk_$INC_RUNK.done" ]; then INC_DONE=$INC_RUNK
  elif [ -n "$PROBEPID" ] && [ "$PROBE_INC" = 1 ]; then
    # a live INCREMENTAL probe is purging dd parts of chunks 1..PROBE_N (read twice): truncating them now
    # would hand purge a silent subset (false SHORT / smaller matrix) or break its pass 2. Stop all
    # incremental work (INC_BROKEN) but defer the truncation to probe_reap.
    INC_BROKEN=1; INC_FREE_PENDING=1
    say "WARNING: incdup chunk $INC_RUNK failed (rc=$rc$([ "$rc" = 124 ] || [ "$rc" = 137 ] && echo ", watchdog timeout ${INC_TIMEOUT}s")): $(grep -a FAIL "$INC_CACHE.log" | tail -1) -- one-shot probes from the next chunk on; incremental cache freed once probe $PROBE_I ends"
  else
    inc_free
    say "WARNING: incdup chunk $INC_RUNK failed (rc=$rc$([ "$rc" = 124 ] || [ "$rc" = 137 ] && echo ", watchdog timeout ${INC_TIMEOUT}s")): $(grep -a FAIL "$INC_CACHE.log" | tail -1) -- incremental cache freed, one-shot probes from the next chunk on"
  fi; }
inc_start(){   # never blocks: start the next incdup step if a landed chunk is waiting
  [ "$INC_BROKEN" = 0 ] && [ -z "$INCPID" ] || return 0
  [ "$INC_DONE" -lt "$LANDED" ] || return 0     # NOT $i: during sieving chunk $i is not in its memfd yet
  if [ -n "${FRPID:-}" ]; then kill -0 "$FRPID" 2>/dev/null && return 0; frwait; fi
  if ! { [ -s "$WORK/freerel/renumber$SL" ] && grep -q c5filter=done "$WORK/freerel/freerel.log" 2>/dev/null; }; then
    inc_free; say "incdup disabled: no freerel cache"; return 0; fi
  INC_RUNK=$((INC_DONE+1))
  local want=$INC_PAR; [ -z "$PROBEPID" ] && [ "${HAVE_AT[$INC_RUNK]}" -ge "$NEXT_PROBE" ] && want=$INC_PAR_PROBE
  inc_set_par "$(inc_par_budget "$want")"
  # own session + PDEATHSIG (INC_PYL): inc_kill takes down timeout + its dup1/dup2 as ONE group; the
  # watchdog (timeout -k) kills the same group on a hang; our death TERMs it. Whole tree at nice INC_NICE
  # (INC_NICE=0 inside). The pgid is recorded for the next attempt's startup check (top of script).
  CADO_BIN="$CF" FRC="$WORK/freerel" INC_PAR=$INC_RUN_PAR INC_PAR_FILE="$INC_PARF" INC_GROUPS=$INC_GROUPS INC_NICE=0 \
    nice -n "$INC_NICE" python3 -c "$INC_PYL" "$$" \
    timeout -k 10 "$INC_TIMEOUT" bash "$CF/cado_incdup.sh" "$INC_CACHE" "$MEM" "$WORK/poly.cado" "$INC_RUNK" "$MEM/$(printf c%03d $INC_RUNK)" \
    >> "$INC_CACHE.log" 2>&1 & INCPID=$!
  echo "$INCPID" > "$WORK/inc.pgid"
}
probe_launch(){   # $1 = land|poll. Launch a probe if due, none is running and the prefix is NEW.
  [ -z "$PROBEPID" ] || return 0
  local K NT=$PROBE_NT INCENV=()
  if [ "$INC_BROKEN" = 0 ]; then
    # INCREMENTAL: the committed prefix 1..INC_DONE, at a commit (incdup idle -> its threads are free).
    # Wait for the chunk that just landed to be committed rather than probe a stale prefix -- unless
    # incdup has fallen >= 2 chunks behind the sieve -- but wait AT MOST ONCE: if an earlier commit was
    # already held back for this reason (INC_WAIT_K), incdup is not catching up with the sieve (step
    # time ~ chunk interval), and waiting again could stall probing for many chunks: probe now.
    K=$INC_DONE; [ "$K" -ge 1 ] && [ -z "$INCPID" ] || return 0
    if [ "$K" != "$LANDED" ] && [ $(( LANDED - K )) -lt 2 ]; then
      [ "$K" -gt "$LAST_PK" ] && [ "${HAVE_AT[$K]}" -ge "$NEXT_PROBE" ] || return 0   # not due: nothing to wait for
      if [ "$INC_WAIT_K" = 0 ] || [ "$INC_WAIT_K" = "$K" ]; then INC_WAIT_K=$K; return 0; fi
    fi
    INCENV=(INC_CACHE="$INC_CACHE" INC_K="$K")
    # thread budget: sieve host + one dup2 group for the next chunk's incdup + 1 for purge's `gzip -dc`
    # input reader (not counted in -t). incdup's dup1 phase (16 `gzip -c --fast` writers, ~3 cores for
    # a few seconds per c160 chunk, measured) is NOT in the budget: it precedes the dup2 groups, its CPU
    # demand fits the INC_TPG it later uses, and it runs at nice 15 (CFS weight 36 vs 335 for this
    # nice-5 probe), so the brief oversubscription in RUNNABLE threads costs the probe ~nothing.
    NT=$(( NCPU - SIEVE_HOST_T - INC_TPG - 1 )); [ "$NT" -le "$PROBE_NT" ] || NT=$PROBE_NT; [ "$NT" -ge 2 ] || NT=2
  else
    # ONE-SHOT (INCDUP=0, or after a fallback): only when a chunk LANDS -- production cadence exactly
    [ "$1" = land ] || return 0
    K=$LANDED
  fi
  [ "$K" -gt "$LAST_PK" ] && [ "${HAVE_AT[$K]}" -ge "$NEXT_PROBE" ] || return 0
  frwait; PROBE_I=$((PROBE_I+1)); PROBE_N=$K; PROBE_AT=${HAVE_AT[$K]}; LAST_PK=$K; mkdir -p "$WORK/probe$PROBE_I"
  PROBE_INC=0; [ "${#INCENV[@]}" -gt 0 ] && PROBE_INC=1; INC_WAIT_K=0
  CH=(); for k in $(seq 1 "$K"); do CH+=("$MEM/$(printf c%03d $k)"); done
  : > "$MEM/rels_out"; rm -f "$MEM/rels_out.cyc" "$MEM/rels_out.purged"
  ( exec env "${INCENV[@]}" MERGE_ONLY=$PROBE_MO LPB0=$LPB LPB1=$LPB NT=$NT DUP2_PAR=$DUP2_PAR TD=$TD FRC="$WORK/freerel" \
      nice -n 5 bash "$CF/cado_filter160.sh" "$WORK/probe$PROBE_I" "$MEM" "$WORK/poly.cado" "$N" "$MEM/rels_out" "${CH[@]}" \
      > "$WORK/probe$PROBE_I/filter.out" 2>&1 ) & PROBEPID=$!; PROBE_RUN_NT=$NT
  say "probe $PROBE_I launched on chunks 1..$K (${HAVE_AT[$K]} raw$([ "$PROBE_INC" = 1 ] && echo ", incremental cache, NT=$NT"))"
}
# ---- CPU co-sieve: CADO las in bounded q-chunks on the idle cores (see LAS= above) ---------------
# States: off (LAS=0 / not started) | fb (makefb running) | run (a las chunk running) | dead (failed:
# GPU-only from here) | retired (the GPU reached LAS_QMIN) | ended (MATRIX).
# LAS_READY: validated, comment-stripped las outputs waiting to land (in q order); LAS_QDONE: end of the
# contiguous las q-range whose relations are ready or landed (the GPU leaps over [LAS_QMIN, LAS_QDONE)).
LASD="$WORK/las"; LAS_STATE=off; LASPID=""; LAS_FBPID=""; LAS_J=0; LAS_Q=$LAS_QMIN; LAS_Q1=0; LAS_W=$LAS_W0
LAS_QDONE=$LAS_QMIN; LAS_QLANDED=$LAS_QMIN; LAS_READY=(); LAS_READY_N=(); LAS_GATED=0; LAS_EMPTY=0
LAS_T0=0; LAS_STOPPED=0; LAS_STOP_T0=0; LAS_STOP_SUM=0; LAS_STOP_TOT=0; LAS_PROG_T=0; LAS_PROG_SZ=-1
LAS_AFF_K=0; LAS_AFF_AT=0; LAS_AFF_BROKEN=0; LAS_RSS_MAX=0; LAS_MEM_AT=0; LAS_TOT=0; LAS_NLAND=0; LAS_SLOTFULL=0
lsay(){ say "[las] $*"; }
ltlog(){ echo "$(el) $*" >> "$WORK/las_throttle.log" 2>/dev/null; }   # outside LASD: survives las_end
las_budget(){   # CPUs las may use right now: whatever sieve host + probe + incdup + freerel leave free
  local b=$(( NCPU - SIEVE_HOST_T - LAS_RESERVE - ( COFAC_CPU == 1 ? COFAC_T : 0 ) ))
  [ -n "${PROBEPID:-}" ] && b=$(( b - PROBE_RUN_NT - 1 ))
  [ -n "${INCPID:-}" ] && b=$(( b - INC_RUN_PAR * INC_TPG ))
  # FRPID stays set until frwait reaps it (the first incdup start, i.e. after chunk 1 lands): count freerel
  # only while it is really alive (measured: otherwise las sat at NCPU-2-FR_T CPUs for the whole chunk 1)
  [ -n "${FRPID:-}" ] && kill -0 "$FRPID" 2>/dev/null && b=$(( b - FR_T ))
  [ "$b" -gt 0 ] || b=0; echo "$b"; }
las_stop(){   # kill makefb / the running las chunk (its partial output is LOST by design) and remove it
  local p
  for p in ${LAS_FBPID:-} ${LASPID:-}; do
    kill -CONT "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null
    for _ in $(seq 1 50); do kill -0 "$p" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$p" 2>/dev/null; wait "$p" 2>/dev/null
  done
  LAS_FBPID=""; LASPID=""; LAS_STOPPED=0
  rm -f "$LASD"/c*.out "$LASD"/c*.out.tmp.* 2>/dev/null; return 0; }
# (las_drop_ready / las_take return their count in LAS_RET: they must NOT run in a $(...) subshell,
#  which would lose their updates of LAS_READY / LAS_QDONE / LAS_TOT)
las_drop_ready(){ local x; LAS_RET=0; for x in "${LAS_READY_N[@]}"; do LAS_RET=$((LAS_RET+x)); done
  rm -f "${LAS_READY[@]}" 2>/dev/null; LAS_READY=(); LAS_READY_N=(); LAS_QDONE=$LAS_QLANDED; }
las_fail(){   # $1 = keep|discard (the validated, not yet landed outputs), $2 = reason. Never fatal.
  [ "$LAS_STATE" = dead ] && return 0
  las_stop; LAS_STATE=dead; rm -f "$LASD/fb1"
  local d=""; [ "$1" = discard ] && { las_drop_ready; d=", $LAS_RET unlanded relations discarded"; }
  say "WARNING: [las] $2 -- CPU co-sieve DISABLED for the rest of the run (GPU-only)$d; ${#LAS_READY[@]} validated output(s) still to land"; }
las_fb_start(){
  [ "$LAS" = 1 ] && [ "$LAS_STATE" = off ] || return 0
  mkdir -p "$LASD" || { LAS_STATE=dead; say "WARNING: [las] cannot create $LASD -- CPU co-sieve DISABLED"; return 0; }
  local t; t=$(las_budget); [ "$t" -ge 1 ] || t=1
  ( exec nice -n "$LAS_NICE" "$CF/makefb" -poly "$WORK/poly.cado" -side 1 -lim "$LAS_LIM1" -maxbits "$(( (LAS_A + 1) / 2 ))" \
        -t "$t" -out "$LASD/fb1" > "$LASD/makefb.log" 2>&1 ) & LAS_FBPID=$!
  LAS_STATE=fb
  lsay "makefb side 1 lim $LAS_LIM1 -t $t started; co-sieve q from $LAS_QMIN, -t $LAS_T nice $LAS_NICE, landing=$LAS_LAND"; }
las_launch(){   # the next q-chunk [LAS_Q, LAS_Q + LAS_W)
  LAS_J=$((LAS_J+1)); LAS_Q1=$(( LAS_Q + LAS_W ))
  rm -f "$LASD/c$LAS_J.out" "$LASD/c$LAS_J.out".tmp.*
  ( exec nice -n "$LAS_NICE" "$CF/las" -poly "$WORK/poly.cado" -fb1 "$LASD/fb1" -lim0 "$LAS_LIM0" -lim1 "$LAS_LIM1" \
        -lpb0 "$LAS_LPB" -lpb1 "$LAS_LPB" -mfb0 "$LAS_MFB0" -mfb1 "$LAS_MFB1" -A "$LAS_A" -q0 "$LAS_Q" -q1 "$LAS_Q1" \
        -sqside 1 -t "$LAS_T" -out "$LASD/c$LAS_J.out" > "$LASD/c$LAS_J.log" 2>&1 ) & LASPID=$!
  LAS_T0=$(date +%s); LAS_STOP_SUM=0; LAS_STOPPED=0; LAS_PROG_T=$LAS_T0; LAS_PROG_SZ=-1; LAS_AFF_K=0; LAS_STATE=run
  ltlog "chunk $LAS_J q[$LAS_Q,$LAS_Q1) pid $LASPID"
  las_throttle; }
las_throttle(){   # confine the running las to las_budget() CPUs; SIGSTOP it below LAS_MIN_CPUS
  [ "$LAS_STATE" = run ] && [ -n "$LASPID" ] || return 0
  local b now need r; b=$(las_budget); now=$(date +%s)
  need=$LAS_MIN_CPUS; [ "$LAS_AFF_BROKEN" = 1 ] && need=$LAS_T   # cannot confine: run only when it fits whole
  if [ "$b" -lt "$need" ]; then
    [ "$LAS_STOPPED" = 1 ] && return 0
    kill -STOP "$LASPID" 2>/dev/null && { LAS_STOPPED=1; LAS_STOP_T0=$now; ltlog "pause (budget $b < $need)"; }
    return 0
  fi
  if [ "$LAS_STOPPED" = 1 ]; then
    kill -CONT "$LASPID" 2>/dev/null; LAS_STOPPED=0; LAS_STOP_SUM=$(( LAS_STOP_SUM + now - LAS_STOP_T0 ))
    LAS_PROG_T=$now; ltlog "resume (budget $b)"
  fi
  [ "$LAS_AFF_BROKEN" = 1 ] && return 0
  [ "$b" != "$LAS_AFF_K" ] || [ $(( now - LAS_AFF_AT )) -ge 30 ] || return 0
  if r=$(python3 "$HERE/las_ctl.py" aff "$LASPID" "$b" 2>&1); then
    [ "$b" != "$LAS_AFF_K" ] && ltlog "confine to $b CPUs ($r)"; LAS_AFF_K=$b; LAS_AFF_AT=$now
  elif kill -0 "$LASPID" 2>/dev/null; then   # (a chunk that just exited is not an affinity failure)
    LAS_AFF_BROKEN=1
    say "WARNING: [las] cannot confine las to $b CPUs ($r) -- degraded: las runs only while its full -t $LAS_T fits the budget"
    las_throttle
  fi; }
las_watch(){   # running chunk: progress/wall watchdogs, RSS sample, memory headroom guard, throttle
  local now sz k v _ h
  now=$(date +%s)
  if [ "$LAS_STOPPED" = 0 ]; then
    sz=$(stat -c%s "$LASD/c$LAS_J.out".tmp.* 2>/dev/null | head -1)
    if [ "${sz:-0}" != "$LAS_PROG_SZ" ]; then LAS_PROG_SZ=${sz:-0}; LAS_PROG_T=$now
    elif [ $(( now - LAS_PROG_T )) -ge "$LAS_STALL" ]; then las_fail keep "chunk $LAS_J: output has not grown for ${LAS_STALL}s (hung?)"; return 0; fi
    [ $(( now - LAS_T0 - LAS_STOP_SUM )) -lt "$LAS_CHUNK_MAX" ] || { las_fail keep "chunk $LAS_J exceeded LAS_CHUNK_MAX=${LAS_CHUNK_MAX}s of active time"; return 0; }
  fi
  while read -r k v _; do [ "$k" = VmRSS: ] && { [ "$v" -gt "$LAS_RSS_MAX" ] && LAS_RSS_MAX=$v; break; }; done < "/proc/$LASPID/status" 2>/dev/null
  if [ $(( now - LAS_MEM_AT )) -ge 30 ]; then
    LAS_MEM_AT=$now; h=$(python3 "$HERE/las_ctl.py" mem 2>/dev/null | grep -oE 'headroom_mb=[0-9]+' | cut -d= -f2)
    [ -n "$h" ] && [ "$h" -lt "$LAS_MEM_MB" ] && { las_fail keep "cgroup memory headroom ${h} MB < LAS_MEM_MB=$LAS_MEM_MB (las RSS ${LAS_RSS_MAX} kB freed)"; return 0; }
  fi
  las_throttle; }
las_finish(){   # $1 = las exit status of chunk LAS_J
  local f="$LASD/c$LAS_J.out" r="$LASD/r$LAS_J.rels" n g dt now; now=$(date +%s)
  [ "$LAS_STOPPED" = 1 ] && LAS_STOP_SUM=$(( LAS_STOP_SUM + now - LAS_STOP_T0 )); LAS_STOPPED=0
  LAS_STOP_TOT=$(( LAS_STOP_TOT + LAS_STOP_SUM )); dt=$(( now - LAS_T0 - LAS_STOP_SUM )); [ "$dt" -ge 1 ] || dt=1
  if [ "$1" != 0 ] || [ ! -s "$f" ]; then las_fail keep "chunk $LAS_J q[$LAS_Q,$LAS_Q1) las rc=$1$([ "$1" -ge 128 ] 2>/dev/null && echo ' (killed -- OOM?)'): $(grep -av '^#' "$LASD/c$LAS_J.log" 2>/dev/null | tail -1)"; return 0; fi
  [ -z "$(tail -c1 "$f")" ] || { las_fail keep "chunk $LAS_J output ends in a torn line"; return 0; }
  # strip las's comment lines (per-q stats: ~1/3 of its bytes); grep rc 2 = read/write error (ENOSPC on /tmp)
  local grc=0; grep -av '^#' "$f" > "$r" || grc=$?
  if [ "$grc" -ge 2 ] || [ -n "$(tail -c1 "$r" 2>/dev/null)" ]; then rm -f "$r" "$f"; las_fail keep "chunk $LAS_J: stripping its output failed (grep rc=$grc; /tmp full?)"; return 0; fi
  n=$(wc -l < "$r"); rm -f "$f"
  if [ "${n:-0}" -gt 0 ] && [ "$LAS_GATED" = 0 ]; then
    # EXACT check (norms factor completely, primes < 2^lpb) BEFORE any las relation reaches the pool
    if g=$(python3 "$HERE/las_ctl.py" gate "$WORK/poly.cado" "$LAS_LPB" "$LAS_LPB" "$r" 2000 2>&1); then LAS_GATED=1; lsay "relation gate ${g#GATE }"
    else rm -f "$r"; las_fail discard "relation gate FAILED on chunk $LAS_J: $g"; return 0; fi
  fi
  lsay "chunk $LAS_J q[$LAS_Q,$LAS_Q1) +${n:-0} in ${dt}s active ($(( ${n:-0} / dt )) rel/s, paused ${LAS_STOP_SUM}s, rss max $(( LAS_RSS_MAX / 1024 )) MB)"
  if [ "${n:-0}" -gt 0 ]; then LAS_READY+=("$r"); LAS_READY_N+=("$n"); LAS_EMPTY=0
  else rm -f "$r"; LAS_EMPTY=$((LAS_EMPTY+1)); [ "$LAS_EMPTY" -ge 2 ] && { las_fail keep "two consecutive las chunks produced no relations"; return 0; }; fi
  # next width: this chunk's active q-rate x LAS_CHUNK_SECS, at most doubling, at least LAS_WMIN
  local wprev=$(( LAS_Q1 - LAS_Q ))
  LAS_QDONE=$LAS_Q1; LAS_Q=$LAS_Q1
  LAS_W=$(( wprev * LAS_CHUNK_SECS / dt ))
  [ "$LAS_W" -le $(( 2 * wprev )) ] || LAS_W=$(( 2 * wprev )); [ "$LAS_W" -ge "$LAS_WMIN" ] || LAS_W=$LAS_WMIN
  [ "$LAS_LAND" = slot ] && las_land_slot
  [ "$LAS_STATE" = run ] && las_launch; return 0; }
las_poll(){   # never blocks (called every poll and after every landing)
  case "$LAS_STATE" in
    fb) kill -0 "$LAS_FBPID" 2>/dev/null && return 0
        local rc=0; wait "$LAS_FBPID" || rc=$?; LAS_FBPID=""
        if [ "$rc" = 0 ] && [ -s "$LASD/fb1" ]; then lsay "factor base ready ($(stat -c%s "$LASD/fb1") bytes)"; las_launch
        else las_fail discard "makefb rc=$rc: $(tail -1 "$LASD/makefb.log" 2>/dev/null)"; fi ;;
    run) if kill -0 "$LASPID" 2>/dev/null; then las_watch; return 0; fi
         local rc=0; wait "$LASPID" || rc=$?; LASPID=""; las_finish "$rc" ;;
  esac; return 0; }
las_take(){   # append every ready output to the chunk slot memfd $1; LAS_RET = relations appended
  local x n=0; LAS_RET=0
  for x in "${LAS_READY_N[@]}"; do n=$((n+x)); done
  cat "${LAS_READY[@]}" >> "$1" || return 1
  rm -f "${LAS_READY[@]}"; LAS_READY=(); LAS_READY_N=(); LAS_QLANDED=$LAS_QDONE
  LAS_TOT=$(( LAS_TOT + n )); LAS_NLAND=$(( LAS_NLAND + 1 )); LAS_RET=$n; }
las_land_slot(){   # LAS_LAND=slot: the ready outputs become ONE new chunk slot now
  [ "${#LAS_READY[@]}" -gt 0 ] || return 0
  if [ $(( LANDED + 1 )) -gt $(( MAXCHUNKS - LAS_SLOT_KEEP )) ]; then
    [ "$LAS_SLOTFULL" = 1 ] || lsay "chunk slots nearly used up ($LANDED of $MAXCHUNKS) -- las outputs now ride with the next GPU chunk"
    LAS_SLOTFULL=1; return 0; fi
  local s=$(( LANDED + 1 )) f n; printf -v f '%s/c%03d' "$MEM" "$s"
  : > "$f"; if ! las_take "$f"; then : > "$f"; las_fail discard "landing into slot $s failed"; return 0; fi; n=$LAS_RET
  have=$(( have + n )); HAVE_AT[$s]=$have; LANDED=$s
  [ -n "${DEBUG_CHUNK_COPY:-}" ] && cat "$f" > "$DEBUG_CHUNK_COPY/$(printf c%03d "$s")"
  say "las -> slot $s +$n  total $have"
  inc_consumed; inc_reap; probe_launch land; inc_start; }
las_band_guard(){   # before a GPU chunk [q,qb): never re-sieve las's band -- retire las, then leap over it
  [ "$LAS" = 1 ] && [ "$LAS_BAND_GUARD" = 1 ] || return 0
  if [ "$qb" -gt "$LAS_QMIN" ] && { [ "$LAS_STATE" = run ] || [ "$LAS_STATE" = fb ]; }; then
    las_stop; LAS_STATE=retired; rm -f "$LASD/fb1"
    lsay "RETIRED: the GPU's next q-range [$q,$qb) reaches the las band at $LAS_QMIN (running chunk discarded; ${#LAS_READY[@]} output(s) still to land)"
  fi
  if [ "$q" -ge "$LAS_QMIN" ] && [ "$q" -lt "$LAS_QDONE" ]; then
    say "GPU leaps over the las-covered q-band [$q,$LAS_QDONE)"; q=$LAS_QDONE; qb=$(( q + w ))
  elif [ "$q" -lt "$LAS_QMIN" ] && [ "$qb" -gt "$LAS_QMIN" ] && [ "$LAS_QDONE" -gt "$LAS_QMIN" ]; then
    qb=$LAS_QMIN; w=$(( qb - q )); say "GPU chunk clipped to [$q,$qb) (las covers [$LAS_QMIN,$LAS_QDONE))"
  fi; return 0; }
las_end(){   # MATRIX: stop las, discard its partial chunk and anything not landed, free /tmp
  [ "$LAS" = 1 ] || return 0
  local st=$LAS_STATE d; las_stop; las_drop_ready; d=$LAS_RET
  [ "$st" = run ] || [ "$st" = fb ] && LAS_STATE=ended
  lsay "summary: $LAS_TOT relations landed ($LAS_NLAND landings, $(( have > 0 ? LAS_TOT * 1000 / have : 0 ))/1000 of raw), chunks started $LAS_J, final state $st, discarded at stop: running chunk + $d ready, paused ${LAS_STOP_TOT}s total, rss max $(( LAS_RSS_MAX / 1024 )) MB, q reached $LAS_QDONE"
  rm -rf "$LASD"; }
[ -z "$MATRIX_DIR" ] && [ -z "${RELS_IN:-}" ] && las_fb_start
while [ -z "$MATRIX_DIR" ]; do
  # i = GPU chunk number; its SLOT in the chunk pool is assigned when it lands (LANDED+1): with LAS_LAND=slot,
  # las chunks take slots in between. Every consumer (HAVE_AT, LANDED, incdup's INC_DONE, probe prefixes
  # 1..K) is indexed by SLOT, in landing order.
  i=$((i+1)); [ $(( LANDED + 1 )) -le "$MAXCHUNKS" ] || die "ran out of chunk slots ($MAXCHUNKS) without a matrix"
  cr=$CHUNK_RELS; [ "$have" -ge $(( FIRST_PROBE - CHUNK_RELS )) ] && cr=$CHUNK_RELS_LATE
  if [ -n "$relq" ]; then w=$(python3 -c "print(max(20000,min(2*$w,int($cr/$relq))))"); fi
  qb=$(( q + w )); las_band_guard; out="$WORK/chunk.dat"; rm -f "$out" "$out".part "$out".part.ckpt "$out".lock "$STOP"
  CAND="$WORK/cand.dat"; COFREL="$WORK/cofac.rel"; rm -f "$CAND" "$CAND".part "$CAND".done "$COFREL"
  COFARGS=(--cofactor); [ "$COFAC_CPU" = 1 ] && COFARGS=(--candidates "$CAND")
  t0=$(date +%s); RESUMED=0; COFOK=1
  "$CS/bench" --pipeline "${COFARGS[@]}" "${ALLOWARGS[@]}" --fill-concurrent --poly "$WORK/c160.job" --fb1 "$WORK/fb.roots1" \
      --logI "$LOGI" --J "$JJ" --sq-side 1 --qrange "$q:$((qb-1))" --relations "$out" --stop-file "$STOP" \
      > "$WORK/bench.log" 2>&1 & BENCHPID=$!
  # CPU cofactorisation of this band's candidates, concurrently (tails $CAND.part until $CAND.done exists)
  [ "$COFAC_CPU" = 1 ] && { "$CS/cofac_cpu" -lpb0 "$LPB" -lpb1 "$LPB" -mfb0 "$MFB0" -mfb1 "$MFB1" -fbb0 "$RLIM" -fbb1 "$ALIM" -t "$COFAC_T" \
      -follow "$CAND.done" "$CAND.part" "$COFREL" 2> "$WORK/cofac.log" & COFPID=$!; }
  # while the GPU sieves this chunk, watch the probe; a MATRIX verdict stops the sieve early
  while kill -0 "$BENCHPID" 2>/dev/null; do
    sleep 3
    if probe_reap && [ -n "$MATRIX_DIR" ]; then touch "$STOP"; kill "$BENCHPID" 2>/dev/null; break; fi
    inc_consumed; inc_reap; probe_launch poll; inc_start; las_poll
  done
  wait "$BENCHPID"; rc=$?; BENCHPID=""
  [ -n "$MATRIX_DIR" ] && { [ -n "${COFPID:-}" ] && { kill "$COFPID" 2>/dev/null; wait "$COFPID" 2>/dev/null; COFPID=""; }
                           say "MATRIX found -- sieve stopped mid-chunk $i"; break; }
  qnext=$qb
  if [ "$rc" != 0 ] && [ ! -s "$out" ]; then
    say "WARNING: bench rc=$rc on q[$q,$qb): $(tail -1 "$WORK/bench.log") -- resuming once from its checkpoint"
    rm -f "$out".lock; RESUMED=1
    # no checkpoint yet (a crash inside the band's first ~30 s; without --cofactor bench writes no seed
    # checkpoint): a plain resume REFUSES a non-empty uncheckpointed .part -> restart the band instead
    RESTART=(); [ -e "$out".part.ckpt ] || RESTART=(--restart)
    "$CS/bench" --pipeline "${COFARGS[@]}" "${ALLOWARGS[@]}" "${RESTART[@]}" --fill-concurrent --poly "$WORK/c160.job" --fb1 "$WORK/fb.roots1" \
        --logI "$LOGI" --J "$JJ" --sq-side 1 --qrange "$q:$((qb-1))" --relations "$out" --stop-file "$STOP" \
        > "$WORK/bench.resume.log" 2>&1; rc=$?
    if [ "$rc" != 0 ] && [ ! -s "$out" ] && [ -s "$out".part ]; then
      if [ -s "$out".part.ckpt ]; then
        rb=$(awk -F' = ' '$1=="rel_bytes"{print $2}' "$out".part.ckpt); nq=$(awk -F' = ' '$1=="next_q"{print $2}' "$out".part.ckpt)
        truncate -s "${rb:-0}" "$out".part; [ -n "$nq" ] && [ "$nq" -gt "$q" ] && qnext=$nq
        # the candidates too, to the same checkpoint (else q >= next_q is split twice: here and next chunk)
        cb=$(awk -F' = ' '$1=="cand_bytes"{print $2}' "$out".part.ckpt)
        [ "$COFAC_CPU" = 1 ] && [ -n "$cb" ] && [ -e "$CAND".part ] && truncate -s "$cb" "$CAND".part
      else
        python3 -c "import sys;p=sys.argv[1];d=open(p,'rb').read();open(p,'wb').write(d[:d.rfind(b'\\n')+1])" "$out".part
      fi
      mv -f "$out".part "$out"
      say "WARNING: resume also failed (rc=$rc) -- kept checkpointed relations, continuing at q=$qnext"
    fi
  fi
  [ -s "$out" ] || { [ -s "$out".part ] && mv -f "$out".part "$out"; }
  [ -z "$(tail -c1 "$out" 2>/dev/null)" ] || { say "WARNING: chunk $i ends in a torn line -- dropping it"
      python3 -c "import sys;p=sys.argv[1];d=open(p,'rb').read();open(p,'wb').write(d[:d.rfind(b'\\n')+1])" "$out"; }
  if [ "$COFAC_CPU" = 1 ]; then
    touch "$CAND.done"; wait "$COFPID"; crc=$?; COFPID=""
    # after a bench resume the follower may have read past the checkpoint truncation: redo the whole
    # final candidates file in one pass (follow mode on an already-finished file: drains it, drops a
    # torn last record). Same after a follower failure.
    if [ "$RESUMED" = 1 ] || [ "$crc" != 0 ]; then
      cin="$CAND"; [ -s "$cin" ] || cin="$CAND.part"
      [ "$crc" != 0 ] && say "WARNING: cofac_cpu rc=$crc on chunk $i: $(tail -1 "$WORK/cofac.log") -- redoing the candidates in one pass"
      # a batch job on a finished file with the sieve stopped (GPU idle until it ends): nice 0 outranks the
      # nice-5 probe and nice-15 incdup, so take every core but the sieve host's
      rt=$(( NCPU - SIEVE_HOST_T )); [ "$rt" -ge "$COFAC_T" ] || rt=$COFAC_T; [ "$rt" -le 256 ] || rt=256
      "$CS/cofac_cpu" -lpb0 "$LPB" -lpb1 "$LPB" -mfb0 "$MFB0" -mfb1 "$MFB1" -fbb0 "$RLIM" -fbb1 "$ALIM" -t "$rt" -follow "$CAND.done" "$cin" "$COFREL" \
          2> "$WORK/cofac.log"; crc=$?
      if [ "$crc" != 0 ]; then
        COFAC_CPU=0; COFOK=0; : > "$COFREL"
        say "WARNING: cofac_cpu failed again (rc=$crc): $(tail -1 "$WORK/cofac.log") -- chunk $i keeps only its TD relations; GPU --cofactor from the next chunk on"
      fi
    fi
    [ -s "$COFREL" ] && { cat "$COFREL" >> "$out" || die "cannot append the CPU-cofactored relations of chunk $i"; }
    say "  cofac_cpu chunk $i: $(grep -a 'cofac_cpu: rec' "$WORK/cofac.log" | tail -1 | sed 's/^cofac_cpu: //')"
  fi
  dt=$(( $(date +%s)-t0 ))
  # bench (BN_LIMBS=8) SKIPS a special-q whose exact norm exceeds 256 bits -- not expected at c160, so say it LOUDLY
  _sk=$(grep -aoE '[0-9]+ special-q skipped for norm width' "$WORK/bench.log" "$WORK/bench.resume.log" 2>/dev/null | head -1)
  [ -n "$_sk" ] && say "WARNING: chunk $i: $_sk (bench built with too few BN_LIMBS for this polynomial -- yield lost)"
  n=$(grep -avc '^#' "$out" 2>/dev/null); n=${n:-0}
  [ "$n" -gt 0 ] || die "chunk $i produced no relations (rc=$rc): $(tail -3 "$WORK/bench.log")"
  slot=$(( LANDED + 1 )); printf -v sf '%s/c%03d' "$MEM" "$slot"
  cat "$out" > "$sf" || die "copy of chunk $i into memfd slot $slot failed"
  # finished, validated las outputs ride along in this slot (LAS_LAND=ride; or slot mode with the pool nearly
  # full / las already stopped). A failed append restores the slot to the GPU chunk alone.
  nl=0
  if [ "${#LAS_READY[@]}" -gt 0 ]; then
    if las_take "$sf"; then nl=$LAS_RET
    else cat "$out" > "$sf" || die "copy of chunk $i into memfd slot $slot failed"; las_fail discard "appending las output to slot $slot failed"; fi
  fi
  [ -n "${DEBUG_CHUNK_COPY:-}" ] && cat "$sf" > "$DEBUG_CHUNK_COPY/$(printf c%03d "$slot")"
  if [ "$i" = 1 ]; then
    { head -150000 "$out"; tail -100000 "$out"; } > "$WORK/gate.dat"   # TD relations first, CPU-cofactored ones last
    "$CS/bench" --check-relations "$WORK/gate.dat" --poly "$WORK/c160.job" --lpb "$LPB" --lpb0 "$LPB" > "$WORK/gate.log" 2>&1
    grep -qa PASS "$WORK/gate.log" || die "relation gate FAILED: $(tail -2 "$WORK/gate.log")"
    say "relation gate PASS"; rm -f "$WORK/gate.dat"
  fi
  rm -f "$out" "$out".part "$out".part.ckpt "$out".lock "$CAND" "$CAND".part "$CAND".done "$COFREL"
  # relq (relations per unit q-width) sizes the next band: not from a chunk that lost its CPU-cofactored half
  have=$(( have + n + nl )); HAVE_AT[$slot]=$have; LANDED=$slot; [ "$rc" = 0 ] && [ "$COFOK" = 1 ] && relq=$(python3 -c "print($n/$w)")
  say "chunk $i q[$q,$qnext) +$n in ${dt}s ($(( n / (dt>0?dt:1) )) rel/s)$([ "$nl" -gt 0 ] && echo " + las $nl") -> slot $slot  total $have  load $(cut -d' ' -f1 /proc/loadavg)"
  q=$qnext
  # launch a probe if due and none is running (immutable chunk list); then feed incdup the new chunk
  probe_reap || true
  [ -n "$MATRIX_DIR" ] && break
  inc_consumed; inc_reap; probe_launch land; inc_start; las_poll
done
# a probe may still be running when the loop exits via MATRIX (it is the one that found it) -- no.
[ -n "$MATRIX_DIR" ] || die "sieve loop ended without a matrix"
inc_kill   # an incdup step for a later chunk is now useless: stop it (and its dup1/dup2) before freeing RAM
las_end    # ... and so is the CPU co-sieve: its running chunk and unlanded outputs are discarded
say "MATRIX from $MATRIX_DIR: $(grep -a VERDICT "$MATRIX_DIR/filter.out" | tail -1)"
# raw chunks are no longer needed (cado2msieve2 already extracted the purged relations): free ~20 GB
for k in $(seq 1 "$MAXCHUNKS"); do : > "$MEM/$(printf c%03d $k)"; done
for k in $(seq 0 15); do : > "$MEM/d1/$k/rel.0000$SL"; : > "$MEM/d2/$k/rel.0000$SL"; done
inc_free   # incremental parts + key shadows (~8 + ~2.5 GB at c160), unless already freed
[ -n "${BENCHPID:-}" ] && { wait "$BENCHPID" 2>/dev/null; BENCHPID=""; }
rm -f "$WORK"/chunk.dat* "$WORK"/cand.dat* "$WORK/cofac.rel" "$STOP" "$WORK/freerel/renumber$SL" "$WORK/freerel/freerel.gz" "$WORK/fb.roots1" "$WORK/raw.p"
for d in "$WORK"/probe*; do [ "$DS_MODE" = cado ] && [ "$d" = "$MATRIX_DIR" ] && continue; rm -f "$d"/purged.gz* "$d"/history*; done
say "/tmp before downstream: $(df -Pm "$WORK" | awk 'NR==2{print $3" MB used, "$4" MB free"}')"

# =============================================================================================
DS="$WORK/ds"; mkdir -p "$DS"
DS_RC=0
if [ "$DS_MODE" = cado ]; then
  echo "### [4/4] downstream (CADO-native): replay -> GPU block Lanczos on the CADO matrix -> characters -> CADO sqrt $(date +%T)"
  cd "$MSV" || die "no $MSV"
  SPARSE="$MEM/c.sparse.bin" REPLAY="$CF/replay" NT="$NCPU" LPB0=$LPB LPB1=$LPB \
    bash ./ds_cado.sh "$MATRIX_DIR" "$WORK/poly.cado" "$N" "$DS" || DS_RC=$?
else
  echo "### [4/4] downstream: msieve -nc2 (cado_filter=1) -> GPU block Lanczos -> CADO sqrt $(date +%T)"
  mv -f "$MEM/rels_out.cyc" "$DS/m.dat.cyc" && mv -f "$MEM/rels_out.purged" "$DS/m.dat.purged" || die "cannot stage cycles/purge map"
  cd "$MSV" || die "no $MSV"
  SKIP_NC1=1 NT="$NCPU" ./msv_launcher "$MEM/rels_out" "$WORK/c.fb" "$DS" ./msvrun_gpu.sh "$N" "$DS" || DS_RC=$?
fi
[ "$DS_RC" = 0 ] || say "downstream rc=$DS_RC -- continuing to factor extraction"
say "/tmp after downstream: $(df -Pm "$WORK" | awk 'NR==2{print $3" MB used"}')"
grep -aE 'matrix is|LA done|dependencies|FACTOR on dep|GPU linear algebra' "$DS/msieve.log" "$DS"/*.log 2>/dev/null | tail -6 | sed 's/^/  /'
echo "### FACTORS $(date +%T)  (total $(( $(date +%s)-T0 ))s)"
python3 - "$N" "$DS/msieve.log" <<'PY' || true
import sys, os, re
N = int(sys.argv[1]); fs = set()
if os.path.isfile(sys.argv[2]):
    for m in re.findall(r'factor:\s*([0-9]+)', open(sys.argv[2]).read()):
        f = int(m)
        if 1 < f < N and N % f == 0: fs.add(f)
if fs:
    p = min(fs); q = N // p
    print(f"p={p}"); print(f"q={q}"); print(f"FACTORED: p*q==N {p*q == N}")
else:
    print("NO FACTOR (downstream produced no factor -- see stage logs above)")
PY
