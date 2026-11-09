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

# ---- CONTAINER=1: the validator profile (docs/rsa560_container_audit.md) ----------------------
# The submission image sets CONTAINER=1 (docker/Dockerfile.c169-*); molab/sandbox runs never do, so nothing below changes
# them. Every knob is only DEFAULTED here (an explicit environment value still wins). Validator constraints this answers:
# read-only root + /tmp the only writable place (10 GiB tmpfs, noexec,nosuid), --network none, non-root user, 85 GB
# cgroup memory (memfds AND tmpfs pages are both charged to it), nofile 4096.
CONTAINER=${CONTAINER:-0}
if [ "$CONTAINER" = 1 ]; then
  # the measured production configuration (gpu/run_key1b.sh + gpu/key1i.env, key12f/key17f runs) -- not new tuning
  # polyselect as measured in every production run (gpu/s92_fresh.sh sets PS_BATCHES=3; the script default is 2); PS_ADAPT builds on it
  : "${PS_BATCHES:=3}"
  # RSA560_LPB1=33 (the submission image's ONE large-prime switch, docker/Dockerfile.c169-gpu ENV): side-1 lpb 33 with
  # lambda1 2.0379 (same side-1 survivor bound as 2.1 x 32; gpu/key1l.env, blind GCP keys 27/28: 3.99 / 3.83 h).
  # 32 (or unset) = the 31/32 production profile below. Only when LPB1 is not given explicitly (the small-key
  # profile of breaking_rsa.py sets its own LPB1/LAMBDA1 and is unaffected).
  # S3 regime switches (docker ENV RSA560_MFB1 / RSA560_LAMBDA1 / RSA560_ALIM, 2026-10-25 screens sb100-103): like
  # RSA560_LPB1, each applies ONLY when the variable is not given explicitly -- the small-key profile of breaking_rsa.py
  # (c100/c115 gates, the workbench's c330) sets MFB1/LAMBDA1/ALIM itself and must not inherit them (a plain ENV MFB1=66
  # ALIM=600000000 gave the c100 gate mfb1 66 / lim1 600M and a relation-gate failure, S3 chain attempt 1).
  [ -n "${RSA560_MFB1:-}" ] && [ -z "${MFB1:-}" ] && MFB1=$RSA560_MFB1
  [ -n "${RSA560_LAMBDA1:-}" ] && [ -z "${LAMBDA1:-}" ] && LAMBDA1=$RSA560_LAMBDA1
  [ -n "${RSA560_ALIM:-}" ] && [ -z "${ALIM:-}" ] && ALIM=$RSA560_ALIM
  case "${RSA560_LPB1:-32}" in
    33) [ -z "${LPB1:-}" ] && { LPB1=33; : "${LAMBDA1:=2.0379}"; } ;;
    32) ;;
    *) die "RSA560_LPB1 must be 32 or 33 (got '${RSA560_LPB1}')" ;;
  esac
  # RSA560_LPB0=32 (default off = 31): side-0 lpb 32 with lambda0 2.0352 (same side-0 survivor bound 521 as 2.1 x 31;
  # gpu/key1m.env, blind GCP key 31 at 32/33: 14,774 s wall, 485M raw, N 16.8M; docs/research/agents_2026-10-25/regime.md).
  # Same scope rule as RSA560_LPB1: only when LPB0 is not given explicitly (the small-key profile sets its own LPB0).
  case "${RSA560_LPB0:-31}" in
    32) [ -z "${LPB0:-}" ] && { LPB0=32; : "${LAMBDA0:=2.0352}"; } ;;
    31) ;;
    *) die "RSA560_LPB0 must be 31 or 32 (got '${RSA560_LPB0}')" ;;
  esac
  # RSA560_CPU_LAS=1 (docker ENV; default unset = exactly the profile below; docs/research/agents_2026-10-26/idle_cpu.md):
  # CADO las co-sieves on the cores the GPU sieve leaves idle (the LAS=1 block below, built for the c160 bench siever and
  # never run beside glas at c169). Measured ceiling at c169 (s37/s49/s50, molab 20 vCPU): glas -t 16 needs ~3.3 cores,
  # las A 29 -t 15 gives ~630 rel/s alone and 240-412 beside glas (+0.6..1.1% of the GPU's 37.3k rel/s), glas +0.1..0.9%
  # slower -- a net +0.5% there; the 24-vCPU validator layout is unmeasured (gpu/r13_cosieve.sh is the decisive test).
  # Profile: LAS_A 29 (A 28 = 145 vs 234 rel/s, s50), LAS_QMIN 20M (above the GPU's final q, ~18.5-19.5M: las is never
  # retired and the GPU leaps over its band if it does get there), the las thread budget from the 24-CPU quota (LAS_NCPU;
  # the sieve-phase budget NCPU stays 20), glas host threads RSA560_CPU_LAS_GT (default 4: s49 span identical at -t 2..16;
  # s37e las +70% beside glas -t 4 vs -t 16). Every value is a default: an explicit LAS_*/SIEVE_T still wins.
  if [ "${RSA560_CPU_LAS:-0}" = 1 ]; then
    : "${LAS:=1}" "${LAS_A:=29}" "${LAS_QMIN:=20000000}" "${LAS_NCPU:=24}" "${SIEVE_T:=${RSA560_CPU_LAS_GT:-4}}"
  fi
  : "${NCPU:=20}" "${SIEVE_T:=16}" "${SIEVE_HOST_T:=4}" "${INCDUP:=1}" "${INC_PAR:=3}" "${PROBE_NT:=12}" "${DUP2_PAR:=4}" \
    "${INC_TPG:=3}" "${LATE_SECS:=0}" "${INC_TIMEOUT:=4000}"
  # RSA560_CPU24=1 (docker ENV; default unset = exactly the profile above): the CPU-bound stages that run ALONE with the GPU
  # idle -- polyselect tail (sopt/ropt), makefb, the stop-time incdup + purge, merge, replay, characters/abx and the sqrt
  # window (SQRT_TPD = NT/SQRT_WIN) -- take all 24 vCPUs of the validator's quota (STAGE_NT) instead of the sieve phase's
  # NCPU=20 concurrency budget, which stays 20 (glas host threads + incdup + probes were measured at that budget). Every
  # value is a default: an explicit NCPU/PS_NCPU/FB_NT/STOP_FINAL_NT/MERGE_NT/REPLAY_T/DS_NT still wins. Candidate only --
  # no measured c169 adoption (docs/research/agents_2026-10-25/container.md, idea 1; Codex's validator_cpu_candidate.md).
  if [ "${RSA560_CPU24:-0}" = 1 ]; then
    : "${STAGE_NT:=24}"
    : "${PS_NCPU:=$STAGE_NT}" "${FB_NT:=$STAGE_NT}" "${STOP_FINAL_NT:=$STAGE_NT}" "${MERGE_NT:=$STAGE_NT}" "${REPLAY_T:=$STAGE_NT}" "${DS_NT:=$STAGE_NT}"
    export REPLAY_T   # ds_cado.sh reads it from the environment (replay -t ${REPLAY_T:-$NT})
  fi
  # RAM: raw chunks are truncated as incdup commits them (without it ~42 GB of raw relations stay resident at c169)
  : "${FREE_RAW:=1}"
  # /tmp: the probes' purged + merge history live in memfds (MEMFD_PURGED below); dependency files gzip'd (ABX_GZ)
  : "${MEMFD_PURGED:=1}" "${PURGE_PLAIN_OUT:=1}"
  # the purge must be the ordered-output build (gpu/purge_parts_patch.py): the stock pass 2 wrote 7-8 GB of temporary
  # parts BESIDE the memfd output, i.e. on this 10 GiB /tmp (S1 validation failure 2026-10-25); cado_filter160.sh fails closed
  export PURGE_ORDERED_REQUIRED=${PURGE_ORDERED_REQUIRED:-1}
  # 100 chunk slots (key12f/key17f used ~46) keep memhold under nofile 4096 with the 2 x PP_SLOTS purged/history memfds
  : "${MAXCHUNKS:=100}"
  # downstream: abx writes c.dep.<k>.gz (plain c169 deps are ~0.6 GB each, 10 = ~6 GB of the 10 GiB /tmp); /dev/shm is
  # Docker's 64 MB default, so the /dev/shm gunzip paths (only taken for a purged.gz) are switched off
  export ABX_GZ=${ABX_GZ:-1} PURGED_PLAIN=${PURGED_PLAIN:-0} CHARS_PLAIN=${CHARS_PLAIN:-0}
  # --network none: never checkpoint or save off the box (gpu/ckpt_push.sh, gpu/ds_save.sh)
  unset CKPT_URL CKPT_KEY CKPT_HF_REPO CKPT_ENV HF_TOKEN DS_SAVE; CKPT_PUSH=/nonexistent
  # read-only root: HOME and every cache under the writable tmpfs. Caches hold DATA only (CuPy reads its cubins and
  # cuModuleLoadData()s them from memory; nothing is ever exec'd or dlopen'ed from /tmp, which is noexec)
  [ -w "${HOME:-/nonexistent}" ] || export HOME=${TMPDIR:-/tmp}
  export TMPDIR=${TMPDIR:-/tmp} CUPY_CACHE_DIR=${CUPY_CACHE_DIR:-${TMPDIR:-/tmp}/.cupy} \
         CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${TMPDIR:-/tmp}/.nv} PYTHONDONTWRITEBYTECODE=1
  # GPU NTT multiply in sqrt (A19): the image builds it next to sqrt; loaded by CADO's par_mul via dlopen (from /app)
  [ -z "${PAR_MUL_GPU:-}" ] && [ -s "$MSV/gpu_mul.so" ] && export PAR_MUL_GPU=$MSV/gpu_mul.so
fi
[ -n "${PAR_MUL_GPU:-}" ] && [ ! -s "$PAR_MUL_GPU" ] && { say "WARNING: PAR_MUL_GPU=$PAR_MUL_GPU missing -- CPU-only sqrt"; unset PAR_MUL_GPU; }
# MEMFD_PURGED=1 (default 0; CONTAINER=1 sets it): each probe directory's purged file and merge history are symlinks to
# memhold memfds $MEM/pp/<iii>/{purged,history} (probe i -> slot i, PP_SLOTS slots). At c169 a plain purged file is
# 7.1-7.9 GB and the cost rule keeps one probe's file while the next probe writes its own: ~16 GB, which cannot sit on
# a 10 GiB /tmp. CADO purge/merge write THROUGH a symlinked output (vendor/cado-symlink-writethrough-cognition.patch);
# every deletion goes through rmf (truncate, then unlink), so a dropped file gives its RAM back.
MEMFD_PURGED=${MEMFD_PURGED:-0}; PP_SLOTS=${PP_SLOTS:-48}
# RES_LOG=1 (default with CONTAINER=1): one "[res]" line per chunk landing and stage -- /tmp used (and its max so far),
# the cgroup's memory.current / memory.peak, the memfd pool's size -- the numbers the validator limits apply to
RES_LOG=${RES_LOG:-$CONTAINER}; RES_TMP_MAX=0

# ---- parameters (env-overridable) -----------------------------------------------------------
NCPU=${NCPU:-$(python3 -c "
try:
    q,p=open('/sys/fs/cgroup/cpu.max').read().split()
    print(max(1,round(int(q)/int(p))) if q!='max' else __import__('os').cpu_count())
except Exception: print(__import__('os').cpu_count())")}
# Research-only stage override: leave the concurrent sieve/incdup/probe budget
# untouched while testing more workers during polynomial selection. Default is
# bit-for-bit the previous thread allocation; candidate partitioning can change
# with PS_NCPU, so compare the selected polynomial as well as elapsed time.
PS_NCPU=${PS_NCPU:-$NCPU}
case "$PS_NCPU" in ''|*[!0-9]*) die "PS_NCPU must be a positive integer" ;; esac
[ "$PS_NCPU" -ge 1 ] && [ "$PS_NCPU" -le 256 ] || die "PS_NCPU must be in 1..256"
# c169 production set, measured on the RTX PRO 6000 (docs/rsa560_ledger.md "X2 stack, part 2", 2026-10-17): lpb 32/33,
# 2 large primes (mfb 63/65), lambda 2.1, ncurves 17/17, the 530 lims 100M/150M, with glas's adjust-strategy port
# (GLAS_ADJUST=3) -- ~15,200 s of sieving at 420M raw relations. The earlier default (c173cog: 30/31, mfb 60/90)
# overflows glas's GPU buffers (ledger bake-off 4). Every knob is per side; the 530 script assumed lpb0 = lpb1.
LPB0=${LPB0:-31}; LPB1=${LPB1:-32}; MFB0=${MFB0:-62}; MFB1=${MFB1:-64}   # production profile since 2026-10-23: 31/32 beats 32/33 on equal terms (s63); mfb 62/64 +2% rel/s (s69/s75)
LAMBDA0=${LAMBDA0:-2.1}; LAMBDA1=${LAMBDA1:-2.1}; NCURVES0=${NCURVES0:-17}; NCURVES1=${NCURVES1:-17}
# gps1 fingerprint collision sort (gpu/gps1_fastsort_patch.py, in the bootstrap + image builds): search -54%, identical
# matches/polynomial (r7_gps1_fastsort sb80/sb81, r7_ps_fastsort key 22: 538 -> 334 s). An unpatched gps1 ignores it.
export GPS1_FASTSORT=${GPS1_FASTSORT:-2}
# LA hot-B word-major accumulator (segspmv.py LA_SEG_HOTT, r8): 52.78 -> 50.71 ms/it on key17f, identical iteration-800
# Lanczos state hashes in all 4 A/B runs (gpu/r8_hott_ab.sh, sb81, data/rsa560/audit_r8/hott_ab_sb81.txt)
export LA_SEG_HOTT=${LA_SEG_HOTT:-1}
# LA deferred dense constraints (gpu_la.py LA_DEFER_ROWS, Codex review 8 rank 2): the 64 heaviest rows leave the Lanczos
# iteration and are enforced in the final combine against the original matrix. key17f 52.745 -> 51.275 ms/it alone, stacks with
# HOTT (50.43 -> 48.88); key12f HOTT+defer -3.85 ms/it; full solve with HOTT on sb84: VERIFY all rows, 64 rank-64 vectors, FACTOR.
# Adopted 2026-10-24 18:25 MDT. The sb84 run needed 8 sqrt dependencies, but ds_cado feeds sqrt random GF(2) mixes of the
# kernel (SQRT_MIX=1): success is a linear functional on the kernel, non-zero because that run factored, so every mix succeeds
# with p = 1/2 independently -- 8 was 1/128 bad luck, not a dependency-quality defect.
export LA_DEFER_ROWS=${LA_DEFER_ROWS:-64}
# SQRT_WIN=2 (2026-10-24, gpu/r7_sqrtwin.sh on key17f, distinct dependencies, sb80): pairs at OMP=10 take ~201 s vs 147 s for
# one dependency at OMP=20 -> key17f's 4 dependencies 590 -> 402 s; mean (p=1/2) ~294 -> ~269 s, and it halves the bad-luck tail
export SQRT_WIN=${SQRT_WIN:-2}
RLIM=${RLIM:-200000000}; ALIM=${ALIM:-450000000}   # lim 200M/450M: +8.8% / +13.4% / +15.1% rel/s at q 2.2M / 12.5M / 22.4M vs 100M/150M, no overflow (sandbox #20, gpu/s20_lim.sh)
export GLAS_RHITS=${GLAS_RHITS:-0}   # flat resieve-hit list: -2.6% ms/q, same relations (sandboxes #12/#13)
export GLAS_FILL_COPRIME=${GLAS_FILL_COPRIME:-3}   # fill also drops 5,7,11,13 | gcd(i,j) hits (gpu/glas_coprime_patch.py): -4.7% wall, same relations (sandbox #18)
export GLAS_REGION_PREFETCH=${GLAS_REGION_PREFETCH:-1}   # region kernel prefetches its bucket into L2 (gpu/glas_prefetch_patch.py): -1.5% wall over 4 alternating pairs, same relations (sandbox #19)
export GLAS_XFUSE=${GLAS_XFUSE:-1}   # transform fused into the large-tier fill (gpu/glas_xfuse_patch.py): -7.9% wall, identical bucket entries (sandbox #21)
export GLAS_XFUSE_GRID=${GLAS_XFUSE_GRID:-752}   # grid-stride fused fill capped at 752 blocks (gpu/glas_xfgrid_patch.py): s82 sb59 paired span 77.52 -> 76.34 s (-1.5%), wall -1.9%, relation sets identical
export GLAS_NARROW=${GLAS_NARROW:-1}; export GLAS_IDIV=${GLAS_IDIV:-1}   # exact bounded-width norms + integer exact division in the survivor filter (gpu/glas_narrow_patch.py): s86 sb61 paired span -1.70%, wall -1.77%, relation sets within NTOL (diff 0)
export GLAS_P1=${GLAS_P1:-3}   # survivor-indexed factor lists + exact rational-first cascade (gpu/glas_p1_patch.py, review2 3.1): paired span/wall -0.55/-1.44% (q 12.5M, s94) and -1.03/-0.74% (q 19M, s95), relations identical, relcheck clean
export GLAS_REGION_HALF=${GLAS_REGION_HALF:-1}   # even-row half sieve (gpu/glas_region_patch.py): -1.1% span alone, exact (sandbox -3, 3 pairs)
export GLAS_REGION_RX=${GLAS_REGION_RX:-1}   # side-1 resieve extraction while L2-hot (gpu/glas_region_patch.py): -1.2% alone, HALF+RX -2.8% span (82.69 vs 85.07 s), same relation set (sandbox -3)
export SQRT_L00=${SQRT_L00:-2}   # sqrt level-00 fragment products without CADO's per-product overhead (gpu/sqrt_l00_patch.py): 132 vs 765 s per dependency, same square root (sandbox #21)
export GLAS_ADJUST=${GLAS_ADJUST:-3}   # per-special-q basis/shape choice (pipeline_c169/vendor/glas-adjust-strategy.patch)
LOGI=${LOGI:-15}; JJ=${JJ:-16384}; QMIN=${QMIN:-2200000}
# SIEVER: glas (Cognition GPU las, CADO relation format, the production candidate) | bench (cuda-sieve,
# the 530 path) | las (CADO's CPU siever: the SAME orchestration exercised end to end on a CPU-only box).
# las and glas take the las command line; SIEVE_A is their -A (2^A cells per special-q; 29 = logI 15 x J 16384).
SIEVER=${SIEVER:-glas}; SIEVE_A=${SIEVE_A:-29}
GLAS=${GLAS:-$APP/glas/glas-gpu}
case "$SIEVER" in glas|bench|las) ;; *) die "SIEVER must be glas, bench or las (got '$SIEVER')";; esac
# GLAS_SESSION=1 (default 0; proposal 1 of docs/rsa560_codex_review.md, gpu/glas_session_patch.py, gpu/s84_session.sh):
# ONE glas process for the whole sieve (FB, CUDA context, device/pinned buffers, strategies initialised once). Bands go
# over a control FIFO ('BAND i q0 q1 out'); glas publishes each COMPLETE band atomically (out.tmp -> out, relation lines
# only) and answers 'ACK i nrels next_q' on an ack FIFO. The driver blocks on the ack (3-s read timeout = the old
# housekeeping cadence, no exit polling), sends band i+1 THE MOMENT band i is acknowledged (before landing i), lands the
# file with the reported count (no strip, no re-count), keeps the complete-band stop rule (an in-flight band is dropped
# at MATRIX, as today) and on a crash restarts the session from the last acknowledged band (once per band, as today's
# single resume). Needs a glas-gpu built with the session patch; GLAS_SESSION=0 is the unchanged per-band process path.
GLAS_SESSION=${GLAS_SESSION:-$([ "${SIEVER:-glas}" = glas ] && echo 1 || echo 0)}; SES_PID=""; SES_CFD=""; SES_AFD=""; SES_GEN=0; SES_PRE_I=""; SES_PRE_W=""; SES_PRE_QB=""; SES_PRE_T0=""; SES_N=0
if [ "$GLAS_SESSION" = 1 ]; then
  [ "$SIEVER" = glas ] || die "GLAS_SESSION=1 needs SIEVER=glas (got $SIEVER)"
  [ "${COFAC_CPU:-0}" = 0 ] || die "GLAS_SESSION=1 does not support COFAC_CPU=1 (bench only)"
fi
# c169 at lpb 32/33 (~420M raw relations, ledger X2 stack part 2): 8M-relation chunks (~53 chunks; 16 slices
# each = ~850 open files, under nofile 4096), 2M late chunks. (At the old 30/31 3LP set: 155M, 4M chunks.)
# 4M-relation chunks (~39 chunks), 1.5M late chunks so
# the MATRIX stop overshoots by ~1-2 min; MAXCHUNKS 100 keeps memhold at ~3,350 memfds under nofile 4096
CHUNK_RELS=${CHUNK_RELS:-8000000}; MAXCHUNKS=${MAXCHUNKS:-120}
# CHUNK_RELS_LATE: chunk size once `have` is within one chunk of FIRST_PROBE. Probes see whole committed
# chunks, so the MATRIX probe overshoots the crossover by ~half a chunk on average: 4M halves that
# (~140 s of sieve at c160). incdup/probe per-chunk overheads stay off the critical path.
# c169 two-phase stop (STOP_MIN_EXCESS 11M, ~525M raw): 2M late chunks from ~357M need ~84 slots on top of ~45 early
# ones -> 129 > MAXCHUNKS (key1f died "ran out of chunk slots (120)" at 508M raw); 4M keeps the run at ~87 slots.
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
# FIRST_PROBE ~87% of the expected crossover so that two SHORT probes land before it; PREDICT_PROBE=1 then
# extrapolates purge's singleton excess (linear in relations past the knee) from the last two SHORT probes
# and places the next probe at the predicted crossover x PRED_FACTOR, clamped to [PROBE_STEP/2, 4 x
# PROBE_STEP] past the last probe. Saves the 2-4 blind 3.5M steps the 530 run spent (~300-600 s).
# FIRST_PROBE_AUTO=1 (default 0 = off; docs/research/agents_2026-10-25/regime.md): when FIRST_PROBE is not given, place it at
# FIRST_PROBE_FRAC (default 0.92) x the regime's expected raw crossover from `stop_cost.py xover LPB0 LPB1` (measured
# 31/32 293M, 31/33 366M, 32/33 441M; modelled 31/34 434M, 32/34 514M, 33/33 506M, 33/34 588M). The fixed 270M default
# is 0.92 x the 31/32 crossover; at 31/33 it buys 1 and at 32/33 3-4 extra SHORT probes (key 31: 274/286/330/373/417M),
# each ~22 s of GPU slowdown (STOP_PROBE_TAX) plus 35-47 s of CPU, and it also ends the 8M early chunks 100-170M raw early.
if [ "${FIRST_PROBE_AUTO:-0}" = 1 ] && [ -z "${FIRST_PROBE:-}" ]; then
  FIRST_PROBE=$(python3 "$HERE/stop_cost.py" xover "$LPB0" "$LPB1" "${FIRST_PROBE_FRAC:-0.92}") || die "FIRST_PROBE_AUTO: stop_cost.py xover $LPB0 $LPB1 failed"
  say "FIRST_PROBE_AUTO: first probe at $FIRST_PROBE raw (lpb $LPB0/$LPB1, frac ${FIRST_PROBE_FRAC:-0.92})"
fi
FIRST_PROBE=${FIRST_PROBE:-270000000}; PROBE_STEP=${PROBE_STEP:-10000000}   # ~87% of the 420M raw crossover at 32/33
PREDICT_PROBE=${PREDICT_PROBE:-1}; PRED_FACTOR=${PRED_FACTOR:-}; PRED_TARGET_EXCESS=${PRED_TARGET_EXCESS:-}   # defaults resolved after STOP_MIN_EXCESS
# STOP_MIN_EXCESS (default 0 = off: the first probe with a matrix stops the sieve and merges inside the probe, as
# before). >0 = TWO-PHASE stop (gpu/s23_cp.sh merge/stopcurve; DS_MODE=cado only): probes run purge only
# (PURGE_ONLY=1 -- cado_filter160.sh exits before it consumes the incremental cache); a probe with a matrix whose
# singleton excess is < STOP_MIN_EXCESS counts as SHORT (the predictor then aims at PRED_TARGET_EXCESS, default
# STOP_MIN_EXCESS) and the sieve goes on; the first one >= it stops the sieve, and merge runs here with the GPU
# sieve gone, on MERGE_NT threads (default NCPU, vs the probe's PROBE_NT beside the sieve). STOP_MIN_EXCESS=1 = the
# two-phase merge alone (any matrix accepted). Over-sieving pays while the LA saved per extra relation exceeds its
# sieve cost: at c169 ~2.3M excess (~2.9% of the singleton columns, estimate from key 1 / key 1b).
# TP_KEEP=1 (measurement only, gpu/s23_cp.sh stopcurve): a probe counted as SHORT keeps its purged file as keep.purged.gz
# default 11M since the key-1e stop curve (sandbox 2, 2026-10-21, data/rsa560/audit_s23/stopcurve_key1e.txt): TD-130 N fell
# 19.45M -> 17.79M from 486M to 502M raw (excess 4.24M -> 7.69M, +0.215M per M raw), LA ~53 s saved per M raw vs ~27.5 s
# of late sieve per M raw (reference box): the 486 -> 502M step alone nets ~-420 s and the slope had not flattened.
# 11M (~517M raw) is a short extrapolation; the next full run measures it.
STOP_MIN_EXCESS=${STOP_MIN_EXCESS:-11000000}; MERGE_NT=${MERGE_NT:-$NCPU}; TP_KEEP=${TP_KEEP:-0}; TP_SEEN=0; PROBE_LATE_NT=${PROBE_LATE_NT:-16}
# PURGE_PLAIN_OUT=1 (default): purge writes its output plain ($PG = purged, else purged.gz; exported to cado_filter160.sh
# as PURGED_NAME): merge and replay then take CADO's parallel reader (s43, key1bds: merge -t 20 172.3 -> 93.0 s) and the
# PURGED_PLAIN gunzip is skipped; purge itself skips the gzip pipe (s44, key1bds -t 20: 212.8 -> 180.6 s, same
# relations). Peak memory falls (plain 7.4 GB alone vs gz 4.2 GB + a plain copy in /dev/shm). c100 smoke sb39 FACTOR OK.
PURGE_PLAIN_OUT=${PURGE_PLAIN_OUT:-1}; [ "$PURGE_PLAIN_OUT" = 1 ] && PG=purged || PG=purged.gz; export PURGED_NAME=$PG
# STOP_RULE: excess (default: the STOP_MIN_EXCESS threshold above, unchanged) | cost (pipeline_c169/stop_cost.{sh,py}; needs
# the two-phase purge-only probes, i.e. STOP_MIN_EXCESS > 0, which stays the backstop). cost: every matrix probe gives
# (raw, purged rows); N ~ STOP_NFAC x rows, LA ~ N/255.5 its x (STOP_C_NNZ_PS x STOP_NNZ_ROW x N + STOP_DENSE_NS x N) +
# STOP_POST_US x rows; the secant between the last two matrix probes, decaying by STOP_LAMBDA per M raw, is the LA saved per
# raw relation; the sieve stops when the next step's saving < its sieve seconds at the measured rel/s (+ STOP_PROBE_TAX).
# Why: a fixed excess overshoots -- key1g probes 4 -> 5 were 24M raw / ~1,336 s of sieve apart, and the s61 stop curve on its
# own chunks (N 15.85 / 14.18 / 12.90M at 301 / 317 / 333M raw) puts the cost optimum near ~308M (LA saves ~58 s per M raw
# there vs ~53.5 s of sieve per M raw at 18.7k rel/s), not at 11M excess (~329M). The model's defaults: gpu/s66_stoprule.sh.
STOP_RULE=${STOP_RULE:-cost}   # default since 2026-10-23 (s66, key1h)
STOP_NFAC=${STOP_NFAC:-0.25}; STOP_C_NNZ_PS=${STOP_C_NNZ_PS:-25}; STOP_DENSE_NS=${STOP_DENSE_NS:-0.2}; STOP_NNZ_ROW=${STOP_NNZ_ROW:-$(( ${TD:-130} + 1 ))}
STOP_POST_US=${STOP_POST_US:-1.8}; STOP_LAMBDA=${STOP_LAMBDA:-0.023}; STOP_MARGIN=${STOP_MARGIN:-0}; STOP_PROBE_TAX=${STOP_PROBE_TAX:-20}
# STOP_LA_SCALE (default 1 = unchanged): this card's LA speed relative to the GCP g4 card the rule's LA constants were
# calibrated against; the rule multiplies its LA term by it (slower LA -> oversieve more). molab RTX PRO 6000 boxes
# measure x1.28 at N 17M (audit_ops/laparts_sb111.txt); modelled worth -33 s/key on such a card, 0 on GCP, never set
# it to the TRUE kappa ratio (0.73 x): the rule's x1.37 LA overprice cancels its secant decay (slow_card.md).
STOP_LA_SCALE=${STOP_LA_SCALE:-1}
STOP_RATE=${STOP_RATE:-}; STOP_RATE_N=${STOP_RATE_N:-4}; STOP_RATE_DEFAULT=${STOP_RATE_DEFAULT:-18700}   # key1g 31/32 chunks 50-60, gpu_ratio 1.159 box
STOP_PRED_EXCESS=${STOP_PRED_EXCESS:-1000000}; STOP_PREDICTIVE=${STOP_PREDICTIVE:-1}; STOP_PROBE_EVERY=${STOP_PROBE_EVERY:-1}
# STOP_SHAPE=1 (STOP_RULE=cost only; default 0 = the secant rule above, unchanged; docs/research/agents_2026-10-26/stop_rule.md):
# stop_cost.py prices the stop from a fitted curve instead of the secant of the last two probes: crossover X_c from the
# matrix probes' singleton-excess line (one point: knee slope STOP_SX), N(A) = N0 (1 + u/STOP_SHAPE_A)^-STOP_SHAPE_B with
# u = A/X_c - 1 (33 purge-only points of keys 25-29/31/arm B to 0.67% rms), LA = STOP_KAPPA x (N/1e6)^2 (9.62-10.02 s per
# (M rows)^2 measured on 7 keys). ONE matrix point decides (STOP_MINPTS 1): a first matrix already past the optimum stops
# at its probe instead of one chunk later; with the curve known, STOP_PROBE_EVERY=2 is enough after the first matrix
# (in-container the matrix-era probes cost the GPU +4.4% ms/q, arm B). KAT: pipeline_c169/tests/test_stop_shape.py.
STOP_SHAPE=${STOP_SHAPE:-0}; STOP_KAPPA=${STOP_KAPPA:-9.9}; STOP_SX=${STOP_SX:-0.22}; STOP_SHAPE_A=${STOP_SHAPE_A:-0.102}; STOP_SHAPE_B=${STOP_SHAPE_B:-0.44}
STOP_MINPTS=${STOP_MINPTS:-$([ "$STOP_SHAPE" = 1 ] && echo 1 || echo 2)}
# STOP_FINAL_NT: purge threads of the final (stop-time) purge, the sieve gone; STOP_FINAL_GROUPS (default INC_GROUPS): dup2
# groups of the final incdup step, ALL concurrent at nice 0 (beside glas -t 16 the budget gives par 1 at nice 15, so the two
# groups ran one after the other: key1g chunk 60 -> probe 5 lag 97 s). Each dup2 loads the renumber table: 4 = 4 copies.
STOP_FINAL_GROUPS=${STOP_FINAL_GROUPS:-}; STOP_FINAL_NT=${STOP_FINAL_NT:-$NCPU}
# DELTA_PURGE=1 (STOP_RULE=cost only; default 0 = off): every purge-only MATRIX probe also leaves a SINGLETON-ONLY purge
# of its input (purge -nsteps 0; cado_filter160.sh DELTA_SING) in probe<i>/sing/; the stop_final handoff then purges that
# base + only the incdup steps committed after it (DELTA_BASE) instead of all ~210M unique relations. Verified in the
# CADO source (filter/purge.cpp pass 2 prints rel.line = the input line verbatim; filter_io skips the '#' header): purge
# re-reads its own output. The probe's CLIQUE-pruned output is not reusable (its excess is spent -> N of the probe).
# Approximate: relations the base dropped as singletons that a later relation would revive stay dropped
# (gpu/s71_deltapurge.sh measures N vs the full purge). Live: the base runs in the background (nice 10, DELTA_SING_NT
# threads, ~8 GB RSS) after the probe's verdict; the handoff waits <= DELTA_WAIT s for a running one, else kills it and
# uses the previous base (or the full purge). Any delta-purge failure falls back to the full purge.
DELTA_PURGE=${DELTA_PURGE:-0}; DELTA_SING_NT=${DELTA_SING_NT:-8}; DELTA_WAIT=${DELTA_WAIT:-60}; export DELTA_SING_NT
DPS=0; [ "$DELTA_PURGE" = 1 ] && [ "$STOP_RULE" = cost ] && DPS=1
# SPEC_MERGE=1 (STOP_RULE=cost + DS_MODE=cado only; default 0 = off; codex review proposal 4, pipeline_c169/spec_merge.sh):
# after each purge-only MATRIX probe the cost rule sieves on from, merge TD + replay that immutable snapshot in the
# background (nice SPEC_NICE, SPEC_NT threads on the last physical cores of the cpuset, RSS-budgeted, one at a time); at
# the stop, `stop_cost.py spec` compares C_ready = LA(N_ready, W_ready) (+ the remaining preparation of one in flight)
# with C_new = handoff + merge + replay + LA(N_new, W_new) and uses the prepared matrix (ds_cado.sh SKIP_REPLAY=1 with a
# matching SKIP_EPOCH) or proceeds exactly as SPEC_MERGE=0. gpu/s97_specmerge.sh replays key1g with 1 vs 0.
SPEC_MERGE=${SPEC_MERGE:-0}; SPEC_USE=""
[ "$SPEC_MERGE" = 1 ] && { [ "$STOP_RULE" = cost ] && [ "${DS_MODE:-cado}" = cado ] || { echo "WARNING: SPEC_MERGE=1 needs STOP_RULE=cost and DS_MODE=cado -- off"; SPEC_MERGE=0; }; }
# predictor target: two-phase aims at STOP_MIN_EXCESS with no raw margin (the excess threshold is the margin); before
# 2026-10-21 PRED_TARGET_EXCESS defaulted to 0 here, so once a matrix existed the prediction was skipped and probes fell
# back to PROBE_STEP: key1f probe 13 (9.93M @ 512M) -> next probe at 536M (15.7M), stop at 548M raw instead of ~517M.
# STOP_RULE=cost: the predictor only has to find the FIRST matrix (STOP_PRED_EXCESS, just past the crossover); probes
# then follow every chunk and the cost rule, not an excess target, places the stop.
if [ "$STOP_RULE" = cost ]; then PRED_TARGET_EXCESS=${PRED_TARGET_EXCESS:-$STOP_PRED_EXCESS}; PRED_FACTOR=${PRED_FACTOR:-1.0}
elif [ "$STOP_MIN_EXCESS" -gt 1 ]; then PRED_TARGET_EXCESS=${PRED_TARGET_EXCESS:-$STOP_MIN_EXCESS}; PRED_FACTOR=${PRED_FACTOR:-1.0}
else PRED_TARGET_EXCESS=${PRED_TARGET_EXCESS:-0}; PRED_FACTOR=${PRED_FACTOR:-1.02}; fi
[ "$STOP_MIN_EXCESS" -gt 0 ] 2>/dev/null && [ "$PRED_TARGET_EXCESS" = 0 ] && PRED_TARGET_EXCESS=$STOP_MIN_EXCESS
# LATE_SECS: wall seconds after which a still-matrixless run drops the prediction margin and halves
# PROBE_STEP (0 = off). 9000 s leaves ~4,000 s for the downstream inside FULL_RUN_SECS 13,000.
LATE_SECS=${LATE_SECS:-9000}; LATE_ON=0
declare -a PRB_R PRB_E
PROBE_NT=${PROBE_NT:-$(( NCPU > 6 ? NCPU - 4 : 2 ))}; DUP2_PAR=${DUP2_PAR:-6}; TD=${TD:-130}
PS_SECS=${PS_SECS:-300}; PS_MAX=${PS_MAX:-900}; PS_MIN_CANDS=${PS_MIN_CANDS:-200}   # 300 s: gps1 finds the same best poly at 300 and 1,200 s (X5); sopt-9 recovers quality
ROPT_K=${ROPT_K:-20}; ROPT_MAX=${ROPT_MAX:-420}   # top-20 at effort 6 (s46g, keys 4-11: ~120 s vs ~95 s for top-12; key 9 +4.1% E from a rank 17-20 candidate, others equal); 40 at effort 10 took 668 s (W7)
# ROPT_EFF: do NOT lower it to 5. Laptop, key 1: the polys that became c0 / c1 / c2 score the same at
# effort 5, 10 and 32 (4.162 / 4.099 / 3.814e-11), but the SOPT_EFF-re-optimized c0 reaches 4.233e-11 only at
# effort 10; at effort 5 / 20 / 32 it gets 4.011 / 4.068 / 4.009e-11. ropt's result is NOT monotone in effort on
# that poly (+-5%), so part of the sopt gain on key 1 is ropt scatter; effort 10 reproduces the glas-tested poly.
# Default 6 since 2026-10-22 (s46b-f, sopt-9 + top-12, 11 keys): effort 6 keeps the effort-10 poly on keys 1-3 and 6-11
# and beats it on keys 4/5 (1.246/1.233e-9 vs 1.186/1.173e-9, +5.1%), never worse, ~50 s faster (ropt ~150 -> ~95 s).
ROPT_EFF=${ROPT_EFF:-6}
# SOPT_EFF (default 9; 0 = off): before topk, re-size-optimize the SOPT_K best candidates of raw.p with CADO sopt
# at -sopteffort SOPT_EFF (the search itself runs CADO's default effort 0, gps1 its own GPU size-optimizer) and
# append the results to raw.p next to the originals (topk drops identical polys). Key 1: c0's raw form exp_E
# 45.23 -> 45.09, after ropt 4.162 -> 4.233e-11 (standard scale 4.349e-13 vs c1 4.198e-13); glas sandbox #20:
# +2.5% rel/s at q 12.5M, +5.1% at q 5M vs c1. gps1's best for the same c5 (85680) was exp_E 45.48 / 4.011e-11.
# ~0.37 s per candidate on one core (laptop), split over NCPU; SOPT_MAX caps the wall. Never fatal.
SOPT_EFF=${SOPT_EFF:-9}; SOPT_K=${SOPT_K:-500}; SOPT_MAX=${SOPT_MAX:-120}; SOPT_BIN=${SOPT_BIN:-$CF/sopt}
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
INC_PAR_PROBE=${INC_PAR_PROBE:-4}; INC_GROUPS=${INC_GROUPS:-2};   # 2 + the decompressed renumber memfd: 111-122 s per 7.7M chunk vs 209-226 at 4 + gz (sandboxes #12/#13)
 INC_TIMEOUT=${INC_TIMEOUT:-1200}
# COFAC_CPU=1 (default): the GPU sieve does NOT cofactorise (bench without --cofactor). It writes its 2LP
# candidates (--candidates) and cofac_cpu (CADO facul ECM/P-1/P+1, rho fallback) splits them on COFAC_T
# host threads WHILE the band runs (follow mode: it tails the candidates .part), so a chunk lands seconds
# after its band ends. Measured c160 q=5M: 26.2 vs 28.2 ms/q (-7%), relation SET identical to --cofactor,
# CPU ~0.7 core. COFAC_CPU=0 = the GPU cofactorisation queue (previous production path).
# cofac_cpu is a 2LP splitter (<= 64-bit cofactors): only the bench path at 2LP can use it. With 3LP
# (mfb1 90) bench cofactorises on the GPU (its ECM queue); las/glas cofactorise with CADO facul themselves.
COFAC_CPU=${COFAC_CPU:-$([ "$SIEVER" = bench ] && [ "$MFB0" -le 64 ] && [ "$MFB1" -le 64 ] && echo 1 || echo 0)}; COFAC_T=${COFAC_T:-2}
# SIEVE_T: threads the siever itself uses (las/glas: facul cofactorisation + host work; bench: its host
# thread). SIEVE_HOST_T is what the incdup/probe thread budget subtracts for it.
if [ "$SIEVER" = bench ]; then SIEVE_HOST_T=${SIEVE_HOST_T:-2}; SIEVE_T=${SIEVE_T:-$SIEVE_HOST_T}
else SIEVE_T=${SIEVE_T:-$(( NCPU > 6 ? NCPU - 4 : 2 ))}; SIEVE_HOST_T=${SIEVE_HOST_T:-$SIEVE_T}; fi
INC_TPG=${INC_TPG:-5}
# FREE_RAW=1: truncate each raw chunk's memfd as soon as incdup commits it (its deduplicated, renumbered
# part lives in dd/; purged.gz carries (a,b) for sqrt). Halves peak RAM, but removes the one-shot
# dup1/dup2 fallback: an incdup failure is then fatal. Requires DS_MODE=cado (MERGE_ONLY probes never
# read raw chunks) and INCDUP=1.
FREE_RAW=${FREE_RAW:-0}
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
# Default ON only beside the bench siever (the c160 production path). Beside glas it is OFF by default: glas's -t 16 host
# threads are ~3.3 cores busy (s49), so the cores ARE idle, but the measured net at c169 is only ~+0.5% (s37, molab
# 20 vCPU) -- see RSA560_CPU_LAS in the CONTAINER=1 profile and docs/research/agents_2026-10-26/idle_cpu.md. LAS=1 with
# SIEVER=glas runs the same code path (GLAS_SESSION=1 polls las_poll from ses_band; ses_prelaunch stays off with LAS=1).
LAS=${LAS:-$([ "$SIEVER" = bench ] && echo 1 || echo 0)}; LAS_QMIN=${LAS_QMIN:-12000000}; LAS_A=${LAS_A:-28}; LAS_LAND=${LAS_LAND:-ride}
LAS_LIM0=${LAS_LIM0:-$RLIM}; LAS_LIM1=${LAS_LIM1:-$ALIM}; LAS_LPB0=${LAS_LPB0:-$LPB0}; LAS_LPB1=${LAS_LPB1:-$LPB1}; LAS_MFB0=${LAS_MFB0:-$MFB0}; LAS_MFB1=${LAS_MFB1:-$MFB1}
LAS_NICE=${LAS_NICE:-10}; LAS_RESERVE=${LAS_RESERVE:-1}; LAS_MIN_CPUS=${LAS_MIN_CPUS:-2}
# LAS_NCPU (default NCPU): the CPU count las's budget is taken from. The container profile keeps NCPU=20 for the sieve-phase
# consumers (glas host + incdup + probes, measured at that budget) while the cgroup quota is 24: LAS_NCPU=24 lets las have
# the 4 CPUs nothing else is budgeted for (las_budget still subtracts every live consumer, so the cgroup's runnable load
# stays <= its quota -- CFS throttling would stop the GPU feeder threads too).
LAS_NCPU=${LAS_NCPU:-$NCPU}; case "$LAS_NCPU" in ''|*[!0-9]*) die "LAS_NCPU must be a positive integer (got '$LAS_NCPU')";; esac
LAS_T=${LAS_T:-$(( LAS_NCPU - SIEVE_HOST_T - LAS_RESERVE - ( COFAC_CPU == 1 ? COFAC_T : 0 ) ))}
# LAS_FB_REUSE=1 (default): with las/glas as the main siever and LAS_A == SIEVE_A, LAS_LIM1 == ALIM, las's factor base IS
# the sieve's $WORK/fb1 (same makefb, same -lim/-maxbits): hard-link it instead of a second makefb (~11 s x 20 threads)
LAS_FB_REUSE=${LAS_FB_REUSE:-1}
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
echo "### c169 GNFS: $(echo -n "$N" | wc -c) digits, $NCPU CPUs, siever $SIEVER, lpb $LPB0/$LPB1 mfb $MFB0/$MFB1 lim $RLIM/$ALIM $([ "$SIEVER" = bench ] && echo "logI $LOGI J $JJ allowance ${ALLOWARGS[*]:-default} cofac $([ "$COFAC_CPU" = 1 ] && echo "cpu x$COFAC_T" || echo gpu)" || echo "A $SIEVE_A lambda $LAMBDA0/$LAMBDA1 ncurves $NCURVES0/$NCURVES1 -t $SIEVE_T") qmin $QMIN"
# the per-stage thread profile, resolved (the c100 gate's stdout shows it: RSA560_CPU24 plumbing check, container.md idea 1)
say "threads: sieve-phase budget $NCPU (siever -t $SIEVE_T, host $SIEVE_HOST_T, probe $PROBE_NT/$PROBE_LATE_NT, incdup $INC_PAR x $INC_TPG); alone: polyselect $PS_NCPU makefb ${FB_NT:-$NCPU} final purge $STOP_FINAL_NT merge $MERGE_NT replay ${REPLAY_T:-${DS_NT:-$NCPU}} downstream ${DS_NT:-$NCPU}$([ "$CONTAINER" = 1 ] && [ "${RSA560_CPU24:-0}" = 1 ] && echo " [RSA560_CPU24]")"

# ---- preflight: fail fast and BY NAME -------------------------------------------------------
case "$N" in ''|*[!0-9]*) die "N is not a decimal integer";; esac
[ "${#N}" -ge 30 ] || die "N has only ${#N} digits"
for b in "$CF/polyselect" "$CF/polyselect_ropt" "$CF/cado_filter160.sh" \
         "$CF/freerel" "$CF/dup1" "$CF/dup2" "$CF/purge" "$CF/merge" "$CF/replay" "$CF/cado2msieve2" \
         "$HERE/memhold.py" "$HERE/polysel160.py" "$HERE/las_ctl.py"; do
  [ -e "$b" ] || die "missing $b"
done
case "$SIEVER" in
  bench) for b in "$CS/bench" "$CS/fbgen_gpu"; do [ -x "$b" ] || die "missing $b (SIEVER=bench)"; done ;;
  glas)  [ -x "$GLAS" ] || die "missing $GLAS (SIEVER=glas; set GLAS=<path to glas-gpu>)"; [ -x "$CF/makefb" ] || die "missing $CF/makefb" ;;
  las)   for b in "$CF/las" "$CF/makefb"; do [ -x "$b" ] || die "missing $b (SIEVER=las)"; done ;;
esac
[ "$DS_MODE" = msieve ] && for b in "$MSV/msv_launcher" "$MSV/msvrun_gpu.sh" "$MSV/msieve_la"; do [ -e "$b" ] || die "missing $b (DS_MODE=msieve)"; done
if [ "$FREE_RAW" = 1 ]; then [ "$DS_MODE" = cado ] && [ "$INCDUP" = 1 ] || die "FREE_RAW=1 needs DS_MODE=cado and INCDUP=1"; fi
[ "$MEMFD_PURGED" = 1 ] && [ "$DS_MODE" != cado ] && die "MEMFD_PURGED=1 needs DS_MODE=cado"
case "$DS_MODE" in cado|msieve) ;; *) die "DS_MODE must be cado or msieve (got '$DS_MODE')";; esac
case "$STOP_MIN_EXCESS" in ''|*[!0-9]*) die "STOP_MIN_EXCESS must be a non-negative integer (got '$STOP_MIN_EXCESS')";; esac
TP_PO=0; if [ "$STOP_MIN_EXCESS" -gt 0 ]; then [ "$DS_MODE" = cado ] || die "STOP_MIN_EXCESS needs DS_MODE=cado"; TP_PO=1; fi
case "$STOP_RULE" in
  excess) ;;
  cost) [ "$TP_PO" = 1 ] || die "STOP_RULE=cost needs the two-phase purge-only probes (STOP_MIN_EXCESS > 0, DS_MODE=cado)"
        for b in "$HERE/stop_cost.sh" "$HERE/stop_cost.py"; do [ -r "$b" ] || die "missing $b (STOP_RULE=cost)"; done
        command -v pgrep > /dev/null || die "pgrep missing (STOP_RULE=cost kills a superseded probe's tree)"
        STOP_LA_SCALE="$STOP_LA_SCALE" python3 "$HERE/stop_cost.py" la 1e7 > /dev/null || die "stop_cost.py does not run (check the STOP_* knobs; STOP_LA_SCALE must be in [0.5, 2])"
        [ "$STOP_LA_SCALE" = 1 ] || say "STOP_LA_SCALE=$STOP_LA_SCALE: the cost rule prices the LA x$STOP_LA_SCALE (slower-LA card: oversieve more; STOP_SHAPE kappa scaled too)"
        case "$STOP_PROBE_EVERY" in ''|*[!0-9]*|0) die "STOP_PROBE_EVERY must be a positive integer";; esac
        case "$STOP_SHAPE" in 0|1) ;; *) die "STOP_SHAPE must be 0 or 1 (got '$STOP_SHAPE')";; esac
        case "$STOP_MINPTS" in ''|*[!0-9]*|0) die "STOP_MINPTS must be a positive integer";; esac
        [ "$STOP_SHAPE" = 1 ] && say "stop rule (cost): STOP_SHAPE=1 (kappa $STOP_KAPPA s/(M rows)^2, knee slope $STOP_SX, shape a $STOP_SHAPE_A b $STOP_SHAPE_B, min points $STOP_MINPTS, probe every $STOP_PROBE_EVERY chunks after the first matrix)" ;;
  *) die "STOP_RULE must be excess or cost (got '$STOP_RULE')";;
esac
case "$INCDUP" in 0|1) ;; *) die "INCDUP must be 0 or 1 (got '$INCDUP')";; esac
case "$LAS" in 0|1) ;; *) die "LAS must be 0 or 1 (got '$LAS')";; esac
case "$LAS_LAND" in ride|slot) ;; *) die "LAS_LAND must be ride or slot (got '$LAS_LAND')";; esac
if [ "$LAS" = 1 ]; then   # fail-SOFT: a missing co-siever costs ~4% of the sieve, never the run
  for b in "$CF/las" "$CF/makefb"; do [ -x "$b" ] || { say "WARNING: [las] $b missing -- CPU co-sieve DISABLED, GPU-only"; LAS=0; }; done
  [ -r "$HERE/las_ctl.py" ] || { say "WARNING: [las] $HERE/las_ctl.py missing -- CPU co-sieve DISABLED, GPU-only"; LAS=0; }
  [ "$LAS_T" -ge 1 ] 2>/dev/null || { say "WARNING: [las] LAS_T=$LAS_T < 1 -- CPU co-sieve DISABLED"; LAS=0; }
  [ "$LAS" = 1 ] && say "[las] CPU co-sieve ON: q from $LAS_QMIN, A $LAS_A, -t $LAS_T nice $LAS_NICE, budget $LAS_NCPU - host $SIEVE_HOST_T - reserve $LAS_RESERVE - live probe/incdup, landing=$LAS_LAND, fb reuse $LAS_FB_REUSE$([ "$CONTAINER" = 1 ] && [ "${RSA560_CPU_LAS:-0}" = 1 ] && echo " [RSA560_CPU_LAS: siever -t $SIEVE_T]")"
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
_nfd=$(( MAXCHUNKS + 34 + 16 )); [ "$INCDUP" = 1 ] && _nfd=$(( _nfd + 32 * MAXCHUNKS )); [ "$MEMFD_PURGED" = 1 ] && _nfd=$(( _nfd + 2 * PP_SLOTS ))
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
# Research opt-in: CADO's native binary table is mmap-shared by dup2 readers.
# Keep this independent of relation-slice compression; never gzip a binary table.
RENUMBER_FORMAT=${RENUMBER_FORMAT:-text}
case "$RENUMBER_FORMAT" in
  text) RN=$SL ;;
  binary) RN=.bin ;;
  *) die "RENUMBER_FORMAT must be text or binary" ;;
esac
export RENUMBER_SUFFIX=$RN
say "renumber format $RENUMBER_FORMAT"
python3 -c 'import os; os.memfd_create("t")' 2>/dev/null || die "memfd_create unavailable"
# LA=gpu (default: gpu_la.py block Lanczos) | bwc (CADO's CPU block Wiedemann: the CPU-only validation path)
LA=${LA:-gpu}
# 256-bit GPU block Lanczos by default (W6, 2026-10-17: half the iterations at ~1.1-1.5x the pass; ds_cado falls back to 64/prod on a second try)
[ "$LA" = gpu ] && export LA_BLOCK=${LA_BLOCK:-256}
if [ "$SIEVER" != las ] || [ "$LA" = gpu ]; then
  nvidia-smi -L >/dev/null 2>&1 || python3 -c 'import cupy; cupy.cuda.runtime.getDeviceCount()' >/dev/null 2>&1 || die "no GPU visible (SIEVER=$SIEVER LA=$LA)"
fi
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
cleanup(){ [ -n "${BENCHPID:-}" ] && kill "$BENCHPID" 2>/dev/null; [ -n "${SES_PID:-}" ] && kill "$SES_PID" 2>/dev/null; [ -n "${PROBEPID:-}" ] && kill "$PROBEPID" 2>/dev/null; inc_kill
           for _p in ${LASPID:-} ${LAS_FBPID:-}; do kill -CONT "$_p" 2>/dev/null; kill -KILL "$_p" 2>/dev/null; done
           [ -n "${COFPID:-}" ] && kill "$COFPID" 2>/dev/null
           [ -n "${PSPID:-}" ] && kill "$PSPID" 2>/dev/null; [ -n "${OVPID:-}" ] && { pkill -TERM -P "$OVPID" 2>/dev/null; kill "$OVPID" 2>/dev/null; }; [ -n "${FRPID:-}" ] && kill "$FRPID" 2>/dev/null; [ -n "${MHPID:-}" ] && kill "$MHPID" 2>/dev/null
           [ -n "${_ppid:-}" ] && kill "$_ppid" 2>/dev/null; [ -n "${_ppf:-}" ] && rm -f "$_ppf"
           [ -n "${SPEC_PID:-}" ] && { kill -CONT -- -"$SPEC_PID" 2>/dev/null; kill -TERM -- -"$SPEC_PID" 2>/dev/null; }; }
trap cleanup EXIT

# ---- memfd pool -----------------------------------------------------------------------------
# c.sparse.bin (DS_MODE=cado): the merged CADO matrix replay writes for gpu_la (~5 GB at c160).
# dd/<kkk>/s<ii>.gz + kk/<kkk>/s<ii> (INCDUP=1): incremental per-(chunk,slice) dedup parts and their
# key shadows (cado_incdup.sh). 32*MAXCHUNKS more memfds (1920 at 60; nofile 4096, preflight-checked:
# INCDUP=1 caps MAXCHUNKS at ~122) -- sparse, free until written.
python3 "$HERE/memhold.py" "$MEM" "c%03d:1:$MAXCHUNKS" "d1/%d/rel.0000$SL:0:15" "d2/%d/rel.0000$SL:0:15" rels_out \
    $([ "$DS_MODE" = cado ] && echo c.sparse.bin) $([ "$INCDUP" = 1 ] && echo renumber.plain) \
    $([ "$MEMFD_PURGED" = 1 ] && echo "pp/%03d/$PG:1:$PP_SLOTS pp/%03d/history:1:$PP_SLOTS") \
    $([ "$INCDUP" = 1 ] && for _c in $(seq 1 "$MAXCHUNKS"); do printf 'dd/%03d/s%%02d'"$SL"':0:15 kk/%03d/s%%02d:0:15 ' "$_c" "$_c"; done) \
    > "$WORK/memhold.out" 2>&1 & MHPID=$!
for _i in $(seq 1 50); do grep -q READY "$WORK/memhold.out" 2>/dev/null && break; sleep 0.1; done
grep -q READY "$WORK/memhold.out" || die "memhold did not start: $(cat "$WORK/memhold.out")"
# rmf <file>...: rm -f that gives a memfd's RAM back first (a bare rm of a memhold link only drops the link: the memfd,
# still open in memhold, would keep its pages). Identical to rm -f for regular files.
rmf(){ local f; for f in "$@"; do [ -L "$f" ] && : > "$f" 2>/dev/null; rm -f "$f"; done; return 0; }
# pp_link <probe dir> <probe number> (MEMFD_PURGED=1): point the probe's purged + history at memfd slot <n>. Out of
# slots -> a loud warning and the probe writes regular files (the 530 behaviour, /tmp)
pp_link(){ [ "$MEMFD_PURGED" = 1 ] || return 0
  local s; printf -v s '%s/pp/%03d' "$MEM" "$2"
  if [ -L "$s/$PG" ] && [ -L "$s/history" ]; then
    : > "$s/$PG"; : > "$s/history"; ln -sfn "$s/$PG" "$1/$PG" && ln -sfn "$s/history" "$1/history" && return 0; fi
  say "WARNING: MEMFD_PURGED: no memfd slot $2 (PP_SLOTS=$PP_SLOTS) -- probe $2 writes its purged/history to $1 (/tmp)"; return 0; }
# res_note <tag> (RES_LOG=1): /tmp + cgroup memory + memfd pool, the quantities the validator limits
res_note(){ [ "$RES_LOG" = 1 ] || return 0
  local t m p pool; t=$(df -Pm "$WORK" 2>/dev/null | awk 'NR==2{print $3}'); [ "${t:-0}" -gt "$RES_TMP_MAX" ] && RES_TMP_MAX=${t:-0}
  m=$(cat /sys/fs/cgroup/memory.current 2>/dev/null); p=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null)
  pool=$(find "$MEM" -type l -exec stat -L -c %s {} + 2>/dev/null | awk '{s+=$1} END{printf "%d", s/1048576}')
  # RES_CPU=1 (default 0): the cgroup's CFS quota accounting (cpu.stat) -- cumulative CPU seconds used and how often / how
  # long the WHOLE cgroup (the GPU feeder threads included) was throttled at the 24-CPU quota (container.md idea 2)
  local c=""; [ "${RES_CPU:-0}" = 1 ] && [ -r /sys/fs/cgroup/cpu.stat ] && \
    c=$(awk '/^usage_usec/{u=$2} /^nr_throttled/{n=$2} /^throttled_usec/{t=$2} END{printf ", cpu used %ds throttled %dx %.1fs", u/1e6, n, t/1e6}' /sys/fs/cgroup/cpu.stat 2>/dev/null)
  say "[res] $1: /tmp used ${t:-?} MB (max ${RES_TMP_MAX}), cgroup mem $(( ${m:-0} / 1048576 )) MB (peak $(( ${p:-0} / 1048576 )) MB), memfd pool ${pool:-?} MB$c"; }
res_note "memhold up"

# =============================================================================================
echo "### [1/4] polyselect (CADO size-opt ${PS_SECS}s..${PS_MAX}s, ropt top-$ROPT_K effort $ROPT_EFF) $(date +%T)"
if [ -n "${POLY_IN:-}" ] && [ -s "$POLY_IN" ]; then
  python3 "$HERE/polysel160.py" norm "$POLY_IN" "$WORK/poly.cado" || die "cannot read supplied polynomial $POLY_IN"
  say "using supplied polynomial $POLY_IN"
else
  # GPS1=<path to Cognition's gps1> (default $APP/glas/gps1 when present and SIEVER=glas): the GPU size-optimised search
  # instead of CADO polyselect. Same raw.p format; 300 s of gps1 found a polynomial within 3.6% of a 2 h CPU search on
  # key 1, and 4x more time found nothing better (ledger W7 GPU polyselect, X5). admax is open-ended: PS_SECS stops it.
  GPS1=${GPS1:-$([ "$SIEVER" = glas ] && [ -x "$APP/glas/gps1" ] && echo "$APP/glas/gps1")}
  if [ -n "$GPS1" ] && [ -x "$GPS1" ]; then
    say "polyselect on the GPU: $GPS1 (P $PS_P, incr $PS_INCR, nq $PS_NQ, ${PS_SECS}s)"
    ( cd "$WORK" && exec "$GPS1" -N "$N" -d 5 -P "$PS_P" -admin "$PS_ADMIN" -admax "${PS_ADMAX_GPU:-20000000}" \
          -incr "$PS_INCR" -nq "$PS_NQ" -sopt-gpu 1 > "$WORK/raw.p" 2> "$WORK/ps.err" ) & PSPID=$!
  else
  ( cd "$WORK" && exec "$CF/polyselect" -N "$N" -degree 5 -P "$PS_P" -admin "$PS_ADMIN" -admax "$PS_ADMAX" \
        -incr "$PS_INCR" -nq "$PS_NQ" -t "$PS_NCPU" > "$WORK/raw.p" 2> "$WORK/ps.err" ) & PSPID=$!
  fi
  # PS_BATCHES=2 (default, gps1 only; 0 = the PS_SECS clock): gps1 writes its candidates in batches (s47: key 1/2 at
  # 137 / 272-274 / 403-405 s, ~2,300 each) and both keys' winners came in batch 2 (s45: 150/200 s = batch 1 only lose
  # E -14.3% / -8.9%). Stop once batch PS_BATCHES is complete (count stable for 2 s): ~26 s before the 300 s clock
  # here, and a slower GPU still gets batch 2 (PS_MAX caps). PS_SECS then applies only without gps1.
  # PS_OVERLAP=1 (default 0, gps1 + PS_BATCHES>=2 only): when batch 1 lands, ps_overlap.sh a runs sopt + topk + ropt
  # on it on the CPU while gps1 searches batch 2 on the GPU; after the stop, ps_overlap.sh b re-runs sopt / ropt only
  # for polys not done in pass a (memo keyed by input polynomial) and picks with the same topk / best as below, so
  # the same raw.p gives the same poly.cado (s72_psoverlap.sh checks). Any failure falls back to the path below.
  PS_OVERLAP=${PS_OVERLAP:-0}; OVPID=""; _ovok=0
  ps_ov(){ N="$N" CF="$CF" NCPU="$PS_NCPU" ALIM="$ALIM" RLIM="$RLIM" SOPT_EFF="$SOPT_EFF" SOPT_K="$SOPT_K" SOPT_MAX="$SOPT_MAX" \
           SOPT_BIN="$SOPT_BIN" ROPT_K="$ROPT_K" ROPT_EFF="$ROPT_EFF" ROPT_MAX="$ROPT_MAX" HERE="$HERE" \
           exec bash "$HERE/ps_overlap.sh" "$1" "$WORK" ${2:+"$2"}; }
  _nb=0; _lc=0; _bat=0; [ -n "$GPS1" ] && [ -x "$GPS1" ] && _bat=${PS_BATCHES:-2}
  # PS_ADAPT=1 (default 0, gps1 + PS_BATCHES>=1 only; r9, data/rsa560/audit_r6/ps_more_b6_sb81.txt + sb86): extra search
  # pays only on weak incumbents (b3 -> b6 E: key14 +14.3%, key10 +9.9%, key18 +10.5%; keys 11/15/16/17 +0). When batch
  # PS_BATCHES lands, ps_overlap.sh a runs the production tail (sopt + topk + ropt) on a snapshot of raw.p on the CPU while
  # gps1 keeps searching on the GPU; `best` on its ropt output = the polynomial production would pick (b_E). b_E >=
  # PS_ADAPT_E -> stop: gps1 is killed, the snapshot's artifacts are installed (same poly as PS_ADAPT=0). Else extend:
  # gps1 runs to PS_ADAPT_MAXB batches (PS_MAX still caps), ps_overlap.sh b (memo of pass a) tails the final raw.p, and
  # the better of that pick and the b_E incumbent is kept. PS_ADAPT=1 replaces PS_OVERLAP's batch-1 pass.
  # PS_ADAPT default 1 since 2026-10-24 18:00 MDT (gate gpu/r9_ps_adapt.sh on sb86: key15 stop 332 s same poly; key14 extend 1.090 -> 1.246e-9
  # at 519 s; 12-key study data/rsa560/audit_r6/ps_more2_sb86.txt: b3 E<1.15 -> +7.2% E average, >=1.15 no gain)
  PS_ADAPT=${PS_ADAPT:-1}; PS_ADAPT_E=${PS_ADAPT_E:-1.15e-9}; PS_ADAPT_MAXB=${PS_ADAPT_MAXB:-6}; _add=""; _adE=""; _adF=""; _adpick=""; _adt=0
  [ "$PS_ADAPT" = 1 ] && [ "$_bat" -lt 1 ] && { say "WARNING: PS_ADAPT=1 needs gps1 batches (PS_BATCHES>=1) -- off"; PS_ADAPT=0; }
  ps_adapt_decide(){   # after pass a: the snapshot's pick and the stop/extend decision (_add = stop | extend | fail)
    local _o
    if [ -f "$WORK/ov/A.done" ] && _o=$(python3 "$HERE/polysel160.py" best "$WORK/ov/roptA.out" "$N" "$WORK/ov/poly_b.cado"); then
      _adE=$(echo "$_o" | awk '/^MurphyE/{print $2; exit}')
      if [ -n "$_adE" ] && awk -v e="$_adE" -v t="$PS_ADAPT_E" 'BEGIN{exit !(e+0 >= t+0)}'; then _add=stop; else _add=extend; fi
      say "PS_ADAPT: batch-$_bat pick $_o -> $_add (threshold $PS_ADAPT_E, pass a $(( $(date +%s)-T0-_adt ))s)"
    else _add=fail; say "WARNING: PS_ADAPT pass a failed ($(tail -1 "$WORK/ps_overlap.a.log" 2>/dev/null)) -- production path"; fi; }
  # PS_MEMO_ROLL=1 (default 0; PS_ADAPT extend only): rolling polyselect memo. Measured on arm B (s2b/s4*): pass b re-ran
  # sopt on 235 of 500 and ropt on all of the final top-20 for 54 s with the GPU idle. While gps1 searches batches
  # PS_BATCHES+1 .. PS_ADAPT_MAXB-1, ps_overlap.sh r runs the CPU tail on each newly completed batch's snapshot, re-running
  # only what no earlier pass memoized; pass b then re-runs only the last batch's newcomers. Memo keyed by the exact input
  # polynomial (memo v2): same final raw.p -> same poly.cado as without it. A pass starts only when it is predicted
  # (duration of the previous pass, batch cadence so far) to end by the final batch + PS_MEMO_ROLL_WAIT s; one still
  # running when the search stops is waited for if its predicted remainder is <= PS_MEMO_ROLL_WAIT s, else killed (its
  # incomplete outputs are never reused).
  PS_MEMO_ROLL=${PS_MEMO_ROLL:-0}; PS_MEMO_ROLL_WAIT=${PS_MEMO_ROLL_WAIT:-15}; _rlb=0; _rlt=0; _rlD=0; _bt1=0; _btn=0
  [ "$PS_MEMO_ROLL" = 1 ] && [ "$PS_ADAPT" != 1 ] && { say "WARNING: PS_MEMO_ROLL=1 needs PS_ADAPT=1 -- off"; PS_MEMO_ROLL=0; }
  ps_kill_tree(){ local _c; for _c in $(pgrep -P "$1" 2>/dev/null); do ps_kill_tree "$_c"; done; kill -TERM "$1" 2>/dev/null; return 0; }
  ps_roll_done(){   # reap a finished rolling pass
    wait "$OVPID"; OVPID=""; _rlD=$(( $(date +%s)-_rlt ))
    say "PS_MEMO_ROLL: batch-$_rlb pass done in ${_rlD}s: $(grep -a 'ropt' "$WORK/ps_overlap.r$_rlb.log" | tail -1)"; }
  ps_adapt_install(){  # the batch-PS_BATCHES snapshot's tail artifacts = what the PS_ADAPT=0 tail writes for that raw.p
    cp -f "$WORK/ov/raw1.p" "$WORK/raw.p" && cp -f "$WORK/ov/topA.poly" "$WORK/top.poly" && \
    cp -f "$WORK/ov/roptA.out" "$WORK/ropt.out" && cp -f "$WORK/ov/poly_b.cado" "$WORK/poly.cado"; }
  while :; do
    sleep "$([ "$_bat" -gt 0 ] && echo 1 || echo 5)"; _el=$(( $(date +%s)-T0 )); _nc=$(grep -c '^# Size-optimized' "$WORK/raw.p" 2>/dev/null)
    kill -0 "$PSPID" 2>/dev/null || break
    if [ "$_bat" -gt 0 ]; then
      if [ "${_nc:-0}" -gt "$_lc" ]; then
        sleep 2; _n2=$(grep -c '^# Size-optimized' "$WORK/raw.p" 2>/dev/null)
        [ "${_n2:-0}" = "$_nc" ] && { _nb=$((_nb+1)); _lc=$_nc; say "gps1 batch $_nb complete: $_nc candidates at ${_el}s"
          [ "$_nb" = 1 ] && _bt1=$_el; _btn=$_el; }
        [ "$PS_ADAPT" != 1 ] && [ "$PS_OVERLAP" = 1 ] && [ "$_nb" = 1 ] && [ "$_bat" -ge 2 ] && [ -z "$OVPID" ] && {
          ps_ov a > "$WORK/ps_overlap.a.log" 2>&1 & OVPID=$!; say "PS_OVERLAP: sopt + ropt on batch 1 started on the CPU"; }
      fi
      if [ "$PS_ADAPT" = 1 ]; then
        [ -z "$_add" ] && [ "$_nb" -ge "$_bat" ] && [ "${_nc:-0}" -ge "$PS_MIN_CANDS" ] && {
          ps_ov a > "$WORK/ps_overlap.a.log" 2>&1 & OVPID=$!; _add=pending; _adt=$(( $(date +%s)-T0 ))
          say "PS_ADAPT: production tail on the batch-$_nb snapshot started on the CPU; gps1 keeps searching"; }
        [ "$_add" = pending ] && ! kill -0 "$OVPID" 2>/dev/null && { wait "$OVPID"; OVPID=""; ps_adapt_decide; _rlD=$(( $(date +%s)-T0-_adt )); }
        case "$_add" in stop|fail) break ;; esac
        if [ "$PS_MEMO_ROLL" = 1 ] && [ "$_add" = extend ]; then
          [ -n "$OVPID" ] && ! kill -0 "$OVPID" 2>/dev/null && ps_roll_done
          # start on the newest complete batch if predicted to end by the final batch (+ PS_MEMO_ROLL_WAIT)
          if [ -z "$OVPID" ] && [ "$_nb" -gt "$_rlb" ] && [ "$_nb" -lt "$PS_ADAPT_MAXB" ] && [ "$_nb" -gt "$_bat" ] && \
             awk -v n="$_nb" -v m="$PS_ADAPT_MAXB" -v t1="$_bt1" -v tn="$_btn" -v e="$(( $(date +%s)-T0 ))" -v d="$_rlD" -v w="$PS_MEMO_ROLL_WAIT" \
               'BEGIN{ c = (n > 1) ? (tn - t1) / (n - 1) : tn; exit !(e + d <= tn + (m - n) * c + w) }'; then
            _rlb=$_nb; _rlt=$(date +%s); ps_ov r "$_nb" > "$WORK/ps_overlap.r$_nb.log" 2>&1 & OVPID=$!
            say "PS_MEMO_ROLL: CPU tail on the batch-$_nb snapshot started (previous pass ${_rlD}s)"
          fi
        fi
        [ -n "$_add" ] && [ "$_nb" -ge "$PS_ADAPT_MAXB" ] && break
      else
      [ "$_nb" -ge "$_bat" ] && [ "${_nc:-0}" -ge "$PS_MIN_CANDS" ] && break
      fi
    else
      [ "$_el" -ge "$PS_SECS" ] && [ "${_nc:-0}" -ge "$PS_MIN_CANDS" ] && break
    fi
    [ "$_el" -ge "$PS_MAX" ] && break
  done
  kill "$PSPID" 2>/dev/null; wait "$PSPID" 2>/dev/null; PSPID=""
  say "size-opt search stopped: $(grep -c '^# Size-optimized' "$WORK/raw.p") candidates"
  if [ "$PS_ADAPT" = 1 ]; then
    # MAXB / PS_MAX / a gps1 exit can come before pass a finishes: wait for it and decide now
    [ "$_add" = pending ] && { _st=$(date +%s); wait "$OVPID"; OVPID=""; say "PS_ADAPT: waited $(( $(date +%s)-_st ))s for pass a"; ps_adapt_decide; }
    if [ "$PS_MEMO_ROLL" = 1 ] && [ -n "$OVPID" ]; then   # a rolling pass still running: wait <= PS_MEMO_ROLL_WAIT s if
      _st=$(date +%s); _rem=$(( _rlD - (_st-_rlt) ))    # predicted to end within it, else (or on overrun) kill it
      [ "$_rem" -le "$PS_MEMO_ROLL_WAIT" ] && while kill -0 "$OVPID" 2>/dev/null && [ $(( $(date +%s)-_st )) -lt "$PS_MEMO_ROLL_WAIT" ]; do sleep 0.2; done
      if ! kill -0 "$OVPID" 2>/dev/null; then ps_roll_done; say "PS_MEMO_ROLL: waited $(( $(date +%s)-_st ))s for it"
      else ps_kill_tree "$OVPID"; wait "$OVPID" 2>/dev/null; OVPID=""
        say "PS_MEMO_ROLL: batch-$_rlb pass killed after $(( $(date +%s)-_rlt ))s (predicted ${_rlD}s; its incomplete outputs are not reused)"; fi
    fi
    case "$_add" in
      stop) if ps_adapt_install; then _ovok=1; _adpick=b$_bat; else say "WARNING: PS_ADAPT install failed -- production path"; _add=fail; fi ;;
      extend) if ( ps_ov b ) > "$WORK/ps_overlap.b.log" 2>&1; then _ovok=1; say "PS_ADAPT: $(grep -a 'sopt\|ropt' "$WORK/ps_overlap.b.log" | tr '\n' ' ')"
              else say "WARNING: PS_ADAPT pass b failed ($(tail -1 "$WORK/ps_overlap.b.log")) -- sequential sopt/ropt on the final raw.p"; fi ;;
      *) say "PS_ADAPT: no batch-$_bat decision (${_add:-batch $_bat never completed}) -- production path on the current raw.p" ;;
    esac
  elif [ -n "$OVPID" ]; then
    _st=$(date +%s); wait "$OVPID"; OVPID=""; say "PS_OVERLAP: batch-1 pass done (waited $(( $(date +%s)-_st ))s): $(grep -a 'ropt top' "$WORK/ps_overlap.a.log" | tail -1)"
    if ( ps_ov b ) > "$WORK/ps_overlap.b.log" 2>&1; then _ovok=1; say "PS_OVERLAP: $(grep -a 'sopt\|ropt' "$WORK/ps_overlap.b.log" | tr '\n' ' ')"
    else say "WARNING: PS_OVERLAP pass b failed ($(tail -1 "$WORK/ps_overlap.b.log")) -- sequential sopt/ropt"; fi
  fi
  if [ "$_ovok" != 1 ]; then
  # SOPT_EFF re-optimization (see the parameter block). Waits on its own PIDs only: a bare `wait` would
  # also wait for memhold. Any failure (no binary, no input, no output) leaves raw.p as the search wrote it.
  if [ "$SOPT_EFF" -gt 0 ] 2>/dev/null; then
    if [ -x "$SOPT_BIN" ]; then
      _st=$(date +%s); rm -rf "$WORK/sopt"; mkdir -p "$WORK/sopt"
      if python3 "$HERE/polysel160.py" soptin "$WORK/raw.p" "$SOPT_K" "$WORK/sopt/in" "$PS_NCPU" > "$WORK/sopt/in.log" 2>&1; then
        _sp=()
        for _f in "$WORK/sopt"/in*.poly; do
          timeout --foreground "$SOPT_MAX" "$SOPT_BIN" -inputpolys "$_f" -sopteffort "$SOPT_EFF" \
              > "${_f%.poly}.out" 2> "${_f%.poly}.err" & _sp+=($!)
        done
        wait "${_sp[@]}"
        if python3 "$HERE/polysel160.py" soptout "$WORK/raw.p" "$WORK/sopt"/in*.out > "$WORK/sopt/out.log" 2>&1; then
          say "sopt effort $SOPT_EFF: $(cat "$WORK/sopt/in.log"); $(cat "$WORK/sopt/out.log") in $(( $(date +%s)-_st ))s"
        else say "WARNING: sopt produced nothing usable ($(tail -1 "$WORK/sopt/out.log")) -- continuing with the search's candidates"; fi
      else say "WARNING: sopt input failed ($(tail -1 "$WORK/sopt/in.log")) -- continuing with the search's candidates"; fi
    else say "WARNING: $SOPT_BIN missing -- sopt re-optimization skipped"; fi
  fi
  python3 "$HERE/polysel160.py" topk "$WORK/raw.p" "$ROPT_K" "$WORK/top.poly" || die "no size-optimized candidates"
  # --foreground: plain timeout setpgid()s itself out of our process group, so breaking_rsa's killpg
  # would leave ropt (-t NCPU) running; polyselect_ropt is a single process, nothing else to reap.
  ( cd "$WORK" && timeout --foreground "$ROPT_MAX" "$CF/polyselect_ropt" -inputpolys "$WORK/top.poly" -ropteffort "$ROPT_EFF" \
        -t "$PS_NCPU" -area 2.4e16 -Bf "$ALIM" -Bg "$RLIM" > "$WORK/ropt.out" 2> "$WORK/ropt.err" )
  say "ropt rc=$? ($(grep -c 'MurphyE' "$WORK/ropt.out") MurphyE lines)"
  python3 "$HERE/polysel160.py" best "$WORK/ropt.out" "$N" "$WORK/poly.cado" || die "ropt produced no usable polynomial"
  fi   # _ovok
  if [ "$PS_ADAPT" = 1 ]; then
    if [ "$_add" = extend ]; then   # never worse than the incumbent: keep the batch-PS_BATCHES pick if the final tail lost
      _adF=$(python3 "$HERE/polysel160.py" best "$WORK/ropt.out" "$N" "$WORK/ov/poly_f.cado" | awk '/^MurphyE/{print $2; exit}')
      if [ -n "$_adF" ] && awk -v f="$_adF" -v b="$_adE" 'BEGIN{exit !(f+0 >= b+0)}'; then _adpick=b$_nb
      elif ps_adapt_install; then say "PS_ADAPT: final pick ${_adF:-?} < incumbent $_adE -- batch-$_bat polynomial kept"; _adF=$_adE; _adpick=b$_bat
      else _adpick=b$_nb; fi
    elif [ "$_add" = stop ]; then _adF=$_adE
    else mkdir -p "$WORK/ov"; _adF=$(python3 "$HERE/polysel160.py" best "$WORK/ropt.out" "$N" "$WORK/ov/poly_f.cado" 2>/dev/null | awk '/^MurphyE/{print $2; exit}'); _adpick=b$_nb; fi
    # s102 conversion (docs/rsa560_codex_review7.md 4.2, central gamma 0.826): gross sieve saving vs the batch-PS_BATCHES
    # incumbent = 13,000 s x (1 - (E_final/E_b)^-0.826); +10% E ~ 984 s
    _adS=$(awk -v f="${_adF:-0}" -v b="${_adE:-0}" 'BEGIN{ if (f>0 && b>0) printf "%.0f", 13000*(1-(f/b)^(-0.826)); else print "?" }')
    say "PS_ADAPT: b$_bat E=${_adE:-?} -> decision ${_add:-none} (threshold $PS_ADAPT_E, $_nb batches, pick ${_adpick:-?}) final E=${_adF:-?} wall=$(( $(date +%s)-T0 ))s est_sieve_saving=${_adS}s"
  fi
  # POLY_TS=1 (default): GPU test-sieve of the TS_K best root-optimized polys, pick by modelled sieve time
  # (polytest160.sh; never fatal, keeps the MurphyE pick unless a candidate models >= TS_MIN_GAIN faster).
  if [ "${POLY_TS:-1}" = 1 ] && [ "$SIEVER" = bench ]; then
    say "test-sieving the top-${TS_K:-5} root-optimized polynomials (GPU, ~2 min)"
    bash "$HERE/polytest160.sh" "$WORK" "$N" "$WORK/ropt.out" "$WORK/poly.cado" "$CS" "$HERE" \
        "$RLIM" "$ALIM" "$LPB1" "$MFB0" "$MFB1" "$LOGI" "$JJ" "$QMIN" -- "${ALLOWARGS[@]}" 2>&1 | grep -a '\[polytest\]'
  elif [ "${POLY_TS:-0}" = 1 ]; then   # las/glas (default OFF since 2026-10-20: ~3,000-q windows rank with ~4.5% noise, gain ~0.1% for up to 300 s; polynomial research)
    say "test-sieving the top-${TS_K:-5} root-optimized polynomials ($SIEVER)"
    _sb="$CF/las"; [ "$SIEVER" = glas ] && _sb="$GLAS"
    TS_RELS="${TS_RELS:-$FIRST_PROBE}" bash "$HERE/polytest_las.sh" "$WORK" "$N" "$WORK/ropt.out" "$WORK/poly.cado" "$SIEVER" "$_sb" "$CF/makefb" "$HERE" \
        "$RLIM" "$ALIM" "$LPB0" "$LPB1" "$MFB0" "$MFB1" "$LAMBDA0" "$LAMBDA1" "$NCURVES0" "$NCURVES1" "$SIEVE_A" "$QMIN" "$SIEVE_T" 2>&1 | grep -a '\[polytest\]'
    rm -rf "$WORK/ts"
  fi
fi
grep -q "^n: $N\$" "$WORK/poly.cado" || die "poly.cado N mismatch"
python3 "$HERE/polysel160.py" job  "$WORK/poly.cado" "$WORK/c160.job" "$RLIM" "$ALIM" "$LPB0" "$LPB1" "$MFB0" "$MFB1"
python3 "$HERE/polysel160.py" msfb "$WORK/poly.cado" "$WORK/c.fb" "$RLIM" "$ALIM" "$LPB0" "$LPB1"
say "polynomial: $(grep -E '^(c5|skew):' "$WORK/poly.cado" | tr '\n' ' ')"
# POLY_ONLY=<file> (fresh-key stitched validation, 2026-10-23): copy the selected polynomial to <file> and stop; the
# measured polyselect seconds are the "total" on the next line. The sieve then runs stitched via run_key1b.sh (POLYB=<file>).
if [ -n "${POLY_ONLY:-}" ]; then cp -f "$WORK/poly.cado" "$POLY_ONLY"; say "POLY_ONLY: polynomial written to $POLY_ONLY (polyselect total $(( $(date +%s) - T0 ))s)"; exit 0; fi

# freerel + renumber depend only on (poly, lpb): build them NOW, in the background, while the GPU
# sieves -- measured 141 s at 4 threads in the sandbox, which would otherwise sit inside the first
# probe on the critical path. cado_filter160.sh finds the cache via FRC and skips its own freerel.
mkdir -p "$WORK/freerel"
( nice -n 5 "$CF/freerel" -poly "$WORK/poly.cado" -lpb0 "$LPB0" -lpb1 "$LPB1" -out "$WORK/freerel/freerel.gz" \
     -renumber "$WORK/freerel/renumber$RN" -t "$FR_T" > "$WORK/freerel/freerel.build.log" 2>&1 \
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
if [ "$SIEVER" = bench ]; then
  echo "### [2/4] factor base (fbgen_gpu, alim $ALIM) $(date +%T)"
  "$CS/fbgen_gpu" --poly "$WORK/c160.job" --lim "$ALIM" --maxbits "$LOGI" --out "$WORK/fb.roots1" > "$WORK/fbgen.log" 2>&1 \
      || die "fbgen_gpu failed: $(tail -2 "$WORK/fbgen.log")"
  say "factor base $(stat -c%s "$WORK/fb.roots1") bytes"
else
  echo "### [2/4] factor base (makefb side 1, lim $ALIM, maxbits $(( (SIEVE_A + 1) / 2 ))) $(date +%T)"
  "$CF/makefb" -poly "$WORK/poly.cado" -side 1 -lim "$ALIM" -maxbits "$(( (SIEVE_A + 1) / 2 ))" -t "${FB_NT:-$NCPU}" -out "$WORK/fb1" > "$WORK/fbgen.log" 2>&1 \
      || die "makefb failed: $(tail -2 "$WORK/fbgen.log")"
  say "factor base $(stat -c%s "$WORK/fb1" 2>/dev/null || stat -f%z "$WORK/fb1") bytes"
fi
# siever_launch q0 q1 out log: start one q-chunk in the background; the pid is in $!
# bench honours --stop-file; las/glas are killed at MATRIX (their running chunk is discarded by design).
RESTART=()
siever_launch(){
  case "$SIEVER" in
    bench)
      "$CS/bench" --pipeline "${COFARGS[@]}" "${ALLOWARGS[@]}" "${RESTART[@]}" --fill-concurrent --poly "$WORK/c160.job" --fb1 "$WORK/fb.roots1" \
          --logI "$LOGI" --J "$JJ" --sq-side 1 --qrange "$1:$(( $2 - 1 ))" --relations "$3" --stop-file "$STOP" > "$4" 2>&1 & ;;
    las|glas)
      local bin="$CF/las" extra=(); [ "$SIEVER" = glas ] && { bin="$GLAS"; extra=(-gpu); }
      "$bin" -poly "$WORK/poly.cado" -fb1 "$WORK/fb1" -lim0 "$RLIM" -lim1 "$ALIM" -lpb0 "$LPB0" -lpb1 "$LPB1" -mfb0 "$MFB0" -mfb1 "$MFB1" \
          -lambda0 "$LAMBDA0" -lambda1 "$LAMBDA1" -ncurves0 "$NCURVES0" -ncurves1 "$NCURVES1" -A "$SIEVE_A" -q0 "$1" -q1 "$2" -sqside 1 \
          -t "$SIEVE_T" "${extra[@]}" -out "$3" > "$4" 2>&1 & ;;
  esac
}
# ---- GLAS_SESSION=1 helpers (unused when 0) ----
# ses_up: (re)start the session when none is alive. Fresh FIFOs every start: a stale BAND in the old control FIFO
# would otherwise be sieved twice. glas is launched BEFORE the driver opens its ends (it inherits no FIFO fd, so the
# driver's death gives it EOF -> exit 5); the driver opens both read-write (never blocks on a FIFO open).
ses_up(){
  [ -n "$SES_PID" ] && kill -0 "$SES_PID" 2>/dev/null && return 0
  ses_kill; SES_GEN=$(( SES_GEN + 1 )); local c="$WORK/ses.ctl" a="$WORK/ses.ack"
  rm -f "$c" "$a"; mkfifo "$c" "$a" || die "GLAS_SESSION: mkfifo failed in $WORK"
  "$GLAS" -poly "$WORK/poly.cado" -fb1 "$WORK/fb1" -lim0 "$RLIM" -lim1 "$ALIM" -lpb0 "$LPB0" -lpb1 "$LPB1" -mfb0 "$MFB0" -mfb1 "$MFB1" \
      -lambda0 "$LAMBDA0" -lambda1 "$LAMBDA1" -ncurves0 "$NCURVES0" -ncurves1 "$NCURVES1" -A "$SIEVE_A" -sqside 1 \
      -t "$SIEVE_T" -gpu -session "$c" -session-ack "$a" >> "$WORK/glas_session.log" 2>&1 &
  SES_PID=$!
  exec {SES_CFD}<>"$c" {SES_AFD}<>"$a" || die "GLAS_SESSION: cannot open the session FIFOs"
  say "GLAS_SESSION start #$SES_GEN (pid $SES_PID)"; }
ses_kill(){   # stop the session (an in-flight band is dropped: its .tmp is never published) and close the FIFOs
  [ -n "$SES_PID" ] && { kill "$SES_PID" 2>/dev/null; wait "$SES_PID" 2>/dev/null; }
  [ -n "$SES_CFD" ] && exec {SES_CFD}>&-; [ -n "$SES_AFD" ] && exec {SES_AFD}<&-
  SES_PID=""; SES_CFD=""; SES_AFD=""; SES_PRE_I=""; rm -f "$WORK/ses.ctl" "$WORK/ses.ack"; return 0; }
ses_send(){ printf 'BAND %s %s %s %s\n' "$1" "$2" "$3" "$4" >&"$SES_CFD" || die "GLAS_SESSION: cannot send band $1"; }
# ses_band: band $i [q,qb) -> $out (sent now unless ses_prelaunch already sent it). Returns with rc=0 SES_N qnext set on
# its ACK; with rc=143 when a probe verdict (MATRIX / SC_FINAL) stopped the sieve (session killed, band dropped). A
# session that dies without the ACK is restarted once from this band (the last acknowledged q); twice is fatal.
ses_band(){
  local tries=0 line a b c d
  while :; do
    if [ "$SES_PRE_I" = "$i" ] && [ -n "$SES_PID" ]; then t0=$SES_PRE_T0
    else ses_up; rm -f "$out" "$out.tmp"; ses_send "$i" "$q" "$qb" "$out"; fi
    SES_PRE_I=""
    while :; do
      if IFS= read -r -t 3 -u "$SES_AFD" line; then
        read -r a b c d <<< "$line"
        if [ "$a" = ACK ] && [ "$b" = "$i" ]; then
          [ -s "$out" ] || [ "$c" = 0 ] || die "GLAS_SESSION: band $i acknowledged ($c relations) but $out is missing"
          SES_N=$c; qnext=$d; rc=0; return 0; fi
        [ "$a" = READY ] || say "WARNING: GLAS_SESSION: unexpected ack line '$line' (waiting for band $i)"
        continue
      fi
      if probe_reap && { [ -n "$MATRIX_DIR" ] || [ "$SC_FINAL" = 1 ]; }; then ses_kill; rc=143; return 0; fi
      inc_consumed; inc_reap; probe_launch poll; inc_start; las_poll; spec_poll
      kill -0 "$SES_PID" 2>/dev/null || break
    done
    # the session died: an ACK written just before the exit is still in the FIFO
    if IFS= read -r -t 0.2 -u "$SES_AFD" line; then read -r a b c d <<< "$line"
      [ "$a" = ACK ] && [ "$b" = "$i" ] && [ -s "$out" ] && { SES_N=$c; qnext=$d; rc=0; ses_kill; return 0; }; fi
    wait "$SES_PID" 2>/dev/null; local src=$?; SES_PID=""; tries=$(( tries + 1 ))
    [ "$tries" -lt 2 ] || die "GLAS_SESSION died twice on band $i q[$q,$qb) (rc=$src): $(grep -av '^#' "$WORK/glas_session.log" | tail -2 | tr '\n' ' ')"
    say "WARNING: GLAS_SESSION died on band $i q[$q,$qb) (rc=$src): $(grep -av '^#' "$WORK/glas_session.log" | tail -1) -- restarting from the last acknowledged band"
    ses_kill
  done; }
# ses_prelaunch <n>: band i was just acknowledged with n relations -- size and SEND band i+1 now, before band i is
# landed, with exactly the loop-top arithmetic (have+n, relq=n/w). Not with las (its band guard), STOP_RULE=cost (the
# predictive stop decides after landing), a resume pause, or near MAXCHUNKS: those keep the send at the loop top.
ses_prelaunch(){
  [ "$GLAS_SESSION" = 1 ] && [ -n "$SES_PID" ] && [ "$LAS" != 1 ] && [ "$STOP_RULE" != cost ] && [ -z "${RESUME_PAUSE_AT:-}" ] || return 0
  [ -z "$MATRIX_DIR" ] && [ "$SC_FINAL" = 0 ] && [ "$rc" = 0 ] && [ "$COFOK" = 1 ] && [ $(( LANDED + 2 )) -le "$MAXCHUNKS" ] || return 0
  local h2=$(( have + $1 )) cr2=$CHUNK_RELS r w2
  [ "$h2" -ge $(( FIRST_PROBE - CHUNK_RELS )) ] && cr2=$CHUNK_RELS_LATE
  r=$(python3 -c "print($1/$w)"); w2=$(python3 -c "print(max(20000,min(2*$w,int($cr2/$r))))")
  SES_PRE_W=$w2; SES_PRE_QB=$(( qnext + w2 )); SES_PRE_I=$(( i + 1 )); SES_PRE_T0=$(date +%s)
  ses_send "$SES_PRE_I" "$qnext" "$SES_PRE_QB" "$WORK/sband.$SES_PRE_I"; }

# =============================================================================================
echo "### [3/4] GPU lattice sieve + concurrent CADO probes (first probe at $FIRST_PROBE raw) $(date +%T)"
# CHUNK_W0: q-width of the FIRST chunk (later chunks are sized from the measured relations per unit q-width).
# 200000 is the c160/c169 value (~2M relations at q=2.2M); small test keys need far less (E0.7 on a c115 at
# 2 threads: 200000 was a 6.4M-relation, 5.5 h first chunk).
CHUNK_W0=${CHUNK_W0:-200000}
have=0; q=$QMIN; i=0; relq=""; w=$CHUNK_W0; NEXT_PROBE=$FIRST_PROBE; PROBEPID=""; PROBE_I=0; MATRIX_DIR=""; TOOLFAIL=0
# LANDED: chunks 1..LANDED are in their memfds.  LAST_PK: the longest prefix 1..K ever probed (a prefix
# is never probed twice).  INC_DONE: chunks 1..INC_DONE committed by incdup.  PROBE_RUN_NT / INC_RUN_PAR:
# threads of the running probe / dup2 groups allowed to the running incdup (the thread budget).
LANDED=0; LAST_PK=0; PROBE_RUN_NT=0; PROBE_INC=0; declare -a HAVE_AT; HAVE_AT[0]=0
INC_CACHE="$WORK/inc"; INC_PARF="$WORK/inc/par"; INC_DONE=0; INC_RUNK=0; INC_STEP=0; INC_FROM=0; INC_TO=0; declare -A STEP_END=(); INCPID=""; INC_RUN_PAR=0; INC_BROKEN=0; INC_FREED=0
# INC_FREE_PENDING: incdup failed while an INC probe was still purging the dd parts -- the truncation
# waits until that probe is reaped.  INC_WAIT_K: the commit at which an INC probe was held back once to
# wait for the just-landed chunk (probe_launch waits at most once).
INC_FREE_PENDING=0; INC_WAIT_K=0
[ "$INCDUP" = 1 ] && mkdir -p "$INC_CACHE" || INC_BROKEN=1
# matrix_ok <probe dir>: did this probe produce what the downstream consumes? (defined BEFORE the replay
# block, which uses it)
matrix_ok(){
  if [ "$DS_MODE" = cado ]; then
    grep -qa 'VERDICT MATRIX' "$1/filter.out" 2>/dev/null && [ -s "$1/$PG" ] && [ -s "$1/history" ]
  else [ -s "$MEM/rels_out.cyc" ]; fi
}
[ "$DS_MODE" = cado ] && PROBE_MO=1 || PROBE_MO=0
TP_MERGE=0
tp_merge(){   # STOP_MIN_EXCESS: $1 = accepted purge-only probe dir, $2 = threads. merge exactly as cado_filter160.sh step 5
  local d=$1 t0 fm v; t0=$(date +%s)
  "$CF/merge" -out "$d/history" -mat "$d/$PG" -target_density "$TD" -t "$2" > "$d/merge.log" 2>&1 || return 1
  fm=$(grep -a 'Final matrix' "$d/merge.log" | tail -1); [ -n "$fm" ] && [ -s "$d/history" ] || return 1
  v=$(grep -a 'VERDICT MATRIX' "$d/filter.out" | tail -1)
  echo "${v% purge-only} merge-only (two-phase, NT=$2, $(( $(date +%s)-t0 ))s) $fm" >> "$d/filter.out"
  say "two-phase merge on $2 threads in $(( $(date +%s)-t0 ))s: $fm"; }
tp_excess(){ grep -a 'VERDICT MATRIX' "$1/filter.out" 2>/dev/null | tail -1 | grep -a 'purge-only' | grep -oE 'sing_excess=-?[0-9]+' | cut -d= -f2; }
SC_N=0; SC_FINAL=0; SC_KEEP=""; [ "$STOP_RULE" = cost ] && . "$HERE/stop_cost.sh"   # sc_* (STOP_RULE=cost)
if [ "$SPEC_MERGE" = 1 ]; then . "$HERE/spec_merge.sh"   # spec_* (SPEC_MERGE=1)
else spec_poll(){ :; }; spec_choose(){ return 1; }; spec_end(){ :; }; spec_probe_hook(){ :; }; spec_kill(){ :; }; fi
# sc_after_probe <probe dir> <landed raw>: STOP_RULE=cost on a purge-only MATRIX probe below the backstop. Records the
# point, keeps its purged file, decides. Returns 0 = stop (MATRIX_DIR+TP_MERGE set, or SC_FINAL=1), 1 = sieve on.
sc_after_probe(){
  [ "$STOP_RULE" = cost ] || return 1
  sc_record "$1" || return 1; sc_keep "$1"; sc_decide probe "$2"
  # SPEC_MERGE: a prepared matrix may beat the stop's own path (its handoff + merge + replay)
  case "$SC_ACT" in stop_probe|stop_final) spec_choose "$SC_ACT" "$2" && return 0;; esac
  case "$SC_ACT" in
    stop_probe) sc_say "stop: merge the matrix of $1 (raw ${PROBE_AT:-?}) now"; MATRIX_DIR=$1; TP_MERGE=1; return 0;;
    stop_final) sc_say "stop: chunks beyond the probe are worth a final purge"; SC_FINAL=1; return 0;;
  esac; spec_poll; return 1; }
# sc_after_land <landed raw>: STOP_PREDICTIVE (cost) before the next chunk starts. 0 = stop (see above), 1 = sieve on.
sc_after_land(){
  [ "$STOP_RULE" = cost ] && [ "$STOP_PREDICTIVE" = 1 ] && [ "$SC_FINAL" = 0 ] && [ "$SC_N" -ge "${STOP_MINPTS:-2}" ] || return 1
  sc_decide land "$1"
  case "$SC_ACT" in stop_probe|stop_final) spec_choose "$SC_ACT" "$1" && return 0;; esac
  case "$SC_ACT" in
    stop_probe) sc_probe_kill; sc_say "predictive stop at landing: merge the kept matrix $SC_KEEP"; MATRIX_DIR=$SC_KEEP; TP_MERGE=1; return 0;;
    stop_final) sc_say "predictive stop at landing ($1 raw): final purge on every landed chunk"; SC_FINAL=1; return 0;;
  esac; return 1; }
inc_free(){   # truncate every incremental part + key shadow (their RAM back); the cache is dead after this
  [ "$INCDUP" = 1 ] && [ "$INC_FREED" = 0 ] || return 0
  local c s f; for c in $(seq 1 "$MAXCHUNKS"); do for s in $(seq 0 15); do
    printf -v f '%s/dd/%03d/s%02d'"$SL" "$MEM" "$c" "$s"; : > "$f"; printf -v f '%s/kk/%03d/s%02d' "$MEM" "$c" "$s"; : > "$f"; done; done
  INC_FREED=1; INC_BROKEN=1; }
inc_par_budget(){   # $1 = wanted dup2 groups -> allowed (>=1) under SIEVE_HOST_T + probe NT + groups*INC_TPG <= NCPU
  local p=$(( (NCPU - SIEVE_HOST_T - PROBE_RUN_NT) / INC_TPG )); [ "$p" -ge 1 ] || p=1; [ "$p" -le "$1" ] || p=$1; echo "$p"; }
inc_set_par(){ INC_RUN_PAR=$1; echo "$1" > "$INC_PARF.tmp" && mv -f "$INC_PARF.tmp" "$INC_PARF"; }
# INCDUP: each dup2 group loads the renumber table; a decompressed copy (12.8 GB at c169, too big for the 10 GiB /tmp)
# loads ~2x faster, so it is inflated ONCE into a memhold memfd in the background; incdup steps use it as soon as the
# marker exists (before that, renumber.gz). inc_renum prints the path to pass as INC_RENUMBER ('' = cado_incdup's default).
RNPID=""
renum_inflate(){ [ "$RENUMBER_FORMAT" != binary ] || return 0
  [ "$INCDUP" = 1 ] && [ -z "$RNPID" ] && [ ! -e "$WORK/renumber.plain.ok" ] || return 0
  ( t0=$(date +%s); gzip -dc "$WORK/freerel/renumber$RN" > "$MEM/renumber.plain" 2>/dev/null && [ -s "$MEM/renumber.plain" ] \
      && echo "$(( $(date +%s)-t0 ))" > "$WORK/renumber.plain.ok" || : > "$MEM/renumber.plain" ) & RNPID=$!; }
inc_renum(){
  if [ "$RENUMBER_FORMAT" = binary ]; then echo "$WORK/freerel/renumber$RN"
  else [ -e "$WORK/renumber.plain.ok" ] && echo "$MEM/renumber.plain"; fi
}
frwait(){   # never fatal: a bad background cache is deleted and cado_filter160.sh rebuilds it (~2 min)
  [ -n "${FRPID:-}" ] || return 0
  local frc=0; wait "$FRPID" || frc=$?; FRPID=""
  if [ "$frc" = 0 ] && grep -q nprimes= "$WORK/freerel/freerel.log" 2>/dev/null && grep -q c5filter=done "$WORK/freerel/freerel.log"; then
    say "freerel cache ready"
    renum_inflate
  else
    say "WARNING: background freerel failed (rc=$frc: $(tail -1 "$WORK/freerel/freerel.build.log" 2>/dev/null)) -- discarding it; the filter will rebuild"
    rm -f "$WORK/freerel/freerel.gz" "$WORK/freerel/freerel.gz.f" "$WORK/freerel/renumber$RN" "$WORK/freerel/freerel.log"
  fi; }
# ---- REPLAY MODE (test harness only): RELS_IN="<file> [<file> ...]" feeds pre-sieved relation
# files (in sieve order) through the IDENTICAL probe + downstream path, one file per chunk slot,
# instead of running the GPU sieve. Probes then run back-to-back as if the chunks had just arrived.
if [ -n "${RELS_IN:-}" ]; then
  # RESUME_FAST: decompress the checkpointed chunks PRELOAD_PAR at a time (sequential gzip -dc was ~10 s per chunk)
  if [ "${RESUME_FAST:-0}" = 1 ] && [ -n "${RESUME_Q:-}" ]; then
    # wait on THESE jobs only: a bare `wait` also waits for memhold / the freerel builder and never returns
    _j=0; _t0=$(date +%s); _pp=(); for f in $RELS_IN; do _j=$((_j+1)); [ "$_j" -le "$MAXCHUNKS" ] || die "replay: more files than chunk slots"
      { case "$f" in *.gz) gzip -dc "$f";; *) cat "$f";; esac > "$MEM/$(printf c%03d $_j)" || touch "$WORK/preload.fail"; } & _pp+=($!)
      [ "${#_pp[@]}" -ge "${PRELOAD_PAR:-10}" ] && { wait "${_pp[@]}"; _pp=(); }; done; [ "${#_pp[@]}" -gt 0 ] && wait "${_pp[@]}"
    [ -e "$WORK/preload.fail" ] && die "replay: parallel preload failed"
    say "resume (RESUME_FAST): $_j chunks decompressed in $(( $(date +%s)-_t0 ))s (${PRELOAD_PAR:-10} in parallel)"
  fi
  for f in $RELS_IN; do
    i=$((i+1)); [ "$i" -le "$MAXCHUNKS" ] || die "replay: more files than chunk slots"
    if [ "${RESUME_FAST:-0}" = 1 ] && [ -n "${RESUME_Q:-}" ]; then :; else
    case "$f" in *.gz) gzip -dc "$f";; *) cat "$f";; esac > "$MEM/$(printf c%03d $i)" || die "replay: cannot load $f"; fi
    n=$(grep -avc '^#' "$MEM/$(printf c%03d $i)"); have=$(( have + n ))
    # REPLAY_PACE_RATE=<rel/s> (measurement only, gpu/s97_specmerge.sh): from FIRST_PROBE on, a chunk "lands" no earlier
    # than n / rate s after the previous one, as if the GPU sieved it (probes on the previous prefix overlap that wait),
    # so background work (SPEC_MERGE) sees live-like time between probes
    if [ -n "${REPLAY_PACE_RATE:-}" ] && [ $(( have - n )) -ge "$FIRST_PROBE" ] && [ -n "${_PACE_T:-}" ]; then
      _tl=$(( _PACE_T + n / REPLAY_PACE_RATE )); while [ "$(date +%s)" -lt "$_tl" ]; do sleep 2; spec_poll; done; fi
    _PACE_T=$(date +%s)
    LANDED=$i; HAVE_AT[$i]=$have
    say "replay chunk $i <- $(basename "$f") +$n total $have"
    # RESUME_FAST=1 (resume only): just preload -- no synchronous incdup catch-up and no replay probes (they kept
    # the GPU idle for 45-80 min per resumed sandbox: key1f segment 3 spent ~80 min here). The live loop below
    # restarts the GPU sieve at once; incdup catches up in the background (INC_MAXBATCH-chunk steps) and the
    # first probe waits for the whole preloaded prefix (NEXT_PROBE >= have, set after this loop).
    [ "${RESUME_FAST:-0}" = 1 ] && [ -n "${RESUME_Q:-}" ] && continue
    SC_STEP=$n   # STOP_RULE=cost: the next step is about one more chunk like this one
    if sc_after_land "$have"; then   # predictive stop (cost): the replay has no sieve to kill
      if [ "$SC_FINAL" = 1 ]; then sc_final; SC_FINAL=0; fi
      [ -n "$SPEC_USE" ] || { tp_merge "$MATRIX_DIR" "$NCPU" || die "two-phase merge failed (see $MATRIX_DIR/merge.log)"; }; TP_MERGE=0; break
    fi
    [ "$have" -ge "$NEXT_PROBE" ] || continue
    if [ "$INC_BROKEN" = 0 ]; then   # replay: catch the cache up synchronously (same watchdog as live)
      frwait; [ -s "$WORK/freerel/renumber$RN" ] || { inc_free; say "incdup disabled: no freerel cache"; }
      while [ "$INC_BROKEN" = 0 ] && [ "$INC_DONE" -lt "$i" ]; do
        k=$((INC_STEP+1)); INC_FROM=$((INC_DONE+1)); INC_TO=$i      # batched: one step over chunks INC_FROM..i
        CADO_BIN="$CF" FRC="$WORK/freerel" INC_RENUMBER="$(inc_renum)" INC_PAR=$INC_PAR INC_GROUPS=$INC_GROUPS INC_NICE=0 \
          nice -n "$INC_NICE" python3 -c "$INC_PYL" "$$" timeout -k 10 "$INC_TIMEOUT" bash "$CF/cado_incdup.sh" "$INC_CACHE" "$MEM" \
          "$WORK/poly.cado" "$k" $(for _c in $(seq $INC_FROM $INC_TO); do printf '%s ' "$MEM/$(printf c%03d $_c)"; done) >> "$INC_CACHE.log" 2>&1 & INCPID=$!
        echo "$INCPID" > "$WORK/inc.pgid"; wait "$INCPID"; irc=$?; INCPID=""
        if [ "$irc" = 0 ] && [ -s "$INC_CACHE/chunk_$k.done" ]; then INC_STEP=$k; STEP_END[$k]=$INC_TO; INC_DONE=$INC_TO
          # FREE_RAW: same as the live reap path -- the replay kept every raw chunk's memfd and OOM-killed purge at probe 10
          # of the key-29 container replay (74.5 GB before purge; the live run's pool is ~72 MB per M raw) -- 2026-10-25
          [ "$FREE_RAW" = 1 ] && for _c in $(seq "$INC_FROM" "$INC_TO"); do : > "$MEM/$(printf c%03d "$_c")"; done
        else inc_free; say "WARNING: incdup chunk $k failed (rc=$irc$([ "$irc" = 124 ] && echo ', watchdog timeout')) -- falling back to one-shot probes: $(grep -a FAIL "$INC_CACHE.log" | tail -1)"; fi
      done
    fi
    frwait; PROBE_I=$((PROBE_I+1)); PROBE_N=$i; PROBE_AT=$have; LAST_PK=$i; mkdir -p "$WORK/probe$PROBE_I"; pp_link "$WORK/probe$PROBE_I" "$PROBE_I"
    CH=(); for k in $(seq 1 "$i"); do CH+=("$MEM/$(printf c%03d $k)"); done
    : > "$MEM/rels_out"; rm -f "$MEM/rels_out.cyc" "$MEM/rels_out.purged"
    INCENV=(); [ "$INC_BROKEN" = 0 ] && INCENV=(INC_CACHE="$INC_CACHE" INC_K="$INC_STEP")
    env "${INCENV[@]}" DELTA_SING=$DPS DELTA_SING_BG=0 MERGE_ONLY=$PROBE_MO PURGE_ONLY=$TP_PO LPB0=$LPB0 LPB1=$LPB1 NT=$PROBE_NT DUP2_PAR=$DUP2_PAR TD=$TD FRC="$WORK/freerel" \
      bash "$CF/cado_filter160.sh" "$WORK/probe$PROBE_I" "$MEM" "$WORK/poly.cado" "$N" "$MEM/rels_out" "${CH[@]}" \
      > "$WORK/probe$PROBE_I/filter.out" 2>&1; prc=$?
    say "replay probe $PROBE_I ($have raw$([ "${#INCENV[@]}" -gt 0 ] && echo ', incremental cache')) rc=$prc: $(grep -a VERDICT "$WORK/probe$PROBE_I/filter.out" | tail -1 | sed 's/.*VERDICT //')"
    if [ "$prc" = 0 ] && [ "$TP_PO" = 1 ]; then   # two-phase: purge-only verdict
      _sx=$(tp_excess "$WORK/probe$PROBE_I")
      if [ -n "$_sx" ] && [ "$_sx" -ge "$STOP_MIN_EXCESS" ]; then
        tp_merge "$WORK/probe$PROBE_I" "$NCPU" || die "two-phase merge failed (see $WORK/probe$PROBE_I/merge.log)"
        MATRIX_DIR="$WORK/probe$PROBE_I"; break
      fi
      _rk=0
      if [ -n "$_sx" ] && sc_after_probe "$WORK/probe$PROBE_I" "$have"; then   # cost rule: stop on this prefix (= all landed)
        [ "$SC_FINAL" = 1 ] && { sc_final; SC_FINAL=0; }
        [ -n "$SPEC_USE" ] || { tp_merge "$MATRIX_DIR" "$NCPU" || die "two-phase merge failed (see $MATRIX_DIR/merge.log)"; }; TP_MERGE=0; break
      fi
      [ "$STOP_RULE" = cost ] && [ "$SC_KEEP" = "$WORK/probe$PROBE_I" ] && _rk=1
      say "two-phase: replay probe $PROBE_I excess ${_sx:-?} < STOP_MIN_EXCESS $STOP_MIN_EXCESS -- counted as SHORT"
      # TP_KEEP: a hard link (not mv), so a cost-rule kept purged file stays where the stop would merge it
      [ "$TP_KEEP" = 1 ] && ln -f "$WORK/probe$PROBE_I/$PG" "$WORK/probe$PROBE_I/keep.$PG" 2>/dev/null
      [ "$_rk" = 1 ] || rmf "$WORK/probe$PROBE_I"/purged*; prc=2
    fi
    if [ "$prc" = 0 ] && matrix_ok "$WORK/probe$PROBE_I"; then MATRIX_DIR="$WORK/probe$PROBE_I"; break; fi
    [ "$prc" = 2 ] || die "replay probe tooling failure: $(grep -a FAILED "$WORK/probe$PROBE_I/filter.out" | tail -1)"
    NEXT_PROBE=$(( have + PROBE_STEP ))
    # cost rule, after the first matrix: probe every STOP_PROBE_EVERY chunks
    [ "$STOP_RULE" = cost ] && [ "$SC_N" -ge 1 ] && NEXT_PROBE=$(( have + 1 + (STOP_PROBE_EVERY - 1) * n ))
  done
  # RESUME (checkpointed run on a reclaimed sandbox): RESUME_Q=<q to continue at> [RESUME_W=<band width>] with
  # RELS_IN = the checkpointed chunks (gpu/ckpt_pull.sh) -> preload them as slots and carry on with the GPU sieve
  if [ -z "$MATRIX_DIR" ] && [ -n "${RESUME_Q:-}" ]; then
    q=$RESUME_Q; w=${RESUME_W:-$w}; say "resume: $i chunks preloaded, $have raw relations, GPU sieve continues at q=$q (band width $w)"
    if [ "${RESUME_FAST:-0}" = 1 ]; then [ "$NEXT_PROBE" -ge "$have" ] || NEXT_PROBE=$have
      INC_MAXBATCH=${INC_MAXBATCH:-24}; say "resume (RESUME_FAST): incdup catches up in the background in <= $INC_MAXBATCH-chunk steps, first probe at >= $NEXT_PROBE raw"; fi
  else
    if [ -z "$MATRIX_DIR" ] && [ "$STOP_RULE" = cost ] && [ -n "$SC_KEEP" ] && [ -s "$SC_KEEP/$PG" ]; then
      sc_say "replay input exhausted before the rule stopped: merging the last kept matrix $SC_KEEP"; spec_end "replay input exhausted"
      tp_merge "$SC_KEEP" "$NCPU" || die "two-phase merge failed (see $SC_KEEP/merge.log)"; MATRIX_DIR=$SC_KEEP; TP_MERGE=0
    fi
    [ -n "$MATRIX_DIR" ] || die "replay: all input relations used, no matrix"
    [ "$DS_MODE" = cado ] || rmf "$MATRIX_DIR"/purged* "$MATRIX_DIR"/history*
  fi
fi
probe_reap(){   # sets MATRIX_DIR on success; returns 0 if a probe finished (either verdict)
  [ -n "$PROBEPID" ] || return 1
  kill -0 "$PROBEPID" 2>/dev/null && return 1
  wait "$PROBEPID"; local rc=$?; PROBEPID=""; spec_probe_hook reap
  local v; v=$(grep -a 'VERDICT' "$WORK/probe$PROBE_I/filter.out" | tail -1)
  say "probe $PROBE_I (chunks 1..$PROBE_N, $PROBE_AT raw) rc=$rc: ${v#*VERDICT }"
  # an incdup step failed while this (incremental) probe was reading the dd parts: free them only NOW
  if [ "$INC_FREE_PENDING" = 1 ]; then INC_FREE_PENDING=0; inc_free; say "incremental cache freed (deferred until probe $PROBE_I ended)"; fi
  PROBE_RUN_NT=0
  local _rk=0
  if [ "$TP_PO" = 1 ] && [ "$rc" = 0 ]; then   # two-phase: a purge-only MATRIX verdict
    local _sx; _sx=$(tp_excess "$WORK/probe$PROBE_I")
    if [ -n "$_sx" ] && [ "$_sx" -ge "$STOP_MIN_EXCESS" ]; then
      say "two-phase: probe $PROBE_I singleton excess $_sx >= STOP_MIN_EXCESS $STOP_MIN_EXCESS -- sieve stops, merge follows (NT=$MERGE_NT)"
      TP_MERGE=1; MATRIX_DIR="$WORK/probe$PROBE_I"; return 0
    fi
    # STOP_RULE=cost: this probe is a point; stop now (its matrix, or a final purge over every landed chunk) or sieve on
    [ -n "$_sx" ] && sc_after_probe "$WORK/probe$PROBE_I" "${HAVE_AT[$LANDED]}" && return 0
    [ "$STOP_RULE" = cost ] && [ "$SC_KEEP" = "$WORK/probe$PROBE_I" ] && _rk=1
    say "two-phase: probe $PROBE_I has a matrix at singleton excess ${_sx:-?} < STOP_MIN_EXCESS $STOP_MIN_EXCESS -- sieving on (counted as SHORT$([ "$_rk" = 1 ] && echo ', purged file kept for the cost rule'))"
    TP_SEEN=1
    [ "$TP_KEEP" = 1 ] && ln -f "$WORK/probe$PROBE_I/$PG" "$WORK/probe$PROBE_I/keep.$PG" 2>/dev/null
    rc=2
  fi
  if [ "$rc" = 0 ] && matrix_ok "$WORK/probe$PROBE_I"; then MATRIX_DIR="$WORK/probe$PROBE_I"; return 0; fi
  [ "$_rk" = 1 ] || rmf "$WORK/probe$PROBE_I"/purged*
  rmf "$WORK/probe$PROBE_I"/history*; rm -f "$MEM/rels_out.cyc" "$MEM/rels_out.purged"; : > "$MEM/rels_out"
  if [ "$rc" = 2 ]; then
    TOOLFAIL=0; NEXT_PROBE=$(( have > PROBE_AT + PROBE_STEP ? have : PROBE_AT + PROBE_STEP ))
    # INCREMENTAL probes only ever see the COMMITTED prefix: requiring `have` (everything landed by now)
    # would make them wait for incdup to catch up with the sieve, and when an incdup step takes longer
    # than a chunk that gap grows every round (measured with stubs: probes on 1..3, 1..5, 1..8, 1..13).
    # The next committed chunk past PROBE_AT + PROBE_STEP is enough; `have` stays the one-shot rule.
    [ "$PROBE_INC" = 1 ] && [ "$INC_BROKEN" = 0 ] && NEXT_PROBE=$(( PROBE_AT + PROBE_STEP ))
    # late-run fallback (c169): past LATE_SECS of wall with no matrix, stop paying for overshoot margin --
    # no prediction factor, half the probe step. (purge's -keep / required_excess are NOT levers: a
    # matrix exists exactly when the singleton excess is positive; keep=160 of ~155M relations is noise.)
    if [ "$LATE_SECS" -gt 0 ] && [ $(( $(date +%s) - T0 )) -ge "$LATE_SECS" ] && [ "$LATE_ON" = 0 ]; then
      LATE_ON=1; PRED_FACTOR=1.0; PROBE_STEP=$(( PROBE_STEP / 2 ))
      say "WARNING: ${LATE_SECS}s of wall without a matrix -- late mode: PRED_FACTOR 1.0, PROBE_STEP $PROBE_STEP"
    fi
    # predictive probing (c169): from the last two SHORT probes' singleton excess, extrapolate the crossover
    if [ "$PREDICT_PROBE" = 1 ]; then
      local _e _p; _e=$(echo "$v" | grep -oE 'sing_excess=-?[0-9]+' | cut -d= -f2)
      if [ -n "$_e" ]; then
        PRB_R+=("$PROBE_AT"); PRB_E+=("$_e")
        if [ "${#PRB_R[@]}" -ge 2 ]; then
          _p=$(python3 -c "
import sys
r1,e1,r2,e2,t,f=map(float,sys.argv[1:7])
print(int((r2+(t-e2)*(r2-r1)/(e2-e1))*f) if e2>e1 and e2<t else 0)" \
              "${PRB_R[-2]}" "${PRB_E[-2]}" "${PRB_R[-1]}" "${PRB_E[-1]}" "$PRED_TARGET_EXCESS" "$PRED_FACTOR" 2>/dev/null || echo 0)
          if [ "${_p:-0}" -gt 0 ] 2>/dev/null; then
            local _lo=$(( PROBE_AT + PROBE_STEP / 2 )) _hi=$(( PROBE_AT + 4 * PROBE_STEP ))
            [ "$_p" -lt "$_lo" ] && _p=$_lo; [ "$_p" -gt "$_hi" ] && _p=$_hi
            say "probe prediction: excess ${PRB_E[-2]} @ ${PRB_R[-2]} -> $_e @ $PROBE_AT; crossover x$PRED_FACTOR = $_p raw (was $NEXT_PROBE)"
            NEXT_PROBE=$_p
          fi
        fi
      fi
    fi
    # STOP_RULE=cost, after the first matrix: the next probe on the next committed chunk (every STOP_PROBE_EVERY chunks);
    # one probe at a time, so a probe longer than a chunk interval just skips commits
    if [ "$STOP_RULE" = cost ] && [ "$SC_N" -ge 1 ]; then
      NEXT_PROBE=$(( PROBE_AT + 1 + (STOP_PROBE_EVERY - 1) * ${SC_STEP:-$CHUNK_RELS_LATE} ))
      say "stop rule (cost): next probe on the first committed prefix >= $NEXT_PROBE raw"
    fi
    # the probe's threads are free again: let a running incdup widen (re-read before each dup2 group)
    [ -n "$INCPID" ] && inc_set_par "$(inc_par_budget "$([ "${HAVE_AT[$INC_TO]}" -ge "$NEXT_PROBE" ] && echo "$INC_PAR_PROBE" || echo "$INC_PAR")")"
  elif [ "$PROBE_INC" = 1 ]; then       # the incremental path failed: fall back for good (one-shot at the next landing)
    # FREE_RAW=1: the committed raw chunks are already truncated, so a one-shot probe would dedup a silent subset and
    # never reach a matrix. Use the cost rule's kept matrix if there is one, else stop loudly.
    if [ "$FREE_RAW" = 1 ]; then
      if [ -n "${SC_KEEP:-}" ] && [ -s "$SC_KEEP/$PG" ]; then
        say "WARNING: incremental probe tooling failure (rc=$rc) with FREE_RAW=1 -- no one-shot fallback; stopping on the kept matrix $SC_KEEP"
        inc_kill; MATRIX_DIR=$SC_KEEP; TP_MERGE=1; return 0; fi
      die "incremental probe tooling failure (rc=$rc) with FREE_RAW=1 (raw chunks freed, no kept matrix): $(grep -a 'FAILED' "$WORK/probe$PROBE_I/filter.out" | tail -1)"
    fi
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
  if [ "$rc" = 0 ] && [ -s "$INC_CACHE/chunk_$INC_RUNK.done" ]; then
    INC_STEP=$INC_RUNK; STEP_END[$INC_STEP]=$INC_TO; INC_DONE=$INC_TO
    # FREE_RAW: the committed chunks' raw relations have no reader left (MERGE_ONLY probes use dd/ parts,
    # sqrt uses purged.gz): give the memfds' RAM back now
    [ "$FREE_RAW" = 1 ] && for _c in $(seq "$INC_FROM" "$INC_TO"); do : > "$MEM/$(printf c%03d "$_c")"; done
  elif [ "$FREE_RAW" = 1 ]; then
    die "incdup chunk $INC_RUNK failed (rc=$rc$([ "$rc" = 124 ] || [ "$rc" = 137 ] && echo ", watchdog timeout ${INC_TIMEOUT}s")): $(grep -a FAIL "$INC_CACHE.log" | tail -1) -- FREE_RAW=1 has no one-shot fallback"
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
  [ $(( LANDED - INC_DONE )) -ge "${INC_LAG:-1}" ] || return 0   # INC_LAG (test only): force multi-chunk steps
  if [ -n "${FRPID:-}" ]; then kill -0 "$FRPID" 2>/dev/null && return 0; frwait; fi
  if ! { [ -s "$WORK/freerel/renumber$RN" ] && grep -q c5filter=done "$WORK/freerel/freerel.log" 2>/dev/null; }; then
    inc_free; say "incdup disabled: no freerel cache"; return 0; fi
  INC_RUNK=$((INC_STEP+1)); INC_FROM=$((INC_DONE+1)); INC_TO=$LANDED   # batched: every landed, uncommitted chunk
  # INC_MAXBATCH (set by RESUME_FAST): cap a step so a resumed backlog of ~100 chunks never meets INC_TIMEOUT in one step
  [ -n "${INC_MAXBATCH:-}" ] && [ $(( INC_TO - INC_FROM + 1 )) -gt "$INC_MAXBATCH" ] && INC_TO=$(( INC_FROM + INC_MAXBATCH - 1 ))
  local want=$INC_PAR; [ -z "$PROBEPID" ] && [ "${HAVE_AT[$INC_TO]}" -ge "$NEXT_PROBE" ] && want=$INC_PAR_PROBE
  inc_set_par "$(inc_par_budget "$want")"
  # own session + PDEATHSIG (INC_PYL): inc_kill takes down timeout + its dup1/dup2 as ONE group; the
  # watchdog (timeout -k) kills the same group on a hang; our death TERMs it. Whole tree at nice INC_NICE
  # (INC_NICE=0 inside). The pgid is recorded for the next attempt's startup check (top of script).
  CADO_BIN="$CF" FRC="$WORK/freerel" INC_RENUMBER="$(inc_renum)" INC_PAR=$INC_RUN_PAR INC_PAR_FILE="$INC_PARF" INC_GROUPS=$INC_GROUPS INC_NICE=0 \
    nice -n "$INC_NICE" python3 -c "$INC_PYL" "$$" \
    timeout -k 10 "$INC_TIMEOUT" bash "$CF/cado_incdup.sh" "$INC_CACHE" "$MEM" "$WORK/poly.cado" "$INC_RUNK" \
    $(for _c in $(seq "$INC_FROM" "$INC_TO"); do printf '%s ' "$MEM/$(printf c%03d "$_c")"; done) \
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
    INCENV=(INC_CACHE="$INC_CACHE" INC_K="$INC_STEP")   # K chunks committed in INC_STEP (batched) steps
    # thread budget: sieve host + one dup2 group for the next chunk's incdup + 1 for purge's `gzip -dc`
    # input reader (not counted in -t). incdup's dup1 phase (16 `gzip -c --fast` writers, ~3 cores for
    # a few seconds per c160 chunk, measured) is NOT in the budget: it precedes the dup2 groups, its CPU
    # demand fits the INC_TPG it later uses, and it runs at nice 15 (CFS weight 36 vs 335 for this
    # nice-5 probe), so the brief oversubscription in RUNNABLE threads costs the probe ~nothing.
    NT=$(( NCPU - SIEVE_HOST_T - INC_TPG - 1 )); [ "$NT" -le "$PROBE_NT" ] || NT=$PROBE_NT; [ "$NT" -ge 2 ] || NT=2
    # once a probe has had a matrix (two-phase SHORT), the next one may stop the sieve: its purge is on the critical
    # path, so it takes PROBE_LATE_NT threads (default 16 = one per incremental dd slice; 12 read them in two waves).
    # s53b (key1f chunks, 524M raw): purge 230 s at -t 12 -> 197 s at -t 16, same matrix. glas's host threads are not on
    # the sieve's critical path (s49: -t 2..16 same span), so the oversubscription only costs the 4 reserved cores.
    # Kept at 16 (2026-10-23, key1g run_seg2_sb48): a matrix-era probe slows the GPU chunk it overlaps by the SAME ~22 s
    # at NT 12 (probe 4, 147 s overlap: chunk 55 234 s) and NT 16 (probe 5, 146 s: chunk 61 241 s) vs 209-218 s
    # neighbours, while the ~50 s SHORT probes at NT 12 cost nothing visible (chunks 40/43: 208/204 s) -- the tax
    # tracks purge's pass-2 memory traffic, not its thread count, so NT 8 would lengthen the probe (and the stop
    # lag) without evidence of a smaller tax. STOP_RULE=cost charges it per step (STOP_PROBE_TAX 20 s) instead.
    [ "$TP_SEEN" = 1 ] && [ "$NT" -lt "$PROBE_LATE_NT" ] && NT=$PROBE_LATE_NT
  else
    # ONE-SHOT (INCDUP=0, or after a fallback): only when a chunk LANDS -- production cadence exactly
    [ "$1" = land ] || return 0
    K=$LANDED
  fi
  [ "$K" -gt "$LAST_PK" ] && [ "${HAVE_AT[$K]}" -ge "$NEXT_PROBE" ] || return 0
  frwait; PROBE_I=$((PROBE_I+1)); PROBE_N=$K; PROBE_AT=${HAVE_AT[$K]}; LAST_PK=$K; mkdir -p "$WORK/probe$PROBE_I"; pp_link "$WORK/probe$PROBE_I" "$PROBE_I"
  PROBE_INC=0; [ "${#INCENV[@]}" -gt 0 ] && PROBE_INC=1; INC_WAIT_K=0
  CH=(); for k in $(seq 1 "$K"); do CH+=("$MEM/$(printf c%03d $k)"); done
  : > "$MEM/rels_out"; rm -f "$MEM/rels_out.cyc" "$MEM/rels_out.purged"
  ( exec env "${INCENV[@]}" DELTA_SING=$DPS DELTA_SING_BG=1 MERGE_ONLY=$PROBE_MO PURGE_ONLY=$TP_PO LPB0=$LPB0 LPB1=$LPB1 NT=$NT DUP2_PAR=$DUP2_PAR TD=$TD FRC="$WORK/freerel" \
      nice -n 5 bash "$CF/cado_filter160.sh" "$WORK/probe$PROBE_I" "$MEM" "$WORK/poly.cado" "$N" "$MEM/rels_out" "${CH[@]}" \
      > "$WORK/probe$PROBE_I/filter.out" 2>&1 ) & PROBEPID=$!; PROBE_RUN_NT=$NT; spec_probe_hook launch
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
las_budget(){   # CPUs las may use right now: whatever sieve host + probe + incdup + freerel leave free (of LAS_NCPU)
  local b=$(( LAS_NCPU - SIEVE_HOST_T - LAS_RESERVE - ( COFAC_CPU == 1 ? COFAC_T : 0 ) ))
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
  # LAS_FB_REUSE: the sieve's own factor base (las/glas main siever, same A and lim1) -- no second makefb
  if [ "$LAS_FB_REUSE" = 1 ] && [ "$SIEVER" != bench ] && [ "$LAS_A" = "$SIEVE_A" ] && [ "$LAS_LIM1" = "$ALIM" ] && [ -s "$WORK/fb1" ] \
     && { ln -f "$WORK/fb1" "$LASD/fb1" 2>/dev/null || cp -f "$WORK/fb1" "$LASD/fb1" 2>/dev/null; } && [ -s "$LASD/fb1" ]; then
    lsay "factor base: the sieve's fb1 reused ($(stat -c%s "$LASD/fb1" 2>/dev/null || echo ?) bytes); co-sieve q from $LAS_QMIN, -t $LAS_T nice $LAS_NICE, landing=$LAS_LAND"
    las_launch; return 0
  fi
  local t; t=$(las_budget); [ "$t" -ge 1 ] || t=1
  ( exec nice -n "$LAS_NICE" "$CF/makefb" -poly "$WORK/poly.cado" -side 1 -lim "$LAS_LIM1" -maxbits "$(( (LAS_A + 1) / 2 ))" \
        -t "$t" -out "$LASD/fb1" > "$LASD/makefb.log" 2>&1 ) & LAS_FBPID=$!
  LAS_STATE=fb
  lsay "makefb side 1 lim $LAS_LIM1 -t $t started; co-sieve q from $LAS_QMIN, -t $LAS_T nice $LAS_NICE, landing=$LAS_LAND"; }
las_launch(){   # the next q-chunk [LAS_Q, LAS_Q + LAS_W)
  LAS_J=$((LAS_J+1)); LAS_Q1=$(( LAS_Q + LAS_W ))
  rm -f "$LASD/c$LAS_J.out" "$LASD/c$LAS_J.out".tmp.*
  ( exec nice -n "$LAS_NICE" "$CF/las" -poly "$WORK/poly.cado" -fb1 "$LASD/fb1" -lim0 "$LAS_LIM0" -lim1 "$LAS_LIM1" \
        -lpb0 "$LAS_LPB0" -lpb1 "$LAS_LPB1" -mfb0 "$LAS_MFB0" -mfb1 "$LAS_MFB1" -lambda0 "$LAMBDA0" -lambda1 "$LAMBDA1" \
        -ncurves0 "$NCURVES0" -ncurves1 "$NCURVES1" -A "$LAS_A" -q0 "$LAS_Q" -q1 "$LAS_Q1" \
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
    if g=$(python3 "$HERE/las_ctl.py" gate "$WORK/poly.cado" "$LAS_LPB0" "$LAS_LPB1" "$r" 2000 2>&1); then LAS_GATED=1; lsay "relation gate ${g#GATE }"
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
while [ -z "$MATRIX_DIR" ] && [ "$SC_FINAL" = 0 ]; do
  # RESUME_PAUSE_AT (raw, resume only): sieve up to the predicted stop, then let the background incdup catch up and a
  # probe see every landed chunk before sieving further -- no overshoot past the stop a single-box run would make.
  # The GPU is idle during the pause: restart overhead, excluded from the stitched benchmark.
  if [ -n "${RESUME_PAUSE_AT:-}" ] && [ "$have" -ge "$RESUME_PAUSE_AT" ] && [ "$INC_BROKEN" = 0 ]; then
    [ "$NEXT_PROBE" -le "$have" ] || NEXT_PROBE=$have   # the probe on chunks 1..LANDED must be due, or the pause never ends
    tp=$(date +%s); say "resume: $have raw >= RESUME_PAUSE_AT $RESUME_PAUSE_AT -- GPU waits for incdup ($INC_DONE/$LANDED) and a probe on chunks 1..$LANDED"
    while [ -z "$MATRIX_DIR" ] && [ "$INC_BROKEN" = 0 ] && { [ "$INC_DONE" -lt "$LANDED" ] || [ -n "$PROBEPID" ] || [ "$LAST_PK" -lt "$LANDED" ]; }; do
      sleep 5; probe_reap || true; { [ -n "$MATRIX_DIR" ] || [ "$SC_FINAL" = 1 ]; } && break; inc_consumed; inc_reap; probe_launch poll; inc_start; spec_poll
    done
    say "resume: pause over after $(( $(date +%s)-tp ))s [restart overhead]"; RESUME_PAUSE_AT=""
    { [ -n "$MATRIX_DIR" ] || [ "$SC_FINAL" = 1 ]; } && break
  fi
  # i = GPU chunk number; its SLOT in the chunk pool is assigned when it lands (LANDED+1): with LAS_LAND=slot,
  # las chunks take slots in between. Every consumer (HAVE_AT, LANDED, incdup's INC_DONE, probe prefixes
  # 1..K) is indexed by SLOT, in landing order.
  i=$((i+1)); [ $(( LANDED + 1 )) -le "$MAXCHUNKS" ] || die "ran out of chunk slots ($MAXCHUNKS) without a matrix"
  cr=$CHUNK_RELS; [ "$have" -ge $(( FIRST_PROBE - CHUNK_RELS )) ] && cr=$CHUNK_RELS_LATE
  if [ "$SES_PRE_I" = "$i" ]; then w=$SES_PRE_W; qb=$SES_PRE_QB   # GLAS_SESSION: already sized and sent at band i-1's ACK
  else
  if [ -n "$relq" ]; then w=$(python3 -c "print(max(20000,min(2*$w,int($cr/$relq))))"); fi
  qb=$(( q + w )); las_band_guard
  fi
  out="$WORK/chunk.dat"; rm -f "$out" "$out".part "$out".part.ckpt "$out".lock "$STOP"
  [ "$GLAS_SESSION" = 1 ] && out="$WORK/sband.$i"   # one file per band: band i+1 is written while band i lands
  CAND="$WORK/cand.dat"; COFREL="$WORK/cofac.rel"; rm -f "$CAND" "$CAND".part "$CAND".done "$COFREL"
  COFARGS=(--cofactor); [ "$COFAC_CPU" = 1 ] && COFARGS=(--candidates "$CAND")
  t0=$(date +%s); RESUMED=0; COFOK=1; RESTART=()
  if [ "$GLAS_SESSION" = 1 ]; then ses_band   # send (unless prelaunched) + wait for the ACK; rc 143 = stopped by a probe
  else
  siever_launch "$q" "$qb" "$out" "$WORK/bench.log"; BENCHPID=$!
  # CPU cofactorisation of this band's candidates, concurrently (tails $CAND.part until $CAND.done exists)
  [ "$COFAC_CPU" = 1 ] && { "$CS/cofac_cpu" -lpb0 "$LPB0" -lpb1 "$LPB1" -mfb0 "$MFB0" -mfb1 "$MFB1" -fbb0 "$RLIM" -fbb1 "$ALIM" -t "$COFAC_T" \
      -follow "$CAND.done" "$CAND.part" "$COFREL" 2> "$WORK/cofac.log" & COFPID=$!; }
  # while the GPU sieves this chunk, watch the probe; a MATRIX verdict stops the sieve early
  while kill -0 "$BENCHPID" 2>/dev/null; do
    sleep 3
    # (STOP_RULE=cost: SC_FINAL = stop now, a final purge over every landed chunk follows; the in-flight chunk is dropped)
    if probe_reap && { [ -n "$MATRIX_DIR" ] || [ "$SC_FINAL" = 1 ]; }; then touch "$STOP"; kill "$BENCHPID" 2>/dev/null; break; fi
    inc_consumed; inc_reap; probe_launch poll; inc_start; las_poll; spec_poll
  done
  wait "$BENCHPID"; rc=$?; BENCHPID=""
  fi
  { [ -n "$MATRIX_DIR" ] || [ "$SC_FINAL" = 1 ]; } && { [ -n "${COFPID:-}" ] && { kill "$COFPID" 2>/dev/null; wait "$COFPID" 2>/dev/null; COFPID=""; }
                           say "$([ -n "$MATRIX_DIR" ] && echo 'MATRIX found' || echo 'stop rule: final purge') -- sieve stopped mid-chunk $i (in-flight chunk dropped)"; break; }
  qnext=$qb
  if [ "$rc" != 0 ] && [ ! -s "$out" ]; then
    say "WARNING: bench rc=$rc on q[$q,$qb): $(tail -1 "$WORK/bench.log") -- resuming once from its checkpoint"
    rm -f "$out".lock; RESUMED=1
    # no checkpoint yet (a crash inside the band's first ~30 s; without --cofactor bench writes no seed
    # checkpoint): a plain resume REFUSES a non-empty uncheckpointed .part -> restart the band instead
    RESTART=(); [ -e "$out".part.ckpt ] || RESTART=(--restart)
    siever_launch "$q" "$qb" "$out" "$WORK/bench.resume.log"; wait $!; rc=$?
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
  # las/glas append their per-q statistics as '#' lines (~1/3 of the bytes): strip them before landing
  if [ "$SIEVER" != bench ] && [ "$GLAS_SESSION" != 1 ] && [ -s "$out" ]; then grep -av '^#' "$out" > "$out.s" && mv -f "$out.s" "$out"; fi
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
      "$CS/cofac_cpu" -lpb0 "$LPB0" -lpb1 "$LPB1" -mfb0 "$MFB0" -mfb1 "$MFB1" -fbb0 "$RLIM" -fbb1 "$ALIM" -t "$rt" -follow "$CAND.done" "$cin" "$COFREL" \
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
  if [ "$GLAS_SESSION" = 1 ]; then n=$SES_N   # glas counted the lines it published (relation lines only)
    [ "${GLAS_SESSION_VERIFY:-0}" = 1 ] && { _vn=$(grep -avc '^#' "$out" 2>/dev/null); [ "${_vn:-0}" = "$n" ] || die "GLAS_SESSION: band $i ACK says $n relations, file has ${_vn:-0}"; }
  else
  n=$(grep -avc '^#' "$out" 2>/dev/null); n=${n:-0}
  fi
  [ "$n" -gt 0 ] || die "chunk $i produced no relations (rc=$rc): $(tail -3 "$WORK/bench.log")"
  ses_prelaunch "$n"   # GLAS_SESSION: band i+1 starts on the GPU now, while band i is landed below
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
    if [ "$SIEVER" = bench ]; then
      "$CS/bench" --check-relations "$WORK/gate.dat" --poly "$WORK/c160.job" --lpb "$LPB1" --lpb0 "$LPB0" > "$WORK/gate.log" 2>&1
    else   # exact check: the listed primes multiply to |G(a,b)| and |F(a,b)|, all below 2^lpb (las_ctl.py)
      python3 "$HERE/las_ctl.py" gate "$WORK/poly.cado" "$LPB0" "$LPB1" "$WORK/gate.dat" 2000 > "$WORK/gate.log" 2>&1
    fi
    grep -qa PASS "$WORK/gate.log" || die "relation gate FAILED: $(tail -2 "$WORK/gate.log")"
    say "relation gate PASS"; rm -f "$WORK/gate.dat"
  fi
  rm -f "$out" "$out".part "$out".part.ckpt "$out".lock "$CAND" "$CAND".part "$CAND".done "$COFREL"
  # relq (relations per unit q-width) sizes the next band: not from a chunk that lost its CPU-cofactored half
  have=$(( have + n + nl )); HAVE_AT[$slot]=$have; LANDED=$slot; [ "$rc" = 0 ] && [ "$COFOK" = 1 ] && relq=$(python3 -c "print($n/$w)")
  say "chunk $i q[$q,$qnext) +$n in ${dt}s ($(( n / (dt>0?dt:1) )) rel/s)$([ "$nl" -gt 0 ] && echo " + las $nl") -> slot $slot  total $have  load $(cut -d' ' -f1 /proc/loadavg)"; res_note "chunk $i landed"
  q=$qnext
  SC_RATES+=("$(( n / (dt>0?dt:1) ))"); SC_STEP=$n   # STOP_RULE=cost: measured rel/s and step size
  # CHECKPOINT (CKPT_URL/CKPT_KEY set, gpu/ckpt_push.sh): ship the landed slot and the resume state off the box in the
  # background (serialised by the pusher's lock); a reclaimed sandbox then resumes with RELS_IN + RESUME_Q
  CKPT_PUSH=${CKPT_PUSH:-$HERE/../gpu/ckpt_push.sh}
  if { [ -n "${CKPT_URL:-}" ] || [ -n "${CKPT_HF_REPO:-}" ] || [ -f "${CKPT_ENV:-/nonexistent}" ]; } && [ -f "$CKPT_PUSH" ]; then
    # CKPT_SLOT0/CKPT_HAVE0: a sieve-only continuation run (gpu/resume_key1.sh phase A) numbers its slots after the
    # checkpointed ones, so the store stays one consistent sequence c001.. + state
    _gs=$(( slot + ${CKPT_SLOT0:-0} ))
    printf 'q=%s have=%s landed=%s i=%s relq=%s w=%s\n' "$q" $(( have + ${CKPT_HAVE0:-0} )) "$_gs" "$i" "$relq" "$w" > "$WORK/ckpt.state"
    ( bash "$CKPT_PUSH" "$(printf c%03d "$_gs")" "$sf" && bash "$CKPT_PUSH" state "$WORK/ckpt.state" raw ) > /dev/null 2>&1 < /dev/null &
  fi
  # launch a probe if due and none is running (immutable chunk list); then feed incdup the new chunk
  probe_reap || true
  { [ -n "$MATRIX_DIR" ] || [ "$SC_FINAL" = 1 ]; } && break
  # STOP_RULE=cost + STOP_PREDICTIVE: decide from the model BEFORE the next chunk starts (no probe needed at the stop point)
  sc_after_land "$have" && break
  inc_consumed; inc_reap; probe_launch land; inc_start; las_poll; spec_poll
done
ses_kill   # GLAS_SESSION: the sieve is over -- a prelaunched band still in flight is dropped (complete-band rule)
# STOP_RULE=cost final handoff: the sieve is stopped (no chunk in flight); las first, then incdup + purge on every core
if [ "$SC_FINAL" = 1 ]; then las_end; LAS_ENDED=1; sc_final; SC_FINAL=0; fi
# a probe may still be running when the loop exits via MATRIX (it is the one that found it) -- no.
[ -n "$MATRIX_DIR" ] || die "sieve loop ended without a matrix"
inc_kill   # an incdup step for a later chunk is now useless: stop it (and its dup1/dup2) before freeing RAM
[ "${LAS_ENDED:-0}" = 1 ] || las_end    # ... and so is the CPU co-sieve: its running chunk and unlanded outputs are discarded
say "MATRIX from $MATRIX_DIR: $(grep -a VERDICT "$MATRIX_DIR/filter.out" | tail -1)"; res_note "sieve stopped"
# raw chunks are no longer needed (cado2msieve2 already extracted the purged relations): free ~20 GB
for k in $(seq 1 "$MAXCHUNKS"); do : > "$MEM/$(printf c%03d $k)"; done
for k in $(seq 0 15); do : > "$MEM/d1/$k/rel.0000$SL"; : > "$MEM/d2/$k/rel.0000$SL"; done
inc_free   # incremental parts + key shadows (~8 + ~2.5 GB at c160), unless already freed
[ -n "${BENCHPID:-}" ] && { wait "$BENCHPID" 2>/dev/null; BENCHPID=""; }
# LA_WARM=1 (default; 2026-10-23, gpu/s65_lastart.sh): the sieve and its cofactoring are gone, so the GPU is idle until
# the LA -- compile every LA kernel into the CuPy cache now (`ds_cado.sh --la-warm` = gpu_la_fast.py --warm with
# ds_cado's own LA environment and interpreter; ~20-30 s, nice 5, one core) while merge + replay run on the CPUs. A cold
# LA start measured +17 s. ds_cado.sh sees LA_WARM_PID, does not start a second one, and waits for it (LA_WARM_WAIT)
# before the LA. Exact: it only writes CuPy cache files.
if [ "$DS_MODE" = cado ] && [ "$LA" = gpu ] && [ "${LA_WARM:-1}" = 1 ] && [ -z "${LA_WARM_PID:-}" ]; then
  ( cd "$MSV" && LA="$LA" exec nice -n 5 bash ./ds_cado.sh --la-warm ) > "$WORK/la_warm.log" 2>&1 &
  export LA_WARM_PID=$! LA_WARM_LOG=$WORK/la_warm.log
  say "LA kernel-cache warm-up started (sieve stopped, GPU idle; pid $LA_WARM_PID -> $WORK/la_warm.log)"
fi
[ "$INCDUP" = 1 ] && : > "$MEM/renumber.plain"   # free the 12.8 GB table before the downstream
rm -f "$WORK"/chunk.dat* "$WORK"/cand.dat* "$WORK/cofac.rel" "$STOP" "$WORK/freerel/renumber$RN" "$WORK/freerel/freerel.gz" "$WORK/fb.roots1" "$WORK/raw.p"
[ "$CONTAINER" = 1 ] && rm -f "$WORK/fb1"   # the sieve's factor base (~380 MB at c169): no reader after the sieve
spec_end "sieve over"   # SPEC_MERGE: kill a preparation in flight, drop a READY matrix that is not SPEC_USE
for d in "$WORK"/probe*; do rm -rf "$d/sing"; [ "$DS_MODE" = cado ] && [ "$d" = "$MATRIX_DIR" ] && continue; rmf "$d"/purged* "$d"/history*; done
if [ -n "$SPEC_USE" ]; then   # SPEC_MERGE: the prepared matrix is used -- no merge, no replay; the purged file is its probe's
  export DS_PURGED=${DS_PURGED:-$MATRIX_DIR/$PG}; say "spec-merge: downstream on the prepared matrix $SPEC_USE (epoch $(cat "$SPEC_USE/c.sparse.epoch")), purged $DS_PURGED"
fi
# PURGE_PLAIN_OUT=1: purge already wrote it plain -- the downstream reads $MATRIX_DIR/purged directly, no gunzip. On EVERY
# path: the replay loop (RELS_IN) merges inline with TP_MERGE=0 and used to skip this inside the TP_MERGE block below, so
# ds_cado looked for purged.gz (key 29 container replay, 2026-10-25). Live stops all set TP_MERGE=1 (unaffected).
if [ "$PG" = purged ] && [ "$DS_MODE" = cado ] && [ -z "${DS_PURGED:-}" ] && [ -s "$MATRIX_DIR/purged" ]; then
  export DS_PURGED=$MATRIX_DIR/purged; say "plain purged (PURGE_PLAIN_OUT) for the downstream: $DS_PURGED ($(stat -Lc %s "$DS_PURGED") B)"
fi
if [ "$TP_MERGE" = 1 ]; then   # two-phase: the sieve, incdup and las are gone; merge on MERGE_NT threads
  # PURGED_PLAIN=1 (default; DS_MODE cado): gunzip purged.gz to $PURGED_PLAIN_DIR (/dev/shm) BESIDE the merge and hand
  # it to ds_cado as DS_PURGED: replay then takes CADO's parallel reader (seekable input only; s43, key1bds: replay
  # -t 20 136.1 -> 79.7 s, output md5-identical; gunzip 64 s hidden under merge's ~170 s); characters, sqrt -ab and
  # abx read the same plain file (ds_cado's own CHARS_PLAIN gunzip is then skipped). Any failure keeps purged.gz.
  _ppid=""; _ppf=""
  if [ "${PURGED_PLAIN:-1}" = 1 ] && [ "$DS_MODE" = cado ] && [ -z "${DS_PURGED:-}" ] && [ -s "$MATRIX_DIR/purged.gz" ]; then
    _ppf=${PURGED_PLAIN_DIR:-/dev/shm}/purged.$$
    ( exec nice -n 5 gzip -dc "$MATRIX_DIR/purged.gz" > "$_ppf" ) & _ppid=$!
  fi
  tp_merge "$MATRIX_DIR" "$MERGE_NT" || die "two-phase merge failed (see $MATRIX_DIR/merge.log)"
  if [ -n "$_ppid" ]; then
    if wait "$_ppid" && [ -s "$_ppf" ]; then export DS_PURGED=$_ppf; say "plain purged for the downstream: $_ppf ($(stat -c %s "$_ppf") B)"
    else rm -f "$_ppf"; say "WARNING: gunzip of purged.gz failed -- downstream reads purged.gz"; fi
  fi
fi
# DS_SAVE=<script> (gpu/ds_save.sh in the stitched benchmarks): hardlink the final purged.gz + history + poly and push
# them in the background (nice 19), so a sandbox lost during the LA only costs the downstream on the next box
HISD=$MATRIX_DIR; [ -n "$SPEC_USE" ] && HISD=$SPEC_USE
if [ -n "${DS_SAVE:-}" ] && [ -s "$HISD/history" ]; then
  rm -rf "$WORK/dssave"; mkdir -p "$WORK/dssave"; ln -f "$MATRIX_DIR/$PG" "$HISD/history" "$WORK/dssave/" && cp "$WORK/poly.cado" "$WORK/dssave/" &&
    { ( exec nice -n 19 ionice -c3 bash "$DS_SAVE" "$WORK/dssave" > "$WORK/dssave.log" 2>&1 ) & say "DS_SAVE: final matrix inputs pushing in the background ($WORK/dssave.log)"; } ||
    say "WARNING: DS_SAVE hardlinks failed"
fi
say "/tmp before downstream: $(df -Pm "$WORK" | awk 'NR==2{print $3" MB used, "$4" MB free"}')"; res_note "merge done"
# STOP_AFTER_MERGE=1 (measurement only, gpu/s66_stoprule.sh): the stop point and its merged matrix are the result
# STOP_AFTER_MERGE=2 (measurement only, gpu/s97_specmerge.sh): run the downstream up to the end of replay (= LA start)
[ "${STOP_AFTER_MERGE:-0}" = 2 ] && export DS_STOP_AFTER_REPLAY=1
[ -n "${SC_T_STOP:-}" ] && export SC_T_STOP
if [ "${STOP_AFTER_MERGE:-0}" = 1 ]; then
  say "STOP_AFTER_MERGE=1: $(grep -a 'Final matrix' "$MATRIX_DIR/merge.log" 2>/dev/null | tail -1) -- exiting before the downstream"
  echo "NO FACTOR (STOP_AFTER_MERGE=1)"; exit 0
fi

# =============================================================================================
DS="$WORK/ds"; mkdir -p "$DS"
DS_RC=0
if [ "$DS_MODE" = cado ]; then
  echo "### [4/4] downstream (CADO-native): replay -> GPU block Lanczos on the CADO matrix -> characters -> CADO sqrt $(date +%T)"
  cd "$MSV" || die "no $MSV"
  DSFW=$MATRIX_DIR; DSSP="$MEM/c.sparse.bin"; DSENV=()
  if [ -n "$SPEC_USE" ]; then   # SPEC_MERGE: replay is skipped only if ds_cado sees the same epoch on the sparse file
    DSFW=$SPEC_USE; DSSP=$SPEC_USE/c.sparse.bin; DSENV=(SKIP_REPLAY=1 SKIP_EPOCH="$(cat "$SPEC_USE/c.sparse.epoch")")
    for _f in c.index.gz replay.log; do ln -f "$SPEC_USE/$_f" "$DS/$_f" 2>/dev/null || cp -f "$SPEC_USE/$_f" "$DS/$_f"; done
  fi
  env "${DSENV[@]}" SPARSE="$DSSP" REPLAY="$CF/replay" NT="${DS_NT:-$NCPU}" LPB0=$LPB0 LPB1=$LPB1 LA="$LA" BWC_DIR="${BWC_DIR:-$APP/bwc}" \
    bash ./ds_cado.sh "$DSFW" "$WORK/poly.cado" "$N" "$DS" || DS_RC=$?
  [ "${STOP_AFTER_MERGE:-0}" = 2 ] && { say "STOP_AFTER_MERGE=2: downstream stopped at the LA start (rc=$DS_RC)"; echo "NO FACTOR (STOP_AFTER_MERGE=2)"; exit 0; }
else
  echo "### [4/4] downstream: msieve -nc2 (cado_filter=1) -> GPU block Lanczos -> CADO sqrt $(date +%T)"
  mv -f "$MEM/rels_out.cyc" "$DS/m.dat.cyc" && mv -f "$MEM/rels_out.purged" "$DS/m.dat.purged" || die "cannot stage cycles/purge map"
  cd "$MSV" || die "no $MSV"
  SKIP_NC1=1 NT="$NCPU" ./msv_launcher "$MEM/rels_out" "$WORK/c.fb" "$DS" ./msvrun_gpu.sh "$N" "$DS" || DS_RC=$?
fi
[ "$DS_RC" = 0 ] || say "downstream rc=$DS_RC -- continuing to factor extraction"
say "/tmp after downstream: $(df -Pm "$WORK" | awk 'NR==2{print $3" MB used"}')"; res_note "downstream done"
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
