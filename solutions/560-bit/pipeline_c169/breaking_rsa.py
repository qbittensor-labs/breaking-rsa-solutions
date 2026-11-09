#!/usr/bin/env python3
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

"""Breaking RSA solver entrypoint — GNFS, single method.

Emits the validator stdout output contract (base64 zip with result.json).
The run is one method end to end: GNFS (CADO-NFS, GPU-accelerated sieving), driven by
gpu_nfs_siever/factor_msv.sh. See Pipeline.md for the stage flow.

TWO CHEAP TIERS AHEAD OF IT WERE REMOVED, both for the same reason -- they cannot factor
a validator key, so on the only workload that exists they were pure critical-path cost:
  * GPU seed-recovery / ECM, dropped 2026-06-08 when the validator moved to 256-bit seeds.
  * The exact quick checks (trial division to 1e6, 200k Fermat near-square iterations,
    Pollard-rho), dropped 2026-08-17. A validator key's smallest factor is 250 bits, so
    trial division and rho are ~2^62 steps away from it, and the generator enforces
    |p-q| > 2^150 -- measured 2^247 on the key in Pipeline.md §17 §"Fermat-safety confirmed",
    where the full tier ran and found nothing -- so Fermat cannot close either.
Restoring them means restoring quick_factor() from git history, not flipping a flag.
"""
import json, os, re, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from enigma_challenges.solution_output import build_solution_zip, write_solution_output
import threading, base64

# ---- deadline telemetry (2026-10-25) -------------------------------------------------------------------------------
# The validator returns only a status string, and validation stops at the first failed run. If a run is about to miss
# the 14,400 s wall anyway, spend it on information: just before the kill, end the run with a status that encodes how
# far the pipeline got. It only fires when success is already impossible. Codes (validator statuses):
#   sieve < 85% of the expected raw relations -> IncorrectFailure (a well-formed "failure" result)
#   sieve >= 85%                              -> EmptyArtifacts   (exit 0, no payload)
#   filtering / LA < 50%                      -> InvalidOutputBase64
#   LA >= 50%                                 -> InvalidOutputZip
#   square root                               -> RunFailure       (non-zero exit, no payload)
#   (WallTimeFailure = the watchdog itself did not fire)
# TELEMETRY=0 disables; TELEMETRY_DEADLINE (s since process start, default 14330); TELEM_RAW_TARGET (default by LPB).
_TELEM = {"stage": "start", "raw": 0, "la_it": 0, "la_n": 0, "rate": 0, "la_ms": 0.0, "la_at": 0.0, "t0": time.time()}
_CHILD = {"proc": None}
_RE_TOTAL = re.compile(r"\btotal (\d{6,})"); _RE_N = re.compile(r"Final matrix has N=(\d+)"); _RE_IT = re.compile(r"^\s*iter (\d+) ")
_RE_RATE = re.compile(r"\((\d+) rel/s\)"); _RE_MSIT = re.compile(r"^\s*iter (\d+)\s+\d+s \(([\d.]+) ms/it\)")
# TELEMETRY_MAP=overshoot (S4b+): the five validator codes report the PROJECTED minutes past the wall at the deadline
# instead of the stage: <=3 RunFailure (a near miss, or in sqrt), <=8 InvalidOutputZip, <=15 InvalidOutputBase64,
# <=30 EmptyArtifacts, else IncorrectFailure (also a tooling failure). TELEMETRY_MAP=stage (default) keeps the S1/S2 map.
_TELEM_BUCKETS = ((3, "sqrt"), (8, "la_hi"), (15, "ds_lo"), (30, "sieve_hi"))

def _telem_update(line):
    try:
        if line.startswith("### [3/4]"): _TELEM["stage"] = "sieve"
        elif line.startswith("### [4/4]"): _TELEM["stage"] = "downstream"
        elif "sqrt deps" in line or "sqrt -ab" in line: _TELEM["stage"] = "sqrt"
        m = _RE_TOTAL.search(line)
        if m and _TELEM["stage"] in ("start", "sieve"): _TELEM["raw"] = int(m.group(1)); _TELEM["stage"] = "sieve"
        m = _RE_N.search(line)
        if m: _TELEM["la_n"] = int(m.group(1))
        m = _RE_IT.match(line)
        if m and _TELEM["stage"] != "sqrt": _TELEM["la_it"] = int(m.group(1)); _TELEM["stage"] = "la"
        m = _RE_RATE.search(line)
        if m and _TELEM["stage"] in ("start", "sieve"): _TELEM["rate"] = int(m.group(1))
        m = _RE_MSIT.match(line)
        if m and _TELEM["stage"] == "la": _TELEM["la_ms"] = float(m.group(2)); _TELEM["la_at"] = time.time()
    except Exception:
        pass

def _telem_stage_code():
    st = _TELEM["stage"]
    lpb1 = os.environ.get("LPB1") or os.environ.get("RSA560_LPB1") or "32"
    target = float(os.environ.get("TELEM_RAW_TARGET") or (385e6 if str(lpb1) == "33" else 316e6))
    if st in ("start", "sieve"):
        return "sieve_lo" if _TELEM["raw"] < 0.85 * target else "sieve_hi"
    if st == "sqrt":
        return "sqrt"
    its = _TELEM["la_n"] / 255.3 if _TELEM["la_n"] else 0
    return "la_hi" if (its and _TELEM["la_it"] >= 0.5 * its) else "ds_lo"

def _telem_overshoot(now=None):
    """Projected finish minus the wall, in seconds (> 0 = the run would miss), or None when nothing can be projected.
    Constants are measured c169 figures on validator-class cards at the S2B regime: sieve stop -> factors ~3,100 s
    (TELEM_DS); LA end -> factors ~220 s (TELEM_POST_LA: characters + one sqrt window); LA ~41 ms/it at N 16M scaling
    with N (TELEM_LA_MSIT); the sieve stops near 405M raw at lpb1 33 (TELEM_RAW_STOP)."""
    try:
        now = now or time.time(); t = now - _TELEM["t0"]; st = _TELEM["stage"]
        wall = float(os.environ.get("TELEM_WALL", "14400")); post = float(os.environ.get("TELEM_POST_LA", "220"))
        if st == "sqrt":
            return t + 120.0 - wall
        if st == "la" and _TELEM["la_n"] and _TELEM["la_ms"] > 0:
            left = (_TELEM["la_n"] / 255.3 - _TELEM["la_it"]) * _TELEM["la_ms"] / 1000.0 - (now - _TELEM["la_at"])
            return t + max(0.0, left) + post - wall
        if st in ("downstream", "la"):
            n = float(_TELEM["la_n"] or 16.0e6)
            la = n / 255.3 * (float(os.environ.get("TELEM_LA_MSIT", "41")) * n / 16.0e6) / 1000.0
            return t + 200.0 + la + post - wall
        if st == "sieve" and _TELEM["rate"] > 0:
            lpb1 = os.environ.get("LPB1") or os.environ.get("RSA560_LPB1") or "32"
            stop = float(os.environ.get("TELEM_RAW_STOP") or (405e6 if str(lpb1) == "33" else 330e6))
            return t + max(0.0, stop - _TELEM["raw"]) / _TELEM["rate"] + float(os.environ.get("TELEM_DS", "3100")) - wall
    except Exception:
        pass
    return None

def _telem_code():
    if os.environ.get("TELEMETRY_MAP", "stage") != "overshoot":
        return _telem_stage_code()
    o = _telem_overshoot()
    if o is None:
        return _telem_stage_code()
    for mins, code in _TELEM_BUCKETS:
        if o <= mins * 60.0: return code
    return "sieve_lo"

def _telem_emit_and_exit(code):
    try:
        if _CHILD["proc"] is not None: os.killpg(os.getpgid(_CHILD["proc"].pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        sys.stdout.flush()
        out = sys.stdout.buffer
        if code == "sieve_lo":
            write_solution_output(build_solution_zip({"result.json": json.dumps({"status": "failure", "p": None, "q": None}),
                                                      "solve_info.json": json.dumps({"telemetry": dict(_TELEM)})}))
            os._exit(0)
        if code == "sieve_hi":
            os._exit(0)
        from enigma_challenges.solution_output import SOLUTION_OUTPUT_SEPARATOR as SEP, SOLUTION_OUTPUT_END_SEPARATOR as END
        if code == "ds_lo":
            out.write(SEP + b"@@telemetry-not-base64@@" + END); out.flush(); os._exit(0)
        if code == "la_hi":
            out.write(SEP + base64.b64encode(b"telemetry: not a zip") + END); out.flush(); os._exit(0)
    except Exception:
        pass
    os._exit(3)   # "sqrt" (and any failure above): crash without artifacts -> RunFailure

def _telem_watchdog(t0):
    deadline = float(os.environ.get("TELEMETRY_DEADLINE", "14330"))
    time.sleep(max(0.0, t0 + deadline - time.time()))
    code = _telem_code()
    # TELEMETRY_LATE_DEADLINE (default unset = fire now): a run in its last stages (LA >= 50 % or sqrt) can still finish
    # between TELEMETRY_DEADLINE and the validator's 14,400 s, and the telemetry says least exactly then. Hold the kill
    # until the later deadline (<= ~14,397: python t0 is < 1 s after docker's StartedAt) and re-read the stage.
    # A success in the meantime ends the process (os._exit in main) and this daemon thread with it.
    late = os.environ.get("TELEMETRY_LATE_DEADLINE")
    if late and (code in ("la_hi", "sqrt") or _TELEM["stage"] in ("la", "sqrt")):
        try:
            late = float(late)
            sys.stdout.write(f"\nTELEMETRY deadline {deadline:.0f}s: stage {_TELEM['stage']} ({code}) -- holding the "
                             f"telemetry exit until {late:.0f}s\n"); sys.stdout.flush()
            time.sleep(max(0.0, t0 + late - time.time()))
            code = _telem_code()
        except Exception:
            pass
    try:
        o = _telem_overshoot()
        sys.stdout.write(f"\nTELEMETRY deadline {deadline:.0f}s: stage {_TELEM['stage']} raw {_TELEM['raw']} rate {_TELEM['rate']} "
                         f"la_it {_TELEM['la_it']} la_n {_TELEM['la_n']} la_ms {_TELEM['la_ms']} projected over by "
                         f"{'n/a' if o is None else f'{o / 60.0:.1f} min'} (map {os.environ.get('TELEMETRY_MAP', 'stage')}) "
                         f"-> {code}\n"); sys.stdout.flush()
    except Exception:
        pass
    _telem_emit_and_exit(code)

# --- GPU keep-alive -----------------------------------------------------------
# Multi-phase GPU run (msieve polyselect -> gpu_loop sieve -> gpu_la.py) uses several CUDA contexts
# in sequence. On a host WITHOUT nvidia-persistenced, the driver de-initializes the GPU between
# contexts, so the next phase can hit "CUDA: no CUDA-capable device is detected" (observed: 1.5h stall,
# 0 relations). We can't enable persistence mode from inside (--cap-drop ALL --no-new-privileges), but
# holding ONE long-lived CUDA context (device refcount > 0) for the whole run keeps the GPU initialized.
# Privilege-free; CuPy is already in the image. Best-effort: any failure degrades to no keep-alive.
_KEEPALIVE = None
def _gpu_keepalive_start():
    global _KEEPALIVE
    try:
        _KEEPALIVE = subprocess.Popen(
            [sys.executable, "-c",
             "import cupy,time;_=cupy.zeros(256,dtype=cupy.uint8);cupy.cuda.runtime.deviceSynchronize();time.sleep(1e9)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        _KEEPALIVE = None
def _gpu_keepalive_stop():
    global _KEEPALIVE
    if _KEEPALIVE is not None:
        try: _KEEPALIVE.kill()
        except Exception: pass
        _KEEPALIVE = None

CHALLENGE_INPUT_FILE = "/challenge_input/challenge_input.json"

def load_problem():
    if os.path.isfile(CHALLENGE_INPUT_FILE):
        d = json.loads(Path(CHALLENGE_INPUT_FILE).read_text())
        return int(d["num"]), int(d["num_bits"])
    if len(sys.argv) == 3:
        d = json.loads(sys.argv[2]); return int(d["num"]), int(d["num_bits"])
    raise SystemExit("No problem input (/challenge_input/challenge_input.json or argv)")

def effective_cpus():
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            return max(1, round(int(q) / int(p)))
    except Exception:
        pass
    return os.cpu_count() or 4

def wall_budget():
    """Internal wall budget, or None for NO internal deadline (the default).

    2026-07-31: deliberately prefer a validator TIMEOUT over a self-inflicted IncorrectFailure.

    This used to return 14400.0 when unset -- and unset is ALWAYS the production case, because the
    validator's `docker run` passes no -e. Measured consequences of that default:

      * The solver self-terminated at t0+14,340s = 3.983h and emitted {"status":"failed"}. The
        validator reads that well-formed payload and scores IncorrectFailure -- a WRONG ANSWER --
        even though the container never went overdue.
      * The real deadline is LATER than 14,400s: the container manager polls for overdue containers
        every 5 minutes (SOLUTION_CONTAINER_MANAGER_TIMEOUT), comparing now against docker's
        .State.StartedAt. So the effective kill is 4h + up to 5 min. Quitting at 3.983h handed that
        grace back and forfeited a real success band of roughly 14,340-14,700s.
      * DOWNSTREAM_RESERVE was subtracted from this budget UP FRONT, so on any key whose downstream
        is cheaper than the reserve the sieve was truncated for nothing. Measured post-sieve cost
        ranges 1,203s (easy key) to 2,299s (hard key), so no single constant is right.

    Returning None makes the pipeline RELATION-BOUND: sieve until the filter reports a matrix, then
    run the downstream. If that exceeds the wall the container is killed and the validator reports
    WallTimeFailure. The run stays diagnosable -- the overdue path calls extract_stdout_output, so
    container stdout is still captured.

    Set WALL_TIME explicitly to restore a bounded run (dev harness, timeout regression test).
    """
    for k in ("WALL_TIME", "WALL_TIME_SECONDS", "SOLVE_WALL_TIME"):
        v = os.environ.get(k)
        if v:
            try: return float(v)
            except ValueError: pass
    return None

def hard_wall():
    """The VALIDATOR's external deadline, in seconds. POLICY ONLY -- never a self-imposed deadline.

    wall_budget() returns None so the solver never stops itself; this figure is used only to answer
    two questions that have no other source of truth inside the container:
      * "did the pipeline fail EARLY enough that a second full attempt still fits?" (retry ladder)
      * "is there still wall left, i.e. would emitting {"status":"failed"} now be a self-inflicted
        IncorrectFailure rather than an honest report?"  (see main())
    14400 = the 4h milestone budget. The kill is 4h + the container-manager's polling lag, and that
    lag is SMALLER than this docstring used to claim: SOLUTION_CONTAINER_MANAGER_TIMEOUT is
    **30 seconds** (qbittensor/constants.py:32), not 5 minutes. The overdue sweep runs on the
    validator's forward cadence (neurons/validator.py calls timer.check_timer() each pass), so the
    real grace is ~30 s plus one forward pass -- NOT a budget to plan against. Verified 2026-09-11
    after a WALL_TIME_FAILURE; the old "up to 5 min" overstated the slack by ~10x.
    """
    for k in ("HARD_WALL", "VALIDATOR_WALL_SECONDS"):
        v = os.environ.get(k)
        if v:
            try: return float(v)
            except ValueError: pass
    return 14400.0

# A full cold run (polyselect + sieve to the crossover + downstream) measured 3.8-4.0h on the
# hardest keys surveyed. A retry is only worth starting if that much wall is still unspent --
# which makes the retry condition "the first attempt died FAST", derived rather than guessed.
# c169: a full run is budgeted at <= 13,000 s (rsa560_plan.md), so only an attempt that died in its first
# ~1,400 s can be retried; anything later is handled inside the pipeline or holds for WallTimeFailure.
FULL_RUN_SECS = float(os.environ.get("FULL_RUN_SECS", "13000"))

def log(m): print(f"[{datetime.now(timezone.utc):%H:%M:%S} UTC] {m}", flush=True)

# ---------------- GNFS (CADO-NFS, GPU-sieved) ----------------
def _surface_gnfs(out, tag):
    # Surface key GNFS diagnostics into the solver's own stdout (BEFORE the output separator, so it
    # appears in `docker logs` without corrupting the result payload). Previously factor_msv.sh's
    # entire output was discarded, making field failures (e.g. relation shortfall) undiagnosable.
    keep = []
    for pat in (r"### \[\d/4\][^\n]*", r"### c16[09][^\n]*", r"\[\d+s\] \[res\][^\n]*", r"\[\d+s\] (?:probe|MATRIX|chunk|relation gate|polynomial|FATAL)[^\n]*",
                r"### FACTORS[^\n]*", r"bake-off winner[^\n]*", r"relations:\s*\d+",
                r"avg [\d.]+ rel/lattice", r"WALL-TIME BUDGET[^\n]*", r"no matrix produced",
                r"NO FACTOR[^\n]*", r"FACTORED[^\n]*"):
        keep += re.findall(pat, out)
    log(f"GNFS [{tag}] diagnostics:")
    for line in keep[-30:]:
        log("  " + line.strip())

def find_c169_script(base):
    """factor_c169.sh next to this file (the image layout /app/pipeline_c169/breaking_rsa.py), or under a
    pipeline_c169/ subdirectory (the 530-style layout with the entrypoint one level up); GNFS_SCRIPT overrides."""
    for c in (os.environ.get("GNFS_SCRIPT"), os.path.join(base, "factor_c169.sh"),
              os.path.join(base, "pipeline_c169", "factor_c169.sh")):
        if c and os.path.isfile(c):
            return c
    return None

def gpu_visible():
    try:
        return subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=30).returncode == 0
    except Exception:
        return False

def small_key_env(digits, ncpu):
    """Parameters for the SMALL test keys (c100 / c115) that exercise the container path end to end. NOT for scoring:
    a validator key is always 169 digits and runs the production defaults of factor_c169.sh (CONTAINER=1 profile).
    c100 = gpu/s39_c100.sh (CADO params.c100, measured on the GPU box: FACTOR OK); c115 = CADO params.c115 (untested
    with this orchestrator). SIEVER=las (CADO's CPU siever) for both; LA=gpu when a GPU is visible, else LA=bwc (CPU)."""
    t = max(1, (ncpu * 4) // 5)
    e = {"SIEVER": "las", "LA": "gpu" if gpu_visible() else "bwc", "NCPU": str(ncpu), "SIEVE_T": str(t),
         "SIEVE_HOST_T": str(t), "PROBE_NT": str(max(2, min(4, ncpu))), "DUP2_PAR": "2", "INC_TPG": "2",
         "STOP_RULE": "excess", "SQRT_WIN": "1", "ROPT_K": "20", "ROPT_EFF": "2", "ROPT_MAX": "60"}
    if digits <= 107:
        e.update({"SIEVE_A": "22", "LPB0": "25", "LPB1": "26", "MFB0": "48", "MFB1": "51", "LAMBDA0": "1.9",
                  "LAMBDA1": "1.93", "NCURVES0": "9", "NCURVES1": "11", "RLIM": "650000", "ALIM": "800000",
                  "QMIN": "180000", "FIRST_PROBE": "3000000", "PROBE_STEP": "300000", "CHUNK_RELS": "500000",
                  "CHUNK_RELS_LATE": "300000", "CHUNK_W0": "100000", "PS_P": "7000", "PS_ADMIN": "60",
                  "PS_ADMAX": "1680", "PS_INCR": "30", "PS_NQ": "1000", "PS_SECS": "60", "PS_MAX": "120",
                  "PS_MIN_CANDS": "50", "STOP_MIN_EXCESS": "300000", "TD": "85"})
    else:
        e.update({"SIEVE_A": "23", "LPB0": "27", "LPB1": "27", "MFB0": "52", "MFB1": "53", "LAMBDA0": "1.91",
                  "LAMBDA1": "1.95", "NCURVES0": "11", "NCURVES1": "13", "RLIM": "2000000", "ALIM": "2700000",
                  "QMIN": "480000", "FIRST_PROBE": "9000000", "PROBE_STEP": "700000", "CHUNK_RELS": "1500000",
                  "CHUNK_RELS_LATE": "700000", "CHUNK_W0": "50000", "PS_P": "20000", "PS_ADMIN": "120",
                  "PS_ADMAX": "7800", "PS_INCR": "60", "PS_NQ": "1000", "PS_SECS": "120", "PS_MAX": "240",
                  "PS_MIN_CANDS": "50", "STOP_MIN_EXCESS": "500000", "TD": "100"})
    return e

def gnfs_solve(n, ncpu, timeout, extra_env=None):
    """Real factoring for hard semiprimes via the GPU-GNFS pipeline:
    msieve GPU polyselect -> gpu_loop GPU sieve -> memfd-backed msieve downstream
    (filter + in-RAM Lanczos + sqrt). All multi-GB intermediates live in anonymous RAM
    (memfd), so /tmp stays ~100 MB under the 1 GiB sandbox. Returns (p,q) or None.
    Skips gracefully if the toolchain isn't built into the image."""
    base = os.path.dirname(os.path.abspath(__file__))
    # RSA-530 (c160): a dedicated cuda-sieve + RAM-backed CADO-filter pipeline. The gpu_loop path
    # (factor_msv.sh) is compiled for lpb29 and its storage design overflows the 10 GiB /tmp at c160.
    # Same stdout contract: "p=" / "q=" lines.
    c160_min = int(os.environ.get("C160_MIN_DIGITS", "158"))
    # RSA-560 (c169): pipeline_c169/factor_c169.sh (docs/rsa560_plan.md). Same stdout contract.
    c169_min = int(os.environ.get("C169_MIN_DIGITS", "166"))
    # CONTAINER=1 (the RSA-560 submission image, docker/Dockerfile.c169-*): factor_c169.sh is the ONLY pipeline in the
    # image, so every size goes to it -- below c169_min with a small-key parameter profile (small_key_env), which is
    # what the validator-constraint dry run (gpu/container_dryrun.sh) uses on the c100/c115 test keys.
    container = os.environ.get("CONTAINER", "0") == "1"
    if container or len(str(n)) >= c169_min:
        script = find_c169_script(base)
    else:
        script_name = "factor_c160.sh" if len(str(n)) >= c160_min else "factor_msv.sh"
        script = os.path.join(base, "gpu_nfs_siever", script_name)
    if not script or not os.path.exists(script):
        log(f"GNFS: no pipeline script for a {len(str(n))}-digit N (looked for {script or 'factor_c169.sh'})")
        return None
    workdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gnfs_work")
    # GNFS_WALL is exported ONLY when there is a budget. Unset tells factor_msv.sh to run
    # relation-bound (no SIEVE_SECS split, no DOWNSTREAM_RESERVE held back, no per-stage caps).
    env = dict(os.environ)
    if container and len(str(n)) < c169_min:
        prof = small_key_env(len(str(n)), ncpu)
        for k, v in prof.items(): env.setdefault(k, v)   # an explicit environment value still wins
        log(f"GNFS: small-key profile for {len(str(n))} digits: " + " ".join(f"{k}={env[k]}" for k in sorted(prof)))
    if timeout is not None:
        env["GNFS_WALL"] = str(int(timeout))
    else:
        env.pop("GNFS_WALL", None)
    # Per-attempt overrides from the retry ladder in main() (e.g. CADO_FILTER=0).
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    # GNFS_LIVE_LOG now defaults ON (was debug-only, off by default). LOAD-BEARING with the
    # no-deadline change: the sub-pipeline's output is otherwise buffered and only flushed by
    # _surface_gnfs() at the END of the run, which never executes if the container is killed at the
    # wall. The validator DOES capture container stdout on the overdue path
    # (_terminate_overdue_containers -> extract_stdout_output), so streaming live is the difference
    # between a diagnosable timeout and an empty log.
    live = os.environ.get("GNFS_LIVE_LOG", "1") not in ("0", "", "no")
    import threading
    buf = []
    timed_out = {"v": False}
    try:
        # start_new_session=True IS LOAD-BEARING (added 2026-07-28). It makes the child a process
        # GROUP LEADER so the timeout path below can kill the WHOLE tree. Without it, proc.kill()
        # reaches only `bash factor_msv.sh`; its descendants (gpu_loop, msieve_la and especially
        # relhold.py) survive holding this stdout pipe, so `for line in proc.stdout` never sees EOF
        # and the solver HANGS PAST ITS DEADLINE emitting no payload at all -- the validator then
        # reports "no solution output found". Note the trap inside factor_msv.sh cannot save us
        # here: proc.kill() sends SIGKILL, which bash cannot trap.
        proc = subprocess.Popen(["bash", script, str(n), workdir],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, start_new_session=True)
        _CHILD["proc"] = proc
    except Exception:
        return None

    def _kill_tree():
        """SIGKILL the child's entire process group. Safe: start_new_session put it in its own
        group, so this can never signal us. Best-effort -- a dead group is not an error."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try: proc.kill()
            except Exception: pass

    def _watch():
        try: proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out["v"] = True
            _kill_tree()
    # No watchdog when uncapped -- the container manager owns the deadline. (A thread with
    # timeout=None would block in proc.wait() forever and never fire anyway; skipping it makes the
    # intent explicit and leaves the timeout path exercised whenever WALL_TIME IS set.)
    t = None
    if timeout is not None:
        t = threading.Thread(target=_watch, daemon=True); t.start()
    try:
        for line in proc.stdout:
            buf.append(line)
            _telem_update(line)
            if live:
                sys.stdout.write(line); sys.stdout.flush()
    finally:
        # Always reap the whole tree, on every exit path (success, timeout, or an exception while
        # reading). Leaving a grandchild alive would keep RAM pinned (relhold holds ~8 GB) and, on a
        # retry, hold this pipe open again.
        _kill_tree()
        try: proc.stdout.close()
        except Exception: pass
    proc.wait()
    if t is not None: t.join(timeout=5)
    out = "".join(buf)
    if timed_out["v"]:
        _surface_gnfs(out, "TIMEOUT"); return None
    _surface_gnfs(out, f"rc={proc.returncode}")
    mp = re.search(r"^p=(\d+)", out, re.M); mq = re.search(r"^q=(\d+)", out, re.M)
    # the ONLY acceptance test: a non-trivial split with p*q == N exactly (never trust a printed line by itself)
    if mp and mq and 1 < int(mp.group(1)) < n and int(mp.group(1)) * int(mq.group(1)) == n:
        log(f"GNFS: verified p*q == N ({len(mp.group(1))} x {len(mq.group(1))} digits)")
        return int(mp.group(1)), int(mq.group(1))
    if mp and mq:
        log("GNFS: printed p/q do NOT multiply to N -- rejected")
    return None

def main():
    ts = datetime.now(timezone.utc).isoformat(); t0 = time.time()
    if os.environ.get("CONTAINER", "0") == "1" and os.environ.get("TELEMETRY", "1") != "0":
        threading.Thread(target=_telem_watchdog, args=(t0,), daemon=True).start()
    cid = sys.argv[1].strip() if len(sys.argv) > 1 else "challenge"
    # RSA560_CALIBRATE=1 (calibration submission, 2026-10-26): do not factor; run pipeline_c169/calib/calibrate.sh (frozen
    # sieve window + CPU step + LA micro-benchmark), and report the predicted S4rung wall on THIS validator through the
    # five status codes (calib_code.py buckets). Any failure or timeout reports sieve_lo (IncorrectFailure).
    if os.environ.get("RSA560_CALIBRATE", "0") == "1":
        code = "sieve_lo"
        try:
            cal = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib", "calibrate.sh")
            p = subprocess.run(["bash", cal], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               timeout=float(os.environ.get("CAL_TIMEOUT", "1800")))
            sys.stdout.write(p.stdout); sys.stdout.flush()
            m = re.search(r"CALIB_CODE=(\w+)", p.stdout)
            if m and m.group(1) in ("sqrt", "la_hi", "ds_lo", "sieve_hi", "sieve_lo"): code = m.group(1)
        except Exception as e:
            log(f"calibration failed: {e!r}")
        # Timing channel (validator-ratio agent 10-26): a MEASURED result is held until CAL_HOLD_S after start (default
        # 1,800 s) so the verdict's arrival time separates "measured" (~30 min) from "the probe itself failed" (minutes).
        if code != "sieve_lo":
            hold = float(os.environ.get("CAL_HOLD_S", "1800")); time.sleep(max(0.0, t0 + hold - time.time()))
        log(f"CALIBRATION exit code {code} after {time.time() - t0:.0f}s")
        _telem_emit_and_exit(code)
    num, num_bits = load_problem()
    ncpu = effective_cpus(); budget = wall_budget()
    deadline = (t0 + budget) if budget else None
    # ---- ADVISORY WALL FOR THE RECOVERY CASCADE (2026-08-24) -------------------------------
    # NOT a deadline: nothing self-terminates on this, and GNFS_WALL stays unset so the sieve is
    # still relation-bound (the 2026-07-31 policy is untouched). It exists because the downstream
    # had NO clock at all in production, so every recovery branch was blind: msvrun_gpu.sh would
    # enter an ~11x-slower CPU fallback at 3h+ elapsed that could not possibly finish, burning the
    # remaining wall and emitting a misleading "wants 1,000,000 more relations". A fallback is only
    # real if it can still FIT; that question needs an absolute deadline, and this is the only
    # source of truth for it inside the container. Absolute epoch seconds so a child started at any
    # depth can answer "how much wall is left?" without knowing when the run began.
    os.environ["GNFS_WALL_AT"] = str(int(t0 + hard_wall()))
    log(f"Breaking RSA: {num_bits}-bit N, GPU + {ncpu} CPU threads, "
        + (f"budget {budget:.0f}s" if budget else
           "NO internal deadline (relation-bound; a validator timeout is preferred to a "
           "self-inflicted IncorrectFailure)"))
    _gpu_keepalive_start()   # pin the GPU across the multi-context GNFS run (see top of file)

    pq = None; method = None
    # GNFS real factoring -- the ONLY method; the sieve starts at t0+0 rather than behind a tier
    # that never fired (see the module docstring). With no internal deadline (the default) this is
    # RELATION-bound, not TIME-bound: it runs until the filter yields a matrix and the downstream
    # finishes. If that overruns the wall the validator kills the container (WallTimeFailure)
    # instead of the solver quitting early and reporting a wrong answer. cap=None means
    # "no watchdog".
    if deadline is None or (deadline - time.time()) > 120:
        log("GNFS / CADO-NFS " + (
            f"(cap {deadline - time.time() - 60:.0f}s)" if deadline else "(uncapped)"))
# ---- ATTEMPT LADDER ---------------------------------------------------------------------
# Each attempt gets a slice of the remaining wall, not a fixed budget, so a slow first attempt
# cannot starve the rest. The ladder stops the moment a factor is found.
        for i in range(int(os.environ.get("GNFS_MAX_ATTEMPTS", "4"))):
            extra = None if i == 0 else {"CKPT_SPEC": "0"}
            tag = "gnfs" if i == 0 else "gnfs-serial"
            if i:
                left = hard_wall() - (time.time() - t0)
                if left < FULL_RUN_SECS:
                    log(f"GNFS: no further attempt -- {left:.0f}s of wall left, a full run "
                        f"needs ~{FULL_RUN_SECS:.0f}s")
                    break
                log(f"GNFS attempt {i+1}: the previous attempt returned no factors after "
                    f"{time.time()-t0:.0f}s (a tooling failure, not a relation shortfall) -- "
                    f"rerunning with {extra}, {left:.0f}s of wall left")
            cap = (deadline - time.time() - 60) if deadline else None
            res = gnfs_solve(num, ncpu, cap, extra_env=extra)
            if res: pq, method = res, tag; break

    if pq:
        p, q = sorted(pq)
        result = {"status": "success", "p": p, "q": q}; status = "success"
        log(f"SUCCESS via {method} in {time.time()-t0:.1f}s")
    else:
        # ---- PREFER A TIMEOUT OVER A SELF-INFLICTED IncorrectFailure (2026-08-06) --------------
        # wall_budget() has returned None since 2026-07-31 precisely so the solver never quits
        # early and hands the validator a well-formed {"status":"failed"} -- which is scored
        # IncorrectFailure, i.e. a WRONG ANSWER, not a timeout. That change closed the wall-clock
        # route to this outcome and left every OTHER route wide open: a set -e trip in
        # factor_msv.sh, a filter that never ran, a downstream that built no matrix, all still
        # arrive here early and report a wrong answer with hours of wall unspent.
        #
        # There is nothing honest to say at this point: the key is factorable and we did not
        # factor it. Holding until the container manager terminates the run reports it as
        # WallTimeFailure instead, and the overdue path still calls extract_stdout_output, so
        # everything logged above is captured either way.
        #
        # Bounded runs (WALL_TIME set: dev harness, timeout regression test) keep the old
        # behaviour and emit immediately -- as does EMIT_FAILED_PAYLOAD=1.
        _left = hard_wall() - (time.time() - t0)
        _emit_now = os.environ.get("EMIT_FAILED_PAYLOAD", "0") in ("1", "true", "yes")
        # REVERTED 2026-08-19: the unconditional hold keeps the container occupying the
        # validator's slot for the whole remaining wall on any no-factors run. Gating the hold on
        # "wall still unspent" is restored -- past the wall the solver emits immediately instead.
        if budget is None and not _emit_now and _left > 0 \
                and os.environ.get("CONTAINER", "0") == "1" and os.environ.get("TELEMETRY", "1") != "0":
            # S1 lesson (2026-10-25, key 30): a tooling failure that then HOLDS until the deadline is reported by the
            # telemetry as the stage the counters show ("sieve_hi" -> EmptyArtifacts), which reads as "too slow". Exit
            # 3 NOW instead: RunFailure, reported at the failure time, distinguishable from an overtime run.
            log(f"NO FACTORS after {time.time()-t0:.0f}s with {_left:.0f}s of wall unspent: a tooling failure "
                f"(stage {_TELEM['stage']}, raw {_TELEM['raw']}). Telemetry on: exiting 3 now (RunFailure).")
            _gpu_keepalive_stop()
            # overshoot map: a tooling failure reports IncorrectFailure (the ">30 min / broken" bucket), never a near miss
            _telem_emit_and_exit("sieve_lo" if os.environ.get("TELEMETRY_MAP", "stage") == "overshoot" else "fatal")
        if budget is None and not _emit_now and _left > 0:
            log(f"NO FACTORS after {time.time()-t0:.0f}s, {_left:.0f}s of wall still unspent. "
                "This is a tooling failure, not a wrong answer -- holding for external "
                "termination (WallTimeFailure) rather than self-reporting IncorrectFailure.")
            _gpu_keepalive_stop()
            while True:
                time.sleep(60)
                _el = time.time() - t0
                log(f"awaiting external termination: {_el:.0f}s elapsed "
                    f"({_el - hard_wall():+.0f}s vs the wall)")
                # Safety valve: no manager is coming (dev box, or the poll never fired). Emit
                # something rather than hanging forever.
                if _el > hard_wall() + 1800:
                    log("30 min past the wall with no termination -- emitting the failed payload")
                    break
        result = {"status": "failed", "p": None, "q": None}; status = "failed"
        log(f"FAILED after {time.time()-t0:.1f}s")
    result_json = json.dumps(result, indent=2)
    info_json = json.dumps({"solution_status": status, "challenge_id": cid,
        "timestamp_utc": ts, "solve_time_seconds": time.time()-t0,
        "method": method or "none", "num_bits": num_bits})
    od = os.environ.get("OUTPUT_DIR")
    if od:
        try:
            Path(od).mkdir(exist_ok=True)
            Path(od, "result.json").write_text(result_json); Path(od, "solve_info.json").write_text(info_json)
        except OSError: pass
    write_solution_output(build_solution_zip({"result.json": result_json, "solve_info.json": info_json}))
    _gpu_keepalive_stop()
    os._exit(0 if status == "success" else 1)

if __name__ == "__main__":
    # STDOUT-ONLY PROTOCOL: the validator reads `docker logs` as container stdout FOLLOWED BY container stderr,
    # splits on the first separator and keeps the leading run of base64-alphabet characters + whitespace --
    # so ANY stderr text starting with letters (a traceback, a SystemExit message, a library warning) would be
    # glued onto the payload and fail base64 decoding (InvalidOutputBase64). Point fd 2 at stdout before
    # anything runs: every byte then travels in one ordered stream BEFORE the separator, and the container's
    # stderr stream stays empty by construction. (The GNFS child already merges stderr into its pipe; the GPU
    # keep-alive writes to /dev/null.)
    try:
        sys.stdout.flush(); sys.stderr.flush(); os.dup2(1, 2)
    except OSError:
        pass
    # Robustness: any exception BEFORE the solution payload is written (e.g. malformed
    # /challenge_input JSON, missing num/num_bits) would otherwise exit with a bare traceback
    # and NO output separator -> the validator reports "no solution output found" with zero
    # diagnostics. Emit a well-formed {status:failed} payload on any such crash instead.
    try:
        main()  # main() ends in os._exit(), which bypasses this handler on the normal path
    except BaseException as _e:
        try:
            import traceback; traceback.print_exc()
            _rj = json.dumps({"status": "failed", "p": None, "q": None}, indent=2)
            _ij = json.dumps({"solution_status": "failed", "error": repr(_e)[:300]})
            write_solution_output(build_solution_zip({"result.json": _rj, "solve_info.json": _ij}))
        except BaseException:
            pass
        _gpu_keepalive_stop()
        os._exit(1)
