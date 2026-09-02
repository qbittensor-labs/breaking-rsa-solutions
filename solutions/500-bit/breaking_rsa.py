#!/usr/bin/env python3
# Copyright (C) 2026 qBitTensor Labs.
# Original author: Xdev (Enigma / Breaking RSA competition).
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
    14400 = the 4h milestone budget; the real kill is 4h + up to 5 min of container-manager polling
    (SOLUTION_CONTAINER_MANAGER_TIMEOUT), so this is the conservative end.
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
FULL_RUN_SECS = float(os.environ.get("FULL_RUN_SECS", "12000"))

def log(m): print(f"[{datetime.now(timezone.utc):%H:%M:%S} UTC] {m}", flush=True)

# ---------------- GNFS (CADO-NFS, GPU-sieved) ----------------
def _surface_gnfs(out, tag):
    # Surface key GNFS diagnostics into the solver's own stdout (BEFORE the output separator, so it
    # appears in `docker logs` without corrupting the result payload). Previously factor_msv.sh's
    # entire output was discarded, making field failures (e.g. relation shortfall) undiagnosable.
    keep = []
    for pat in (r"### \[\d/4\][^\n]*", r"bake-off winner[^\n]*", r"relations:\s*\d+",
                r"avg [\d.]+ rel/lattice", r"WALL-TIME BUDGET[^\n]*", r"no matrix produced",
                r"NO FACTOR[^\n]*", r"FACTORED[^\n]*"):
        keep += re.findall(pat, out)
    log(f"GNFS [{tag}] diagnostics:")
    for line in keep[-30:]:
        log("  " + line.strip())

def gnfs_solve(n, ncpu, timeout, extra_env=None):
    """Real factoring for hard semiprimes via the GPU-GNFS pipeline:
    msieve GPU polyselect -> gpu_loop GPU sieve -> memfd-backed msieve downstream
    (filter + in-RAM Lanczos + sqrt). All multi-GB intermediates live in anonymous RAM
    (memfd), so /tmp stays ~100 MB under the 1 GiB sandbox. Returns (p,q) or None.
    Skips gracefully if the toolchain isn't built into the image."""
    base = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(base, "gpu_nfs_siever", "factor_msv.sh")
    if not os.path.exists(script): return None
    workdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gnfs_work")
    # GNFS_WALL is exported ONLY when there is a budget. Unset tells factor_msv.sh to run
    # relation-bound (no SIEVE_SECS split, no DOWNSTREAM_RESERVE held back, no per-stage caps).
    env = dict(os.environ)
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
    if mp and mq and int(mp.group(1)) * int(mq.group(1)) == n:
        return int(mp.group(1)), int(mq.group(1))
    return None

def main():
    ts = datetime.now(timezone.utc).isoformat(); t0 = time.time()
    cid = sys.argv[1].strip() if len(sys.argv) > 1 else "challenge"
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
        # ---- ATTEMPT LADDER (2026-08-06) -------------------------------------------------------
        # A GNFS attempt that comes back with NO FACTORS is not the same event as "this key is too
        # hard". The pipeline is relation-bound (wall_budget() is None), so it does not stop itself
        # for lack of time -- it only returns empty-handed on a tooling/integration failure or on a
        # downstream that could not build a matrix. Both used to fall straight through to
        # {"status":"failed"}, which the validator scores IncorrectFailure -- a WRONG ANSWER
        # reported while hours of wall time were still unspent.
        #
        # So: if the attempt died fast enough that a whole second run still fits inside the wall,
        # run one. It keeps going for as long as a full run still fits -- idling out the remaining
        # wall would be strictly worse than trying again.
        #
        # WHAT THE RETRY CHANGES, AND WHAT IT MUST NOT.  The retry drops CKPT_SPEC, NOT the CADO
        # filter. It is tempting to fall back to CADO_FILTER=0 as "the configuration with the most
        # validator evidence", and an earlier draft of this did -- that is WRONG on exactly the keys
        # that matter. msieve's filter is the binding constraint on hard keys: on key03 it had an
        # excess SURPLUS and still could not build a matrix (Pipeline.md §11), so retrying a hard
        # key with CADO_FILTER=0 spends 3.3h to arrive at a filter that provably cannot succeed.
        # The speculative probe is the right thing to shed instead: it is the newest and most
        # concurrent machinery in the pipeline (a second memfd snapshot, a background probe racing
        # a running sieve, a thread split), while costing only the ~1,000s it saves when it wins.
        # Attempt 2+ is therefore the SAME filter on a strictly simpler, serial control flow.
        #
        # NOTE ON REACH: with FULL_RUN_SECS=12000 against a 14,400s wall, a retry only starts if
        # attempt 1 died inside ~40 min -- i.e. in polyselect or early sieve. Filter/downstream
        # failures happen at 3h+ and can never be retried; for those the hold below is the only
        # protection, and the real fix is that they stop happening.
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
