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

"""Breaking RSA solver entrypoint — TIERED.

Emits the validator stdout output contract (base64 zip with result.json).
Tries, in order of cost, within the wall-time budget:
  Tier 0  exact quick checks (trial division, Fermat near-square, Pollard-rho)  -- seconds
  Tier 1  GNFS (CADO-NFS, GPU-accelerated sieving) -- real factoring of hard semiprimes

Scope: Tier 0 catches small-factor / close-prime / malformed N quickly. A genuinely
random 460-bit semiprime requires Tier 1 (GNFS); see gpu_nfs_siever/ and SUBMISSION.md.
(A GPU seed-recovery / ECM tier was removed: the validator switched to 256-bit seeds on
2026-06-08, killing seed-recovery, and ECM only catches vanishingly-rare weak keys.)
"""
import json, math, os, re, subprocess, sys, time
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
    for k in ("WALL_TIME", "WALL_TIME_SECONDS", "SOLVE_WALL_TIME"):
        v = os.environ.get(k)
        if v:
            try: return float(v)
            except ValueError: pass
    return 14400.0   # validator default: 4h

def log(m): print(f"[{datetime.now(timezone.utc):%H:%M:%S} UTC] {m}", flush=True)

# ---------------- Tier 0: exact, cheap, pure-Python ----------------
def _small_trial(n, bound=1_000_000):
    if n % 2 == 0: return 2
    f = 3
    while f <= bound and f * f <= n:
        if n % f == 0: return f
        f += 2
    return None

def _fermat(n, iters=200_000):
    a = math.isqrt(n)
    if a * a < n: a += 1
    for _ in range(iters):
        b2 = a * a - n
        b = math.isqrt(b2)
        if b * b == b2:
            f = a - b
            if 1 < f < n: return f
        a += 1
    return None

def _rho(n, t_deadline):
    if n % 2 == 0: return 2
    for c in range(1, 12):
        x = y = 2; d = 1
        while d == 1:
            if time.time() > t_deadline: return None
            x = (x * x + c) % n
            y = (y * y + c) % n; y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if 1 < d < n: return d
    return None

def quick_factor(n, t_deadline):
    for fn in (lambda: _small_trial(n), lambda: _fermat(n), lambda: _rho(n, t_deadline)):
        if time.time() > t_deadline: break
        try:
            f = fn()
        except Exception:
            f = None
        if f and 1 < f < n and n % f == 0:
            return f, n // f
    return None

# ---------------- Tier 1: GNFS (CADO-NFS, GPU-sieved) ----------------
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

def gnfs_solve(n, ncpu, timeout):
    """Real factoring for hard semiprimes via the GPU-GNFS pipeline:
    msieve GPU polyselect -> gpu_loop GPU sieve -> memfd-backed msieve downstream
    (filter + in-RAM Lanczos + sqrt). All multi-GB intermediates live in anonymous RAM
    (memfd), so /tmp stays ~100 MB under the 1 GiB sandbox. Returns (p,q) or None.
    Skips gracefully if the toolchain isn't built into the image."""
    base = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(base, "gpu_nfs_siever", "factor_msv.sh")
    if not os.path.exists(script): return None
    workdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gnfs_work")
    env = dict(os.environ, GNFS_WALL=str(int(timeout)))   # tell the sieve its wall budget so it uses
    # the full time instead of quitting at QMAX. GNFS_LIVE_LOG (debug only): when set, stream the
    # sub-pipeline's progress (polyselect / sieve / filter) to our stdout live so it can be watched in
    # real time; unset (the validator default) keeps the original silent-capture + end-of-run digest.
    live = bool(os.environ.get("GNFS_LIVE_LOG"))
    import threading
    buf = []
    timed_out = {"v": False}
    try:
        proc = subprocess.Popen(["bash", script, str(n), workdir],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
    except Exception:
        return None
    def _watch():
        try: proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out["v"] = True
            proc.kill()
    t = threading.Thread(target=_watch, daemon=True); t.start()
    for line in proc.stdout:
        buf.append(line)
        if live:
            sys.stdout.write(line); sys.stdout.flush()
    proc.wait(); t.join(timeout=5)
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
    deadline = t0 + budget
    log(f"Breaking RSA: {num_bits}-bit N, GPU + {ncpu} CPU threads, budget {budget:.0f}s")
    _gpu_keepalive_start()   # pin the GPU across the multi-context GNFS run (see top of file)

    pq = None; method = None
    # Tier 0 — cheap exact checks (cap 30s)
    pq = quick_factor(num, min(deadline, t0 + 30))
    if pq: method = "quick"
    # Tier 1 — GNFS real factoring (remaining budget, leave 60s slack for output)
    if not pq and (deadline - time.time()) > 120:
        cap = deadline - time.time() - 60
        log(f"Tier 1: GNFS / CADO-NFS (cap {cap:.0f}s)")
        res = gnfs_solve(num, ncpu, cap)
        if res: pq, method = res, "gnfs"

    if pq:
        p, q = sorted(pq)
        result = {"status": "success", "p": p, "q": q}; status = "success"
        log(f"SUCCESS via {method} in {time.time()-t0:.1f}s")
    else:
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
