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

#
# Breaking RSA solver — multithreaded Self-Initializing Quadratic Sieve (YAFU).
#
# Why SIQS (YAFU) and not GNFS:
#   The validator runs solutions with a HARD sandbox: --read-only root, a single
#   writable /tmp that is a 256 MB *noexec* tmpfs, --user miner, --network none,
#   and a 4-hour wall-time. Empirically, GNFS (CADO) for a 103-digit semiprime
#   needs ~900 MB of scratch (relations + the filtering step doubles it), so it
#   ENOSPCs in 256 MB no matter how it is tuned. msieve's SIQS fits on disk but
#   sieves single-threaded (~10 h). YAFU's SIQS is the one engine that fits both
#   constraints at once: multithreaded sieving finishes a 103-digit factorization
#   in ~7 min on ~24 cores, and its compact relation file peaks at ~150 MB —
#   comfortably inside 256 MB. It is a single self-contained binary, so the
#   noexec tmpfs is a non-issue (nothing is copied to scratch and exec'd).
#
# Pipeline:
#   Stage 0  cheap exact methods (trial division + bounded Pollard's rho) — for
#            small/degenerate inputs and the workbench's low-bit test cases.
#   Stage 1  YAFU SIQS — the engine for the real challenge size.
#
# Input (live validator): the problem is delivered as a read-only mounted file
#   /challenge_input/challenge_input.json containing {difficulty, num, num_bits}.
#   The workbench instead passes (challenge_id, problem_json) as argv. We support
#   BOTH: prefer the mounted file, fall back to argv.
#
# Output contract: logs to stdout, then a magic separator, then a base64 zip of
#   result.json + solve_info.json (see enigma_challenges.solution_output).
#
# Env overrides:
#   YAFU_BIN      path to the yafu binary (default: search common locations)
#   YAFU_THREADS  thread count (default: os.cpu_count())
#   SIQS_WORKDIR  scratch dir for yafu (default: $TMPDIR or /tmp)
#   RHO_BUDGET    Pollard's rho iteration budget before handing off to SIQS

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import *

import gmpy2
from gmpy2 import mpz, gcd

from enigma_challenges.breaking_rsa import Problem, Solution
from enigma_challenges.solution_output import build_solution_zip, write_solution_output

CHALLENGE_INPUT_FILE = "/challenge_input/challenge_input.json"


def _printlog(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stage 0: cheap exact methods
# ---------------------------------------------------------------------------

def _trial_division(n: mpz, bound: int = 1_000_000) -> Optional[int]:
    if n % 2 == 0:
        return 2
    d = 3
    while d <= bound and d * d <= n:
        if n % d == 0:
            return int(d)
        d += 2
    return None


def _pollard_rho_brent(n: mpz, budget: int) -> Optional[int]:
    if n % 2 == 0:
        return 2
    iters = 0
    for c in range(1, 12):
        y, r, q = mpz(2), 1, mpz(1)
        d = mpz(1)
        c_m, x, ys = mpz(c), mpz(0), mpz(0)
        while d == 1 and iters < budget:
            x = y
            for _ in range(r):
                y = (y * y + c_m) % n
            k = 0
            while k < r and d == 1 and iters < budget:
                ys = y
                m = min(128, r - k)
                for _ in range(m):
                    y = (y * y + c_m) % n
                    q = (q * abs(x - y)) % n
                d = gcd(q, n)
                k += m
                iters += m
            r *= 2
        if d == n:
            while True:
                ys = (ys * ys + c_m) % n
                d = gcd(abs(x - ys), n)
                if d > 1:
                    break
        if 1 < d < n:
            return int(d)
        if iters >= budget:
            break
    return None


def _stage0(n: mpz, log) -> Optional[Tuple[int, int]]:
    f = _trial_division(n, 1_000_000)
    if f:
        log(f"Stage 0: trial division found factor {f}")
        return f, int(n // f)
    r = gmpy2.isqrt(n)
    if r * r == n:
        log("Stage 0: N is a perfect square")
        return int(r), int(r)
    budget = int(os.environ.get("RHO_BUDGET", 2_000_000))
    log(f"Stage 0: Pollard's rho (budget {budget})...")
    f = _pollard_rho_brent(n, budget)
    if f and 1 < f < n:
        log(f"Stage 0: Pollard's rho found factor {f}")
        return int(f), int(n // f)
    return None


# ---------------------------------------------------------------------------
# Stage 1: YAFU SIQS
# ---------------------------------------------------------------------------

def _find_yafu() -> Optional[str]:
    env = os.environ.get("YAFU_BIN")
    if env and os.path.isfile(env):
        return env
    for c in ["/opt/yafu/yafu", "/usr/local/bin/yafu",
              os.path.expanduser("~/yafu/yafu"), shutil.which("yafu")]:
        if c and os.path.isfile(c):
            return c
    return None


def _run_yafu_siqs(n: mpz, log) -> Optional[Tuple[int, int]]:
    yafu = _find_yafu()
    if not yafu:
        log("YAFU not found (set YAFU_BIN); skipping")
        return None

    threads = os.environ.get("YAFU_THREADS") or str(os.cpu_count() or 8)
    workdir = os.environ.get("SIQS_WORKDIR") or os.environ.get("TMPDIR") or "/tmp"
    rundir = os.path.join(workdir, "yafu_run")
    os.makedirs(rundir, exist_ok=True)

    log(f"Stage 1: YAFU SIQS, threads={threads}, workdir={rundir}")
    # YAFU reads the expression from stdin; force SIQS (its factor() picks GNFS
    # above 95 digits, which cannot fit the sandbox's disk).
    proc = subprocess.Popen(
        [yafu, "-threads", str(threads)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=rundir,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(f"siqs({int(n)})\n")
    proc.stdin.flush()
    proc.stdin.close()

    captured: List[str] = []
    last_log = 0.0
    for raw in proc.stdout:
        # YAFU rewrites the progress line with \r; split so we can log milestones.
        for line in raw.replace("\r", "\n").splitlines():
            captured.append(line)
            s = line.strip()
            if not s:
                continue
            if s.startswith("***") or s.startswith("P") and " = " in s or "SIQS" in s or "elapsed" in s.lower():
                log(f"  yafu: {s[:160]}")
            elif "rels found" in s:
                now = time.time()
                if now - last_log > 30:  # throttle the noisy progress line
                    log(f"  yafu: {s[:160]}")
                    last_log = now
    proc.wait()

    text = "\n".join(captured)
    factors = [int(x) for x in re.findall(r"\bP\d+\s*=\s*(\d+)", text)]
    # de-dup, keep those that actually divide n
    facs = sorted(set(f for f in factors if 1 < f < n and n % f == 0))
    if len(facs) >= 2 and facs[0] * (n // facs[0]) == n:
        p = facs[0]
        return int(p), int(n // p)
    if len(facs) == 1 and facs[0] * (n // facs[0]) == n:
        return int(facs[0]), int(n // facs[0])
    log("Stage 1: YAFU did not yield a valid factorization")
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def factor_semiprime(n_int: int, num_bits: int, log) -> Tuple[Optional[int], Optional[int], str]:
    n = mpz(n_int)
    log(f"Factoring {num_bits}-bit ({len(str(n_int))}-digit) semiprime")
    res = _stage0(n, log)
    if res:
        return res[0], res[1], "stage0"
    res = _run_yafu_siqs(n, log)
    if res:
        return res[0], res[1], "yafu_siqs"
    return None, None, "failed"


def _load_problem(log) -> Tuple[str, "Problem"]:
    """Live validator delivers the problem as a mounted file; workbench via argv."""
    # Preferred: mounted challenge input file (live validator)
    if os.path.isfile(CHALLENGE_INPUT_FILE):
        try:
            data = json.loads(Path(CHALLENGE_INPUT_FILE).read_text())
            prob = Problem(int(data["difficulty"]), int(data["num"]), int(data["num_bits"]))
            cid = (sys.argv[1].strip() if len(sys.argv) > 1 else "") or "challenge"
            log(f"Loaded problem from {CHALLENGE_INPUT_FILE}")
            return cid, prob
        except Exception as e:
            log(f"Failed to parse {CHALLENGE_INPUT_FILE}: {e}")
    # Fallback: argv (workbench)  <challenge_id> <problem_json>
    if len(sys.argv) == 3:
        cid = sys.argv[1].strip()
        prob = Problem.from_json(sys.argv[2].strip())
        log("Loaded problem from argv")
        return cid, prob
    raise SystemExit("No problem input: expected /challenge_input/challenge_input.json "
                     "or <challenge_id> <problem_json> argv")


def main() -> None:
    timestamp_start = datetime.now(timezone.utc).isoformat()
    start = time.time()
    try:
        challenge_id, problem = _load_problem(_printlog)
    except SystemExit as e:
        print(str(e))
        sys.exit(1)
    if problem.num < 6:
        print("Error: number must be a positive non-trivial semiprime")
        sys.exit(1)

    _printlog(f"Starting Breaking RSA challenge: {challenge_id}")
    _printlog(f"Number size: {problem.num_bits} bits")
    numstr = str(problem.num)
    _printlog(f"N = {numstr[:40]}{'...' if len(numstr) > 40 else ''}")

    p, q, method = factor_semiprime(problem.num, problem.num_bits, log=_printlog)
    solve_time = time.time() - start

    # Self-verify before claiming success: both prime, product == N.
    ok = (
        p is not None and q is not None
        and mpz(p) * mpz(q) == problem.num
        and gmpy2.is_prime(mpz(p)) and gmpy2.is_prime(mpz(q))
    )
    if ok:
        _printlog(f"SUCCESS via {method} in {solve_time:.2f}s")
        solution = Solution("success", int(p), int(q))
    else:
        _printlog(f"FAILED after {solve_time:.2f}s")
        solution = Solution("failed", None, None)

    result_json = json.dumps(solution.to_dict(), indent=2)
    solve_info_json = json.dumps({
        "solution_status": solution.status,
        "challenge_id": challenge_id,
        "timestamp_utc": timestamp_start,
        "solve_time_seconds": solve_time,
        "method": method,
        "num_bits": problem.num_bits,
    })

    output_dir = os.environ.get("OUTPUT_DIR")
    if output_dir:
        try:
            Path(output_dir).mkdir(exist_ok=True)
            Path(output_dir, "result.json").write_text(result_json)
            Path(output_dir, "solve_info.json").write_text(solve_info_json)
        except OSError:
            pass

    zip_bytes = build_solution_zip({
        "result.json": result_json,
        "solve_info.json": solve_info_json,
    })
    write_solution_output(zip_bytes)
    os._exit(0 if solution.status == "success" else 1)


if __name__ == "__main__":
    main()
