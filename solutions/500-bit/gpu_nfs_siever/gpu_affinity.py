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

"""Pin a GPU-driving process to the CPUs local to the GPU's NUMA node, then exec it.

WHY THIS EXISTS (2026-08-12). A submission that SUCCEEDED on an AMD EPYC 9555 returned
WallTimeFailure on an Intel Xeon 6980. Nothing in this pipeline had ever done any NUMA or
GPU-affinity handling, and the two hosts differ in exactly the way that makes that matter:

  * EPYC 9555 is single-socket / 64C. A 24-CPU cpuset is one NUMA domain, and it is the domain
    that owns the GPU's PCIe root complex. Locality is accidental but total.
  * Xeon 6980P is 128C and is commonly dual-socket and/or SNC-partitioned. A 24-CPU cpuset can
    straddle NUMA domains, and can land wholly or partly on the socket that does NOT own the GPU.
    Every host buffer then allocates remote (first-touch) and every H2D/D2H crosses the
    inter-socket link.

The sieve is 88% of the wall (12,453s of 14,104s on key08) and issues per-lattice transfers, so a
remote-socket placement is charged ~2.1M times over a run. This module makes the placement explicit
instead of leaving it to whatever cpuset the validator's `docker run` happens to hand us.

WHAT IT DOES NOT DO. It only ever NARROWS the affinity mask to a subset of what we were already
given -- it cannot grant CPUs the cgroup did not. On a single-node host (every box these timings
were measured on) it is a NO-OP by construction, so it cannot regress the EPYC result.

WHY ONLY THE GPU STAGES ARE PINNED. Narrowing the mask costs threads. That is safe for gpu_loop and
gpu_la.py and NOT safe for the CPU-bound stages:
  * gpu_loop: its CPU stage hides under the 7.9 ms GPU stage. Pipeline.md 14b measured 8.215 ms/lat
    at OMP=4 vs 8.230 at OMP=24 -- a 3% thread budget costs 0.2%. That measurement is the reason
    MIN_CPUS below is 4, and it is why this is nearly free even when the local node is half the set.
  * gpu_la.py drives CuPy from one host thread; its host cost is the matrix upload.
  * polyselect (32 slices), the CADO filter probe (NT=20) and the sqrt (OMP window) are CPU-bound
    and scale with thread count, so they are deliberately left on the full cpuset. Their gain from
    locality is real but unmeasured, and it is not worth trading measured threads for it blind.

usage:
  gpu_affinity.py -- <cmd> [args...]   set affinity, then exec (PID is PRESERVED -- factor_msv.sh
                                       kills the sieve by PID, so this MUST exec, never fork)
  gpu_affinity.py --print              print the CPU list it would pin to (empty = no-op)

env:
  GPU_NUMA_PIN=0        disable entirely (exec straight through / print nothing)
  GPU_NUMA_MIN_CPUS=N   refuse to pin below N local CPUs (default 4; see the 14b note above)
"""
import os, sys

MIN_CPUS_DEFAULT = 4


def _log(msg):
    sys.stderr.write(f"  [affinity] {msg}\n")
    sys.stderr.flush()


def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _parse_cpulist(s):
    """'0-3,8,12-13' -> {0,1,2,3,8,12,13}. Returns an empty set on anything unparseable."""
    out = set()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = part.split("-", 1)
                out.update(range(int(lo), int(hi) + 1))
            else:
                out.add(int(part))
        except ValueError:
            return set()
    return out


def gpu_numa_node():
    """The NUMA node owning the NVIDIA GPU(s), or None if it cannot be determined.

    sysfs is the host's PCI tree even inside the container, and the cpu ids in
    /sys/devices/system/node/*/cpulist are the same namespace sched_getaffinity reports, so the
    intersection below is meaningful. Class 0x0300 = VGA controller, 0x0302 = 3D controller (the
    class a datacenter card without a display engine presents).
    """
    base = "/sys/bus/pci/devices"
    try:
        devices = sorted(os.listdir(base))
    except OSError:
        return None
    nodes = set()
    for bdf in devices:
        if (_read(f"{base}/{bdf}/vendor") or "").lower() != "0x10de":   # NVIDIA
            continue
        cls = (_read(f"{base}/{bdf}/class") or "").lower()
        if not (cls.startswith("0x0300") or cls.startswith("0x0302")):
            continue
        node = _read(f"{base}/{bdf}/numa_node")
        try:
            node = int(node)
        except (TypeError, ValueError):
            continue
        if node >= 0:            # -1 = "not reported", the normal VM answer
            nodes.add(node)
    # Several GPUs on DIFFERENT nodes: we cannot tell which one CUDA will pick (ordering depends on
    # CUDA_VISIBLE_DEVICES and the driver's enumeration), and pinning to the wrong one is worse than
    # not pinning. The validator spec is a single RTX PRO 6000, so this is a guard, not a case.
    return nodes.pop() if len(nodes) == 1 else None


def target_cpus():
    """The CPU set to pin to, or None for 'do nothing'."""
    if os.environ.get("GPU_NUMA_PIN", "1") in ("0", "no", "false", ""):
        return None
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return None
    if not allowed:
        return None

    try:
        nnodes = len([d for d in os.listdir("/sys/devices/system/node")
                      if d.startswith("node") and d[4:].isdigit()])
    except OSError:
        nnodes = 0
    if nnodes < 2:
        return None              # single-node host: nothing to be local to

    node = gpu_numa_node()
    if node is None:
        _log(f"{nnodes} NUMA nodes but the GPU's node is not reported — not pinning")
        return None

    local = _parse_cpulist(_read(f"/sys/devices/system/node/node{node}/cpulist"))
    pin = allowed & local
    if not pin:
        # The cpuset has NO cpu on the GPU's node. Narrowing is impossible; this is worth saying
        # out loud because it is the worst placement and only the host operator can fix it.
        _log(f"WARNING: none of our {len(allowed)} CPUs are on the GPU's NUMA node {node} — "
             "every host buffer and transfer is remote. Not pinnable from inside the container.")
        return None
    if pin == allowed:
        return None              # already entirely local — the EPYC case, stay silent-ish

    try:
        floor = int(os.environ.get("GPU_NUMA_MIN_CPUS", MIN_CPUS_DEFAULT))
    except ValueError:
        floor = MIN_CPUS_DEFAULT
    if len(pin) < floor:
        _log(f"GPU is on NUMA node {node} but only {len(pin)} of our {len(allowed)} CPUs are "
             f"local (floor {floor}) — not pinning, the thread loss would cost more")
        return None

    _log(f"GPU on NUMA node {node}: pinning to {len(pin)} local CPUs of {len(allowed)} "
         f"({','.join(str(c) for c in sorted(pin))})")
    return pin


def main(argv):
    if len(argv) >= 2 and argv[1] == "--print":
        pin = target_cpus()
        print(",".join(str(c) for c in sorted(pin)) if pin else "")
        return 0
    if len(argv) >= 2 and argv[1] == "--":
        cmd = argv[2:]
    else:
        cmd = argv[1:]
    if not cmd:
        sys.stderr.write(__doc__)
        return 2

    # Best-effort THROUGHOUT: a failure to pin must never be a failure to sieve.
    try:
        pin = target_cpus()
        if pin:
            os.sched_setaffinity(0, pin)
    except Exception as e:                                    # noqa: BLE001 - see above
        _log(f"pinning failed ({e!r}) — running unpinned")

    # execvp, NOT a fork: factor_msv.sh backgrounds this and later does `kill "$SVPID"` to stop the
    # sieve on a MATRIX verdict. exec keeps the PID, so that kill still reaches gpu_loop itself.
    # A wrapper that forked would swallow the signal and the sieve would run on to REL_TARGET.
    try:
        os.execvp(cmd[0], cmd)
    except OSError as e:
        sys.stderr.write(f"  [affinity] exec {cmd[0]} failed: {e}\n")
        return 127


if __name__ == "__main__":
    sys.exit(main(sys.argv))
