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

"""las_ctl.py -- small helpers for the CADO las CPU co-siever driven by factor_c160.sh.

  las_ctl.py gate <poly.cado> <lpb0> <lpb1> <relfile> [nmax]
      EXACT relation check of the first nmax (default 2000) relations of a las output: for every
      relation a,b:<side0 primes>:<side1 primes> the listed primes (hex, with multiplicity) must
      multiply to |G(a,b)| and |F(a,b)| EXACTLY, gcd(a,b)=1, b>0, and every prime must be < 2^lpb.
      This catches a poly/param mismatch between las and the GPU siever (their relations would
      not pool) before a single las relation reaches the chunk pool. Comment lines are skipped.
      exit 0 "GATE PASS n=<k>" | exit 1 "GATE FAIL <why>"
  las_ctl.py aff <pid> <k>
      Confine EVERY thread of <pid> (/proc/<pid>/task/*) to k CPUs taken from this process's own
      allowed set (the orchestrator's). This is how las's CPU share is lowered mid-chunk: las's -t
      is fixed at launch, but k CPUs cap its bandwidth at k whatever its thread count (measured at
      c160: -t 14 confined to 6 CPUs = 95% of a native -t 6). Under a CFS-quota container this is
      what keeps the cgroup's runnable load <= its quota; nice alone does not (quota throttling
      stops every thread of the cgroup, the GPU feeder included).
      exit 0 "AFF k=<k> tids=<n>" | exit 1 on any failure (the caller then degrades to STOP/CONT)
  las_ctl.py mem
      cgroup memory headroom: "headroom_mb=<n> current_mb=<n> max_mb=<n>" or "headroom_mb=inf"
      when no limit is visible (cgroup v2 memory.max / v1 memory.limit_in_bytes). current excludes
      the reclaimable page cache (active_file + inactive_file); memfd/tmpfs pages stay counted.
"""
import os, re, sys
from math import gcd


def read_poly(path):
    t = open(path).read()
    c, g = {}, {}
    for m in re.finditer(r'^c(\d+):\s*(-?\d+)\s*$', t, re.M): c[int(m.group(1))] = int(m.group(2))
    for m in re.finditer(r'^Y(\d+):\s*(-?\d+)\s*$', t, re.M): g[int(m.group(1))] = int(m.group(2))
    if not c or not g:   # CADO "poly0: c0,c1" / "poly1: c0,...,cd" form
        p0 = re.search(r'^poly0:\s*([-\d, ]+)$', t, re.M); p1 = re.search(r'^poly1:\s*([-\d, ]+)$', t, re.M)
        if not (p0 and p1): raise ValueError("cannot parse polynomial " + path)
        g = dict(enumerate(int(x) for x in p0.group(1).split(',')))
        c = dict(enumerate(int(x) for x in p1.group(1).split(',')))
    d = max(c); dg = max(g)
    return [c.get(i, 0) for i in range(d + 1)], [g.get(i, 0) for i in range(dg + 1)]


def hom(coef, a, b):
    d = len(coef) - 1
    return sum(coef[i] * a ** i * b ** (d - i) for i in range(d + 1))


def gate(poly, lpb0, lpb1, relfile, nmax=2000):
    F, G = read_poly(poly)
    L0, L1 = 1 << int(lpb0), 1 << int(lpb1)
    n = 0
    with open(relfile, 'rb') as fh:
        for ln, raw in enumerate(fh, 1):
            if raw.startswith(b'#') or not raw.strip(): continue
            L = raw.decode('ascii', 'replace').strip()
            try:
                ab, s0, s1 = L.split(':')
                a, b = (int(x) for x in ab.split(','))
                p0 = [int(x, 16) for x in s0.split(',')] if s0 else []
                p1 = [int(x, 16) for x in s1.split(',')] if s1 else []
            except ValueError:
                return "GATE FAIL line %d unparsable: %.80s" % (ln, L)
            if b <= 0 or gcd(a, b) != 1: return "GATE FAIL line %d bad (a,b)=(%d,%d)" % (ln, a, b)
            for ps, lim, side in ((p0, L0, 0), (p1, L1, 1)):
                if ps and max(ps) >= lim: return "GATE FAIL line %d side %d prime %x >= 2^lpb" % (ln, side, max(ps))
            m0 = 1
            for p in p0: m0 *= p
            m1 = 1
            for p in p1: m1 *= p
            if m0 != abs(hom(G, a, b)): return "GATE FAIL line %d side 0 primes do not factor |G(a,b)| (poly mismatch?)" % ln
            if m1 != abs(hom(F, a, b)): return "GATE FAIL line %d side 1 primes do not factor |F(a,b)| (poly mismatch?)" % ln
            n += 1
            if n >= nmax: break
    if n == 0: return "GATE FAIL no relations in %s" % relfile
    return "GATE PASS n=%d" % n


def aff(pid, k):
    cpus = sorted(os.sched_getaffinity(0))
    k = max(1, min(int(k), len(cpus)))
    want = set(cpus[-k:])          # the top of the allowed set; everything else is unpinned
    tids = os.listdir('/proc/%d/task' % pid)
    done = 0
    for t in tids:
        try:
            os.sched_setaffinity(int(t), want); done += 1
        except ProcessLookupError:
            pass                    # a thread exited between listdir and the call
    if done == 0: raise OSError("no thread of pid %d could be confined" % pid)
    return "AFF k=%d tids=%d" % (k, done)


def mem():
    for cur, mx in (('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.max'),
                    ('/sys/fs/cgroup/memory/memory.usage_in_bytes', '/sys/fs/cgroup/memory/memory.limit_in_bytes')):
        try:
            m = open(mx).read().strip(); c = int(open(cur).read().strip())
        except (OSError, ValueError):
            continue
        # page cache is reclaimable: count it out (shmem = tmpfs + memfd pages sits on the anon LRU,
        # so it is NOT in active_file/inactive_file and stays counted -- it is the pool itself)
        rec = 0
        try:
            st = dict(l.split() for l in open(os.path.join(os.path.dirname(cur), 'memory.stat')))
            rec = int(st.get('active_file', st.get('total_active_file', 0))) + int(st.get('inactive_file', st.get('total_inactive_file', 0)))
        except (OSError, ValueError):
            pass
        c = max(0, c - rec)
        if m == 'max' or int(m) >= 1 << 60: return "headroom_mb=inf current_mb=%d" % (c >> 20)
        return "headroom_mb=%d current_mb=%d max_mb=%d" % ((int(m) - c) >> 20, c >> 20, int(m) >> 20)
    return "headroom_mb=inf"


def main(a):
    if len(a) >= 5 and a[0] == 'gate':
        r = gate(a[1], a[2], a[3], a[4], int(a[5]) if len(a) > 5 else 2000)
        print(r); return 0 if r.startswith("GATE PASS") else 1
    if len(a) == 3 and a[0] == 'aff':
        print(aff(int(a[1]), a[2])); return 0
    if len(a) == 1 and a[0] == 'mem':
        print(mem()); return 0
    sys.stderr.write(__doc__); return 2


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:     # any helper failure is a named, non-zero exit (the caller is fail-soft)
        print("LASCTL ERROR %s: %s" % (type(e).__name__, e)); sys.exit(1)
