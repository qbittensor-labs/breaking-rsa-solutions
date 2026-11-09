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

"""polysel160.py -- helpers for the c160 polynomial stage (CADO polyselect -> ropt).

  topk   <raw.p> <K> <out>          : write the K best size-optimized polys (lowest exp_E) in the
                                      format polyselect_ropt -inputpolys reads
  best   <ropt_out> <N> <out.cado>  : pick the root-optimized poly with the HIGHEST MurphyE,
                                      normalize c5 > 0 (msieve -nc3 aborts on a negative leading
                                      coefficient), write a CADO .poly; prints "MurphyE <E> c5 <c5>"
  job    <poly.cado> <job> rlim alim lpbr lpba mfbr mfba   : GGNFS .job for cuda-sieve
  msfb   <poly.cado> <c.fb> rlim alim lpbr lpba             : msieve factor-base/poly file
  topn   <ropt_out> <N> <K> <prefix>  : the K best DISTINCT root-optimized polys by MurphyE as
                                      <prefix><i>.cado (c5 > 0), one line each "<file> MurphyE <E>"
  pick   <results> <murphy_name> [min_gain]  : test-sieve model (see polytest160.sh). results lines:
                                      name E q width nq rel_per_q ms_per_q skipped ; prints the winner

NEVER select by exp_E alone: at c160 the exp_E-best size-optimized poly yielded ~4x less than the
MurphyE-best root-optimized one (measured 2026-09-26). A parse miss is FATAL (exit 3), never a
silent fallback -- that exact fallback produced a wrong "5.8x infeasible" verdict once.
"""
import re, sys

def blocks_sizeopt(txt):
    return [b for b in re.split(r'(?=# Size-optimized polynomial)', txt) if b.startswith('# Size-optimized')]

def fields(block):
    p = {}
    for L in block.splitlines():
        m = re.match(r'\s*(n|skew|c\d|Y\d):\s*(-?[\d.]+)\s*$', L)
        if m: p[m.group(1)] = m.group(2)
    return p

def write_poly(p, path, N=None):
    with open(path, 'w') as f:
        f.write(f"n: {N or p['n']}\nskew: {p['skew']}\n")
        for i in range(6): f.write(f"c{i}: {p['c'+str(i)]}\n")
        f.write(f"Y0: {p['Y0']}\nY1: {p['Y1']}\n")

def cmd_topk(raw, K, out):
    txt = open(raw).read(); sc = []
    for b in blocks_sizeopt(txt):
        m = re.search(r'exp_E\s+([\d.]+)', b)
        if m: sc.append((float(m.group(1)), b))
    sc.sort(key=lambda t: t[0])
    if not sc: print("FATAL: no size-optimized candidates in", raw); sys.exit(3)
    with open(out, 'w') as f: f.write(''.join(b for _, b in sc[:int(K)]))
    print(f"candidates {len(sc)} best exp_E {sc[0][0]} kept {min(len(sc), int(K))}")

def cmd_best(ropt, N, out):
    txt = open(ropt).read(); best = None
    for b in re.split(r'### root-optimized polynomial \d+ ###', txt)[1:]:
        m = re.search(r'MurphyE\([^)]*\)=([\d.eE+-]+)', b)
        if not m: continue
        p = fields(b)
        if all(k in p for k in ['skew', 'Y0', 'Y1'] + [f'c{i}' for i in range(6)]):
            e = float(m.group(1))
            if best is None or e > best[0]: best = (e, p)
    if best is None: print("FATAL: no root-optimized poly with a MurphyE in", ropt); sys.exit(3)
    e, p = best
    if int(p['c5']) < 0:
        for i in range(6): p[f'c{i}'] = str(-int(p[f'c{i}']))
        for k in ('Y0', 'Y1'): pass   # the rational side is unchanged: f -> -f keeps the roots
    write_poly(p, out, N)
    print(f"MurphyE {e:.4e} c5 {p['c5']} skew {p['skew']}")

def _ropt_polys(ropt, N):
    txt = open(ropt).read(); c = []
    for b in re.split(r'### root-optimized polynomial \d+ ###', txt)[1:]:
        m = re.search(r'MurphyE\([^)]*\)=([\d.eE+-]+)', b)
        if not m: continue
        p = fields(b)
        if all(k in p for k in ['skew', 'Y0', 'Y1'] + [f'c{i}' for i in range(6)]): c.append((float(m.group(1)), p))
    c.sort(key=lambda t: -t[0]); return c

def cmd_topn(ropt, N, K, prefix):
    seen = set(); n = 0
    for e, p in _ropt_polys(ropt, N):
        key = tuple(p[f'c{i}'] for i in range(6))
        if key in seen: continue
        seen.add(key)
        if int(p['c5']) < 0:
            for i in range(6): p[f'c{i}'] = str(-int(p[f'c{i}']))
        write_poly(p, f"{prefix}{n}.cado", N); print(f"{prefix}{n}.cado MurphyE {e:.4e}"); n += 1
        if n >= int(K): break
    if n == 0: print("FATAL: no root-optimized poly with a MurphyE in", ropt); sys.exit(3)

def cmd_pick(results, murphy, min_gain='0.015', R='180e6', qmin='2.2e6'):
    """Rank test-sieved candidates by MODELLED sieve time to R relations from qmin: per poly, rel/special-q
    ~ A q^-a fitted from two q points, special-q density 1/ln q for every poly (a degree-5 S5 poly has one
    root per prime on average; the count in a test window is +-5% root-count noise, so it is NOT fitted),
    ms per special-q COMMON to all candidates (the mean of every measurement): the GPU sieve cost per q does not
    depend on the poly (sieve-only ms 23.7-24.2 across 5 key-3 polys, warm GPU) but a per-poly mean picks up
    GPU warm-up (c0 is always measured first: 24.25 vs ~23.0 on a cold GPU) and CPU-load noise of +-2-4%.
    Print the winner; switch away from the MurphyE pick only for a modelled gain >= min_gain."""
    import math
    d = {}
    for L in open(results):
        f = L.split()
        if len(f) < 8: continue
        name, E, q, w, nq, relq, ms, sk = f[0], float(f[1]), float(f[2]), float(f[3]), int(f[4]), float(f[5]), float(f[6]), int(f[7])
        d.setdefault(name, {'E': E, 'pts': {}, 'skipped': 0})
        d[name]['pts'][q] = (nq / w, relq, ms); d[name]['skipped'] += sk
    R, qmin, min_gain = float(R), float(qmin), float(min_gain)
    allms = [m for v in d.values() for (_, _, m) in v['pts'].values() if m > 0]
    msc = sum(allms) / len(allms) if allms else 1.0
    def T(pts):
        (q1, (d1, r1, m1)), (q2, (d2, r2, m2)) = sorted(pts.items())[:2]
        a = math.log(r1 / r2) / math.log(q2 / q1); A = r1 * q1 ** a
        ms = msc; q = qmin; rel = 0.0; t = 0.0; dq = 20000.0
        while rel < R and q < 1e9:
            n = dq / math.log(q); rel += n * A * q ** -a; t += n * ms / 1000; q += dq
        return t
    rows = []
    for name, v in d.items():
        if len(v['pts']) < 2 or v['skipped']: print(f"  {name}: {'norm-width skips' if v['skipped'] else 'incomplete'} -- excluded"); continue
        if any(r <= 0 or n <= 0 or m <= 0 for (n, r, m) in v['pts'].values()): print(f"  {name}: empty band -- excluded"); continue
        rows.append((T(v['pts']), name, v['E']))
    rows.sort()
    tm = next((t for t, n, e in rows if n == murphy), None)
    for t, n, e in rows:
        print(f"  {n:16s} modelled sieve {t:7.0f} s  MurphyE {e:.4e}" + (f"  ({100 * (t / tm - 1):+.1f}% vs MurphyE pick)" if tm else ""))
    if not rows: print("PICK", murphy, "reason=no-results"); return
    if tm is None:   # leave the MurphyE pick only when IT showed norm-width skips (it would lose special-q);
        # any other gap in its measurement (fbgen/bench failure, budget) is no evidence against it
        if d.get(murphy, {}).get('skipped'): print("PICK", rows[0][1], "reason=murphy-pick-skips-special-q"); return
        print("PICK", murphy, "reason=murphy-pick-incomplete"); return
    best_t, best, _ = rows[0]
    if best != murphy and best_t <= tm * (1 - min_gain): print("PICK", best, f"reason=gain={100*(1-best_t/tm):.1f}%")
    else: print("PICK", murphy, f"reason=best-gain={100*(1-best_t/tm):.1f}%<{100*min_gain:.1f}%")

def cmd_norm(src, dst):
    """normalize a supplied CADO poly so c5 > 0 (negating f keeps every relation valid)"""
    p = read_poly(src)
    if int(p['c5']) < 0:
        for i in range(6): p[f'c{i}'] = str(-int(p[f'c{i}']))
        print(f"normalized: c5 was negative, negated c0..c5 -> c5 {p['c5']}")
    write_poly(p, dst)

def read_poly(path):
    p = {}
    for L in open(path):
        m = re.match(r'\s*(n|skew|c\d|Y\d):\s*(-?[\d.]+)\s*$', L)
        if m: p[m.group(1)] = m.group(2)
    return p

def cmd_job(poly, job, rlim, alim, lpbr, lpba, mfbr, mfba):
    p = read_poly(poly)
    with open(job, 'w') as f:
        f.write(f"n: {p['n']}\nskew: {p['skew']}\n")
        for i in range(6): f.write(f"c{i}: {p['c'+str(i)]}\n")
        f.write(f"Y0: {p['Y0']}\nY1: {p['Y1']}\n")
        f.write(f"rlim: {rlim}\nalim: {alim}\nlpbr: {lpbr}\nlpba: {lpba}\nmfbr: {mfbr}\nmfba: {mfba}\n"
                f"rlambda: 2.3\nalambda: 2.3\n")

def cmd_msfb(poly, fb, rlim, alim, lpbr, lpba):
    p = read_poly(poly)
    with open(fb, 'w') as f:
        f.write(f"N {p['n']}\nSKEW {p['skew']}\nR0 {p['Y0']}\nR1 {p['Y1']}\n")
        for i in range(6): f.write(f"A{i} {p['c'+str(i)]}\n")
        f.write(f"FRMAX {rlim}\nFAMAX {alim}\nSRLPMAX {1 << int(lpbr)}\nSALPMAX {1 << int(lpba)}\n")

if __name__ == '__main__':
    c = sys.argv[1]; a = sys.argv[2:]
    {'topk': cmd_topk, 'best': cmd_best, 'job': cmd_job, 'msfb': cmd_msfb, 'norm': cmd_norm, 'topn': cmd_topn, 'pick': cmd_pick}[c](*a)
