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
  soptin <raw.p> <K> <prefix> <nparts> : the K best candidates (raw form when present) as sopt input files
  soptout <raw.p> <sopt.out...>      : append sopt's re-optimized polys to raw.p as size-optimized blocks
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

def _polykey(p):
    return tuple(p.get(k) for k in ('Y0', 'Y1', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5'))

def cmd_topk(raw, K, out):
    # identical polynomials are kept once: the sopt re-optimization step (SOPT_EFF) appends a copy of every
    # candidate it re-ran, and for most of them sopt returns the very same polynomial
    txt = open(raw).read(); sc = []; seen = set(); dup = 0
    for b in blocks_sizeopt(txt):
        m = re.search(r'exp_E\s+([\d.]+)', b)
        if m: sc.append((float(m.group(1)), b))
    sc.sort(key=lambda t: t[0])
    if not sc: print("FATAL: no size-optimized candidates in", raw); sys.exit(3)
    keep = []
    for e, b in sc:
        k = _polykey(fields(b))
        if k in seen: dup += 1; continue
        seen.add(k); keep.append((e, b))
        if len(keep) >= int(K): break
    with open(out, 'w') as f: f.write(''.join(b for _, b in keep))
    print(f"candidates {len(sc)} best exp_E {sc[0][0]} kept {len(keep)}" + (f" ({dup} duplicates skipped)" if dup else ""))

def _sopt_list(raw, K):
    """soptin's candidate list: [(exp_E, sized poly, sopt input q)] for the K best of raw.p, in soptin's order"""
    txt = open(raw).read(); c = []
    for b in blocks_sizeopt(txt):
        m = re.search(r'exp_E\s+([\d.]+)', b)
        if m: c.append([float(m.group(1)), fields(b), None])
    # raw forms: each '# Raw polynomial:' block's commented coefficients, keyed by position before the next sized block
    parts = re.split(r'(?=# Raw polynomial:)', txt)
    rawmap = {}
    for pt in parts:
        if not pt.startswith('# Raw polynomial:'): continue
        rp, _, sz = pt.partition('# Size-optimized polynomial')
        r = {}
        for L in rp.splitlines():
            mm = re.match(r'#\s*(n|c\d|Y\d):\s*(-?\d+)\s*$', L)
            if mm: r[mm.group(1)] = mm.group(2)
        s = fields('# Size-optimized polynomial' + sz.split('# Raw polynomial:')[0]) if sz else {}
        if s and all(k in r for k in ('n', 'Y0', 'Y1', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5')): rawmap[_polykey(s)] = r
    c.sort(key=lambda t: t[0]); c = c[:int(K)]
    return [(e, p, rawmap.get(_polykey(p)) or p) for e, p, _ in c]

def _sopt_inpoly(q):
    return f"n: {q['n']}\n" + ''.join(f"{k}: {q[k]}\n" for k in ('Y0', 'Y1', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5')) + "\n"

def cmd_soptin(raw, K, prefix, nparts):
    """input of the SOPT_EFF step: the K best candidates of a CADO polyselect / gps1 raw.p (lowest exp_E) as
    uncommented polynomials for `sopt -inputpolys`, round-robin into <prefix><i>.poly, i < nparts. Both
    writers emit '# Raw polynomial:' (commented coefficients) right before each '# Size-optimized polynomial:'
    block; the RAW form is re-optimized when present (sopt starts from it, as polyselect does), else the
    size-optimized form."""
    c = _sopt_list(raw, K); nraw = 0
    if not c: print("FATAL: no size-optimized candidates in", raw); sys.exit(3)
    nparts = max(1, min(int(nparts), len(c))); fs = [open(f"{prefix}{i}.poly", 'w') for i in range(nparts)]
    for j, (e, p, q) in enumerate(c):
        if q is not p: nraw += 1
        fs[j % nparts].write(_sopt_inpoly(q))
    for f in fs: f.close()
    print(f"sopt input: {len(c)} candidates (exp_E {c[0][0]} .. {c[-1][0]}), {nraw} from their raw form, {nparts} parts")

def _sopt_block(p, e, lognorm):
    return ("# Size-optimized polynomial:\n" + f"n: {p['n']}\n"
            + ''.join(f"{k}: {p[k]}\n" for k in ('Y0', 'Y1', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5'))
            + f"skew: {p['skew']}\n# exp_E {e:.2f}, lognorm {lognorm}, skew {p['skew']} (sopt)\n\n")

def cmd_soptout(out, *souts):
    """append sopt's re-optimized polynomials ('### Size-optimized polynomial (i) ###' blocks with a
    '# side 1 lognorm L, exp_E E, ...' line) to <out> as raw.p-style '# Size-optimized polynomial:' blocks;
    a block cut short has no exp_E line and is dropped. sopt block-buffers stdout, so a part killed by SOPT_MAX
    usually leaves nothing at all: SOPT_MAX is a hang guard, not a budget"""
    n = 0; best = None; blocks = []
    for so in souts:
        try: txt = open(so).read()
        except OSError: continue
        for b in re.split(r'### Size-optimized polynomial \(\d+\) ###', txt)[1:]:
            b = b.split('### Input raw polynomial')[0]
            m = re.search(r'^# side \d+ lognorm ([\d.]+), exp_E ([\d.]+)', b, re.M)
            p = fields(b)
            if not m or not all(k in p for k in ('n', 'skew', 'Y0', 'Y1', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5')): continue
            e = float(m.group(2)); best = e if best is None else min(best, e); n += 1
            blocks.append(_sopt_block(p, e, m.group(1)))
    if not n: print(f"FATAL: no re-optimized polynomial in {len(souts)} sopt output file(s)"); sys.exit(3)
    with open(out, 'a') as f: f.write("\n" + ''.join(blocks))
    print(f"sopt output: {n} re-optimized polynomials appended, best exp_E {best:.2f}")

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

# ---- PS_OVERLAP (ps_overlap.sh): memoized sopt / ropt, so work done on gps1 batch 1 while batch 2 runs on the GPU
# is reused and the final pick is the one the sequential path makes from the same raw.p. Memo key = the input
# polynomial (Y0 Y1 c0..c5) exactly as sopt / polyselect_ropt echo it; a miss (unparsable echo, killed part) is
# simply re-run, so the memo can cost time but never change the answer.
_KEYS = ('Y0', 'Y1', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5')

def _echoed(txt):
    r = {}
    for L in txt.splitlines():
        mm = re.match(r'#\s*(n|c\d|Y\d):\s*(-?\d+)\s*$', L)
        if mm: r[mm.group(1)] = mm.group(2)
    return tuple(r.get(k) for k in _KEYS) if all(k in r for k in _KEYS) else None

def _sopt_memo(dirs):
    import glob, os
    memo = {}
    for d in dirs:
        for so in sorted(glob.glob(os.path.join(d, '*.out'))):
            txt = open(so).read()
            for ch in re.split(r'### Input raw polynomial \(\d+\) ###', txt)[1:]:
                inp, hdr, out = ch.partition('### Size-optimized polynomial')
                k = _echoed(inp)
                if k is None or not hdr or k in memo: continue
                m = re.search(r'^# side \d+ lognorm ([\d.]+), exp_E ([\d.]+)', out, re.M); p = fields(out)
                if m and all(x in p for x in ('n', 'skew') + _KEYS): memo[k] = (p, float(m.group(2)), m.group(1))
    return memo

def cmd_ovsoptmiss(raw, K, prefix, nparts, *dirs):
    """the soptin list of raw.p minus the memo (sopt outputs under dirs) -> <prefix><i>.poly; prints 'miss M of L'"""
    c = _sopt_list(raw, K)
    if not c: print("FATAL: no size-optimized candidates in", raw); sys.exit(3)
    memo = _sopt_memo(dirs); miss = []; seen = set()
    for e, p, q in c:
        k = _polykey(q)
        if k not in memo and k not in seen: seen.add(k); miss.append(q)
    nparts = max(1, min(int(nparts), len(miss)))
    if miss:
        fs = [open(f"{prefix}{i}.poly", 'w') for i in range(nparts)]
        for j, q in enumerate(miss): fs[j % nparts].write(_sopt_inpoly(q))
        for f in fs: f.close()
    print(f"miss {len(miss)} of {len(c)}")

def cmd_ovsoptnparts(raw, K, nparts):
    print(max(1, min(int(nparts), len(_sopt_list(raw, K)))))

def cmd_ovsoptapply(raw, K, nparts, order, *dirs):
    """append to raw.p exactly what `soptin K nparts` + sopt + `soptout raw <glob order>` appends: candidate j of
    the soptin list sits in part j % nparts, parts are read in `order` (comma list of part indices, the shell's glob
    order of in*.out), candidates in list order within a part; a candidate with no memo entry is dropped, like a
    sopt block cut short. Exit 3 (and nothing appended) when nothing is usable, like soptout."""
    c = _sopt_list(raw, K); memo = _sopt_memo(dirs); n = 0; best = None; blocks = []
    nparts = max(1, min(int(nparts), len(c)))
    for i in [int(x) for x in order.split(',') if x != '']:
        for j in range(i, len(c), nparts):
            r = memo.get(_polykey(c[j][2]))
            if r is None: continue
            p, e, ln = r; best = e if best is None else min(best, e); n += 1
            blocks.append(_sopt_block(p, e, ln))
    if not n: print(f"FATAL: no re-optimized polynomial in the memo"); sys.exit(3)
    with open(raw, 'a') as f: f.write("\n" + ''.join(blocks))
    print(f"sopt output: {n} re-optimized polynomials appended, best exp_E {best:.2f}")

def _ropt_memo(outs):
    """LEGACY, BROKEN -- kept only so gpu/review5_model.py can reproduce the defect; nothing calls it any more.
    It assumes each output follows its input, but multithreaded polyselect_ropt prints ALL input headers first
    and the outputs in completion order: on every saved log it returns ONE entry (the last input) whose body holds
    all 20 outputs. ovroptmiss / ovroptmerge use the ID-joined, validated memo v2 below (review5 s6.1)."""
    memo = {}
    for ro in outs:
        try: txt = open(ro).read()
        except OSError: continue
        for ch in re.split(r'### input polynomial \d+ ###', txt)[1:]:
            inp, hdr, rest = ch.partition('### root-optimized polynomial')
            k = _echoed(inp)
            if k is None or not hdr or k in memo: continue
            b = rest.split('\n', 1)[1] if '\n' in rest else ''
            p = fields(b)
            if re.search(r'MurphyE\([^)]*\)=([\d.eE+-]+)', b) and all(x in p for x in ('skew',) + _KEYS):
                memo[k] = (inp, b)
    return memo

def _topk_polys(top):
    return [fields(b) for b in blocks_sizeopt(open(top).read())]

# ---- ropt memo v2 (review5 s6.1): join input and output blocks BY PRINTED ID within each file, validate every
# block, key by input identity + result-relevant options + executable identity, merge only with exact coverage.
# Any structural defect (duplicate ID, output without input, missing/duplicated coefficient, n mismatch, no common
# root mod n, non-finite E, MurphyE bounds != run options, mixed options/executables, ambiguous duplicate) raises
# RoptCacheError -> exit 3 -> ps_overlap.sh b falls back to the sequential path. A partially written file
# (truncated tail, killed run, no Stat trailer) is never reused: all of its inputs are cache misses.
import math, os, json, hashlib

class RoptCacheError(ValueError): pass

_RO_HDR = re.compile(r'^### (input|root-optimized) polynomial (\d+) ###[ \t]*$', re.M)
_RO_END = re.compile(r'^(?:###|# Stat:)', re.M)      # a block ends at the next header, progress line or Stat trailer
_RO_MURPHY = re.compile(r'^# side 1 MurphyE\(([^)]*)\)=(\S+)[ \t]*$', re.M)
_RO_FIELDS = ('n', 'skew') + _KEYS

def _ro_poly(block, commented, what):
    pat = re.compile(r'^' + (r'#[ \t]*' if commented else '') + r'(n|skew|Y[01]|c[0-5]):[ \t]*(\S+)[ \t]*$', re.M)
    p = {}
    for k, v in pat.findall(block):
        if k in p: raise RoptCacheError(f"{what}: field {k} printed twice")
        p[k] = v
    miss = [k for k in _RO_FIELDS if k not in p]
    if miss: raise RoptCacheError(f"{what}: missing {','.join(miss)}")
    try: q = {k: int(p[k]) for k in ('n',) + _KEYS}; sk = float(p['skew'])
    except ValueError: raise RoptCacheError(f"{what}: unparsable coefficient")
    if not (math.isfinite(sk) and sk > 0): raise RoptCacheError(f"{what}: bad skew {p['skew']}")
    N, y0, y1 = q['n'], q['Y0'], q['Y1']
    if N <= 1 or y1 == 0 or math.gcd(y1, N) != 1: raise RoptCacheError(f"{what}: Y1 not invertible mod n")
    m = (-y0 * pow(y1, -1, N)) % N
    if sum(q[f'c{i}'] * pow(m, i, N) for i in range(6)) % N:
        raise RoptCacheError(f"{what}: f and g have no common root mod n")
    return p

def _ro_skew(s): return f"{float(s):.3f}"

def _ro_input_key(p):
    """input identity: n and the 8 coefficients. Not the skew: gps1's raw.p blocks (hence most of top.poly) carry
    no 'skew:' line and polyselect_ropt echoes one it computed itself (key17 b3 input 0: raw.p comment 3906418.08,
    echo 4081466.841). The echoed skew is checked against a top.poly 'skew:' line when there is one (_ro_lookup),
    and two outputs for one key that differ in anything raise 'ambiguous' (ropt_cache)."""
    return (str(int(p['n'])),) + tuple(str(int(p[k])) for k in _KEYS)

def _ro_opts(cmdline):
    """the result-relevant polyselect_ropt options of a command line (-t and -inputpolys are not)"""
    t = cmdline.split(); o = {}
    for i, a in enumerate(t[:-1]):
        if a in ('-ropteffort', '-area', '-Bf', '-Bg'): o[a[1:]] = t[i + 1]
    if set(o) != {'ropteffort', 'area', 'Bf', 'Bg'}: raise RoptCacheError(f"ropt options incomplete: {sorted(o)}")
    try: return tuple((k, float(o[k])) for k in ('Bf', 'Bg', 'area', 'ropteffort'))
    except ValueError: raise RoptCacheError("unparsable ropt option")

def _ro_bounds(b):
    d = {}
    for kv in b.split(','):
        k, _, v = kv.partition('=')
        try: d[k.strip()] = float(v)
        except ValueError: raise RoptCacheError(f"bad MurphyE bounds '{b}'")
    return d

def parse_ropt(txt, name='<text>'):
    """one polyselect_ropt output -> dict(complete, why, exe, opts, ins={id: poly}, outs={id: dict(poly, E, text)},
    order=[output IDs in completion order]). Raises RoptCacheError on any defect; complete=False only for a
    truncated / killed file, which is never reused."""
    h = re.match(r'# \((\S+)\) \S+ (.*)$', txt, re.M)
    if not h: raise RoptCacheError(f"{name}: no polyselect_ropt command header")
    opts = _ro_opts(h.group(2)); od = dict(opts)
    pre = txt[:txt.index('###')] if '###' in txt else txt
    exe = hashlib.sha1('\n'.join([h.group(1)] + re.findall(r'^# (?:M |Compiled with|Compilation flags).*$', pre, re.M))
                       .encode()).hexdigest()[:16]
    stat = re.search(r'^# Stat: total phase', txt, re.M)
    marks = list(_RO_HDR.finditer(txt)); ins, outs, order = {}, {}, []; truncated = None
    for j, m in enumerate(marks):
        kind, idx = m.group(1), int(m.group(2)); last = j + 1 == len(marks)
        nxt = _RO_END.search(txt, m.end()); stop = nxt.start() if nxt else len(txt)
        if not last: stop = min(stop, marks[j + 1].start())
        block = txt[m.end():stop]; what = f"{name}: {kind} polynomial {idx}"
        if kind == 'input':
            if idx in ins: raise RoptCacheError(f"{what}: duplicate input ID")
            ins[idx] = _ro_poly(block, True, what); continue
        if idx in outs: raise RoptCacheError(f"{what}: duplicate output ID")
        ms = _RO_MURPHY.findall(block)
        if not ms:
            if last and not stat and nxt is None: truncated = f"output {idx} cut short"; break
            raise RoptCacheError(f"{what}: no '# side 1 MurphyE(...)=' line")
        if len(ms) > 1: raise RoptCacheError(f"{what}: {len(ms)} MurphyE lines")
        bd, es = ms[0]
        try: E = float(es)
        except ValueError: raise RoptCacheError(f"{what}: unparsable MurphyE '{es}'")
        if not (math.isfinite(E) and E > 0): raise RoptCacheError(f"{what}: non-finite or non-positive MurphyE {es}")
        bd = _ro_bounds(bd)
        if set(bd) != {'Bf', 'Bg', 'area'} or any(abs(bd[k] - od[k]) > 1e-3 * od[k] for k in bd):
            raise RoptCacheError(f"{what}: MurphyE bounds {bd} differ from the run's options {od}")
        p = _ro_poly(block, False, what)
        outs[idx] = dict(poly=p, E=E, text=block); order.append(idx)
    if not ins: raise RoptCacheError(f"{name}: no input polynomial")
    for idx in order:                                    # the join is by ID, independent of block order
        if idx not in ins: raise RoptCacheError(f"{name}: root-optimized polynomial {idx}: output without an input")
        if int(outs[idx]['poly']['n']) != int(ins[idx]['n']):
            raise RoptCacheError(f"{name}: root-optimized polynomial {idx}: n differs from input {idx}")
    if len({int(p['n']) for p in ins.values()}) != 1: raise RoptCacheError(f"{name}: inputs with different n")
    why = truncated or (None if stat else "no '# Stat: total phase' trailer")
    if why is None and set(outs) != set(ins): why = f"inputs without output: {sorted(set(ins) - set(outs))}"
    if why is None:
        m = re.search(r'^# (\d+) polynomial\(s\) read\.', txt, re.M)
        if m and int(m.group(1)) != len(ins): why = f"{m.group(1)} polynomials read, {len(ins)} input blocks"
    return dict(complete=why is None, why=why, exe=exe, opts=opts, ins=ins, outs=outs, order=order)

def _ro_norm(p):
    """output identity for consistency checks: (n, Y0, Y1, sign-normalized c, skew)"""
    c = [int(p[f'c{i}']) for i in range(6)]
    if c[5] < 0: c = [-v for v in c]
    return (int(p['n']), int(p['Y0']), int(p['Y1']), tuple(c), _ro_skew(p['skew']))

def ropt_cache(paths, warn=None):
    """{(input key, opts, exe): entry} from complete ropt outputs. Missing files are skipped, incomplete files are
    skipped entirely (warned), defects raise. IDs are local to a file: the join is per file, the key is the input.
    The same input seen twice (same opts/exe) must give the same output and E, else 'ambiguous' raises."""
    cache = {}
    for path in paths:
        try: txt = open(path).read()
        except OSError: continue
        r = parse_ropt(txt, path)
        if not r['complete']:
            if warn: warn(f"ropt memo: {path} not reused ({r['why']})")
            continue
        for idx, ip in sorted(r['ins'].items()):
            o = r['outs'][idx]; k = (_ro_input_key(ip), r['opts'], r['exe'])
            e = dict(inp=ip, out=o['poly'], E=o['E'], text=o['text'], src=path, id=idx, opts=r['opts'], exe=r['exe'])
            if k in cache:
                c = cache[k]
                if _ro_norm(c['out']) != _ro_norm(e['out']) or c['E'] != e['E']:
                    raise RoptCacheError(f"ambiguous memo: {c['src']}#{c['id']} and {path}#{idx} (same input) differ")
                continue
            cache[k] = e
    return cache

def _ro_expected_opts():
    """PS_ROPT_OPTS (ps_overlap.sh sets it to the options its ropt_run uses): only entries with these options hit"""
    s = os.environ.get('PS_ROPT_OPTS')
    return _ro_opts(s) if s else None

def _ro_lookup(cache, p, opts):
    """validated entries for one top.poly block (fields p), restricted to opts (None = any); a top.poly skew
    that differs from the skew polyselect_ropt echoed for that input is a miss"""
    if not all(k in p for k in ('n',) + _KEYS): raise RoptCacheError("top.poly block without n/coefficients")
    ik = _ro_input_key(p); sk = _ro_skew(p['skew']) if 'skew' in p else None
    return [e for (k, o, x), e in cache.items()
            if k == ik and (opts is None or o == opts) and (sk is None or sk == _ro_skew(e['inp']['skew']))]

def _ro_atomic_write(path, data):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, 'w') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def cmd_ovroptmiss(top, out, *outs):
    """the polys of top.poly with no validated memo entry in the ropt outputs -> out (topk format); prints
    'miss M of L'. Exit 3 on a defective memo file (ps_overlap.sh b then falls back to the sequential path)."""
    blocks = blocks_sizeopt(open(top).read())
    try:
        cache = ropt_cache(outs, warn=lambda m: print(m, file=sys.stderr)); opts = _ro_expected_opts()
        if opts is None and len({(e['opts'], e['exe']) for e in cache.values()}) > 1:
            raise RoptCacheError("memo mixes ropt options/executables and PS_ROPT_OPTS is unset")
        miss = [b for b in blocks if not _ro_lookup(cache, fields(b), opts)]
    except RoptCacheError as ex: print(f"FATAL: {ex}"); sys.exit(3)
    _ro_atomic_write(out, ''.join(miss))
    print(f"miss {len(miss)} of {len(blocks)}")

def cmd_ovroptmerge(top, out, *outs):
    """ropt.out for top.poly assembled from the memo, in top.poly order with IDs renumbered 0..L-1 (so `best`
    breaks E ties toward the earlier top.poly entry). Every requested input must be covered by exactly one
    validated output, all with one options/executable set; nothing outside top.poly is emitted. Writes
    <out>.manifest.json then <out>, each atomically, or writes nothing and exits 3 (missing / ambiguous)."""
    try:
        polys = _topk_polys(top)
        if not polys: raise RoptCacheError(f"no polynomial in {top}")
        for p in polys: _ro_lookup({}, p, None)            # field check
        if len({_ro_input_key(p) for p in polys}) != len(polys): raise RoptCacheError(f"duplicate polynomial in {top}")
        cache = ropt_cache(outs, warn=lambda m: print(m, file=sys.stderr)); opts = _ro_expected_opts()
        w = ["# ps_overlap: merged root-optimized polynomials (memo v2), top.poly order\n"]; man = []; used = set()
        for i, p in enumerate(polys):
            hit = _ro_lookup(cache, p, opts)
            if not hit: raise RoptCacheError(f"top.poly #{i} has no validated ropt output")
            if len(hit) > 1: raise RoptCacheError(f"top.poly #{i} matched {len(hit)} entries (options/executables differ)")
            e = hit[0]; ip = e['inp']; used.add((e['opts'], e['exe']))
            w.append(f"\n### input polynomial {i} ###\n" + ''.join(f"# {k}: {ip[k]}\n" for k in ('n',) + _KEYS + ('skew',))
                     + f"\n### root-optimized polynomial {i} ###\n{e['text'].strip()}\n")
            man.append(dict(id=i, src=e['src'], src_id=e['id'], E=e['E'],
                            key=hashlib.sha1(repr(_ro_input_key(ip)).encode()).hexdigest()[:16]))
        if len(used) != 1: raise RoptCacheError(f"merged outputs mix ropt options/executables: {sorted(used)}")
        o, x = used.pop()
        _ro_atomic_write(out + '.manifest.json',
                         json.dumps(dict(top=top, opts=dict(o), exe=x, n=len(man), entries=man), indent=1) + '\n')
        _ro_atomic_write(out, ''.join(w))
    except RoptCacheError as ex: print(f"FATAL: {ex}"); sys.exit(3)
    print(f"merged {len(polys)} of {len(polys)}")

if __name__ == '__main__':
    c = sys.argv[1]; a = sys.argv[2:]
    {'topk': cmd_topk, 'best': cmd_best, 'job': cmd_job, 'msfb': cmd_msfb, 'norm': cmd_norm, 'topn': cmd_topn, 'pick': cmd_pick,
     'soptin': cmd_soptin, 'soptout': cmd_soptout, 'ovsoptmiss': cmd_ovsoptmiss, 'ovsoptnparts': cmd_ovsoptnparts,
     'ovsoptapply': cmd_ovsoptapply, 'ovroptmiss': cmd_ovroptmiss, 'ovroptmerge': cmd_ovroptmerge}[c](*a)
