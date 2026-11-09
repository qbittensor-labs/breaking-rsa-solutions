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

"""mock_siever.py -- stand-ins for glas-gpu (GLAS_SESSION mode) and CADO las, for the CPU co-sieve KAT
(pipeline_c169/tests/las_cosieve_harness.sh). No GPU, no CADO: relations are synthetic but EXACT for the harness's tiny
polynomial (every listed prime divides the norm and their product equals |norm|, b > 0, gcd(a,b) = 1, primes < 2^lpb), so
the driver's real relation gates (las_ctl.py gate, on the first GPU chunk and on the first las chunk) run unchanged.

  modes:  glas <las args> -t T -gpu -session CTL -session-ack ACK
              the READY / BAND <id> <q0> <q1> <out> / ACK <id> <n> <q1> protocol of gpu/glas_session_patch.py: writes
              <out>.tmp, renames to <out> (relation lines only), ACKs; EOF on CTL -> exit 5
          las  <las args> -q0 Q0 -q1 Q1 -out OUT -t T
              writes OUT.tmp.<pid> progressively ('#' per-q comment lines interleaved, as las does), renames on completion
  per special-q: the multiples of MOCK_QSTEP in [q0, q1); a PRNG seeded by q draws MOCK_K + (q/step mod 5) pairs (a, b).
  env:    MOCK_DIR        sidecars: <mode>_bands.tsv 'id q0 q1 n', <mode>_rels/<id>.rels (exact copies), <mode>_args.log
          MOCK_QSTEP (1000)  MOCK_K (12)  MOCK_BAND_SECS (glas, per band, 2)  MOCK_LAS_SECS (las, per chunk, 2)
          MOCK_LAS_FAIL   '' | rc1:<j> (chunk j: exit 1 after its .tmp) | badprime:<j> (chunk j's first relation lists a
                          wrong prime -> the gate must reject it) | torn:<j> (chunk j's published file has no final newline)
"""
import math, os, random, re, sys, time

SMALL = [p for p in range(2, 2000) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def is_prime(n):
    if n < 2: return False
    for p in SMALL:
        if n % p == 0: return n == p
    d, s = n - 1, 0
    while d % 2 == 0: d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):   # deterministic below 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1: break
        else: return False
    return True


def rho(n, rng):
    if n % 2 == 0: return 2
    for c in range(1, 60):
        y = rng.randrange(1, n); x = ys = y; r = q = g = 1
        while g == 1:
            x = y
            for _ in range(r): y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(128, r - k)):
                    y = (y * y + c) % n; q = q * abs(x - y) % n
                g = math.gcd(q, n); k += 128
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n; g = math.gcd(abs(x - ys), n)
        if g != n: return g
    return None


def factor(n, rng, out):
    """prime factors of n (> 0) with multiplicity appended to out; False when it cannot be split"""
    for p in SMALL:
        while n % p == 0: out.append(p); n //= p
        if p * p > n: break
    def split(m):
        if m == 1: return True
        if m < 4000000 or is_prime(m): out.append(m); return True   # no factor < 2000 and < 2000^2 -> prime
        d = rho(m, rng)
        if d is None or d == m: return False
        return split(d) and split(m // d)
    return split(n)


def read_poly(path):
    c, g = {}, {}
    for L in open(path):
        m = re.match(r'\s*c(\d+):\s*(-?\d+)\s*$', L)
        if m: c[int(m.group(1))] = int(m.group(2))
        m = re.match(r'\s*Y(\d+):\s*(-?\d+)\s*$', L)
        if m: g[int(m.group(1))] = int(m.group(2))
    return [c.get(i, 0) for i in range(max(c) + 1)], [g.get(i, 0) for i in range(max(g) + 1)]


def hom(coef, a, b):
    d = len(coef) - 1
    return sum(coef[i] * a ** i * b ** (d - i) for i in range(d + 1))


def relations(F, G, q, lpb0, lpb1, kbase):
    """exact relation lines for special-q q (the (a,b) draw is seeded by q: reproducible, q-specific)"""
    rng = random.Random(q * 1000003 + 17); k = kbase + (q // max(1, QSTEP)) % 5; out = []; tries = 0
    while len(out) < k and tries < 40 * k:
        tries += 1
        a = rng.randint(-3000, 3000); b = rng.randint(1, 60)
        if a == 0 or math.gcd(a, b) != 1: continue
        n0 = abs(hom(G, a, b)); n1 = abs(hom(F, a, b))
        if n0 == 0 or n1 == 0: continue
        f0, f1 = [], []
        if not (factor(n0, rng, f0) and factor(n1, rng, f1)): continue
        if (f0 and max(f0) >= 1 << lpb0) or (f1 and max(f1) >= 1 << lpb1): continue
        out.append("%d,%d:%s:%s" % (a, b, ','.join('%x' % p for p in sorted(f0)), ','.join('%x' % p for p in sorted(f1))))
    return out


def band(F, G, q0, q1, lpb0, lpb1):
    lines = []; k = int(os.environ.get('MOCK_K', '12'))
    qs = range(((q0 + QSTEP - 1) // QSTEP) * QSTEP, q1, QSTEP)
    for q in qs: lines.append((q, relations(F, G, q, lpb0, lpb1, k)))
    return lines


def args(argv):
    d = {}; i = 0
    while i < len(argv):
        if argv[i].startswith('-') and i + 1 < len(argv) and not argv[i + 1].startswith('-'): d[argv[i][1:]] = argv[i + 1]; i += 2
        else: d[argv[i].lstrip('-')] = True; i += 1
    return d


def sidecar(mode, ident, q0, q1, lines):
    md = os.environ.get('MOCK_DIR')
    if not md: return
    os.makedirs(os.path.join(md, mode + '_rels'), exist_ok=True)
    with open(os.path.join(md, mode + '_rels', '%s.rels' % ident), 'w') as f:
        for L in lines: f.write(L + '\n')
    with open(os.path.join(md, mode + '_bands.tsv'), 'a') as f: f.write('%s\t%d\t%d\t%d\n' % (ident, q0, q1, len(lines)))


def log_args(mode, argv):
    md = os.environ.get('MOCK_DIR')
    if md:
        os.makedirs(md, exist_ok=True)
        with open(os.path.join(md, mode + '_args.log'), 'a') as f: f.write(' '.join(argv) + '\n')


def main():
    global QSTEP
    mode = sys.argv[1]; a = args(sys.argv[2:]); log_args(mode, sys.argv[2:])
    QSTEP = int(os.environ.get('MOCK_QSTEP', '1000'))
    F, G = read_poly(a['poly']); lpb0 = int(a.get('lpb0', 50)); lpb1 = int(a.get('lpb1', 50))
    if mode == 'glas':
        ack = open(a['session-ack'], 'a'); ack.write('READY %d\n' % os.getpid()); ack.flush()
        with open(a['session'], 'r') as ctl:
            for line in ctl:
                w = line.split()
                if not w or w[0].startswith('#'): continue
                if w[0] == 'QUIT': sys.exit(0)
                if w[0] != 'BAND' or len(w) != 5: sys.stderr.write('mock glas: bad command %r\n' % line); sys.exit(1)
                ident, q0, q1, out = w[1], int(w[2]), int(w[3]), w[4]
                t0 = time.time(); per_q = band(F, G, q0, q1, lpb0, lpb1); lines = [L for _, ls in per_q for L in ls]
                time.sleep(max(0.0, float(os.environ.get('MOCK_BAND_SECS', '2')) - (time.time() - t0)))
                with open(out + '.tmp', 'w') as f:
                    for L in lines: f.write(L + '\n')
                os.replace(out + '.tmp', out); sidecar('glas', ident, q0, q1, lines)
                sys.stderr.write('# GLAS_SESSION band %s q[%d,%d) %d relations\n' % (ident, q0, q1, len(lines)))
                ack.write('ACK %s %d %d\n' % (ident, len(lines), q1)); ack.flush()
        sys.exit(5)   # EOF without QUIT: the driver died
    if mode == 'las':
        q0, q1, out = int(a['q0']), int(a['q1']), a['out']
        ident = os.path.basename(out).split('.')[0]; j = int(re.sub(r'\D', '', ident) or 0)
        fail = os.environ.get('MOCK_LAS_FAIL', ''); fmode, _, fj = fail.partition(':')
        hit = fail and int(fj or 0) == j
        per_q = band(F, G, q0, q1, lpb0, lpb1)
        if hit and fmode == 'badprime':   # append a spurious prime (0x3b=59) to side 1 of the first relation IN per_q,
            for _q, ls in per_q:          # so the file written below carries it and the driver's las_ctl.py gate rejects it
                if ls: ab, s0, s1 = ls[0].split(':'); ls[0] = '%s:%s:%s' % (ab, s0, (s1 + ',3b') if s1 else '3b'); break
        lines = [L for _, ls in per_q for L in ls]
        tmp = '%s.tmp.%d' % (out, os.getpid()); dt = float(os.environ.get('MOCK_LAS_SECS', '2')) / max(1, len(per_q))
        with open(tmp, 'w') as f:
            for q, ls in per_q:
                f.write('# Sieving q=%d: %d relations\n' % (q, len(ls)))
                for L in ls: f.write(L + '\n')
                f.flush(); time.sleep(dt)
            f.write('# Total %d relations\n' % len(lines))
            if hit and fmode == 'torn': f.write('# torn')
        if hit and fmode == 'rc1': sys.stderr.write('mock las: injected failure on chunk %d\n' % j); sys.exit(1)
        os.replace(tmp, out); sidecar('las', ident, q0, q1, lines)
        sys.stderr.write('# mock las chunk %s q[%d,%d) %d relations\n' % (ident, q0, q1, len(lines))); sys.exit(0)
    sys.stderr.write(__doc__); sys.exit(2)


if __name__ == '__main__':
    main()
