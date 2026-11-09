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

"""test_stop_shape.py -- CPU known-answer test of the sieve-stop rule (pipeline_c169/stop_cost.py); no GPU, no CADO.

Replays the saved cost-rule trajectories (arm B in-container 2026-10-25; keys 25-28 GCP; keys 29/31 GCP extracts in
data/rsa560/audit_ops/stop_traj_key*.txt) through the REAL stop_cost.py:

 1. regression (known answers): with the knobs those runs used (STOP_SHAPE=0), every logged 'at land' / 'at probe'
    verdict is reproduced -- action, A_star (+-0.5M), save_next / cost_next (+-2 s).
 2. STOP_SHAPE=1 on the same landings, rates and probe reaps (probes every STOP_PROBE_EVERY chunks after the first
    matrix; landings and probe points past the logged stop are extrapolated from the run's own fitted curve): the stop,
    its distance to the measured-curve optimum A* (must be <= TOL_RAW), the modelled wall delta vs the actual stop
    (measured sieve curve + kappa_run N(A)^2 + post; must lie in [-DT_LO, +DT_HI]), probe count.
 3. one-point case: arm B's final purge point as the ONLY matrix point (a first matrix past the optimum): STOP_SHAPE=1
    stops at once, the secant rule says need_2_points.

usage: python3 -I pipeline_c169/tests/test_stop_shape.py [-v]      exit 0 = PASS
"""
import math, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.dirname(HERE); REPO = os.path.dirname(P)
SC = os.path.join(P, 'stop_cost.py'); V = '-v' in sys.argv
RUNS = {   # name: (log, measured LA s, merged N, box)
  'armB':  ('data/rsa560/audit_s2/s2b_armB/h_full_stdout.txt', 2542, 15990798, 'container 24 vCPU'),
  'key25': ('data/rsa560/key25f_gcp/runlog_final.txt', 2343, 15606303, 'gcp 31/32'),
  'key26': ('data/rsa560/key26f_gcp/runs_key26f_run.txt', 2536, 16030478, 'gcp 31/32'),
  'key27': ('data/rsa560/key27f_gcp/runs_key27f_run.txt', 2566, 16052864, 'gcp 31/33'),
  'key28': ('data/rsa560/key28f_gcp/runs_key28f_run.txt', 2267, 15286637, 'gcp 31/33'),
  'key29': ('data/rsa560/audit_ops/stop_traj_key29.txt', 2899, 17008016, 'gcp 31/33'),
  'key31': ('data/rsa560/audit_ops/stop_traj_key31.txt', 2824, 16822514, 'gcp 32/33'),
}
LEGACY_ENV = dict(STOP_SHAPE='0', STOP_NFAC='0.25', STOP_C_NNZ_PS='25', STOP_DENSE_NS='0.2', STOP_NNZ_ROW='131', STOP_POST_US='1.8',
                  STOP_LAMBDA='0.023', STOP_MARGIN='0', STOP_PROBE_TAX='20', STOP_ITER_DIV='255.5')
SHAPE_ENV = dict(STOP_SHAPE='1', STOP_KAPPA='9.9', STOP_NFAC='0.246', STOP_POST_US='1.8', STOP_PROBE_TAX='6', STOP_SX='0.22',
                 STOP_SHAPE_A='0.102', STOP_SHAPE_B='0.44', STOP_MINPTS='1', STOP_MARGIN='0')
PROBE_EVERY = int(os.environ.get('STOP_PROBE_EVERY', '2'))
TOL_RAW, DT_LO, DT_HI = 8e6, 60.0, 40.0
A_SH, B_SH, NFAC_M = 0.102, 0.44, 0.246

CH = re.compile(r'^\[(\d+)s\] chunk (\d+) q\[(\d+),(\d+)\) \+(\d+) in (\d+)s \((\d+) rel/s\).*total (\d+)')
PR = re.compile(r'^\[(\d+)s\] probe (\d+) \(chunks 1\.\.(\d+), (\d+) raw\) rc=\d+: (SHORT|MATRIX) raw=(\d+) unique=\d+ sing_excess=(-?\d+) .*final=\'nrows=(\d+) .*t=(\d+)s')
FH = re.compile(r'^\[(\d+)s\] stop rule \(cost\): final handoff on chunks 1\.\.(\d+) \((\d+) raw\): incdup (\d+)s \+ purge NT=\d+ (\d+)s: MATRIX raw=\d+ unique=\d+ sing_excess=(-?\d+) .*final=\'nrows=(\d+)')
DEC = re.compile(r'^\[(\d+)s\] stop rule \(cost\): at (land|probe): (continue|stop_final|stop_probe)(.*)')

def kv(s):
    return {k: v for k, v in re.findall(r'(\w+)=([^\s]+)', s)}

def parse(path):
    chunks, probes, dec, final = [], [], [], None
    for L in open(path, errors='replace'):
        m = CH.match(L)
        if m:
            t, i, qlo, qhi, n, dt, rs, tot = map(int, m.groups()); chunks.append(dict(t=t, i=i, qlo=qlo, qhi=qhi, n=n, dt=dt, tot=tot)); continue
        m = PR.match(L)
        if m:
            t, pi, k, at, st, raw, ex, rows, pt = m.groups()
            probes.append(dict(t=int(t), i=int(pi), k=int(k), raw=int(raw), st=st, ex=int(ex), rows=int(rows), pt=int(pt))); continue
        m = FH.match(L)
        if m:
            t, k, raw, inc, pu, ex, rows = m.groups(); final = dict(t=int(t), k=int(k), raw=int(raw), inc=int(inc), purge=int(pu), ex=int(ex), rows=int(rows)); continue
        m = DEC.match(L)
        if m: dec.append(dict(t=int(m.group(1)), where=m.group(2), act=m.group(3), **kv(m.group(4))))
    return chunks, probes, dec, final

def decide(env, pts, rate, al, step):
    """pts: list of (raw, rows, excess, t) -> (action, kv dict); calls the real stop_cost.py"""
    with tempfile.NamedTemporaryFile('w', suffix='.tsv', delete=False) as f:
        for raw, rows, ex, t in pts: f.write('%d\t%d\t%d\tprobe\t%d\n' % (raw, rows, ex, t))
        name = f.name
    try:
        e = dict(os.environ); e.update(env)
        r = subprocess.run([sys.executable, '-I', SC, 'decide', name, str(rate), str(int(al)), str(int(step))], env=e, capture_output=True, text=True)
        out = r.stdout.strip(); err = r.stderr.strip()       # the orchestrator reads stdout; stderr goes to stopcost.log
    finally: os.unlink(name)
    if r.returncode: raise RuntimeError('stop_cost.py rc=%d: %s' % (r.returncode, err))
    act = out.split()[0]; return act, kv(out), out + (' | stderr: ' + err if err else '')

def shape(u): return (1 + u / A_SH) ** -B_SH

def fit_curve(mpts, nfac):
    """mpts [(raw, rows, ex)] -> (xc, sx, N0): excess line + log-space N0, as stop_cost.py does (all points)"""
    n = len(mpts); ma = sum(p[0] for p in mpts) / n; me = sum(p[2] for p in mpts) / n
    sx = sum((p[0] - ma) * (p[2] - me) for p in mpts) / sum((p[0] - ma) ** 2 for p in mpts); xc = ma - me / sx
    N0 = math.exp(sum(math.log(nfac * p[1]) - math.log(shape(p[0] / xc - 1)) for p in mpts) / n)
    return xc, sx, N0

def sieve_t(chunks):
    def t_at(raw):
        pt, ptot = chunks[0]['t'] - chunks[0]['dt'], 0
        for c in chunks:
            if c['tot'] >= raw: return pt + (raw - ptot) / (c['tot'] - ptot) * (c['t'] - pt)
            pt, ptot = c['t'], c['tot']
        rate = sorted(c['n'] / c['dt'] for c in chunks[-4:])[1]; return pt + (raw - ptot) / rate
    return t_at

def median_rate(chunks, upto):
    rs = [c['n'] // c['dt'] for c in chunks[:upto]][-4:]; rs = sorted(rs); n = len(rs)
    return rs[n // 2] if n % 2 else (rs[n // 2 - 1] + rs[n // 2]) // 2

fails = []
def check(cond, msg):
    if not cond: fails.append(msg); print('  FAIL: ' + msg)
    elif V: print('  ok: ' + msg)

def regression(name, chunks, probes, dec):
    """every logged decision reproduced by stop_cost.py with the legacy knobs"""
    mp = [p for p in probes if p['st'] == 'MATRIX']; nok = 0
    for d in dec:
        ak = int(d['A_K']); pts = [(p['raw'], p['rows'], p['ex'], p['pt']) for p in mp if p['raw'] <= ak]
        if 'rate' in d: act, k, out = decide(LEGACY_ENV, pts, float(d['rate']), int(d['A_L']), int(d['step']))
        else: act, k, out = decide(LEGACY_ENV, pts, 30000, ak, 4000000)     # need_2_points lines carry no rate/A_L/step
        same = act == d['act'] and ('reason' not in d or d['reason'] in out)
        if 'A_star' in d and 'A_star' in k:
            same = same and abs(int(k['A_star']) - int(d['A_star'])) <= 5e5 and abs(float(k['save_next']) - float(d['save_next'])) <= 2 and abs(float(k['cost_next']) - float(d['cost_next'])) <= 2
        check(same, '%s regression at %ds (%s): logged %s %s | replay %s' % (name, d['t'], d['where'], d['act'], ' '.join('%s=%s' % (x, d[x]) for x in ('A_star', 'save_next', 'cost_next') if x in d), out[:160]))
        nok += same
    return nok, len(dec)

def shape_replay(name, chunks, probes, final, la_s, n_merge, env, every):
    """STOP_SHAPE=1 on the run's landings. Returns (stop raw, probes used, decision where, T-model dict)."""
    mp = [p for p in probes if p['st'] == 'MATRIX']
    allpts = [(p['raw'], p['rows'], p['ex']) for p in mp] + ([(final['raw'], final['rows'], final['ex'])] if final else [])
    nfac = n_merge / final['rows'] if final else NFAC_M
    xc, sx, N0 = fit_curve(allpts, nfac)                      # the run's own curve (for extrapolated probe points + T(A))
    kappa = la_s / (n_merge / 1e6) ** 2; t_at = sieve_t(chunks); post = 1.8e-6
    def N_of(A): return N0 * shape(A / xc - 1)
    def T(A): return t_at(A) + kappa * (N_of(A) / 1e6) ** 2 + post * N_of(A) / nfac
    # landings: logged, then extrapolated with the last chunk's size and dt
    land = [(c['i'], c['tot'], c['n'], c['t']) for c in chunks]
    last = chunks[-1]
    for j in range(1, 40): land.append((last['i'] + j, last['tot'] + j * last['n'], last['n'], last['t'] + j * last['dt']))
    # probe reaps: logged matrix probes (prefix k reaped at time t); extrapolated: prefix k reaped at the landing of k+1
    first_k = mp[0]['k']; reaps = {p['k']: (p['t'], p['rows'], p['ex'], p['pt']) for p in mp}
    have_of = {i: h for i, h, n, t in land}; t_of = {i: t for i, h, n, t in land}
    for k in range(mp[-1]['k'] + 1, last['i'] + 39):
        reaps[k] = (t_of[k + 1], N_of(have_of[k]) / nfac, sx * (have_of[k] - xc), mp[-1]['pt'])
    pts = []; nprobe = 0; used_k = 0; stop = None
    for idx, (i, have, n, t) in enumerate(land):
        if i < first_k: continue
        rate = median_rate(chunks, min(i, len(chunks))) if i <= len(chunks) else median_rate(chunks, len(chunks))
        # probes reaped before this landing: prefixes k (k < i) due under the cadence, in order
        for k in sorted(kk for kk in reaps if kk < i and kk > used_k and reaps[kk][0] <= t):
            if k != first_k and (k - first_k) % every: continue
            tk, rows, ex, pt = reaps[k]; A = have_of[k]
            pts.append((A, int(rows), int(ex), pt)); nprobe += 1; used_k = k
            al = max(h for ii, h, nn, tt in land if tt <= tk)      # landed raw at the probe's reap
            act, kvs, out = decide(env, pts, rate, al, n)
            if V: print('    probe k=%d A=%.1fM -> %s %s' % (k, A / 1e6, act, ' '.join('%s=%s' % (x, kvs[x]) for x in ('A_star', 'save_next', 'cost_next') if x in kvs)))
            if act != 'continue': stop = (al, act, 'probe k=%d' % k, kvs); break
        if stop: break
        if pts:
            act, kvs, out = decide(env, pts, rate, have, n)
            if V: print('    land i=%d A=%.1fM -> %s %s' % (i, have / 1e6, act, ' '.join('%s=%s' % (x, kvs[x]) for x in ('A_star', 'save_next', 'cost_next') if x in kvs)))
            if act != 'continue': stop = (have, act, 'land i=%d' % i, kvs); break
    As = [xc * (1 + j * 0.0005) for j in range(0, 400)]; astar = min(As, key=T)
    return stop, nprobe, dict(xc=xc, sx=sx, N0=N0, kappa=kappa, T=T, astar=astar, N_of=N_of, nfac=nfac)

def main():
    print('stop_cost.py: %s' % SC)
    tot_ok = tot = 0; rows = []
    for name, (log, la_s, n_merge, box) in RUNS.items():
        path = os.path.join(REPO, log)
        if not os.path.exists(path): fails.append('missing ' + log); print('  FAIL: missing ' + log); continue
        chunks, probes, dec, final = parse(path)
        ok, n = regression(name, chunks, probes, dec); tot_ok += ok; tot += n
        stop, nprobe, M = shape_replay(name, chunks, probes, final, la_s, n_merge, SHAPE_ENV, PROBE_EVERY)
        a_act = final['raw'] if final else [p for p in probes if p['st'] == 'MATRIX'][-1]['raw']
        n_act = sum(1 for p in probes if p['st'] == 'MATRIX')
        check(stop is not None, '%s: STOP_SHAPE=1 reached a stop' % name)
        if stop is None: continue
        a_sh, act, where, kvs = stop; T = M['T']
        dT = T(a_sh) - T(a_act); dstar = (a_sh - M['astar']) / 1e6
        rows.append((name, box, a_act / 1e6, M['astar'] / 1e6, (T(a_act) - T(M['astar'])), a_sh / 1e6, act, where, dT, dstar, n_act, nprobe, float(kvs.get('A_star', 0)) / 1e6))
        check(abs(a_sh - M['astar']) <= TOL_RAW, '%s: STOP_SHAPE=1 stop %.1fM within %.0fM of the optimum %.1fM (%+.1fM)' % (name, a_sh / 1e6, TOL_RAW / 1e6, M['astar'] / 1e6, dstar))
        check(-DT_LO <= dT <= DT_HI, '%s: modelled wall delta vs the actual stop %+.0f s in [-%.0f, +%.0f]' % (name, dT, DT_LO, DT_HI))
    print('\nregression: %d/%d logged verdicts reproduced by the legacy rule' % (tot_ok, tot))
    check(tot_ok == tot, 'all logged verdicts reproduced')
    print('\n%-6s %-16s %8s %8s %6s | %8s %-10s %-12s %6s %6s %5s %5s %8s' % ('run', 'box', 'actual', 'A*', 'regret', 'shape', 'act', 'where', 'dT', 'dA*', 'np', 'np2', 'rule A*'))
    for r in rows: print('%-6s %-16s %7.1fM %7.1fM %+5.0fs | %7.1fM %-10s %-12s %+5.0fs %+5.1fM %5d %5d %7.1fM' % r)
    print('  actual/A*/shape = stop raw (M); regret = T(actual)-T(A*) modelled; dT = T(shape)-T(actual); dA* = shape-A*; np/np2 = matrix-era probes actual / STOP_SHAPE (every %d); rule A* = the rule\'s own estimate at its stop' % PROBE_EVERY)
    # 3. one-point case: arm B's final purge point as the only matrix point (first matrix past the optimum)
    chunks, probes, dec, final = parse(os.path.join(REPO, RUNS['armB'][0]))
    one = [(final['raw'], final['rows'], final['ex'], final['purge'])]; rate = median_rate(chunks, len(chunks))
    a1, k1, o1 = decide(SHAPE_ENV, one, rate, final['raw'], chunks[-1]['n']); a0, k0, o0 = decide(LEGACY_ENV, one, rate, final['raw'], chunks[-1]['n'])
    print('\none-point case (arm B final point only, %.1fM raw, landed = probed):\n  STOP_SHAPE=1: %s\n  secant     : %s' % (final['raw'] / 1e6, o1[:200], o0[:120]))
    check(a1 == 'stop_probe', 'one-point case: STOP_SHAPE=1 stops at the first matrix (got %s)' % a1)
    check(a0 == 'continue' and 'need_2_points' in o0, 'one-point case: the secant rule needs a second point (got %s)' % o0[:60])
    check(abs(int(k1['xc']) - 369.6e6) < 3e6, 'one-point crossover estimate %.1fM within 3M of 369.6M' % (int(k1['xc']) / 1e6))
    # fallback: nonsense excess (crossover beyond A_K) -> secant rule, not a crash
    bad = [(final['raw'], final['rows'], -50000000, final['purge'])]
    ab, kb, ob = decide(SHAPE_ENV, bad, rate, final['raw'], chunks[-1]['n'])
    check(ab == 'continue' and 'need_2_points' in ob, 'fit failure falls back to the secant rule (got %s)' % ob[:80])
    print('\n%s' % ('PASS' if not fails else 'FAIL (%d)' % len(fails)))
    sys.exit(1 if fails else 0)

if __name__ == '__main__': main()
