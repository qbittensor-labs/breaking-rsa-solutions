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

"""KAT for the STOP_LA_SCALE knob (pipeline_c169/stop_cost.py, stop_cost.sh, spec_merge.sh, factor_c169.sh).
CPU only, ~15 s:  python3 -I pipeline_c169/tests/test_stop_la_scale.py
 1. default (unset / 1.0): `la` reproduces the pre-knob formula (13.60 s per (M rows)^2 -> 3482 s at N 16M), and the
    replay of `decide` on the eight saved trajectories reproduces every logged verdict (action, A_star +-0.5M, save/cost
    +-2 s) -- the default path is unchanged.
 2. STOP_LA_SCALE=1.28: `la` is exactly x1.28; on every logged decision event A_star moves later and save_next grows,
    cost_next is unchanged (the knob touches the LA term only).
 3. out of range (0.4, 2.5, 'abc'): stop_cost.py exits non-zero (the driver dies at startup, not mid-run).
 4. the shell wrappers pass the knob to every stop_cost.py call (decide, spec); the driver defaults it to 1 and
    validates it; bash -n on the three files.
"""
import importlib.util, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
PC = os.path.join(REPO, 'pipeline_c169'); SC = os.path.join(PC, 'stop_cost.py')
spec = importlib.util.spec_from_file_location('slow_card_model', os.path.join(REPO, 'docs', 'research', 'agents_2026-10-26', 'slow_card_model.py'))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def la(n, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith('STOP_')}; e.update(env or {})
    r = subprocess.run([sys.executable, '-I', SC, 'la', str(n)], env=e, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def main():
    ok = True
    # 1. default arithmetic
    rc, v = la(16e6); ok &= rc == 0 and v == '3482'
    rc1, v1 = la(16e6, {'STOP_LA_SCALE': '1.0'}); ok &= rc1 == 0 and v1 == '3482'
    print(f'1a. la 16e6 default = {v} (expect 3482, pre-knob formula) STOP_LA_SCALE=1.0 -> {v1}: {"OK" if v == v1 == "3482" else "FAIL"}')
    runs = [M.parse(p) for _, p, _, _ in M.RUNS]; fits = [M.fit_run(r) for r in runs]
    with tempfile.TemporaryDirectory() as td:
        n_ok = 0
        for (name, _, _, _), r, f in zip(M.RUNS, runs, fits):
            v = r['verdict']; g = M.replay_rule(r, f, 1.0, 1.0, 1.0, td, extend=False)
            same = g and g['act'] == v['act'] and g['where'] == v['where'] and g['A_K'] == int(v['A_K']) and g['A_L'] == int(v['A_L']) \
                and abs(float(g['d']['A_star']) - float(v['A_star'])) < 0.5e6 and abs(float(g['d']['save_next']) - float(v['save_next'])) <= 2 \
                and abs(float(g['d']['cost_next']) - float(v['cost_next'])) <= 2
            n_ok += bool(same)
        ok &= n_ok == len(M.RUNS)
        print(f'1b. default replay reproduces {n_ok}/{len(M.RUNS)} logged verdicts: {"OK" if n_ok == len(M.RUNS) else "FAIL"}')
        # 2. scaled
        rc, v = la(16e6, {'STOP_LA_SCALE': '1.28'}); ok &= (rc == 0 and v == '4457')   # round(3482 x 1.28)
        print(f'2a. la 16e6 at STOP_LA_SCALE=1.28 = {v} (expect 4457): {"OK" if v == "4457" else "FAIL"}')
        later = 0
        for (name, _, _, _), r, f in zip(M.RUNS, runs, fits):
            v = r['verdict']
            # the same decision event as logged, with the knob: replay stops at or after the logged event
            g0 = M.replay_rule(r, f, 1.0, 1.0, 1.0, td, extend=False)
            sc = M.load_rule(M.rule_env(r, None, 1.28))
            # evaluate decide at the logged decision point with the knob on the logged points
            pts = [p for p in r['pts'] if p['t'] <= v['t']]
            path = os.path.join(td, 'k.tsv')
            with open(path, 'w') as fh:
                for p in pts: fh.write(f"{p['raw']}\t{p['rows']}\t{p['sx']}\tp\t{p['tp']}\n")
            out = sc.decide(path, float(v['rate']), int(v['A_L']), int(v['step'])); d = M.kv(out)
            cond = float(d['A_star']) > float(g0['d']['A_star']) + 1e6 and float(d['save_next']) > float(g0['d']['save_next']) \
                and abs(float(d['cost_next']) - float(g0['d']['cost_next'])) < 0.5
            later += cond
            print(f"    {name:7s} A* {float(g0['d']['A_star'])/1e6:6.1f}M -> {float(d['A_star'])/1e6:6.1f}M  save {float(g0['d']['save_next']):4.0f} -> {float(d['save_next']):4.0f}  "
                  f"cost {float(g0['d']['cost_next']):4.0f} -> {float(d['cost_next']):4.0f}  action {g0['act']} -> {out.split()[0]}")
        ok &= later == len(M.RUNS)
        print(f'2b. at 1.28 A_star later by > 1M, save_next larger, cost_next unchanged on {later}/{len(M.RUNS)} logged decisions: {"OK" if later == len(M.RUNS) else "FAIL"}')
        # 2c. STOP_SHAPE=1 (s5 merge): the knob scales the shape rule's kappa too -- STOP_LA_SCALE=r must give the same
        # decision as STOP_KAPPA=9.9*r (slow_card.md: the shape rule needs the same card factor), and unset == 1.0 verbatim.
        path = os.path.join(td, 'shape.tsv')
        with open(path, 'w') as fh:   # two matrix points of the 32/33 kind (sx 0.225 per M, crossover ~366M)
            fh.write('370000000\t66000000\t1000000\tp\t60\n390000000\t64500000\t5500000\tp\t65\n')
        base = {k: v for k, v in os.environ.items() if not k.startswith('STOP_')}
        shp = dict(STOP_SHAPE='1', STOP_NFAC='0.246', STOP_PROBE_TAX='6')
        def dec(extra):
            e = dict(base); e.update(shp); e.update(extra)
            r = subprocess.run([sys.executable, '-I', SC, 'decide', path, '35000', '395000000', '4000000'], env=e, capture_output=True, text=True)
            return r.returncode, r.stdout.strip()
        rc0, o0 = dec({}); rc1, o1 = dec({'STOP_LA_SCALE': '1.0'}); rcs, os_ = dec({'STOP_LA_SCALE': '1.28'}); rck, ok_ = dec({'STOP_KAPPA': str(9.9 * 1.28)})
        ds, dk = M.kv(os_), M.kv(ok_)
        same = rcs == rck == 0 and os_.split()[0] == ok_.split()[0] and all(abs(float(ds[k]) - float(dk[k])) <= 1 for k in ('A_star', 'N_star', 'save_next', 'cost_next', 'LA_K'))
        ident = rc0 == rc1 == 0 and o0 == o1 and o0.split()[1] == 'shape=1' and float(M.kv(o0)['LA_K']) < float(ds['LA_K'])
        ok &= same and ident
        print(f'2c. STOP_SHAPE=1: unset == 1.0 verbatim ({o0.split()[0]}), STOP_LA_SCALE=1.28 == STOP_KAPPA=9.9x1.28 (A* {float(ds["A_star"])/1e6:.1f}M vs {float(dk["A_star"])/1e6:.1f}M, '
              f'LA_K {ds["LA_K"]} vs {dk["LA_K"]}; default LA_K {M.kv(o0)["LA_K"]}): {"OK" if same and ident else "FAIL"}')
    # 3. out of range
    bad = [la(16e6, {'STOP_LA_SCALE': x})[0] for x in ('0.4', '2.5', 'abc')]
    ok &= all(rc != 0 for rc in bad)
    print(f'3.  out-of-range exit codes {bad} (all non-zero): {"OK" if all(rc != 0 for rc in bad) else "FAIL"}')
    # 4. wrappers
    def has(f, s): return s in open(os.path.join(PC, f)).read()
    w = has('stop_cost.sh', 'STOP_LA_SCALE="${STOP_LA_SCALE:-1}"') and has('spec_merge.sh', 'STOP_LA_SCALE="${STOP_LA_SCALE:-1}"') \
        and has('factor_c169.sh', 'STOP_LA_SCALE=${STOP_LA_SCALE:-1}') and has('factor_c169.sh', 'STOP_LA_SCALE="$STOP_LA_SCALE" python3 "$HERE/stop_cost.py" la 1e7')
    syn = all(subprocess.run(['bash', '-n', os.path.join(PC, f)]).returncode == 0 for f in ('factor_c169.sh', 'stop_cost.sh', 'spec_merge.sh'))
    ok &= w and syn
    print(f'4.  wrappers pass the knob (decide, spec), driver default + validation, bash -n: {"OK" if w and syn else "FAIL"}')
    print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
