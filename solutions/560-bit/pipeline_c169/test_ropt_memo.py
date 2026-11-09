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

"""Unit tests for the ropt memo v2 (review5 s6.1 / s7.2) on the REAL saved polyselect_ropt logs.

    python3 -B -m unittest discover -s pipeline_c169 -p test_ropt_memo.py -v

No CADO binary is run; every test is pure Python on data/rsa560/polys_s77w and data/rsa560/audit_pstrial/src.
"""
import contextlib, io, json, os, random, re, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polysel160 as ps  # noqa: E402

DATA = HERE.parent / 'data/rsa560'
S77 = DATA / 'polys_s77w'
LOGS = sorted(S77.glob('*.ropt.out')) + sorted((DATA / 'audit_pstrial/src').glob('*.ropt.out'))
# review5 s2.3 table: b2 inputs reusable in b3, per key
SHARED = {15: 14, 16: 12, 17: 13, 18: 20}


def text(name):
    return (S77 / name).read_text()


def topk_text(polys):
    """a top.poly (topk / blocks_sizeopt format) holding these input polys, in order"""
    return ''.join("# Size-optimized polynomial:\n" + f"n: {p['n']}\n" + ''.join(f"{k}: {p[k]}\n" for k in ps._KEYS)
                   + f"skew: {p['skew']}\n# exp_E 45.00, lognorm 52.00, skew {p['skew']}\n\n" for p in polys)


def run(fn, *a, env=None):
    """call a cmd_* helper; returns (exit code, stdout)"""
    out = io.StringIO(); old = dict(os.environ)
    try:
        if env: os.environ.update(env)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try: fn(*a); rc = 0
            except SystemExit as ex: rc = ex.code
    finally:
        os.environ.clear(); os.environ.update(old)
    return rc, out.getvalue()


def assoc(r):
    """input identity -> (normalized output, E): the association that must not depend on block order"""
    return {ps._ro_input_key(r['ins'][i]): (ps._ro_norm(r['outs'][i]['poly']), r['outs'][i]['E']) for i in r['ins']}


def trailer(txt):
    t = txt[txt.index('# 0-th best polynomial found'):]
    return ps._ro_poly(t, True, 'trailer'), float(re.search(r'MurphyE\([^)]*\)=(\S+)', t)[1])


class Tmp(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory(); self.d = Path(self._d.name)

    def tearDown(self):
        self._d.cleanup()

    def put(self, name, s):
        (self.d / name).write_text(s); return str(self.d / name)


class RealLogs(unittest.TestCase):
    def test_key17_b3_join(self):
        r = ps.parse_ropt(text('key17.b3.ropt.out'))
        self.assertTrue(r['complete'])
        self.assertEqual((len(r['ins']), len(r['outs'])), (20, 20))
        self.assertEqual(r['order'][0], 7)                       # first completion is input 7, not input 0/19
        w = max(r['outs'], key=lambda i: r['outs'][i]['E'])
        self.assertEqual(w, 0)
        self.assertAlmostEqual(r['outs'][0]['E'], 1.370e-09, places=15)
        self.assertEqual(r['outs'][0]['poly']['c5'], '425040')   # winner output belongs to input 0 (c5 425040)
        self.assertEqual(r['ins'][0]['c5'], '425040')

    def test_every_log_parses_with_its_printed_winner(self):
        self.assertEqual(len(LOGS), 16)
        for f in LOGS:
            with self.subTest(f.name):
                txt = f.read_text(); r = ps.parse_ropt(txt, f.name)
                self.assertTrue(r['complete'], r['why'])
                self.assertEqual(sorted(r['ins']), list(range(20)))
                self.assertEqual(sorted(r['outs']), list(range(20)))
                w = max(r['outs'], key=lambda i: r['outs'][i]['E'])
                tp, te = trailer(txt)
                self.assertEqual(r['outs'][w]['E'], te)
                self.assertEqual(ps._ro_norm(r['outs'][w]['poly']), ps._ro_norm(tp))
                cado = f.with_name(f.name.replace('.ropt.out', '.poly.cado'))
                if cado.exists():
                    self.assertEqual(ps._ro_norm(r['outs'][w]['poly']), ps._ro_norm(ps.read_poly(cado)))

    def test_legacy_memo_defect_reproduced(self):
        # documents the bug the fix replaces: one entry (last input) holding all 20 outputs
        for k in SHARED:
            old = ps._ropt_memo([str(S77 / f'key{k}.b2.ropt.out')])
            self.assertEqual(len(old), 1)
            (inp, body), = old.values()
            self.assertEqual(1 + body.count('### root-optimized polynomial'), 20)

    def test_shared_b2_b3_inputs_identical_outputs(self):
        for k, n in SHARED.items():
            with self.subTest(key=k):
                a2 = assoc(ps.parse_ropt(text(f'key{k}.b2.ropt.out')))
                a3 = assoc(ps.parse_ropt(text(f'key{k}.b3.ropt.out')))
                sh = set(a2) & set(a3)
                self.assertEqual(len(sh), n)
                self.assertTrue(all(a2[x] == a3[x] for x in sh))

    def test_reorder_blocks_same_association(self):
        txt = text('key17.b3.ropt.out'); base = assoc(ps.parse_ropt(txt))
        marks = list(re.finditer(r'(?m)^### (input|root-optimized) polynomial \d+ ###$', txt))
        stat = txt.index('\n# Stat:')
        head, tail = txt[:marks[0].start()], txt[stat:]
        blocks = [txt[m.start():(marks[j + 1].start() if j + 1 < len(marks) else stat)] for j, m in enumerate(marks)]
        ins = [b for b in blocks if b.startswith('### input')]; outs = [b for b in blocks if b.startswith('### root')]
        rng = random.Random(560)
        for _ in range(5):
            rng.shuffle(ins); rng.shuffle(outs)
            r = ps.parse_ropt(head + ''.join(ins) + ''.join(outs) + tail)
            self.assertTrue(r['complete'])
            self.assertEqual(assoc(r), base)
        # interleaved (input i right before an unrelated output) must not change it either
        mixed = ''.join(a + b for a, b in zip(ins, outs))
        self.assertEqual(assoc(ps.parse_ropt(head + mixed + tail)), base)


class FailClosed(unittest.TestCase):
    T = text('key17.b3.ropt.out')

    def bad(self, txt, msg):
        with self.assertRaisesRegex(ps.RoptCacheError, msg):
            ps.parse_ropt(txt)

    def test_duplicate_ids(self):
        self.bad(self.T.replace('### input polynomial 19 ###', '### input polynomial 18 ###'), 'duplicate input ID')
        self.bad(self.T.replace('### root-optimized polynomial 8 ###', '### root-optimized polynomial 7 ###'),
                 'duplicate output ID')

    def test_output_without_input(self):
        self.bad(self.T.replace('### root-optimized polynomial 8 ###', '### root-optimized polynomial 20 ###'),
                 'output without an input')

    def test_truncated_is_incomplete_never_reused(self):
        t = self.T
        for cut in (t.index('### root-optimized polynomial 15 ###'),            # whole blocks missing
                    t.index('# side 1 MurphyE', t.index('### root-optimized polynomial 17 ###')),  # mid-block
                    t.index('\n# Stat:')):                                    # no trailer
            r = ps.parse_ropt(t[:cut])
            self.assertFalse(r['complete'])
            self.assertEqual(ps.ropt_cache([]), {})
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'cut.out'; f.write_text(t[:cut])
            self.assertEqual(ps.ropt_cache([str(f)]), {})

    def test_missing_coefficient(self):
        b = self.T.index('### root-optimized polynomial 4 ###')
        c3 = self.T.index('\nc3:', b); eol = self.T.index('\n', c3 + 1)
        self.bad(self.T[:c3] + self.T[eol:], 'missing c3')
        i = self.T.index('### input polynomial 3 ###'); c2 = self.T.index('\n# c2:', i)
        self.bad(self.T[:c2] + self.T[self.T.index('\n', c2 + 1):], 'missing c2')

    def test_wrong_modulus(self):
        n = re.search(r'(?m)^# n: (\d+)$', self.T)[1]; n2 = str(int(n) + 2)
        self.bad(self.T.replace('# n: ' + n, '# n: ' + n2, 1), 'no common root|n differs')
        self.bad(self.T.replace(n, n2), 'no common root')           # consistent but wrong N everywhere

    def test_invalid_common_root(self):
        b = self.T.index('### root-optimized polynomial 0 ###')
        m = re.compile(r'(?m)^c0: (-?\d+)$').search(self.T, b)
        self.bad(self.T[:m.start(1)] + str(int(m[1]) + 1) + self.T[m.end(1):], 'no common root')

    def test_non_finite_or_missing_score(self):
        for e in ('nan', 'inf', '-1.0e-09', '0'):
            self.bad(self.T.replace(')=1.370e-09', ')=' + e, 1), 'non-finite|non-positive')
        self.bad(self.T.replace(')=1.370e-09', ')=', 1), "no '# side 1 MurphyE")

    def test_score_parsed_after_paren_equals(self):
        r = ps.parse_ropt(self.T)
        self.assertTrue(all(o['E'] < 1e-8 for o in r['outs'].values()))   # never Bf=4.5e8 / area=2.4e16
        self.bad(self.T.replace(')=1.370e-09', ')=1.370e-09 Bf=4.5e8', 1), "no '# side 1 MurphyE")

    def test_mixed_bounds(self):
        b = self.T.index('### root-optimized polynomial 9 ###')
        t = self.T[:b] + self.T[b:].replace('MurphyE(Bf=4.500e+08', 'MurphyE(Bf=5.000e+08', 1)
        self.bad(t, 'bounds')
        self.bad(self.T.replace('-Bg 200000000', '-Bg 300000000', 1), 'bounds')


class Merge(Tmp):
    def setUp(self):
        super().setUp()
        self.f2 = {k: str(S77 / f'key{k}.b2.ropt.out') for k in SHARED}
        self.f3 = {k: str(S77 / f'key{k}.b3.ropt.out') for k in SHARED}
        self.r3 = {k: ps.parse_ropt(text(f'key{k}.b3.ropt.out')) for k in SHARED}

    def top(self, k, ids):
        return self.put(f'top{k}.poly', topk_text([self.r3[k]['ins'][i] for i in ids]))

    def merged(self, path):
        txt = Path(path).read_text()
        return [ps._ro_poly(b, False, 'merged') for b in re.split(r'### root-optimized polynomial \d+ ###', txt)[1:]], txt

    def test_miss_counts_b3_against_b2_memo(self):
        for k, n in SHARED.items():
            top = self.top(k, range(20)); out = str(self.d / f'miss{k}.poly')
            rc, o = run(ps.cmd_ovroptmiss, top, out, self.f2[k])
            self.assertEqual(rc, 0); self.assertEqual(o.strip(), f"miss {20 - n} of 20")
            self.assertEqual(len(ps.blocks_sizeopt(Path(out).read_text())), 20 - n)
            rc, o = run(ps.cmd_ovroptmiss, top, out, self.f2[k], self.f3[k])
            self.assertEqual(o.strip(), "miss 0 of 20")

    def test_merge_all_20_exactly_20(self):
        for k in SHARED:
            with self.subTest(key=k):
                top = self.top(k, range(20)); out = str(self.d / f'ropt{k}.out')
                rc, o = run(ps.cmd_ovroptmerge, top, out, self.f2[k], self.f3[k])
                self.assertEqual((rc, o.strip()), (0, 'merged 20 of 20'))
                polys, txt = self.merged(out)
                self.assertEqual(len(polys), 20)
                self.assertEqual(txt.count('### input polynomial'), 20)
                man = json.loads(Path(out + '.manifest.json').read_text())
                self.assertEqual(man['n'], 20)
                # cache on (b2 + b3) == cache off (b3 alone): same outputs per input, same best
                want = [ps._ro_norm(self.r3[k]['outs'][i]['poly']) for i in range(20)]
                self.assertEqual([ps._ro_norm(p) for p in polys], want)
                cado = str(self.d / 'm.cado'); ref = str(self.d / 'r.cado')
                rc1, b1 = run(ps.cmd_best, out, self.r3[k]['ins'][0]['n'], cado)
                rc2, b2 = run(ps.cmd_best, self.f3[k], self.r3[k]['ins'][0]['n'], ref)
                self.assertEqual((rc1, rc2), (0, 0)); self.assertEqual(b1, b2)
                self.assertEqual(Path(cado).read_text(), Path(ref).read_text())

    def test_merge_subset_emits_only_subset(self):
        k = 17; ids = [3, 7, 11, 14, 19]               # excludes the winner (input 0)
        top = self.top(k, ids); out = str(self.d / 'sub.out')
        rc, o = run(ps.cmd_ovroptmerge, top, out, self.f2[k], self.f3[k])
        self.assertEqual((rc, o.strip()), (0, 'merged 5 of 5'))
        polys, _ = self.merged(out)
        self.assertEqual([ps._ro_norm(p) for p in polys], [ps._ro_norm(self.r3[k]['outs'][i]['poly']) for i in ids])
        rc, b = run(ps.cmd_best, out, self.r3[k]['ins'][0]['n'], str(self.d / 's.cado'))
        best = max(self.r3[k]['outs'][i]['E'] for i in ids)
        self.assertLess(best, 1.37e-9)
        self.assertIn(f"MurphyE {best:.4e}", b)

    def test_cross_file_ids_do_not_alias(self):
        # key17 b2 and b3 both use IDs 0..19 for different inputs: b3's ID-k output must never be taken from b2's ID k
        k = 17; r2 = ps.parse_ropt(text('key17.b2.ropt.out'))
        cache = ps.ropt_cache([self.f2[k], self.f3[k]])
        for i in range(20):
            e, = ps._ro_lookup(cache, self.r3[k]['ins'][i], None)
            self.assertEqual(ps._ro_norm(e['out']), ps._ro_norm(self.r3[k]['outs'][i]['poly']))
        aliased = sum(ps._ro_input_key(r2['ins'][i]) != ps._ro_input_key(self.r3[k]['ins'][i]) for i in range(20))
        self.assertGreater(aliased, 0)

    def test_missing_fails_and_writes_nothing(self):
        k = 17; top = self.top(k, range(20)); out = str(self.d / 'none.out')
        rc, o = run(ps.cmd_ovroptmerge, top, out, self.f2[k], str(self.d / 'absent.out'))
        self.assertEqual(rc, 3); self.assertIn('FATAL', o)
        self.assertFalse(os.path.exists(out)); self.assertFalse(os.path.exists(out + '.manifest.json'))
        self.assertEqual([p for p in os.listdir(self.d) if '.tmp.' in p], [])

    def test_truncated_b_file_fails_merge(self):
        k = 17; t3 = text('key17.b3.ropt.out')
        cut = self.put('cut.out', t3[:t3.index('### root-optimized polynomial 15 ###')])
        rc, o = run(ps.cmd_ovroptmerge, self.top(k, range(20)), str(self.d / 'x.out'), self.f2[k], cut)
        self.assertEqual(rc, 3)

    def test_ambiguous_duplicate_fails(self):
        k = 17; t3 = text('key17.b3.ropt.out'); r2 = ps.parse_ropt(text('key17.b2.ropt.out'))
        shared = next(i for i in range(20) if ps._ro_input_key(self.r3[k]['ins'][i]) in
                      {ps._ro_input_key(p) for p in r2['ins'].values()})
        b = t3.index(f'### root-optimized polynomial {shared} ###')
        old = re.compile(r'\)=(\S+)').search(t3, b)
        fake = self.put('fake.out', t3[:old.start(1)] + '9.999e-10' + t3[old.end(1):])
        rc, o = run(ps.cmd_ovroptmerge, self.top(k, range(20)), str(self.d / 'x.out'), self.f2[k], fake)
        self.assertEqual(rc, 3); self.assertIn('ambiguous', o)

    def test_stale_options_or_executable(self):
        k = 18; top = self.top(k, range(20)); out = str(self.d / 'o.out')
        good = '-ropteffort 6 -area 2.4e16 -Bf 450000000 -Bg 200000000'
        rc, o = run(ps.cmd_ovroptmerge, top, out, self.f2[k], env={'PS_ROPT_OPTS': good})
        self.assertEqual((rc, o.strip()), (0, 'merged 20 of 20'))
        rc, o = run(ps.cmd_ovroptmiss, top, str(self.d / 'm.poly'), self.f2[k],
                    env={'PS_ROPT_OPTS': good.replace('450000000', '500000000')})
        self.assertEqual(o.strip(), 'miss 20 of 20')                 # other Bf -> explicit misses
        rc, o = run(ps.cmd_ovroptmerge, top, out + '2', self.f2[k], env={'PS_ROPT_OPTS': good.replace('-ropteffort 6', '-ropteffort 8')})
        self.assertEqual(rc, 3)
        # same inputs from a different executable revision: two candidates per input -> refused, not guessed
        t17 = text('key17.b3.ropt.out')
        other = self.put('rev.out', t17.replace('# (b5ca9c4+)', '# (deadbee+)', 1))
        rc, o = run(ps.cmd_ovroptmerge, self.top(17, range(20)), out + '3', self.f3[17], other)
        self.assertEqual(rc, 3)
        rc, o = run(ps.cmd_ovroptmiss, self.top(17, range(20)), str(self.d / 'm2.poly'), self.f3[17], other)
        self.assertEqual(rc, 3)                                       # mixed executables, PS_ROPT_OPTS unset

    def test_real_gps1_topk_without_skew_lines(self):
        # top.poly built by topk from the real gps1 raw.p (no 'skew:' lines; ropt echoes its own skew)
        import tarfile
        with tarfile.open(DATA / 'polysel_raw/psraw_s77w_s77x.tgz') as tar:
            raw = self.put('raw.17.p', tar.extractfile('s77w/raw/raw.17.p').read().decode())
        top = str(self.d / 'top.poly')
        self.assertEqual(run(ps.cmd_topk, raw, '20', top)[0], 0)
        self.assertFalse(any('skew' in p for p in ps._topk_polys(top)))
        rc, o = run(ps.cmd_ovroptmiss, top, str(self.d / 'm.poly'), self.f3[17])
        self.assertEqual((rc, o.strip()), (0, 'miss 7 of 20'))    # 13 of these 20 were ropt'd in b3
        rc, o = run(ps.cmd_ovroptmerge, top, str(self.d / 'x.out'), self.f3[17])
        self.assertEqual(rc, 3)                                     # 7 uncovered -> explicit failure, no partial
        self.assertFalse(os.path.exists(self.d / 'x.out'))

    def test_top_skew_must_match_echo(self):
        k = 17; p = dict(self.r3[k]['ins'][0]); p['skew'] = '1234.5'
        top = self.put('sk.poly', topk_text([p]))
        rc, o = run(ps.cmd_ovroptmiss, top, str(self.d / 'm.poly'), self.f3[k])
        self.assertEqual(o.strip(), 'miss 1 of 1')

    def test_duplicate_request_fails(self):
        k = 17; top = self.top(k, [0, 1, 1])
        rc, o = run(ps.cmd_ovroptmerge, top, str(self.d / 'd.out'), self.f3[k])
        self.assertEqual(rc, 3)


if __name__ == '__main__':
    unittest.main()
