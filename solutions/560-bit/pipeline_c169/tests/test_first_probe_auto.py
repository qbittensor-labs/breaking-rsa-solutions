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

"""test_first_probe_auto.py -- unit test of `stop_cost.py xover` (FIRST_PROBE_AUTO, default-off knob) against the measured
crossovers of the GCP e2e runs. Run: python3 -I pipeline_c169/tests/test_first_probe_auto.py  (no GPU, no CADO, < 1 s).

A FIRST_PROBE must (a) land BELOW the measured crossover of its regime on every key (a probe past the crossover without a
SHORT point before it would skip the two-phase predictor), and (b) not be more than ~15% early (each extra SHORT probe
costs ~22 s of GPU slowdown + 35-47 s of CPU). The fixed production default 270M is 0.92 x the 31/32 crossover."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, '..', 'stop_cost.py')
sys.path.insert(0, os.path.join(HERE, '..')); import stop_cost

# measured first-MATRIX crossovers (raw relations) of the blind GCP runs: SHORT/MATRIX probe brackets, excess-interpolated
MEASURED = {   # key: (lpb0, lpb1, crossover raw)
    25: (31, 32, 294.5e6), 26: (31, 32, 293.3e6),
    27: (31, 33, 365.0e6), 28: (31, 33, 363.0e6), 29: (31, 33, 368.5e6),
    31: (32, 33, 441.0e6),
}
fails = 0
def check(cond, msg):
    global fails
    print(('ok   ' if cond else 'FAIL ') + msg); fails += 0 if cond else 1

for key, (l0, l1, xc) in sorted(MEASURED.items()):
    fp = stop_cost.xover(l0, l1, 0.92)
    check(fp < xc, f'key {key} lpb {l0}/{l1}: FIRST_PROBE_AUTO {fp/1e6:.0f}M < measured crossover {xc/1e6:.1f}M')
    check(fp > 0.85 * xc, f'key {key} lpb {l0}/{l1}: FIRST_PROBE_AUTO {fp/1e6:.0f}M > 0.85 x crossover ({0.85*xc/1e6:.0f}M)')
# the production default is reproduced for 31/32 (so FIRST_PROBE_AUTO=1 changes nothing there)
check(stop_cost.xover(31, 32, 0.92) == 270000000, f'31/32 at 0.92 = production default 270M (got {stop_cost.xover(31, 32, 0.92)})')
# monotone in both lpbs, integer millions, unlisted regimes extrapolate
check(stop_cost.xover(31, 33) > stop_cost.xover(31, 32) and stop_cost.xover(32, 33) > stop_cost.xover(31, 33) and
      stop_cost.xover(32, 34) > stop_cost.xover(32, 33) and stop_cost.xover(33, 34) > stop_cost.xover(32, 34), 'monotone in lpb0/lpb1')
check(all(stop_cost.xover(a, b) % 1000000 == 0 for a, b in stop_cost.XOVER), 'whole millions')
check(abs(stop_cost.xover(30, 31) / (293e6 / 1.44) - 1) < 0.01, f'unlisted 30/31 extrapolates by XOVER_RUNG ({stop_cost.xover(30, 31)})')
check(abs(stop_cost.xover(32, 33) / stop_cost.xover(31, 32) - 1.5) < 0.02, '32/33 needs ~1.50x the 31/32 raw (measured 441/293)')
# CLI path (what factor_c169.sh calls), and frac validation
out = subprocess.run([sys.executable, '-I', P, 'xover', '32', '33', '0.92'], capture_output=True, text=True)
check(out.returncode == 0 and out.stdout.strip() == str(stop_cost.xover(32, 33, 0.92)), f'CLI xover 32 33 0.92 -> {out.stdout.strip()}')
out = subprocess.run([sys.executable, '-I', P, 'xover', '32', '33'], capture_output=True, text=True)
check(out.returncode == 0 and out.stdout.strip() == '441000000', f'CLI xover 32 33 (frac 1) -> {out.stdout.strip()}')
out = subprocess.run([sys.executable, '-I', P, 'xover', '32', '33', '2.0'], capture_output=True, text=True)
check(out.returncode != 0, 'CLI rejects frac 2.0')
out = subprocess.run([sys.executable, '-I', P, 'xover', '32'], capture_output=True, text=True)
check(out.returncode != 0, 'CLI rejects a missing lpb1')
# the other subcommands still work
out = subprocess.run([sys.executable, '-I', P, 'la', '1e7'], capture_output=True, text=True)
check(out.returncode == 0 and float(out.stdout) > 0, f'la 1e7 still runs ({out.stdout.strip()} s)')
print(f'{"PASS" if fails == 0 else "FAIL"}: {fails} failures'); sys.exit(1 if fails else 0)
