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

"""Extract root-optimized polynomials from `polyselect_ropt` output, ranked by MurphyE.

Writes <outdir>/b0.cado, b1.cado, ... in the same format `gpu_loop` and `polyselect_best.sh`
consume (n / skew / c0..c5 / Y0 / Y1), best MurphyE first.

MurphyE tracks gpu_loop yield far better than msieve-E (rho~0.21), which is what lets a top-8
funnel be enough. It is NOT a selector, though: on the 2026-09-04 candidate set the polynomial with
the LOWEST MurphyE of the top three had the HIGHEST measured yield, by 2.4% -- and rel/lat is an
exact count, so that is outside the 1.6% timing noise floor. Rank with this, then bake off.

usage: cado_extract_polys.py <ropt_output.txt> <outdir> [howmany]
"""
import os
import re
import sys

FIELDS = ["c0", "c1", "c2", "c3", "c4", "c5", "Y0", "Y1"]
BLOCK = r"### root-optimized polynomial \d+ ###\n(.*?)MurphyE\([^)]*\)=([0-9.e+-]+)"


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, outdir = sys.argv[1], sys.argv[2]
    howmany = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    os.makedirs(outdir, exist_ok=True)

    polys = []
    for m in re.finditer(BLOCK, open(src).read(), re.S):
        body, E = m.group(1), float(m.group(2))
        d = {}
        for line in body.splitlines():
            mm = re.match(r"^(n|skew|Y0|Y1|c[0-5]):\s*(\S+)", line.strip())
            if mm:
                d[mm.group(1)] = mm.group(2)
        if all(k in d for k in ["n", "skew"] + FIELDS):
            polys.append((E, d))

    if not polys:
        sys.exit("no root-optimized polynomials found")
    polys.sort(key=lambda x: -x[0])          # MurphyE: higher is better
    print(f"{len(polys)} rooted polys, MurphyE {polys[0][0]:.4g} .. {polys[-1][0]:.4g}")
    for i, (E, d) in enumerate(polys[:howmany]):
        with open(os.path.join(outdir, f"b{i}.cado"), "w") as f:
            f.write(f"n: {d['n']}\nskew: {d['skew']}\n")
            for k in FIELDS:
                f.write(f"{k}: {d[k]}\n")
        print(f"  b{i}: MurphyE={E:.4g}  c5={d['c5']}")


if __name__ == "__main__":
    main()
