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

"""Pool CADO `polyselect` raw output and emit the top-K by exp_E for rootsieving.

CADO's `polyselect_ropt -inputpolys` needs polynomial blocks separated by BLANK LINES and with
the trailing `# exp_E ...` comment stripped. Feeding it the raw output verbatim silently parses
only ONE polynomial ("1 polynomial(s) read") -- that is not an error message you will notice.

usage: cado_top_raw.py <out.raw> <K> <raw1> [raw2 ...]
"""
import re
import sys


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    dst, K, srcs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]

    blocks, cur = [], []
    for src in srcs:
        for line in open(src):
            if line.startswith("n:"):
                cur = [line]
            elif cur:
                if line.startswith("# exp_E"):
                    m = re.search(r"exp_E ([0-9.]+)", line)
                    if m:
                        blocks.append((float(m.group(1)), "".join(cur)))
                    cur = []
                else:
                    cur.append(line)

    if not blocks:
        sys.exit("no polynomials parsed")
    blocks.sort(key=lambda x: x[0])          # exp_E: lower is better
    keep = blocks[:K]
    # blank line between blocks is load-bearing -- see the docstring
    open(dst, "w").write("\n".join(b for _, b in keep) + "\n")
    print(f"pooled {len(blocks)} raw polys -> top {len(keep)}; "
          f"exp_E {keep[0][0]:.2f} .. {keep[-1][0]:.2f}")


if __name__ == "__main__":
    main()
