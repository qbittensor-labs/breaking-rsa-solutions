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

# Parse candidate poly pool(s), dedup, sort by E desc, emit top-K as cand0..cand{K-1}.cado.
# Accepts MULTIPLE .p files (for ensemble best-of-K over the union).
# usage: pick_topk.py <N> <outdir> <K> <p_file_1> [p_file_2 ...]
import sys, re
N, outd, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
pfiles = sys.argv[4:]
cands = []
for pf in pfiles:
    try: txt = open(pf).read()
    except OSError: continue
    for b in re.split(r'(?=# norm)', txt):
        m = re.search(r'e ([\d.eE+-]+)', b)
        if not m or 'skew' not in b: continue
        p = {}
        for L in b.splitlines():
            mm = re.match(r'(skew|c\d|Y\d):\s*(\S+)', L.strip())
            if mm: p[mm.group(1)] = mm.group(2)
        if len(p) >= 9: cands.append((float(m.group(1)), tuple(sorted(p.items())), p))
seen = set(); uniq = []
for e, key, p in sorted(cands, key=lambda x: -x[0]):
    if key in seen: continue
    seen.add(key); uniq.append((e, p))
for i, (e, p) in enumerate(uniq[:K]):
    with open(f"{outd}/cand{i}.cado", "w") as f:
        f.write(f"n: {N}\nskew: {p['skew']}\n")
        for j in range(6): f.write(f"c{j}: {p['c'+str(j)]}\n")
        f.write(f"Y0: {p['Y0']}\nY1: {p['Y1']}\n")
    print(f"cand{i} E={e:.4e} c5={p['c5']} skew={p['skew']}")
print(f"# pool={len(cands)} uniq={len(uniq)} emitted={min(K,len(uniq))}")
