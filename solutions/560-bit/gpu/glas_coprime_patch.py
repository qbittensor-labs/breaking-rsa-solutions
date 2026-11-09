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

"""glas_coprime_patch.py <gpu-siever/src/sieve_core_gpu.cu> -- idea 1 of docs/rsa560_nvidia_ideas.md.

GLAS_FILL_COPRIME becomes a level (stock: a flag, 1 = drop the fill hits where 3 divides both i and j):
  0 keep them, 1 drop 3 | gcd(i, j) (stock default), 2 also 5 and 7, 3 also 11 and 13.
Those cells cannot give a relation (the survivor test keeps gcd(i, j) == 1 only), so the relation set is unchanged
and only the bucket hits (~6% at level 2, ~7.3% at level 3) and their L2 requests go away. Default stays 1."""
import sys
p = sys.argv[1]; s = open(p).read()
if 'c_fill_coprime >= 2' in s: print('already patched'); sys.exit(0)

def sub(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b, 1)

sub('    return !c_fill_coprime || j % 3 || off % 3 != h % 3;\n',
    '    if (!c_fill_coprime)\n'
    '        return true;\n'
    '    if (!(j % 3) && off % 3 == h % 3)\n'
    '        return false;\n'
    '    if (c_fill_coprime >= 2) {   /* GLAS_FILL_COPRIME=2: also 5 and 7, =3: also 11 and 13 */\n'
    '        if ((!(j % 5) && off % 5 == h % 5) || (!(j % 7) && off % 7 == h % 7))\n'
    '            return false;\n'
    '        if (c_fill_coprime >= 3 &&\n'
    '            ((!(j % 11) && off % 11 == h % 11) || (!(j % 13) && off % 13 == h % 13)))\n'
    '            return false;\n'
    '    }\n'
    '    return true;\n')
sub('    bool fill_coprime = true; /* GLAS_FILL_COPRIME=0 keeps the 3 | gcd(i, j) hits */\n',
    '    uint32_t fill_coprime = 1; /* GLAS_FILL_COPRIME: 0 keeps the 3 | gcd(i, j) hits, 2 also drops 5, 7, 3 also 11, 13 */\n')
sub('    ctx.fill_coprime = env_flag("GLAS_FILL_COPRIME", true);\n',
    '    ctx.fill_coprime = 1;\n'
    '    if (char const * e = getenv("GLAS_FILL_COPRIME"))\n'
    '        ctx.fill_coprime = (uint32_t)std::min(3, std::max(0, atoi(e)));\n')
open(p, 'w').write(s); print('patched')
