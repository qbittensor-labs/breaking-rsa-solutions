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

"""Default-off review6 experiment, against b5ca9c4 + the production stack.

Usage: python3 gpu/glas_empty_pattern_patch.py PATH/sieve_core_gpu.cu
Build the experimental object with -DGLAS_EMPTY_PATTERN=1; default 0 is old behavior.
This does not change the shared-memory reservation or any barriers.
"""
from pathlib import Path
import sys

MARKER = "/* REVIEW6_EMPTY_PATTERN: compile-time experiment, default off. */"
OLD = """    bool const use_pat = segs == 1 && j0 != 0 &&
                         !(t.skip & 2) && !(t.skip & 16);"""
NEW = """    bool const use_pat = segs == 1 && j0 != 0 &&
                         !(t.skip & 2) && !(t.skip & 16) &&
                         (!GLAS_EMPTY_PATTERN || n_pat > n_skip);"""
ANCHOR = "constexpr uint32_t PAT_LEN = 5040;"
DEFINITION = MARKER + "\n#ifndef GLAS_EMPTY_PATTERN\n#define GLAS_EMPTY_PATTERN 0\n#endif\n"


def patch(source):
    if MARKER in source:
        if source.count(NEW) != 1 or source.count(DEFINITION) != 1:
            raise ValueError("partial or conflicting review6 patch")
        return source
    if "GLAS_P1" not in source or "GLAS_IDIV" not in source:
        raise ValueError("requires production P1/narrow stack")
    if source.count(OLD) != 1 or source.count(ANCHOR) != 1:
        raise ValueError("source mismatch; do not guess at the replacement")
    return source.replace(ANCHOR, DEFINITION + ANCHOR, 1).replace(OLD, NEW, 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    old = path.read_text()
    new = patch(old)
    if new != old:
        path.write_text(new)
    print("already patched" if new == old else "patched; opt in at compile time")
