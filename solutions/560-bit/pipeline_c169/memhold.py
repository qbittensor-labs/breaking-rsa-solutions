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

"""memhold.py -- a POOL of RAM-backed files for the c160 pipeline.

The validator's /tmp is a 10 GiB tmpfs, but a c160 run moves ~20 GB of raw relations plus
~11 GB of dup1 slices and ~8 GB of dup2 slices through the CADO filter. RAM (85 GB) is billed the
same whether a byte sits in /tmp or in a memfd -- only the 10 GiB /tmp CAP differs. So every
large intermediate lives in an anonymous memfd, exposed as a symlink <dir>/<name> ->
/proc/<pid>/fd/<n>. Anything that open()s the symlink for writing (O_TRUNC) lands in RAM:
  * our own shell redirections / cat,
  * CADO dup1/dup2/cado2msieve with the vendor/cado-symlink-writethrough.patch (stock CADO
    writes a temp file and rename()s it over the link, which would put the data back on /tmp).

usage: memhold.py <dir> <name> [<name> ...]      (names may use a {a..b} style range: c%03d:1:96)
Creates <dir>/<name> for each, prints READY, then waits for SIGTERM. Memfds are sparse: an
unwritten one costs nothing. Free one early with `: > <dir>/<name>` (truncates through the link).
Exits non-zero WITHOUT creating links if memfd_create is unavailable -- never falls back to disk.
"""
import os, sys, signal

def expand(spec):
    # "fmt:lo:hi" -> fmt % i for i in [lo, hi]
    if spec.count(':') == 2 and '%' in spec:
        fmt, lo, hi = spec.split(':')
        return [fmt % i for i in range(int(lo), int(hi) + 1)]
    return [spec]

def main():
    if len(sys.argv) < 3:
        sys.exit("usage: memhold.py <dir> <name|fmt:lo:hi> ...")
    d = sys.argv[1]
    if not hasattr(os, "memfd_create"):
        sys.stderr.write("memhold: os.memfd_create unavailable -- refusing to fall back to disk\n")
        sys.exit(1)
    os.makedirs(d, exist_ok=True)
    names = [n for s in sys.argv[2:] for n in expand(s)]
    fds = []
    pid = os.getpid()
    for n in names:
        fd = os.memfd_create(n)          # no MFD_CLOEXEC needed: we never exec
        fds.append(fd)
        link = os.path.join(d, n)
        parent = os.path.dirname(link)
        os.makedirs(parent, exist_ok=True)
        try:
            os.remove(link)
        except FileNotFoundError:
            pass
        os.symlink("/proc/%d/fd/%d" % (pid, fd), link)
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    signal.signal(signal.SIGINT, lambda *_: os._exit(0))
    sys.stdout.write("READY %d\n" % len(fds))
    sys.stdout.flush()
    while True:
        signal.pause()

if __name__ == "__main__":
    main()
