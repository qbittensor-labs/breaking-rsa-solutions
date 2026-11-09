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

# Hold a RAM-backed file and expose it as a symlink at <path>, so a process writing there
# lands in RAM, NOT on the validator's tiny /tmp / ~10 GiB disk budget. The relation file is
# ~12 GB, so writing it to disk would BLOW the storage limit -- RAM backing is mandatory.
# Stays alive until SIGTERM.
#
#  1. Preferred: an anonymous memfd (os.memfd_create) -- pure RAM, no name in any filesystem,
#     auto-freed when this holder exits. This is the path on the validator's Ubuntu CPython.
#  2. Fallback: a real file under /dev/shm (a RAM-backed tmpfs), which we unlink on exit. Used
#     when this Python was built without os.memfd_create (e.g. some conda-forge builds). Still
#     100% RAM -- it does NOT consume the disk budget.
#
# It NEVER falls back to disk: if neither RAM path is viable it exits WITHOUT creating the
# symlink and logs why, so the failure is loud rather than silently overflowing the disk.
import os, sys, signal

path = sys.argv[1]
base = os.path.basename(path)
# headroom the RAM holder must have free for the relation file (~12 GB) + slack. Override via env.
NEED_BYTES = int(os.environ.get("RELHOLD_MIN_FREE_GIB", "14")) * (1 << 30)

shm_file = None  # set only in the /dev/shm fallback, so we can unlink it on exit

if hasattr(os, "memfd_create"):
    fd = os.memfd_create(base)
    target = "/proc/%d/fd/%d" % (os.getpid(), fd)
else:
    shm_dir = "/dev/shm"
    ok = False
    if os.path.isdir(shm_dir):
        try:
            st = os.statvfs(shm_dir)
            ok = st.f_bavail * st.f_frsize >= NEED_BYTES
        except OSError:
            ok = False
    if not ok:
        # No memfd and no room in RAM tmpfs -> do NOT write the big file to disk (blows the
        # storage budget). Bail loudly; the caller must not proceed with a disk-backed rels file.
        sys.stderr.write(
            "relhold: no os.memfd_create and /dev/shm lacks %d GiB free -- refusing to back "
            "the relation file on disk (would exceed the storage limit)\n"
            % (NEED_BYTES >> 30))
        sys.stderr.flush()
        sys.exit(1)
    shm_file = os.path.join(shm_dir, "relhold.%d.%s" % (os.getpid(), base))
    open(shm_file, "wb").close()
    target = shm_file

try:
    os.remove(path)
except FileNotFoundError:
    pass
os.symlink(target, path)


def _cleanup(*_):
    if shm_file is not None:
        try:
            os.remove(shm_file)
        except OSError:
            pass
    os._exit(0)


signal.signal(signal.SIGTERM, _cleanup)
signal.signal(signal.SIGINT, _cleanup)
sys.stdout.write("READY\n")
sys.stdout.flush()
signal.pause()
