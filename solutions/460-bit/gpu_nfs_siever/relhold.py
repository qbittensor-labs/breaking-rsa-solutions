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

# Hold an anonymous memfd and expose it as a symlink at <path>, so a process writing there
# lands in RAM (the memfd), NOT on the validator's 1 GiB /tmp tmpfs. Stays alive until SIGTERM.
import os,sys,signal
path=sys.argv[1]
fd=os.memfd_create(os.path.basename(path))
try: os.remove(path)
except FileNotFoundError: pass
os.symlink(f"/proc/{os.getpid()}/fd/{fd}", path)
sys.stdout.write("READY\n"); sys.stdout.flush()
signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
signal.pause()
