# Copyright (C) 2026 qBitTensor Labs.
# Original author: qBitTensor Labs (Enigma / Breaking RSA competition).
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

# Vendored upstream SDK

These files are copied **verbatim** from the official Enigma challenge repo
(`github.com/qbittensor-labs/enigma`, MIT-licensed, © 2026 qBitTensor Labs):

- `__init__.py` — `Challenge` / `Solver` / `Serde` base classes.
- `breaking_rsa.py` — the Breaking RSA `Problem` / `Solution` / `Verif` dataclasses,
  `BreakingRSA.generate` (instance generator), and `validate_breaking_rsa_solution`
  (**the single source of truth for grading**).
- `solution_output.py` — the stdout base64-zip submission protocol
  (`build_solution_zip`, `write_solution_output`, and the reader-side helpers).

**Why vendored:** our solver must validate its own output against the *exact* grader
logic before emitting, and must speak the *exact* stdout protocol. Upstream explicitly
warns that this logic is "shared to prevent divergence." Do **not** edit these files;
refresh them from upstream if the challenge SDK changes.

**Python version:** the SDK uses PEP 695 generics (`class Challenge[P, S, V]`), so it
requires Python **3.12+**.

Updated 2026-10-25: `solution_output.py` re-synced to upstream 9860c5c (2026-10-13), which adds the optional
SOLUTION_OUTPUT_END_LINE marker so stderr text that `docker logs` appends after the payload cannot corrupt it.
