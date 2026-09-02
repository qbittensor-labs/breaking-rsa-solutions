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

#
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#
# the Software.
#

"""
Solution output protocol for Enigma challenges.

Miner containers communicate their results via stdout using this protocol:

  1. Text logs (human-readable, any format)
  2. A magic separator line (SOLUTION_OUTPUT_SEPARATOR)
  3. Base64-encoded zip containing result.json and other artifacts

Docker's json-file logging driver treats stdout as UTF-8 text and corrupts
raw binary, so the zip MUST be base64-encoded. The validator captures stdout
via ``docker logs`` after the container exits.

This module provides helpers for both sides:
  - Solver side: build_solution_zip(), write_solution_output()
  - Reader side: split_on_separator(), extract_artifacts()
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import sys
import zipfile

# Magic separator line. Must match across solver, validator, and workbench.
SOLUTION_OUTPUT_SEPARATOR: bytes = (
    b"\n----- ENIGMA-SOLUTION-OUTPUT-BEGIN-a8c7f3e2-9d4b-4c5a-8f1e-2b6d3a4e5f7c -----\n"
)


# ---------------------------------------------------------------------------
# Solver-side helpers
# ---------------------------------------------------------------------------

def build_solution_zip(files: dict[str, str | bytes]) -> bytes:
    """Pack solution artifacts into an in-memory zip.

    Args:
        files: mapping of archive filename to content (str or bytes).

    Returns:
        Raw zip bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def write_solution_output(zip_bytes: bytes) -> None:
    """Flush all output streams, then emit separator + base64-encoded zip to stdout.

    This MUST be the last thing written to stdout. Any subsequent output will
    corrupt the payload and cause extraction failures on the validator.
    """
    sys.stdout.flush()
    sys.stderr.flush()

    encoded = base64.b64encode(zip_bytes)
    buf = sys.stdout.buffer
    buf.write(SOLUTION_OUTPUT_SEPARATOR)
    buf.write(encoded)
    buf.write(b"\n")
    buf.flush()


# ---------------------------------------------------------------------------
# Reader-side helpers (validator / workbench)
# ---------------------------------------------------------------------------

def split_on_separator(raw_stdout: bytes) -> tuple[bytes, bytes, bool]:
    """Split raw stdout bytes at the solution output separator.

    Returns:
        (logs_bytes, payload_bytes, separator_found).
        The separator itself is consumed and appears in neither part.
    """
    idx = raw_stdout.find(SOLUTION_OUTPUT_SEPARATOR)
    if idx == -1:
        return raw_stdout, b"", False
    return raw_stdout[:idx], raw_stdout[idx + len(SOLUTION_OUTPUT_SEPARATOR):], True


def extract_artifacts(
    raw_stdout: bytes,
    output_dir: str,
) -> tuple[bool, str | None]:
    """Extract solution artifacts from raw container stdout.

    Splits on the separator, writes stdout.log from the log portion,
    base64-decodes the payload into a zip, and extracts the zip contents
    (result.json, solve_info.json, etc.) into output_dir.

    Args:
        raw_stdout: Raw bytes captured from the container's stdout.
        output_dir: Directory to write extracted files into.

    Returns:
        (success, error_message). error_message is None on success.
    """
    os.makedirs(output_dir, exist_ok=True)

    logs_bytes, payload_b64, found = split_on_separator(raw_stdout)

    # Always write the log portion
    log_path = os.path.join(output_dir, "stdout.log")
    with open(log_path, "wb") as f:
        f.write(logs_bytes)

    if not found:
        return False, (
            "No solution output separator found in container stdout. "
            "The solver must print logs, then the separator line, then "
            "a base64-encoded zip of result.json and other artifacts."
        )

    payload_b64 = payload_b64.strip()
    if not payload_b64:
        return False, "Separator found but base64 payload is empty."

    try:
        payload_bytes = base64.b64decode(payload_b64, validate=True)
    except (binascii.Error, ValueError) as e:
        return False, f"Base64 decode of solution payload failed: {e}"

    if not payload_bytes:
        return False, "Decoded payload is empty."

    zip_path = os.path.join(output_dir, "solution_artifacts.zip")
    with open(zip_path, "wb") as f:
        f.write(payload_bytes)

    if not zipfile.is_zipfile(zip_path):
        return False, "Decoded payload is not a valid zip file."

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            dest = os.path.abspath(output_dir)
            for member in zf.infolist():
                member_path = os.path.normpath(os.path.join(dest, member.filename))
                if not (member_path == dest or member_path.startswith(dest + os.sep)):
                    return False, f"Zip member '{member.filename}' escapes output directory."
            zf.extractall(dest)
    except zipfile.BadZipFile as e:
        return False, f"Bad zip file: {e}"
    except Exception as e:
        return False, f"Zip extraction failed: {e}"

    return True, None
