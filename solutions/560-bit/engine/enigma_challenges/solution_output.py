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
  3. Base64-encoded zip containing the files named by RESULT_JSON_FILENAME etc. (see constants below)
  4. An optional end line (SOLUTION_OUTPUT_END_SEPARATOR). Anything after it is ignored.
     Miners that omit it are read the same way as before.

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
import re
import sys
import zipfile

# Magic separator line. Must match across solver, validator, and workbench.
SOLUTION_OUTPUT_SEPARATOR: bytes = (
    b"\n----- ENIGMA-SOLUTION-OUTPUT-BEGIN-a8c7f3e2-9d4b-4c5a-8f1e-2b6d3a4e5f7c -----\n"
)

# Optional terminator. When present, only the bytes between the begin and end
# lines are the payload. When absent, the reader uses the same base64 cleanup
# it used before this marker existed.
SOLUTION_OUTPUT_END_LINE: bytes = (
    b"----- ENIGMA-SOLUTION-OUTPUT-END-a8c7f3e2-9d4b-4c5a-8f1e-2b6d3a4e5f7c -----"
)
SOLUTION_OUTPUT_END_SEPARATOR: bytes = b"\n" + SOLUTION_OUTPUT_END_LINE + b"\n"

SOLUTION_LOG_FILENAME: str = "container.log"

# Filenames expected inside the solution artifacts zip (the solver contract).
RESULT_JSON_FILENAME: str = "result.json"
SOLVE_INFO_JSON_FILENAME: str = "solve_info.json"

# Filename of the zip that is base64-encoded and emitted after the separator.
SOLUTION_OUTPUT_ZIP_FILENAME: str = "solution_artifacts.zip"


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


def build_solution_result_zip(
    result: str | bytes,
    solve_info: str | bytes | None = None,
    extra_files: dict[str, str | bytes] | None = None,
) -> bytes:
    """Convenience wrapper around build_solution_zip for the common pattern.

    Most solvers emit at minimum:
      - result.json
      - solve_info.json (optional but common)

    Usage:

        zip_bytes = build_solution_result_zip(result_json, solve_info_json)
        # or with more:
        zip_bytes = build_solution_result_zip(result_json, extra_files={"output.txt": txt})

    This reduces duplication of the filename keys in solver code.
    """
    files: dict[str, str | bytes] = {RESULT_JSON_FILENAME: result}
    if solve_info is not None:
        files[SOLVE_INFO_JSON_FILENAME] = solve_info
    if extra_files:
        files.update(extra_files)
    return build_solution_zip(files)


def write_solution_output(zip_bytes: bytes) -> None:
    """Flush all output streams, then emit separator + base64-encoded zip to stdout.

    The end marker is written immediately after the base64 so shutdown warnings
    on stderr, which ``docker logs`` appends, stay outside the payload.
    """
    sys.stdout.flush()
    sys.stderr.flush()

    encoded = base64.b64encode(zip_bytes)
    buf = sys.stdout.buffer
    buf.write(SOLUTION_OUTPUT_SEPARATOR)
    buf.write(encoded)
    buf.write(SOLUTION_OUTPUT_END_SEPARATOR)
    buf.flush()


# ---------------------------------------------------------------------------
# Reader-side helpers (validator / workbench)
# ---------------------------------------------------------------------------

# How much trailing alphabet (stderr that happens to look like base64) we will
# drop while searching for a prefix that is still a zip. A Python warning path
# such as "/usr/local/..." contributes only a few dozen characters.
_BASE64_TAIL_SLACK = 512
_BASE64_LINE = re.compile(rb"[A-Za-z0-9+/=]+")


def clean_base64_payload(payload_b64: bytes | str) -> bytes | str:
    """Keep the base64 payload and drop text docker logs appended after it.

    ``docker logs`` has no stdout-only mode, so a shutdown warning on stderr
    can follow the payload. Whole base64 lines are kept. A later line that is
    not base64 ends the payload. If that line still starts with base64
    characters and nothing has been kept yet, only that leading run is kept.
    """
    as_text = isinstance(payload_b64, str)
    raw = payload_b64.encode("ascii", errors="ignore") if as_text else bytes(payload_b64)
    parts: list[bytes] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _BASE64_LINE.fullmatch(stripped)
        if match:
            parts.append(stripped)
            continue
        if not parts:
            prefix = _BASE64_LINE.match(stripped)
            if prefix:
                parts.append(prefix.group(0))
        break
    cleaned = b"".join(parts) if parts else raw.strip()
    if as_text:
        return cleaned.decode("ascii", errors="ignore")
    return cleaned


def decode_solution_payload(payload_b64: bytes | str) -> bytes | None:
    """Decode a solution payload, ignoring stderr glued on by ``docker logs``.

    When the end marker is present, only the bytes before it are decoded.
    When it is missing, the payload is cleaned the same way as before.

    Returns the longest decoded prefix that opens as a zip. When nothing is a
    zip, returns the longest valid base64 prefix so the caller can report an
    invalid zip. Returns None when no prefix decodes.
    """
    if isinstance(payload_b64, str):
        raw = payload_b64.encode("ascii", errors="ignore")
    else:
        raw = bytes(payload_b64)
    end_at = raw.find(SOLUTION_OUTPUT_END_LINE)
    if end_at != -1:
        payload_b64 = raw[:end_at]
    cleaned = clean_base64_payload(payload_b64)
    if isinstance(cleaned, str):
        cleaned = cleaned.encode("ascii", errors="ignore")
    cleaned = re.sub(rb"\s+", b"", cleaned)
    if len(cleaned) < 4:
        return None

    aligned = len(cleaned) - (len(cleaned) % 4)
    floor = max(4, aligned - _BASE64_TAIL_SLACK)
    first_decoded: bytes | None = None
    for size in range(aligned, floor - 1, -4):
        try:
            decoded = base64.b64decode(cleaned[:size], validate=True)
        except (binascii.Error, ValueError):
            continue
        if first_decoded is None:
            first_decoded = decoded
        if _opens_as_zip(decoded):
            return decoded
    return first_decoded


def _opens_as_zip(data: bytes) -> bool:
    if not data.startswith(b"PK"):
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return archive.testzip() is None and bool(archive.namelist())
    except zipfile.BadZipFile:
        return False


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

    Splits on the separator, writes SOLUTION_LOG_FILENAME from the log portion
    base64-decodes the payload into a zip, and extracts the zip contents
    ({RESULT_JSON_FILENAME}, {SOLVE_INFO_JSON_FILENAME}, etc.) into output_dir.

    Args:
        raw_stdout: Raw bytes captured from the container's stdout.
        output_dir: Directory to write extracted files into.

    Returns:
        (success, error_message). error_message is None on success.
    """
    os.makedirs(output_dir, exist_ok=True)

    logs_bytes, payload_b64, found = split_on_separator(raw_stdout)

    # Always write the log portion using SOLUTION_LOG_FILENAME.
    log_path = os.path.join(output_dir, SOLUTION_LOG_FILENAME)
    with open(log_path, "wb") as f:
        f.write(logs_bytes)

    if not found:
        return False, (
            f"No solution output separator found in container stdout. "
            f"The solver must print logs, then the separator line, then "
            f"a base64-encoded zip of {RESULT_JSON_FILENAME} and other artifacts."
        )

    payload_b64 = payload_b64.strip()
    if not payload_b64:
        return False, "Separator found but base64 payload is empty."

    payload_bytes = decode_solution_payload(payload_b64)
    if payload_bytes is None:
        return False, "Base64 decode of solution payload failed."

    if not payload_bytes:
        return False, "Decoded payload is empty."

    zip_path = os.path.join(output_dir, SOLUTION_OUTPUT_ZIP_FILENAME)
    with open(zip_path, "wb") as f:
        f.write(payload_bytes)

    if not zipfile.is_zipfile(zip_path):
        return False, "Decoded payload is not a valid zip file."

    def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
        mode = (info.external_attr >> 16) & 0o170000
        return mode == 0o120000

    dest = os.path.abspath(output_dir)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                member_path = os.path.normpath(os.path.join(dest, member.filename))
                if not (member_path == dest or member_path.startswith(dest + os.sep)):
                    return False, f"Zip member '{member.filename}' escapes output directory."
                if _is_zip_symlink(member):
                    return False, f"Zip member '{member.filename}' is a symlink (not allowed)."

            # Manual extraction (like _safe_extract_zip) instead of extractall.
            # This gives us control and consistent symlink/traversal guarantees.
            os.makedirs(dest, exist_ok=True)
            for member in zf.infolist():
                member_path = os.path.normpath(os.path.join(dest, member.filename))
                if member.is_dir():
                    os.makedirs(member_path, exist_ok=True)
                    continue
                parent = os.path.dirname(member_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with zf.open(member, "r") as src, open(member_path, "wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)

            # Post-extraction symlink sweep (belt + suspenders)
            for root, dirs, files in os.walk(dest):
                for name in list(dirs) + list(files):
                    full = os.path.join(root, name)
                    if os.path.islink(full):
                        try:
                            os.unlink(full)
                        except Exception:
                            pass
                        return False, "Symlink found in extracted archive (not allowed)."
    except zipfile.BadZipFile as e:
        return False, f"Bad zip file: {e}"
    except Exception as e:
        return False, f"Zip extraction failed: {e}"

    return True, None
