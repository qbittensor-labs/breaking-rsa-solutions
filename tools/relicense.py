#!/usr/bin/env python3
"""Classify and relicense Breaking RSA solver files.

Custom source files (participant / qBitTensor Labs) get a uniform AGPL-3.0 header;
vendored third-party code is left untouched; binaries are skipped. Dry-run by
default. See docs/RELICENSING.md for the policy this implements.

    python tools/relicense.py --src <extracted-solution> --report
    python tools/relicense.py --src <extracted-solution> --dst solutions/500-bit \
        --author "<verified author from the submission's own headers>" --apply

--author is required and must come from evidence (the submission's own copyright
headers, or the winning key's owner); use "an anonymous competition participant"
when the submission does not name its author. Never infer authorship from code
similarity to another milestone — winners are published for successors to build on.
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# --- classification ---------------------------------------------------------

SOURCE_EXT = {".py", ".cu", ".cuh", ".c", ".h", ".sh"}
HASH_STYLE = {".py", ".sh"}                       # '#' line comments (+ Dockerfile)
SLASH_STYLE = {".cu", ".cuh", ".c", ".h"}         # '//' line comments

# Anything under these path components is vendored third-party — never relicensed.
THIRD_PARTY_DIRS = {"vendor", "cub", "licenses", "third_party"}
# Known third-party file/binary names (basename match, case-insensitive).
THIRD_PARTY_NAMES = {
    "msieve", "msieve_la", "msieve_dump", "msv_launcher", "cado_sqrt",
    "dup1", "dup2", "freerel", "merge", "purge", "replay", "cado2msieve",
    "sort_engine.so",
}
THIRD_PARTY_SUFFIX = (".ptx", ".diff", ".patch")
# qBitTensor Labs' own contract package — relicense to AGPL like other custom code.
QBTL_PKG_DIRS = {"enigma_challenges"}

# Lines we DROP from a leading comment header when relicensing (license/copyright
# statements only). Descriptive comments are preserved.
LICENSE_LINE_PAT = re.compile(
    r"copyright|all rights reserved|written by xdev|the mit license|"
    r"permission is hereby|the above copyright|the software is provided|"
    r"without warranty|in no event|out of or in connection|dealings in the software|"
    r"furnished to do so|subject to the following|warranties of merchantability|"
    r"\bliability\b|noninfringement|free software foundation|"
    r"gnu (affero|general) public license", re.I)


def is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(4096)
    except OSError:
        return True


def classify(rel: Path) -> str:
    parts = set(rel.parts)
    if parts & THIRD_PARTY_DIRS:
        return "THIRD_PARTY"
    if rel.name.lower() in THIRD_PARTY_NAMES or rel.name.lower().endswith(THIRD_PARTY_SUFFIX):
        return "THIRD_PARTY"
    if parts & QBTL_PKG_DIRS:
        return "QBTL_PKG"
    if rel.name == "Dockerfile" or rel.suffix in SOURCE_EXT:
        return "CUSTOM"
    return "OTHER"


# --- header rendering -------------------------------------------------------

def agpl_header(author: str, comment: str) -> str:
    lines = [
        f"Copyright (C) 2026 qBitTensor Labs.",
        f"Original author: {author} (Enigma / Breaking RSA competition).",
        f"IP in custom components assigned to qBitTensor Labs under the Enigma rules.",
        f"",
        f"This program is free software: you can redistribute it and/or modify it",
        f"under the terms of the GNU Affero General Public License as published by",
        f"the Free Software Foundation, either version 3 of the License, or (at your",
        f"option) any later version.",
        f"",
        f"This program is distributed in the hope that it will be useful, but WITHOUT",
        f"ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS",
        f"FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more",
        f"details. You should have received a copy of the license with this program;",
        f"if not, see <https://www.gnu.org/licenses/>.",
    ]
    return "\n".join(f"{comment} {ln}".rstrip() for ln in lines) + "\n"


def comment_for(rel: Path) -> str:
    if rel.suffix in SLASH_STYLE:
        return "//"
    return "#"  # .py, .sh, Dockerfile


def _decomment(line: str) -> str:
    """Strip comment markers/dividers from a line to inspect its prose."""
    s = line.strip()
    for p in ("/*", "*/", "//", "*", "#"):
        if s.startswith(p):
            s = s[len(p):].strip()
    return s


def _is_content(line: str) -> bool:
    """True if the de-commented line carries prose (not just a ==== / **** rule)."""
    return bool(_decomment(line).strip("=*-_ ").strip())


def strip_old_header(text: str, comment: str) -> str:
    """Remove only license/copyright LINES from the leading comment header, keeping
    descriptive comments and the code. Handles C block comments and line-comment
    runs; preserves a shebang. If the header has no prose once license lines are
    gone, the whole comment is dropped."""
    shebang = ""
    if text.startswith("#!"):
        nl = text.find("\n") + 1
        shebang, text = text[:nl], text[nl:]
    lines = text.splitlines(keepends=True)
    n = len(lines)

    # Delimit the leading comment region.
    k = 0
    if text.lstrip().startswith("/*"):
        while k < n and lines[k].strip() == "":
            k += 1
        start = k
        while k < n and "*/" not in lines[k]:
            k += 1
        if k < n:
            k += 1                      # include the closing */ line
        pre, region, rest = lines[:start], lines[start:k], lines[k:]
    else:
        while k < n and (lines[k].strip() == "" or lines[k].lstrip().startswith(("//", "#"))):
            k += 1
        pre, region, rest = [], lines[:k], lines[k:]

    kept = [l for l in region if not LICENSE_LINE_PAT.search(_decomment(l))]
    if not any(_is_content(l) for l in kept):
        kept = []                        # header was license-only -> drop it whole
    return shebang + "".join(pre) + "".join(kept) + "".join(rest)


def relicense_text(path: Path, rel: Path, author: str) -> str:
    comment = comment_for(rel)
    text = path.read_text(errors="replace")
    body = strip_old_header(text, comment)
    shebang = ""
    if body.startswith("#!"):
        nl = body.find("\n") + 1
        shebang, body = body[:nl], body[nl:]
    return shebang + agpl_header(author, comment) + "\n" + body.lstrip("\n")


# --- driver -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="extracted solution tree")
    ap.add_argument("--dst", type=Path, help="destination tree (required with --apply)")
    ap.add_argument("--author", required=True,
                    help="original-author credit line — must be verified from the "
                         "submission itself; use 'an anonymous competition participant' "
                         "if the submission does not name its author")
    ap.add_argument("--apply", action="store_true", help="write relicensed copy")
    ap.add_argument("--report", action="store_true", help="print classification report")
    ap.add_argument("--exclude-binaries", action="store_true",
                    help="drop binaries (default: preserve them as submitted)")
    args = ap.parse_args()

    if args.apply and not args.dst:
        ap.error("--apply requires --dst")

    counts: dict[str, int] = {}
    for path in sorted(args.src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(args.src)
        kind = classify(rel)
        binary = is_binary(path)
        tag = "BINARY" if (binary and kind in ("CUSTOM", "QBTL_PKG", "OTHER")) else kind
        counts[tag] = counts.get(tag, 0) + 1
        if args.report:
            print(f"  {tag:12s} {rel}")

        if not args.apply:
            continue
        dst = args.dst / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if tag in ("CUSTOM", "QBTL_PKG"):
            # The enigma_challenges/ contract package is qBitTensor Labs' own code,
            # not the participant's — credit it accordingly.
            author = "qBitTensor Labs" if tag == "QBTL_PKG" else args.author
            dst.write_text(relicense_text(path, rel, author))
        elif tag == "BINARY" and args.exclude_binaries:
            continue  # dropped on request; default preserves the submission as-is
        else:
            shutil.copy2(path, dst)  # third-party / other / binaries (preserve mode)

    print("\nClassification summary ({}):".format(args.src))
    for k in sorted(counts):
        print(f"  {k:12s} {counts[k]}")
    if not args.apply:
        print("\n(dry run — no files written; add --apply --dst to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
