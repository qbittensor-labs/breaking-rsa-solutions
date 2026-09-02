# Relicensing policy and procedure

This repository normalizes the licensing of winning Breaking RSA submissions to
**AGPL-3.0** for the custom components, while leaving bundled third-party software
under its original license. This document states the policy, the automated
classification, and the verification performed before public release.

## Policy

1. **Custom components → AGPL-3.0.** Files authored by the participant (or by
   qBitTensor Labs) — the solver entrypoint, the GPU siever, the pipeline scripts,
   the filtering/linalg drivers — get a uniform AGPL-3.0 header with a
   `© 2026 qBitTensor Labs` copyright and an "Original author" credit line. This is
   authorized by the Enigma rules: winning submissions are AGPL-3.0 and IP in
   custom components is assigned to qBitTensor Labs.
2. **Third-party components → unchanged.** Vendored libraries (CADO-NFS, msieve,
   CUB, zlib, GMP-ECM, YAFU, APRCL, …) keep their own license. Their headers are
   **not** touched, and their license texts are preserved under
   `solutions/<milestone>/licenses/`.
3. **Patches to third-party code inherit the third-party license.** A `.diff` or a
   modified upstream source file is a derivative of that upstream and stays under
   its license, with the modification noted.
4. **Submitted binaries are preserved as submitted.** The submission is the
   artifact of record. Some submitters shipped prebuilt binaries (a custom
   `gpu_loop`, and vendored `msieve`/CADO tools) that the Docker build depends on;
   we keep exactly what each submission included so it builds and runs as judged. A
   binary cannot carry a text header, so it is not "relicensed": a **custom** binary
   is a build artifact of AGPL-licensed source shipped alongside it, and a
   **third-party** binary retains its upstream license (see `NOTICE.md`). Use
   `--exclude-binaries` only if a specific solution's binaries are redundant with a
   reproducible build.

## Automated classification (`tools/relicense.py`)

The tool walks a solution tree and labels each file:

- **CUSTOM** — a text source file (`.py .cu .cuh .c .h .sh`, `Dockerfile`) that is
  **not** under a vendored path and does not match a known third-party binary/name.
  Its existing header (`All rights reserved` / MIT / etc.) is replaced with the
  AGPL header for the extension's comment style.
- **THIRD_PARTY** — anything under `vendor/`, `cub/`, `licenses/`, a vendored
  `msieve/` tree, or matching known third-party names (`msieve`, `cado_*`, `dup1`,
  `dup2`, `freerel`, `merge`, `purge`, `replay`, `sort_engine.so`, `*.ptx`,
  `stage1_core*`, `*.diff`). Left untouched.
- **BINARY** — no text header; skipped (excluded from the source tree by default).
- **QBTL_PKG** — the `enigma_challenges/` contract package (qBitTensor Labs' own);
  normalized to the AGPL header.

Run it dry (default) to get a classification report; run with `--apply` to write a
relicensed copy into a destination tree:

```
python tools/relicense.py --src /path/to/extracted/500-bit --report          # dry run
python tools/relicense.py --src /path/to/extracted/500-bit \
    --dst solutions/500-bit --author "Xdev" --apply                          # write
```

`--author` is **required** and must be verified against the submission itself —
its own copyright/authorship headers, or the identity of the winning key's owner.
Do **not** infer authorship from code similarity to another milestone: Enigma
publishes winning solutions precisely so later competitors can build on them, so
shared code lineage does not imply shared authorship. When the submission does not
name its author, use `--author "an anonymous competition participant"` (the
340-bit and 460-bit solvers are credited this way; the 480/500 submissions carry
Xdev's own authorship headers).

## Verification record

The following was verified before public release (2026-09-02):

- **Authority to relicense.** The published
  [Competition Rules](https://www.qbittensorlabs.com/enigma/rules) (§5.1
  irrevocable IP assignment of all submissions; §5.3 AGPL-3.0 release of winning
  submissions; §5.5 participant warranty that every included component is theirs
  to assign or compatibly licensed) give qBitTensor Labs ownership of the custom
  components and the right to release them under AGPL-3.0 and to dual-license.
- **Custom/third-party split.** Every file was classified by `tools/relicense.py`
  and the highest-risk classifications were individually reviewed: `cado2msieve.c` is
  an original CADO→msieve bridge (correctly custom); `gpu_la.py` is a from-scratch
  port of msieve's public-domain `lanczos.c` (public domain permits relicensing
  the port). Third-party license texts are preserved under each solution's
  `licenses/` directory and listed in `NOTICE.md` (the 340-bit solution bundles no
  third-party code; its Dockerfile builds GMP-ECM and YAFU from upstream).
- **Attribution.** Every "Original author" credit is backed by the submission's
  own authorship headers, cross-checked against the public winning-key record
  (challenges.qbittensorlabs.com): the 480/500 milestones were won by the same key
  whose submissions carry Xdev's headers; the 340/460 milestones were won by
  distinct keys whose submissions name no author and are credited anonymously.
- **Hygiene.** All solver sources and bundled binaries were swept for secrets,
  credentials, endpoints, and embedded paths; clean.

The individual file reviews, the winning-key cross-check, and the secrets sweep
were performed by AI (Claude) under qBitTensor Labs' direction. Approved for
release by qBitTensor Labs, 2026-09-02.

**A note on best effort.** These solvers are user-submitted artifacts, preserved
as judged. Classifying every line of third-party heritage in submissions we did
not write is inherently best-effort; the participants' warranty (Rules §5.5)
covers what we cannot see. If you believe any file is misclassified,
misattributed, or includes your code without proper credit or license, contact
**support@qbittensorlabs.com** and we will correct or remove it.
