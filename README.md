# Breaking RSA — Winning Solutions

Open-source archive of the winning solvers from the **Enigma / Breaking RSA**
challenge ([qbittensorlabs.com/enigma](https://www.qbittensorlabs.com/enigma/challenges/breaking-rsa)),
operated by qBitTensor Labs on Bittensor Subnet 63.

Breaking RSA posts a ladder of increasingly hard balanced RSA semiprimes and pays
a prize for the first solver that factors each, in an identical hardened sandbox
under a fixed wall-clock. Every winning submission is released under **AGPL-3.0**;
this repository collects them, normalizes their licensing, and preserves each
milestone so the next competitor — and the wider research community — can build on
what came before.

## The milestone ladder

| Milestone | Digits | Method | End-to-end | Author |
|---|---|---|---|---|
| ~320 (baseline) | c103 | msieve SIQS cascade (CPU) — operator example | — | qBitTensor Labs |
| 340-bit | c103 | multithreaded YAFU SIQS (CPU) | ~7 min | anonymous participant |
| 460-bit | c140 | GPU-GNFS v1 (custom GPU lattice sieve + CPU linalg) | ~3.5 h | anonymous participant |
| 480-bit | c145 | GPU-GNFS v2 (+ GPU block-Lanczos, parallel sqrt) | ~3 h 12 m | Xdev |
| 500-bit | c151 | GPU-GNFS v3 (fully hardened) | ~3.9 h | Xdev |

The GPU-GNFS solvers (460→480→500) form a single evolving codebase: a from-scratch
CUDA lattice siever (`gpu_loop`) that moves NFS relation collection — the dominant
cost — onto the GPU, with the downstream (filtering, linear algebra, square root)
progressively GPU-accelerated as the modulus grew. Successive milestones building
on earlier winners' published code is by design — Enigma open-sources each winning
solution so the next competitor starts from it — so shared lineage does not imply
shared authorship: the 460-bit and 480-bit milestones were won by different keys.

## Layout

```
solutions/
  340-bit/    # YAFU SIQS solver (anonymous participant)
  460-bit/    # GPU-GNFS v1 (anonymous participant)
  480-bit/    # GPU-GNFS v2 (Xdev)
  500-bit/    # GPU-GNFS v3 (Xdev)  — standing frontier
tools/
  relicense.py   # classifies custom vs vendored files; normalizes headers to AGPL-3.0
docs/
  RELICENSING.md # licensing policy, custom/third-party classification, review checklist
LICENSE          # GNU AGPL-3.0 (full text)
NOTICE.md        # attribution + bundled third-party components and their licenses
```

Each `solutions/<milestone>/` contains the solver source. Prebuilt binaries and
vendored third-party libraries (msieve, CADO-NFS, CUB, …) are handled per
`docs/RELICENSING.md`: third-party code retains its own license and is **not**
relicensed.

## Licensing

Custom components are © qBitTensor Labs and released under **AGPL-3.0** (see
[`LICENSE`](LICENSE)). Under the Enigma rules, winning submissions are AGPL-3.0 and
the IP in custom components is assigned to qBitTensor Labs; bundled third-party
libraries retain their original licenses, reproduced under each solution's
`third_party/` and summarized in [`NOTICE.md`](NOTICE.md).

### Commercial licensing

**The AGPL-3.0 is a strong copyleft license, and it has real obligations.** You are
free to use, study, modify, and run this software — including commercially — *but*
if you convey the software or **make it available to users over a network** (the
"Affero" clause closes the SaaS loophole), you must release the **complete
corresponding source code of your entire application**, including your
modifications, under the AGPL-3.0 to those users.

#### ⚠️ Know the risk before you build on AGPL code

AGPL-3.0 is one of the most aggressive open-source licenses in existence, and
underestimating it is a costly mistake. If you incorporate this code into a
product, the copyleft can reach **your entire application**:

- **It can force you to open-source your own proprietary code.** Combine AGPL code
  with your product and distribute it — or merely run it as a **backend for a
  website, API, or SaaS** — and you can be obligated to publish *all* of your
  application's source, including code you intended to keep secret. There is no
  "internal use / we never shipped a binary" escape hatch; the network clause is
  the point of AGPL.
- **Many organizations ban AGPL outright.** It routinely **fails legal and security
  review, blocks partnerships, and derails acquisition and fundraising due
  diligence** — an AGPL dependency discovered late can tank a deal or force an
  expensive rip-and-replace.
- **Non-compliance is copyright infringement.** Getting the boundary wrong exposes
  you to injunctions and to being compelled to either **disclose your source or
  remove the software** — after you've already built on it.

In short: if you are building anything you intend to keep closed, sell, embed, or
offer as a service, **AGPL is a serious legal risk you should not take on without
advice.**

**A commercial license removes that risk entirely.** qBitTensor Labs holds the
copyright to the custom components (assigned under the Enigma rules), so we can
license the same code to you under proprietary-friendly terms that **lift the
copyleft and the network-source-disclosure requirements completely**. Under a
commercial license you can **embed, modify, and ship these components in
closed-source and SaaS products with no obligation to release your source** — the
safe, worry-free path for commercial use. This is the standard dual-licensing
model: **open source under AGPL-3.0, or a commercial license from us — your
choice.**

To make commercial use safe and remove all copyleft obligations on the components
we own, reach out to **support@qbittensorlabs.com**.

> **Scope.** A commercial license from qBitTensor Labs covers the **custom
> components** we own (the solver entrypoints, the GPU siever, the pipeline). It
> does **not** cover the **bundled third-party libraries** (msieve, CADO-NFS, CUB,
> GMP/GMP-ECM, YAFU, zlib, APRCL, …), which remain under their own licenses (see
> [`NOTICE.md`](NOTICE.md)); you are responsible for complying with those
> separately. Most are permissive (BSD) or LGPL and impose light obligations, but
> you or your counsel should confirm for your specific use. Nothing here is legal
> advice; the governing terms are the [`LICENSE`](LICENSE) text and any commercial
> agreement you sign with us.

## Reproducing a factorization

Each solution builds to a single Docker image and factors an arbitrary balanced
semiprime of its target size on one high-end GPU + CPU node. Per-solution build
and run instructions live in each `solutions/<milestone>/README.md`.

## Credit

The 480-bit and 500-bit GPU-GNFS solvers were built by the competition participant
**Xdev** (per the authorship headers in those submissions). The 340-bit and 460-bit
solvers were submitted by anonymous participants. The ~320-bit baseline is
qBitTensor Labs' reference example. See [`NOTICE.md`](NOTICE.md).
