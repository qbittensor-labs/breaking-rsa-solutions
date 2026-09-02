# NOTICE — Attribution and Third-Party Components

## Custom components

The custom solver components in this repository are © 2026 qBitTensor Labs and are
licensed under the GNU Affero General Public License v3.0 (see `LICENSE`).

Under the published [Enigma Official Competition Rules](https://www.qbittensorlabs.com/enigma/rules)
(§5.1 assignment of submissions; §5.3 open-source release), winning submissions are
released under AGPL-3.0 and the intellectual property in custom components is
assigned to qBitTensor Labs. Original authorship is credited as follows:

| Milestone | Original author | Winning key (public record) |
|---|---|---|
| 340-bit | an anonymous competition participant | `5Hh49arazTCdw7kgenMDjRyLcnceFLjNzpnmZYoGg934dPdj` |
| 460-bit | an anonymous competition participant | `5GKRWqttTzSaDZMUKsqVVVajX6xXu7sLQ1Uxt3dFrUC8GizH` |
| 480-bit, 500-bit | **Xdev** (competition handle) | `5DqYrRh7LP5XscZMEHcA1GSuXqwDjVCLZUNspdR8gFa6oMpj` |
| ~320-bit baseline / example | qBitTensor Labs | — |

> Winning keys are from the public competition record
> (challenges.qbittensorlabs.com). The 480-bit and 500-bit milestones were won by
> the same key, whose submissions carry Xdev's own authorship header
> ("Written by Xdev"). The 460-bit milestone was won by a different key and its
> submission does not name its author, so it is credited anonymously unless the
> author comes forward.

We credit these authors for their work. Relicensing to AGPL-3.0 reflects the
participation terms; it does not diminish original authorship.

## Bundled third-party components

The solvers bundle and/or build on the following third-party software, which
retains its **own** license. These components are **not** relicensed by this
repository; their license texts are preserved under each solution's `licenses/`
directory.

The **340-bit** solution bundles no third-party code in this repository: its
Dockerfile fetches and builds GMP-ECM and YAFU from their upstream sources at
image-build time, so those projects' licenses accompany their own distributions.

| Component | Role in the solvers | License |
|---|---|---|
| **CADO-NFS** | polynomial selection, filtering, square root, reference siever | LGPL-2.1+ |
| **msieve** | GPU polynomial selection, filtering, block-Lanczos, square root | public domain / as stated in its distribution |
| **NVIDIA CUB** | GPU sort/scan primitives (`sort_engine`) | BSD-3-Clause |
| **GMP / gmpy2** | multiprecision arithmetic | LGPL-3.0 / LGPL |
| **GMP-ECM** | elliptic-curve method (early milestones) | GPL / LGPL as stated |
| **YAFU** | self-initializing quadratic sieve (340-bit) | as stated in its distribution |
| **zlib** | compression | zlib license |
| **APRCL** (via msieve) | primality proving | as stated in its distribution |
| **CuPy** | GPU linear-algebra kernels (NVRTC JIT) | MIT |

> Some components are linked or patched rather than copied verbatim; patches to a
> third-party project inherit that project's license.

## Corrections

These solvers are user-submitted artifacts, preserved as judged. Classification
and attribution are done on a best-effort basis (see the verification record in
`docs/RELICENSING.md`). If you believe any file is misclassified, misattributed,
or includes your code without proper credit or license, contact
**support@qbittensorlabs.com** and we will correct or remove it.
