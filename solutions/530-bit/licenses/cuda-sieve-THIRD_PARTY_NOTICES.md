# Third-party notices

Except for the material identified below, this repository is made available
under CC0 1.0 Universal as stated in [`LICENSE`](LICENSE).

## CADO-NFS

The following files contain or are derived from CADO-NFS material:

- `bench/plattice.cuh`
- `bench/prp.cuh`
- `oracle/cado-after-sieve-survdump.patch`
- `oracle/las-threads-work-data.cpp.orig`

CADO-NFS is Copyright the CADO-NFS contributors and is distributed under the
GNU Lesser General Public License, version 2.1 or (at your option) any later
version. The license text is in
[`LICENSES/LGPL-2.1-or-later.txt`](LICENSES/LGPL-2.1-or-later.txt).

Upstream project: <https://gitlab.inria.fr/cado-nfs/cado-nfs>

The oracle material records CADO-NFS revision `0574bc39d`. The CUDA p-lattice
implementation is a port of `sieve/las-plattice.hpp` and
`sieve/las-reduce-plattice-simplistic.hpp`; the probable-prime classification
follows `sieve/las-cofactor.cpp`.

