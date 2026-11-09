# CADO-NFS build configuration used for the binaries in this directory.
# Drop in as `local.sh` at the root of a cado-nfs checkout before `make -j`.
#
#   git clone https://gitlab.inria.fr/cado-nfs/cado-nfs.git && cd cado-nfs
#   cp .../cado-local.sh local.sh
#   <apply the SOURCE PATCH in (2a) below>
#   make -j$(nproc)          # cmake also requires python3-flask + python3-requests
#   objdump -d filter/purge | grep -cE '%zmm|\{%k[0-7]\}'    # MUST print 0
#
# ---------------------------------------------------------------------------
# 1. NEVER -march=native / AVX-512.
# Match the project convention (gpu_nfs_siever/build.sh): x86-64-v3 = AVX2+BMI2.
# CADO's DEFAULT build emits -mavx512f. The Dockerfile COPYs prebuilt binaries into
# an image with no compiler, so an AVX-512 binary SIGILLs (exit 132) on any validator
# CPU without it (e.g. AMD EPYC 7443 / Zen 3) and the run factors nothing, with no way
# to recover at runtime -- the identical trap build.sh documents for gpu_loop.
CFLAGS="-O2 -march=x86-64-v3"
CXXFLAGS="-O2 -march=x86-64-v3"

# ---------------------------------------------------------------------------
# 2. TWO UPSTREAM BUGS block -outfmt (compressed intermediates). Without compression
#    cado_filter.sh needs ~8.6 GB of the validator's 10 GB /tmp for the dup1 split
#    alone, and peaks at ~12.1 GB overall -> ENOSPC ~3.3h into a run.
#    See Pipeline.md §11.
#
# (a) utils/gzip.cpp -- is_supported_compression_format() compares char const*
#     POINTERS, so no user-supplied string can ever match; -outfmt is always rejected
#     with "Error, output compression format unsupported". PATCH REQUIRED:
#
#         -        if (r.suffix == s)
#         +        if (strcmp(r.suffix, s) == 0)
#
#     (<cstring> is already included.)
#
# (b) filter/dup1.cpp -- with -outfmt the output is opened through a shell pipe
#     ("gzip -c --fast > file") BEFORE -mkdir creates the directory, so slice 0 dies
#     with `sh: cannot create ...: Directory nonexistent` and rc=141.
#     NOT patched: cado_filter.sh works around it by pre-creating the slice dirs and
#     omitting -mkdir. Patching upstream instead is fine; the workaround is harmless.
#
# cado_filter.sh PROBES for (a) at runtime on a 100-line sample and falls back to
# uncompressed with a loud warning, so an unpatched CADO degrades rather than breaks
# -- but it will then need the full ~8.6 GB.
