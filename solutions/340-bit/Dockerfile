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

FROM python:3.12-slim

# Build deps for GMP-ECM and YAFU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc make git \
    libgmp-dev zlib1g-dev \
    autoconf automake libtool m4 \
    && rm -rf /var/lib/apt/lists/*

# GMP-ECM from source (provides ecm.h + libecm, required by YAFU's SIQS build)
RUN git clone --depth 1 https://gitlab.inria.fr/zimmerma/ecm.git /tmp/ecm-src \
    && cd /tmp/ecm-src \
    && autoreconf -i && ./configure && make -j"$(nproc)" && make install && ldconfig \
    && rm -rf /tmp/ecm-src

# YAFU (bundles ytools/ysieve/msieve/aprcl). Multithreaded SIQS is the engine:
# it factors a 103-bit-half semiprime in ~7 min on ~24 cores and peaks at
# ~150 MB scratch — fitting the validator's 256 MB noexec tmpfs with margin.
RUN git clone --depth 1 https://github.com/bbuhrow/yafu.git /opt/yafu \
    && cd /opt/yafu \
    && make -f Makefile.gcc yafu USE_SSE41=1 ECM=1 \
    && test -x /opt/yafu/yafu

# Python dep for the solver wrapper (manylinux wheel bundles GMP/MPFR/MPC)
RUN pip install --no-cache-dir gmpy2

# The validator runs containers as a NON-ROOT account named "miner" (docker run
# --user miner). That account must exist in the image or the container won't start.
RUN useradd -m -u 1000 -s /usr/sbin/nologin miner

WORKDIR /app
COPY enigma_challenges /app/enigma_challenges/
COPY breaking_rsa.py /app/

# Root filesystem is mounted read-only at runtime; the only writable path is the
# 256 MB tmpfs at /tmp. Point HOME/TMPDIR/scratch there. YAFU is a single binary
# on the (exec-allowed) read-only root, so the noexec tmpfs is fine for its data.
ENV YAFU_BIN=/opt/yafu/yafu \
    HOME=/tmp \
    TMPDIR=/tmp \
    SIQS_WORKDIR=/tmp \
    PYTHONUNBUFFERED=1

USER miner
ENTRYPOINT ["python", "/app/breaking_rsa.py"]
