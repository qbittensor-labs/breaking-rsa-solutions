#!/usr/bin/env bash
# Build the PARALLEL 'characters' from the dev copy of the CADO tree
# (/tmp/rsa530/dev/chars/cadobuild = copy of /tmp/rsa530/cadobuild + cado-characters-parallel.patch).
# Same image / hostname / flags (x86-64-v3) as /tmp/rsa530/cadobuild/build_chars.sh -> incremental build.
# host:
#   docker run --rm --hostname eace9e82e499 -v /tmp/rsa530/dev/chars/cadobuild:/src \
#     nvidia/cuda:13.0.1-devel-ubuntu24.04 bash /src/build_chars_parallel.sh
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq cmake libgmp-dev zlib1g-dev python3 python3-flask python3-requests >/dev/null
cd /src/cado-nfs
nice -n 19 make -j4 characters 2>&1 | tail -25
B=build/$(hostname)
ls -la $B/linalg/characters
n=$(objdump -d $B/linalg/characters | grep -cE '%zmm|\{%k[0-7]\}' || true); echo "characters zmm/kmask insns: $n"
ldd $B/linalg/characters
