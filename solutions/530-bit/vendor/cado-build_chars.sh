#!/usr/bin/env bash
# Build CADO 'characters' and 'sqrt' from the patched tree (same flags/tree as cadobuild/build.sh).
# --hostname eace9e82e499 reuses the existing configured build dir build/eace9e82e499
# (CADO names its build dir after `hostname`), so this is an incremental build, not a reconfigure.
# Runs inside the docker container; the host invocation is at the bottom (commented).
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq cmake libgmp-dev zlib1g-dev python3 python3-flask python3-requests >/dev/null
cd /src/cado-nfs
nice -n 19 make -j2 characters sqrt 2>&1 | tail -15
B=build/$(hostname)
ls -la $B/linalg/characters $B/sqrt/sqrt
# AVX-512 guard (validator CPUs may lack it -> SIGILL)
for b in $B/linalg/characters $B/sqrt/sqrt; do
  n=$(objdump -d $b | grep -cE '%zmm|\{%k[0-7]\}' || true); echo "$b zmm/kmask insns: $n"
done
# host:
#   docker run --rm --hostname eace9e82e499 -v /tmp/rsa530/cadobuild:/src \
#     nvidia/cuda:13.0.1-devel-ubuntu24.04 bash /src/build_chars.sh
