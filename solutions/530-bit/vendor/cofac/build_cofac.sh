#!/usr/bin/env bash
# builds cofac_cpu (C) + facul_split.cpp (C++, CADO facul) inside nvidia/cuda:13.0.1-devel-ubuntu24.04
# host: docker run --rm -v <cado build tree root>:/src -v <this dir>:/dev_cofac nvidia/cuda:13.0.1-devel-ubuntu24.04 bash /dev_cofac/build_cofac.sh
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq libgmp-dev >/dev/null
C=/src/cado-nfs; B=$C/build/eace9e82e499; E=$B/sieve/ecm/CMakeFiles/facul.dir
# the facul objects come from the configured CADO tree (vendor/cado-local.sh as local.sh):
[ -e "$E/facul.cpp.o" ] || (cd $C && make facul testbench)
cd /dev_cofac
INC="-I$C/. -I$C/utils -I$C/utils/embedded -I$B -I$C/gf2x -I$B/gf2x -I$C/sieve/. -I$C/sieve/ecm/. -I$C/sieve/ecm/.."
g++ -std=c++20 -O2 -march=x86-64-v3 $INC -c facul_split.cpp -o facul_split.o
gcc -O3 -march=x86-64-v3 -DWITH_FACUL -Wall -Wno-misleading-indentation -c cofac_cpu.c -o cofac_cpu.o
g++ -O2 -march=x86-64-v3 -o cofac_cpu_facul cofac_cpu.o facul_split.o $E/*.o $B/utils/libutils.a -lgmp -lpthread -lm -fopenmp
ls -la cofac_cpu_facul    # shipped as cuda-sieve/cofac_cpu
echo "zmm/kmask insns: $(objdump -d cofac_cpu_facul | grep -cE '%zmm|\{%k[0-7]\}' || true)"
