#!/usr/bin/env bash
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

# build.sh -- compile gpu_loop for sm_120.
#
# FLAGS below is the production set. Every entry is measured; anything not listed is either a
# default or a measured regression kept behind its own flag so the result stays reproducible.
# NO GPU is needed to build. Do not restore -march=native: the previous binary was built on an
# AVX-512 machine and SIGILLs on CPUs without it, which is an unrecoverable deployment failure.
#
# ANY KERNEL CHANGE MUST REPRODUCE THE REFERENCE RELATION md5s AT BOTH KEY SIZES BEFORE IT SHIPS.
# Sorted-set md5 over a q-window -- never compare relation COUNTS, a reordered or double-counted
# scatter preserves the count. The smaller key size is the one that matters: these flags land on
# the live solver.
set -e
# MEASURED NEGATIVES -- do not re-try these:
#   LIM 35M / 50M / 60M      40M is the interior optimum.
#   J 10240 / 12288          8192 wins; larger regions lose more to setup than they gain.
#   lpb != 29/29             the axis is flat on sieve; the MATRIX decides, and it gets worse.
#   3 large primes           +14 % relations on paper, ~3 % net once the cofactoring is priced.
#   rational special-q       -46 %.
#   PRIM_SMEM                +0.4 %; scat_lat_coop is register-bound, shared costs it warps.
#   COF_LB                   +7.1 %; cofactor_ff's stalls are intrinsic register pressure.
#   CBITS_SHIFT 4            the coarse array stops being cache-resident.
#   TILE_QUEUE               +18.9 %; the static line mapping was the coalescing guarantee.
#   LAT_QUEUE                0 %; a dynamic 32-line block queue for scat_lat_coop. GRID gives
#                            192,512 warps against only ~75,035 blocks of work, so every warp
#                            already gets <=1 block and there is nothing to steal. Relation-
#                            identical, kept behind its flag. FINDINGS-20260916 §2d.
#   LAT_PAIR                 +1 %; pairing heavy front blocks with light back ones. The
#                            imbalance is INTRA-block, not a gradient across blocks. §2d.
#   SKEWRED_BAND             suppressing the q-lattice basis swap in [1.0,2.0] removes the
#                            scatter step but pays for it in DUPLICATES: unique yield goes
#                            negative by q/skew ~ 3.9 under production dedup. §5d.
FLAGS="${FLAGS:--DSPB=211 -DCOF_SC -DFB_SORT -DTSPLIT=393216 -DFAST_FINAL -DFUSED_SCAN -DFLAT_COOP2 -DSCAT_COOP_LEAN -DLPB_VAL=536870912ULL -DLIM=40000000u -DCFG_MFB1=58 -DCFG_MFB0=58 -DMAXSURV_VAL=600000 -DREDUCE_CAP=32 -DRCOLS=32 -DTILE_COL -DTILE_RESIEVE -DPRIM_SCAN -DPRIM_SCATTER -DPRIM_LDG -DCBITS -DCBITS_SHIFT=5 -DRS_BAND -DTILE_PREFETCH -DTILE_COOP -DTILE_COOP_M=256u -DBA_VEC -DJ=8192 -DTCOLS=4}"
# SPB=211: interior optimum -- 151/211/311 all yield the same rel/lat, 211 is fastest.
# RS_BAND: band-major resieve_colr with the coarse survivor mask staged in SHARED memory.
#   Measured 2026-09-17, relation-IDENTICAL at c151 and c155 (sorted-set md5, also at GRIDMUL
#   1/64/128): resieve 1.50 -> 1.08 ms/lat (-28%), sieve wall -9.2% at BOTH key sizes and flat
#   across the whole c155 production q-range (27.1 / 38.0 / 49.1 M). See gpu_loop.cu for the
#   mechanism; CBITS2 was the obvious alternative and measured worse at every shift.
# COF_SC: short-circuits the two cofactor sides. Small, because cofactor_ff is latency-bound.
# TILE_COOP_M=256: warp-per-line for the heavy prefix of the tiled scatter. See gpu_loop.cu.
GMPINC=""
for _d in /usr/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu; do
  [ -f "$_d/gmp.h" ] && { [ "$_d" = /usr/include ] || GMPINC="-I$_d"; break; }
done
# glibc >= 2.41 declares the C23 reciprocal-sqrt (rsqrt/rsqrtf/rsqrtl) in <bits/mathcalls.h> with
# noexcept, while CUDA 13.0's crt/math_functions.h declares them WITHOUT it -- two errors, and the
# compile never starts. The toolkit is pinned at 13.0 (see install-env.sh: 13.1+ needs a driver
# newer than the validators' 580.x), so the fix goes on the glibc side: compat-include/ shadows
# bits/mathcalls.h with a byte-identical copy minus that one declaration. -D cannot do this --
# libc-header-start.h #undefs the guard macro and re-derives it from __USE_GNU, which g++ always
# sets. Probe rather than version-sniff, so this is a no-op on a box whose headers agree.
_probe_cc(){ echo '#include <math.h>
int main(){return 0;}' | nvcc -arch=sm_120 ${1:-} -x cu -o /dev/null - >/dev/null 2>&1; }
COMPATINC=""
if ! _probe_cc; then
  # GENERATE the shim from THIS box's own <bits/mathcalls.h> rather than using the bundled copy:
  # the bundled one is a snapshot of the header on the machine that shipped it, and grafting it
  # onto a different glibc would be wrong. Bundled copy stays as a fallback only.
  _HERE=$(cd "$(dirname "$0")" && pwd)
  _SYS=""
  for _d in /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include; do
    [ -f "$_d/bits/mathcalls.h" ] && { _SYS="$_d/bits/mathcalls.h"; break; }
  done
  if [ -n "$_SYS" ]; then
    _GEN=$(mktemp -d); mkdir -p "$_GEN/bits"
    # drop ONLY the C23 reciprocal-sqrt declaration; everything else is byte-identical
    sed 's|^__MATHCALL_VEC (rsqrt,, (_Mdouble_ __x));$|/* rsqrt suppressed: CUDA 13.0 declares it without noexcept (build.sh) */|' \
        "$_SYS" > "$_GEN/bits/mathcalls.h"
    if _probe_cc "-I$_GEN"; then
      COMPATINC="-I$_GEN"
      trap 'rm -rf "$_GEN"' EXIT
      echo "note: glibc/CUDA rsqrt clash -- generated shim from $_SYS"
    else
      rm -rf "$_GEN"
    fi
  fi
  if [ -z "$COMPATINC" ] && _probe_cc "-I$_HERE/compat-include"; then
    COMPATINC="-I$_HERE/compat-include"
    echo "note: glibc/CUDA rsqrt clash -- using bundled compat-include (generation failed)"
  fi
  if [ -z "$COMPATINC" ]; then
    echo "FATAL: <math.h> will not compile under nvcc and no rsqrt shim worked." >&2
    echo "       See compat-include/README (glibc >= 2.41 vs CUDA 13.0 C23 rsqrt clash)." >&2
    exit 1
  fi
fi
nvcc -O3 -use_fast_math -arch=sm_120 $COMPATINC $GMPINC -Xcompiler -fopenmp,-march=x86-64-v3 $FLAGS ${EXTRA_FLAGS:-} -o ${OUT:-gpu_loop} gpu_loop.cu -lgmp
echo "built ${OUT:-gpu_loop} (FLAGS=$FLAGS ${EXTRA_FLAGS:-})"
