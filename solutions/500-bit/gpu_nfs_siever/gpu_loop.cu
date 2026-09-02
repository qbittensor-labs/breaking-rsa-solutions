// Copyright (C) 2026 qBitTensor Labs.
// Original author: Xdev (Enigma / Breaking RSA competition).
// IP in custom components assigned to qBitTensor Labs under the Enigma rules.
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
// details. You should have received a copy of the license with this program;
// if not, see <https://www.gnu.org/licenses/>.

//
// gpu_loop.cu -- custom CUDA lattice siever for c151 GNFS, targeting RTX PRO 6000
// Blackwell (sm_120). Factor base is built ONCE (root-finding); then per special-q the GPU runs
// project -> scatter -> scan -> resieve -> cofactor while the CPU (OpenMP, <=24 threads) finalizes
// the previous lattice's relations, fully overlapped. Relations stream into a RAM buffer (no disk).
// Build: ./build.sh (nvcc -O3 -arch=sm_120, -DFAST_FINAL -DFUSED_SCAN). Args:
//   gpu_loop <poly.cado> <qmin> <qmax> [dump_file] [rel_target]
#include "rootfind.c"
#include <math.h>
#include <vector>
#include <algorithm>
#include <gmp.h>
#include <sys/time.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <omp.h>
#include <cuda_runtime.h>
#include "u256.cuh"
#include <cub/cub.cuh>
#define MAXP 24
// Small-prime resieve skip: primes < SPB are NOT resieved (their per-cell walk dominates the
// resieve pass: sum(1/p) for p<256 is ~half of all factor-base hits). Instead the cofactor kernel
// and the CPU finalizer trial-divide each survivor's norm by these primes directly (105k survivors
// x ~54 primes is far cheaper than re-walking ~half the sieve array). Relations are IDENTICAL:
// trial division is the ground truth that resieve merely optimizes.  SPB=0 disables (baseline).
#ifndef SPB
#define SPB 32     /* in-source default; build.sh sets the production value (-DSPB) */
#endif
// Scatter optimization: the smallest factor-base primes
// (2,3,5,...) hit a huge fraction of sieve cells -> they dominate scat_col's atomicAdd traffic. Skip
// primes < SCAT_SKIP in scat_col (they're already trial-divided in cofactor via SPB and skipped in
// resieve), and raise the survivor threshold by SCATOFF to compensate for their un-scattered log mass.
// SCAT_SKIP=0 / SCATOFF=0 -> original behavior (no skip, exact). Pair SCAT_SKIP with SPB.
#ifndef SCAT_SKIP
#define SCAT_SKIP 32   /* in-source default: skip primes <32 in scatter (matches SPB); ~36% faster scatter */
#endif
#ifndef SCATOFF
#define SCATOFF 10     /* in-source default: survivor-threshold compensation for un-scattered small primes */
#endif
/* !! The RAT_* defaults below USED TO LIVE INSIDE the `#ifndef SCATOFF` guard above (2026-08-01).
   That made SCATOFF IMPOSSIBLE TO OVERRIDE from the command line: passing -DSCATOFF=<n> skipped the
   whole block, taking RAT_XSLACK's default with it, and every build died with
   `identifier "RAT_XSLACK" is undefined` at lines 910/1733. Found while trying to sweep
   SCAT_SKIP x SCATOFF -- all nine variants failed to compile. The guards are now separate, so each
   macro can be set independently. Behaviour is UNCHANGED when nothing is overridden: the same
   defaults (SCATOFF=10, RAT_XSLACK=0) are still defined exactly once. */
/* RAT_NONE: remove the rational side from the SIEVE entirely (both groups 0 and 1), do not clear or
   read dLr, and drop the rational gate from the scan -- candidate selection becomes purely algebraic.
   Justified by the §3.3 probe: at +32 both survivor curves converge on the algebraic-only ceiling
   (333,444 vs 331,298), so the rational array is no longer selecting anything. Implies RAT_NOLAT.
   Retention should be >= the +32 case (no rational gate -> no 2-large-prime losses).
   NOTE: cofactor_ff is deliberately left UNCHANGED here, so this measures the sieve/scan/resieve
   saving in isolation. The real design would also drop the rational norm+classify from the cofactor,
   which is additional saving not captured by this flag. */
#if defined(RAT_NONE) && !defined(RAT_NOLAT)
#define RAT_NOLAT 1
#endif
#ifndef RAT_XSLACK
#define RAT_XSLACK 0   /* extra bits of RATIONAL-gate relaxation; pairs with RAT_NOLAT (see Pipeline.md
                          §3). With the rational large primes unsieved, lr[] is short by their logs, so
                          the gate must be opened by ~that much to keep true relations. RAT_XSLACK=32
                          makes it effectively vacuous -> 100% retention, algebraic gate alone selects.
                          RAISE MAXSURV_VAL with it (~333k survivors/lat at 32) or the §1.4 hang fires. */
#endif
// ---- ASYMMETRIC large-prime bounds (added 2026-07-19) ----------------------
// The two sides have very different norm sizes, so a single shared lpb is a
// compromise. Measured with -DPROBE_3LP: the RATIONAL side's residual overflows
// its bound ~2.2x more often than the algebraic side (+5.9% vs +2.7% of survivors),
// which is the signature of a mistuned split. mfb was already per-side
// (CFG_MFB0/CFG_MFB1); lpb was not. LPB0_VAL = side 0 = rational (Nr/MFB0/rf),
// LPB1_VAL = side 1 = algebraic (Na/MFB1/af), matching the CADO side convention.
// Both default to LPB_VAL, so an unmodified build is bit-identical to before.
#ifndef LPB_VAL
#define LPB_VAL 1073741824ULL    /* in-source default; build.sh sets the production value */
#endif
// SIEVE_ADD: the sieve-array accumulate. Normally an atomicAdd. With
// -DPROBE_NOATOMIC it becomes a plain racy write -- RESULTS ARE WRONG, timing only --
// to measure what fraction of scatter is atomic serialization vs the Gauss reduction.
#ifdef PROBE_NOATOMIC
#define SIEVE_ADD(p,v) do{ *(p) += (v); }while(0)
#elif defined(PROBE_COALESCED)
// RESULTS ARE WRONG -- timing only. Keeps the atomic and the full enumeration, but sends every
// update to ONE address per warp instead of 32 scattered ones. PROBE_NOATOMIC removes atomicity
// while KEEPING the scattered addresses, so it measures atomic overhead only; this measures the
// cost of the uncoalesced access pattern itself (32 lanes -> 32 cache-line transactions per
// instruction). The gap between the two is what a bucket/tiled scatter could actually recover.
// *** THIS PROBE IS FLAWED -- superseded by PROBE_COALESCED3. It masks every address into ONE
// 64-byte window, so all 32 lanes contend for the SAME word: it measures atomic CONTENTION, not
// coalescing. Its "no gain" reading (5.12 vs 5.00) is NOT evidence that the access pattern is free.
// Kept only so the mistake is not repeated. Use PROBE_COALESCED3. ***
#define SIEVE_ADD(p,v) atomicAdd((unsigned*)((char*)g_probe_sink+(((unsigned long long)(p))&0xFFC0ull)),(v))
__device__ unsigned long long g_probe_sink_buf[8192];
#define g_probe_sink (g_probe_sink_buf)
#elif defined(PROBE_COALESCED2)
// Corrected coalescing probe. PROBE_COALESCED masked every address into ONE 64-byte window, so all
// 32 lanes contended for the same word -- that measures atomic CONTENTION, not coalescing, and its
// "no gain" result was therefore not evidence about the access pattern. Here each lane writes its
// OWN word, lane-consecutive (32 lanes -> 128 contiguous bytes = 2 cache lines, zero same-word
// contention), cycling over a 256KB L2-resident buffer. RESULTS WRONG -- timing only.
__device__ unsigned g_probe_sink2[65536];
#define SIEVE_ADD(p,v) atomicAdd(&g_probe_sink2[((((unsigned)(unsigned long long)(p))>>6)<<5 | (threadIdx.x&31)) & 65535u],(v))
#elif defined(PROBE_COALESCED3)
// As PROBE_COALESCED2 but targeting the REAL 16MB sieve array, so the working set is identical to
// production and only the ACCESS PATTERN differs. Isolates coalescing from any L2/L1 residency
// effect that the small 256KB sink in COALESCED2 could have contributed. RESULTS WRONG -- timing only.
// *** MEASURED 2026-07-20 (this is the trustworthy coalescing number):
//     base (scattered atomics, 16MB)     scatter 5.01
//     COALESCED3 (coalesced, 16MB)       scatter 4.20   <- coalescing is worth 0.81 ms
//     COALESCED2 (coalesced, 256KB)      scatter 3.41   <- extra 0.79 ms is L1/L2 residency, NOT
//                                                          access pattern; do not credit it
//     PROBE_NOATOMIC (scattered, no atomic) 4.19        <- ~= COALESCED3, so atomicity is nearly
//                                                          free once transactions coalesce
// => a PERFECT zero-overhead bucket/tiled scatter caps at 8.15 -> 7.34 ms/lat = 1.11x. Real
// bucketing must write and re-read ~37M updates/lattice, which is why -DBUCKET_SIEVE measured
// +24%. Worth retrying ONLY if the bucketing overhead can be held under ~0.8 ms/lattice. ***
__device__ unsigned g_probe_sink3[4194304];   // 16MB -- same working set as the real sieve array
#define SIEVE_ADD(p,v) atomicAdd(&g_probe_sink3[((((((unsigned)(unsigned long long)(p))>>6)<<5)|(threadIdx.x&31))&4194303u)],(v))
#elif defined(RACY_SCATTER)
// EXPERIMENT (2026-07-20): drop atomicity on the REAL scattered addresses, everything else intact.
// Unlike PROBE_NOATOMIC (which also strips the write in the box walk -> "timing only"), this keeps
// the full production path and only removes the atomic. On an ADD-ONLY sieve a lost RMW can only
// make a cell sum TOO LOW -> a missed survivor (false negative), NEVER too high -> the relations it
// DOES emit are still cofactor-verified and valid. So this cannot corrupt the matrix; the only cost
// is yield. Measures: speed gain (expect ~0.8ms scatter) vs yield loss from races.
#define SIEVE_ADD(p,v) do{ *(p) += (v); }while(0)
#else
#define SIEVE_ADD(p,v) atomicAdd((p),(v))
#endif
// ---- L2 residency control (2026-07-25) -------------------------------------------------------
// Measured decomposition of scat_lat (FB_BANDS + PROBE_NOENUM/PROBE_NOATOMIC): for the SPARSE
// large primes (band R3, p>2^20, 2.92M lines/side) the box walk's *writes* cost 0.59 ms/side while
// the enumeration around them is ~0.02 -- i.e. a scattered byte-add there is ~4x dearer than the
// same add from band R2. The suspected cause is L2 capacity, not transaction count: the per-lattice
// factor-base traffic (project 78 MB + scat_lat reads/BA 147 MB + resieve_coop 164 MB ~ 390 MB)
// streams through a 128 MB L2 and evicts the two 33.5 MB sieve arrays that every scatter hits.
//
// L2_STREAM: tag the factor-base streams (M/R/L/P/BA -- read once per lattice, never reused) with
// the `.cs` (cache-streaming / evict-first) policy so they cannot displace the sieve arrays.
// L2_PERSIST: additionally pin the sieve arrays themselves via an access-policy window
// (this device: l2=128 MB, persistingL2CacheMaxSize=80 MB, window max 128 MB; the two arrays are
// 67 MB, so they fit). Both are pure cache HINTS -- no change to any value read or written, so
// relations are byte-identical by construction.
#ifdef L2_STREAM
#define FB_LD(p)      __ldcs(p)
#define FB_ST(p,v)    __stcs((p),(v))
#else
#define FB_LD(p)      (*(p))
#define FB_ST(p,v)    (*(p)=(v))
#endif
// ---- SQSIDE_RAT: put the special-q on the RATIONAL side (side 0) instead of the algebraic ----
// Default (unset) = algebraic special-q, the original behaviour. gpu_loop hardcoded this; CADO
// exposes it as -sqside and it is a genuine METHOD choice, not a tuning parameter. Four things
// must move together: (1) the special-q roots come from Y1*x+Y0 (exactly one root per q) instead
// of the degree-5 F (0..5 roots per q); (2) log2(q) is credited to the rational norm, not the
// algebraic one; (3) the cofactor kernel divides Q out of Nr, not Na -- BOTH `cofactor` AND
// `cofactor_ff` (the latter is the live one under FAST_FINAL and also records Q into that side's
// factor list; missing it yields exactly 0 relations because q stays in the rational cofactor
// where it is < LIM and cannot be a valid large prime); (4) the CPU mpz formatter (#else path)
// emits q into rf. SQ_ADJ_* are 1.0f/0.0f constants, so the default build folds them away.
//
// *** MEASURED 2026-07-20 -- WORSE, do not use. lpb29, q=1M, 24 cores, same poly:
//       algebraic sq (default): 54.5 rel/lat, 8.22 ms/lat -> 6630 rel/s
//       rational  sq (this):    34.5 rel/lat, 7.99 ms/lat -> 4318 rel/s  (-35%)
// Slightly cheaper per lattice, but yield collapses. The algebraic norm is the larger/harder
// side at c151, so the special-q's ~20 bits of reduction is worth far more there. This closes
// Pipeline.md 18's "untried method #1" -- it is tried, and it is negative. ***
#ifdef SQSIDE_RAT
#define SQ_ADJ_A 0.0f
#define SQ_ADJ_R 1.0f
#else
#define SQ_ADJ_A 1.0f
#define SQ_ADJ_R 0.0f
#endif
#ifndef LPB0_VAL
#define LPB0_VAL LPB_VAL         /* side 0 = rational */
#endif
#ifndef LPB1_VAL
#define LPB1_VAL LPB_VAL         /* side 1 = algebraic */
#endif
// Cofactor residual size bounds (bits) passed to the cofactor kernel. With lpb=2^29
// a 2-large-prime cofactor reaches ~58 bits; raising lpb to 2^30 pushes that
// to ~60, so MFB MUST rise too or the extra relations the bigger lpb buys are
// rejected by the size filter. Set per key via -DCFG_MFB1 / -DCFG_MFB0 in build.sh.
#ifndef CFG_MFB1
#define CFG_MFB1 60   /* in-source default; build.sh sets the production value */
#endif
#ifndef CFG_MFB0
#define CFG_MFB0 59   /* in-source default; build.sh sets the production value */
#endif
// Max survivors per lattice (buffer cap; scale with sieve-region area).
#ifndef MAXSURV_VAL
#define MAXSURV_VAL 300000   /* in-source default at LIM=35M / J=4096 */
#endif
__constant__ uint32_t cSP[64]; __constant__ int cNSP;   // small primes < SPB (device, cofactor)
static uint32_t hSP[64]; static int hNSP=0;             // small primes < SPB (host, CPU finalizer)
static void build_smallp(){ for(uint32_t p=2;p<(uint32_t)SPB && hNSP<64;p++){ int pr=1; for(uint32_t d=2;d*d<=p;d++) if(p%d==0){pr=0;break;} if(pr) hSP[hNSP++]=p; } }
static double wall_s(){ struct timeval t; gettimeofday(&t,0); return t.tv_sec+t.tv_usec*1e-6; }
long g_nship_sum=0,g_lat_cnt=0;
double g_prof[8]={0,0,0,0,0,0,0,0};   // [PROF] per-group GPU time: project,scatter,scan,resieve,cofactor,nsurv,scat_col,scat_lat
static u256 parse_u256(const char* s){ mpz_t z; mpz_init_set_str(z,s,10); int neg=mpz_sgn(z)<0; mpz_t mag; mpz_init(mag); mpz_abs(mag,z);
  u256 r=u_zero(); size_t cnt=0; mpz_export(r.w,&cnt,-1,8,0,0,mag); if(neg)r=u_neg(r); mpz_clear(z);mpz_clear(mag); return r; }
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
// ---- polynomial: loaded at RUNTIME from a CADO .poly (degree-5 alg + linear rat) ----
static char* C[6]={0,0,0,0,0,0};      // algebraic coeffs c0..c5 (decimal strings)
static char* sY0=0; static char* sY1=0;  // rational poly Y0 + Y1*x
static double G_SKEW=0;                // skew (from .poly)
static char* dupstr(const char* s){ size_t n=strlen(s); char* r=(char*)malloc(n+1); memcpy(r,s,n+1); return r; }
static void rstrip(char* s){ size_t n=strlen(s); while(n&&(s[n-1]=='\n'||s[n-1]=='\r'||s[n-1]==' '||s[n-1]=='\t'))s[--n]=0; }
// Parse a CADO .poly: lines "c0: ..".."c5: ..", "Y0: ..", "Y1: ..", "skew: ..". Other lines ignored.
static void load_poly(const char* path){
  FILE* f=fopen(path,"r"); if(!f){ fprintf(stderr,"poly: cannot open %s\n",path); exit(1); }
  char line[8192];
  while(fgets(line,sizeof(line),f)){
    char* p=line; while(*p==' '||*p=='\t')p++;
    char* colon=strchr(p,':'); if(!colon||p[0]=='#') continue;
    *colon=0; char* key=p; rstrip(key); char* val=colon+1; while(*val==' '||*val=='\t')val++; rstrip(val);
    if(!*val) continue;
    if(key[0]=='c'&&key[1]>='0'&&key[1]<='5'&&key[2]==0) C[key[1]-'0']=dupstr(val);
    else if(!strcmp(key,"Y0")) sY0=dupstr(val);
    else if(!strcmp(key,"Y1")) sY1=dupstr(val);
    else if(!strcmp(key,"skew")) G_SKEW=atof(val);
  }
  fclose(f);
  for(int k=0;k<6;k++) if(!C[k]){ fprintf(stderr,"poly: missing c%d (need a degree-5 .poly)\n",k); exit(1); }
  if(!sY0||!sY1){ fprintf(stderr,"poly: missing Y0/Y1\n"); exit(1); }
  if(G_SKEW<=0){ fprintf(stderr,"poly: missing/invalid skew\n"); exit(1); }
  fprintf(stderr,"poly loaded: c5=%s skew=%.3f Y1=%s\n",C[5],G_SKEW,sY1);
}
// compile-time defaults; all build-tunable, set for c151 by build.sh. Bigger LIM = larger
// factor base = more relations/lattice; bigger I2/J = larger sieve region per
// special-q = more relations/lattice. Both raise rel/lattice so the lattice COUNT
// (and sieve wall time) doesn't blow up when a key needs ~1.7-2x more relations.
#ifndef LIM
#define LIM 35000000u   /* in-source default; build.sh sets the production value */
#endif
#ifndef I2
#define I2 4096
#endif
#ifndef J
#define J  4096
#endif
#define W  (2*I2)
static const size_t NCELL=(size_t)W*J;
#if defined(PRIM_SCAN) || defined(PRIM_SCATTER)
#define PRIM_WHEEL 1
// Coordinate primitivity wheel (expert lead 2026-07-23). Special-q lattice has det=q, so a wheel
// prime p!=q divides BOTH a and b iff it divides BOTH i and j: p|a & p|b <=> p|i & p|j. Hence
// (i,j) sharing a small prime => non-coprime (a,b) => invalid relation. Reject those cells cheaply
// via two constant lookup tables (W+J = 12 KB), skipping the ~38.6% non-primitive lattice points
// BEFORE they cost L2 scatter transactions. Keep the exact gcd(a,b) check downstream (covers q and
// non-wheel primes). Relations must stay byte-identical to baseline.
__constant__ uint8_t c_dmi[W];      // bit k set if wheel-prime[k] | (index - I2)
__constant__ uint8_t c_dmj[J];      // bit k set if wheel-prime[k] | index
__constant__ unsigned c_wmask;      // active wheel-prime subset (sweep via WHEEL_MASK env)
__device__ __forceinline__ bool prim_skip(int i,int j){ return (c_dmi[i+I2] & c_dmj[j] & c_wmask)!=0; }
static void build_wheel(){
  static const int WP[8]={2,3,5,7,11,13,17,19};
  uint8_t* dmi=(uint8_t*)malloc(W); uint8_t* dmj=(uint8_t*)malloc(J);
  for(int x=0;x<W;x++){ int i=x-I2; uint8_t m=0; for(int k=0;k<8;k++) if(i%WP[k]==0) m|=(uint8_t)(1u<<k); dmi[x]=m; }
  for(int j=0;j<J;j++){ uint8_t m=0; for(int k=0;k<8;k++) if(j%WP[k]==0) m|=(uint8_t)(1u<<k); dmj[j]=m; }
  cudaMemcpyToSymbol(c_dmi,dmi,W); cudaMemcpyToSymbol(c_dmj,dmj,J);
  unsigned wm=getenv("WHEEL_MASK")?(unsigned)strtoul(getenv("WHEEL_MASK"),0,0):0xFFu;
  cudaMemcpyToSymbol(c_wmask,&wm,sizeof(wm));
  free(dmi); free(dmj);
  fprintf(stderr,"[PRIM_WHEEL] active wheel mask=0x%02x (of 2,3,5,7,11,13,17,19)\n",wm);
}
#endif
// Bucket sieve for lat (large-prime) scatter: replaces scat_lat's scattered global
// atomicAdds (~358-cycle L2 latency each) with 2-phase shared-memory accumulation.
// Phase 1 bucket_fill_lat: one thread/prime, enumerate lattice points, store
//   (stripe_off<<8|logp) into per-stripe buckets — no writes to arr at all.
// Phase 2 bucket_flush_lat: one block/stripe, accumulate bucket into a 64KB shared-
//   memory tile via fast shmem atomics (~1 cycle), then non-atomic add to global arr
//   (safe: scat_col finished before flush; only one block owns each stripe).
// Off encoding: off = (row%STRIPE_H)*J + col packed as (off<<8)|logp in uint32_t.
#ifndef STRIPE_H
#define STRIPE_H 16              // rows per stripe; tile = STRIPE_H*J bytes (fits Blackwell 128KB shmem)
#endif
#define N_STRIPES (W/STRIPE_H)   // 8192/16 = 512 horizontal stripes
#define STRIPE_SIZE (STRIPE_H*J) // 16*4096 = 65536 bytes per tile
#ifndef BUCKET_CAP
#define BUCKET_CAP 65536         // max entries per stripe; 512*65536*4*2 = 256MB total
#endif
// col/lat boundary. NOTE the original comment ("medium primes were lat-imbalanced") predates
// FLAT_COOP2, the cooperative uniform-work walk added specifically to cure that imbalance -- so
// this value was tuned against a siever that no longer exists. It matters a lot: the column method
// costs W scans per line REGARDLESS of hit count, so its efficiency is J/m, i.e. <1 for every
// prime above J=4096. At T=65536 roughly 90% of the col lines sit in that wasteful range.
// Overridable so it can actually be swept.
// MEASURED 2026-07-20 (lpb29, q=1M, 24 cores, 3 runs each, rel/s):
//   T= 65536 (old): 6610   T= 98304: 7127   T=131072: 7241   T=163840: 7222   T=196608: 7174
//   T= 32768: 6124   T=16384: 5270   T=4096: 3689   T=262144: 6934   T=1048576: 4289
// => T=131072 is the optimum, +9.5% rel/s over the shipped 65536, ranges non-overlapping.
// Relations verified BIT-IDENTICAL by sorted-set diff (T only reassigns a prime between the two
// methods; both visit the same lattice points). Yield unchanged at 54.5 rel/lat.
#ifndef TSPLIT
#define TSPLIT 131072
#endif
static const uint32_t T=TSPLIT;
static inline long long smod(long long a,long long p){ long long r=a%p; return r<0?r+p:r; }
static u64 ginv2(u64 a,u64 m){ long long t=0,nt=1,r=m,nr=a%m; while(nr){long long q=r/nr,tmp;tmp=t-q*nt;t=nt;nt=tmp;tmp=r-q*nr;r=nr;nr=tmp;} if(r>1)return 0; if(t<0)t+=m; return (u64)t; }
static u64 evf(char*const cc[6],u64 x,u64 m){u64 v=0;for(int k=5;k>=0;k--)v=addmod(mulmod(v,x,m),strmod(cc[k],m),m);return v;}
static u64 evfp(char*const cc[6],u64 x,u64 m){u64 v=0;for(int k=5;k>=1;k--)v=addmod(mulmod(v,x,m),mulmod(strmod(cc[k],m),(u64)k%m,m),m);return v;}

// ---- factor base, built ONCE (special-q independent): (m, r, type, logp) per side/size group ----
// groups: g = side*2 + (m>T?1:0) ; side 0=rational,1=algebraic ; type 0=affine,1=projective
std::vector<uint32_t> fM[4],fR[4],fP[4]; std::vector<uint8_t> fT[4],fL[4];
#define FB_MAXT 64
// per-thread factor-base buffers: build_fb() is OpenMP-parallel (each prime independent);
// thread buffers are merged in tid order afterwards, preserving the serial ascending-prime order.
static std::vector<uint32_t> tM[FB_MAXT][4],tR[FB_MAXT][4],tP[FB_MAXT][4];
static std::vector<uint8_t>  tT[FB_MAXT][4],tL[FB_MAXT][4];
static thread_local int g_tid=0;   // owning thread's buffer index (set in the parallel region)
static thread_local u64 g_pbase;   // base prime for current addfb calls (set per prime in build_fb)
static void addfb(int side,u64 m,u64 r,int type,int lp){ int g=side*2+(m>T?1:0); int t=g_tid;
  tM[t][g].push_back((uint32_t)m); tR[t][g].push_back((uint32_t)r); tT[t][g].push_back((uint8_t)type); tL[t][g].push_back((uint8_t)lp);
  tP[t][g].push_back((uint32_t)g_pbase); }
// general prime-power root lifting for algebraic side (handles RAMIFIED primes where Hensel breaks)
static void lift_alg_powers(u64 p,u64* r0,int nr0,int lp){
  if((u64)p*p>LIM) return;
  u64 cur[128]; int nc=0; for(int i=0;i<nr0 && nc<128;i++) cur[nc++]=r0[i]%p;
  u64 pk=p;
  while(pk<=LIM/p){
    u64 pk1=pk*p; u64 nx[128]; int nn=0;
    for(int i=0;i<nc && nn<128;i++){ u64 r=cur[i]; u64 fpp=evfp(C,r,p);
      if(fpp){ u64 fp1=evfp(C,r,pk1), fr=evf(C,r,pk1), iv=ginv2(fp1,pk1);  // unramified: Hensel
        u64 r1=submod(r%pk1,mulmod(fr,iv,pk1),pk1); if(evf(C,r1,pk1)==0){ nx[nn++]=r1; addfb(1,pk1,r1,0,lp); } }
      else for(u64 t=0;t<p && nn<128;t++){ u64 rr=r+t*pk; if(evf(C,rr,pk1)==0){ nx[nn++]=rr; addfb(1,pk1,rr,0,lp); } } // ramified: all lifts
    }
    if(nn==0) break; for(int i=0;i<nn;i++)cur[i]=nx[i]; nc=nn; pk=pk1;
  }
}
static void build_fb(){
  uint8_t* comp=(uint8_t*)calloc(LIM+1,1);
  for(u64 i=2;i*i<=LIM;i++) if(!comp[i]) for(u64 j=i*i;j<=LIM;j+=i) comp[j]=1;
  int nt=omp_get_max_threads(); if(nt>FB_MAXT)nt=FB_MAXT; if(nt<1)nt=1;
  // Parallel root-finding: each prime is independent. schedule(static) gives thread t a
  // contiguous ascending p-range, so merging buffers in tid order reproduces serial order.
  #pragma omp parallel num_threads(nt)
  {
    g_tid=omp_get_thread_num();
    u64 roots[8];
    #pragma omp for schedule(dynamic,256)
    for(long long pp=2; pp<=(long long)LIM; pp++){ u64 p=(u64)pp; if(comp[p])continue;
      int lp=(int)lround(log2((double)p)); g_pbase=p;
      int nr=find_roots(C,p,roots);
      for(int k=0;k<nr;k++) addfb(1,p,roots[k],0,lp);
      lift_alg_powers(p,roots,nr,lp);                                // prime powers (ramified-safe)
      if(strmod(C[5],p)==0) addfb(1,p,0,1,lp);                       // algebraic projective (p | leading coeff)
      u64 Y1m=strmod(sY1,p);
      if(Y1m){ u64 m=mulmod(strmod(sY0,p),ginv2(Y1m,p),p); m=(p-m)%p; addfb(0,p,m,0,lp);
        if(p*p<=LIM){ u64 pk=p; while(pk<=LIM/p){ u64 pk1=pk*p,Y1k=strmod(sY1,pk1); if(!Y1k)break;
          u64 mk=mulmod(strmod(sY0,pk1),ginv2(Y1k,pk1),pk1); mk=(pk1-mk)%pk1; addfb(0,pk1,mk,0,lp); pk=pk1;} } }
      else addfb(0,p,0,1,lp);                                        // rational projective
    }
  }
  // merge per-thread buffers in tid order (preserves ascending-prime order of the serial build)
  for(int g=0;g<4;g++){
    size_t tot=0; for(int t=0;t<nt;t++) tot+=tM[t][g].size();
    fM[g].reserve(tot); fR[g].reserve(tot); fT[g].reserve(tot); fL[g].reserve(tot); fP[g].reserve(tot);
    for(int t=0;t<nt;t++){
      fM[g].insert(fM[g].end(),tM[t][g].begin(),tM[t][g].end());
      fR[g].insert(fR[g].end(),tR[t][g].begin(),tR[t][g].end());
      fT[g].insert(fT[g].end(),tT[t][g].begin(),tT[t][g].end());
      fL[g].insert(fL[g].end(),tL[t][g].begin(),tL[t][g].end());
      fP[g].insert(fP[g].end(),tP[t][g].begin(),tP[t][g].end());
      std::vector<uint32_t>().swap(tM[t][g]); std::vector<uint32_t>().swap(tR[t][g]); std::vector<uint32_t>().swap(tP[t][g]);
      std::vector<uint8_t>().swap(tT[t][g]); std::vector<uint8_t>().swap(tL[t][g]);
    }
  }
  free(comp);
}
// ---- GPU: projection (roots -> lines) ----
__device__ __forceinline__ long long smd(long long a,long long m){ long long r=a%m; return r<0?r+m:r; }
__device__ uint32_t ginv_d(long long a,long long m){ long long t=0,nt=1,r=m,nr=a%m; if(nr<0)nr+=m;
  while(nr){ long long q=r/nr,tmp; tmp=t-q*nt;t=nt;nt=tmp; tmp=r-q*nr;r=nr;nr=tmp; } if(r>1)return 0xFFFFFFFFu; if(t<0)t+=m; return (uint32_t)t; }
__global__ void project(const uint32_t* M,const uint32_t* Rin,const uint8_t* TY,uint32_t* Rout,int n,
                        long long a0,long long b0,long long a1,long long b1){
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
    long long m=FB_LD(&M[i]),A,B;
    if(TY[i]){ A=smd(-b0,m); B=smd(b1,m); }                          // projective: b0*i+b1*j≡0
    else { long long r=Rin[i]; A=smd(-(a0-r*b0),m); B=smd(a1-r*b1,m); }  // affine: j≡R i, R=-(a0-r b0)/(a1-r b1)
    if(B==0){ if(A==0){Rout[i]=0xFFFFFFFFu;continue;}                // vertical: A*i≡0 mod m
      long long g=A,mm=m; while(mm){long long t=g%mm;g=mm;mm=t;}
      Rout[i]=0x80000000u | (uint32_t)(m/g);  continue; }            // flag + step (i≡0 mod step)
    uint32_t inv=ginv_d(B,m); if(inv==0xFFFFFFFFu){ Rout[i]=0xFFFFFFFFu; continue; }
    Rout[i]=(uint32_t)(( (long long)A*inv)%m);
  }
}
// ---- scatter (skip invalid R) ----
__global__ void scat_col(uint8_t* arr,const uint32_t* M,const uint32_t* R,const uint8_t* L,int n){
#ifdef PROBE_COL_NOWRITE
  unsigned long long colsink=0;
#endif
#ifdef SCAT_COL_RECUR
  // *** MEASURED +9% REGRESSION (2026-07-21): 7.33 -> 8.00 ms/lat, relations byte-identical. ***
  // Removes the per-column 64-bit mulmod (recurrence j0 += rbd) but the restructure (block-per-line vs
  // the flat grid-stride over all n*W items) wrecks load balance: the ~25k col-lines have wildly
  // different weight (small m = thousands of writes/line, large m = near-zero), so blocks that draw a
  // heavy line stall while others idle. The mulmod was NOT the bottleneck -- the flat kernel's uniform
  // parallelism is. Same lesson as EXACT_ENUM/FK/bucket: wasteful-but-parallel beats efficient-but-serial.
  // Per-line: threads cover cols [tid, tid+blockDim, ...]. rbd=(r*blockDim) mod m; j0 advances by rbd.
  for(int line=blockIdx.x; line<n; line+=gridDim.x){
    uint32_t m=M[line],r=R[line],lp=L[line]; if(r==0xFFFFFFFFu)continue;
    if(m<(uint32_t)SCAT_SKIP)continue;
    if(r&0x80000000u){ uint32_t step=r&0x7FFFFFFFu;
      for(int col=threadIdx.x;col<W;col+=blockDim.x){ int i=col-I2;
        if((((i%(int)step)+(int)step)%(int)step)==0) for(int j=0;j<J;j++){ size_t idx=(size_t)(i+I2)*J+j; SIEVE_ADD((unsigned*)&arr[idx&~3ull],(unsigned)lp<<(8*(idx&3))); } }
      continue; }
    uint32_t rbd=(uint32_t)(((uint64_t)r*blockDim.x)%m);
    int col0=threadIdx.x,i0=col0-I2;
    uint32_t j0=(uint32_t)(((uint64_t)r*(uint32_t)(((i0%(int)m)+(int)m)%(int)m))%m);
    for(int col=col0;col<W;col+=blockDim.x){ int i=col-I2;
      for(uint32_t j=j0;j<J;j+=m){ size_t idx=(size_t)(i+I2)*J+j; SIEVE_ADD((unsigned*)&arr[idx&~3ull],(unsigned)lp<<(8*(idx&3))); }
      j0+=rbd; if(j0>=m)j0-=m; }
  }
#else
  for(long long w=(long long)blockIdx.x*blockDim.x+threadIdx.x; w<(long long)n*W; w+=(long long)gridDim.x*blockDim.x){
    int line=(int)(w/W),col=(int)(w%W),i=col-I2; uint32_t m=M[line],r=R[line],lp=L[line]; if(r==0xFFFFFFFFu)continue;
    if(m<(uint32_t)SCAT_SKIP)continue;   // small primes skipped in scatter (trial-divided in cofactor); threshold compensated by SCATOFF
    if(r&0x80000000u){ uint32_t step=r&0x7FFFFFFFu;                   // vertical line: i≡0 mod step, all j
      if((((i%(int)step)+(int)step)%(int)step)==0) for(int j=0;j<J;j++){ size_t idx=(size_t)(i+I2)*J+j; SIEVE_ADD((unsigned*)&arr[idx&~3ull],(unsigned)lp<<(8*(idx&3))); } continue; }
    uint32_t im=(uint32_t)(((i%(int)m)+(int)m)%(int)m), j0=(uint32_t)(((uint64_t)r*im)%m);
#ifdef PROBE_COL_NOWRITE
    // Diagnostic (timing only, WRONG results): keep the per-(line,column) modular setup -- the one
    // mulmod that ANY tiling scheme must still perform, since a tile has to span a column's whole
    // j-range to amortise it -- but drop the j-walk and its writes. The remainder is the hard floor
    // for a shared-memory tiled rewrite of the dense/small-prime path.
    colsink+=j0; continue;
#endif
    for(uint32_t j=j0;j<J;j+=m){ size_t idx=(size_t)(i+I2)*J+j; SIEVE_ADD((unsigned*)&arr[idx&~3ull],(unsigned)lp<<(8*(idx&3))); }
  }
#endif
#ifdef PROBE_COL_NOWRITE
  if(colsink==0xDEADBEEFCAFEULL) arr[0]=1;
#endif
}
// ---- TILE_COL: shared-memory tiled scatter for the DENSE (small-prime, p<=T) groups ----------
// Replaces scat_col for groups 0/2 and the sieve-array memset.
//
// Why this shape (measured, see Pipeline.md 5): the dense path splits into two costs that need
// opposite fixes. Band R0 (p<=4096, ~600-800 lines) is WRITE-bound -- 29 M scattered byte-atomics
// per side, only 0.03 ms of address math. Band R1 (4096<p<=T, ~12k lines) is SETUP-bound -- one
// 64-bit mulmod per (line,column) = 96 M mulmods per side, only 0.02 ms of writes. So:
//   * a block owns TCOLS consecutive COLUMNS (a column is J contiguous bytes, so a tile is
//     TCOLS*J contiguous bytes) and accumulates every dense line into SHARED memory, which fixes
//     R0: the scattered adds land in shared instead of L2.
//   * j0 is computed once per (line, tile) and then advanced across the tile's columns by the
//     recurrence j0(i+1) = (j0(i) + r) mod m (r<m, so one conditional subtract) -- TCOLS-fold
//     fewer mulmods, which fixes R1.
//   * the tile is written out with plain coalesced stores, not atomics, so this kernel also
//     INITIALISES the array: the separate 33.5 MB cudaMemset per side is dropped (fused init).
// Block-per-column-group is load-balanced by construction; this is what makes it work where
// SCAT_COL_RECUR (block-per-LINE) failed at -9% -- there the per-line weights differ by 1000x.
// Correctness: the tile starts at zero and receives exactly the same multiset of 32-bit word adds
// (same byte-carry semantics) as memset+scat_col, so the stored array is identical.
#ifndef TCOLS
#define TCOLS 8
#endif
// TCOLS MUST divide W: the grid is W/TCOLS blocks and this kernel is the array's only initialiser
// (the memset is dropped), so a non-dividing TCOLS leaves the last W%TCOLS columns holding stale
// data from the previous lattice. Measured symptom at TCOLS=12: nsurv 86.5k -> 106k, cofactor
// 0.77 -> 1.12 ms, i.e. silent corruption that only shows up as extra false survivors.
static_assert(W%TCOLS==0,"TCOLS must divide W (scat_tile is the sole array initialiser)");
__global__ void scat_tile(uint8_t* arr,const uint32_t* M,const uint32_t* R,const uint8_t* L,int n){
  __shared__ uint8_t tile[TCOLS*J];
  const int col0=blockIdx.x*TCOLS;
  for(int z=threadIdx.x*4; z<TCOLS*J; z+=blockDim.x*4) *(unsigned*)&tile[z]=0u;
  __syncthreads();
  for(int line=threadIdx.x; line<n; line+=blockDim.x){
    uint32_t m=M[line],r=R[line]; unsigned lp=L[line];
    if(r==0xFFFFFFFFu) continue;
    if(m<(uint32_t)SCAT_SKIP) continue;
    if(r&0x80000000u){ uint32_t step=r&0x7FFFFFFFu;                  // vertical line: i=0 mod step
      for(int c=0;c<TCOLS;c++){ int i=col0+c-I2;
        if((((i%(int)step)+(int)step)%(int)step)==0)
          for(int j=0;j<J;j++){ unsigned idx=(unsigned)c*J+j;
            atomicAdd((unsigned*)&tile[idx&~3u],lp<<(8*(idx&3u))); } }
      continue; }
    int i0=col0-I2;
    uint32_t j0=(uint32_t)(((uint64_t)r*(uint32_t)(((i0%(int)m)+(int)m)%(int)m))%m);
    for(int c=0;c<TCOLS;c++){
      for(uint32_t j=j0;j<J;j+=m){ unsigned idx=(unsigned)c*J+j;
        atomicAdd((unsigned*)&tile[idx&~3u],lp<<(8*(idx&3u))); }
      j0+=r; if(j0>=m)j0-=m;                                         // j0(i+1) = (j0(i)+r) mod m
    }
  }
  __syncthreads();
  for(int z=threadIdx.x*4; z<TCOLS*J; z+=blockDim.x*4)
    *(unsigned*)&arr[(size_t)col0*J+z]=*(unsigned*)&tile[z];
}
__device__ __forceinline__ long long fdv(long long a,long long b){ long long q=a/b,r=a%b; if(r!=0&&((r<0)!=(b<0)))q--; return q; }
// int32 floor/ceil division (C truncates toward zero), for the exact-range enumeration below.
// MEASURED: EXACT_C2_RANGE below is a REGRESSION (scatter 7.35->8.40, resieve 5.34->7.07 ms/lat,
// GPU +18%) even though it cuts the walk from 188.2M to 41.2M iterations/lattice. The ~20 int32
// divisions it needs per line cost more than the 4.57x of wasted box iterations they remove --
// i.e. the filtered-out iterations are nearly free (adds+compares, no memory op). Keep the box
// walk. Build -DEXACT_ENUM to re-test. Relations are identical either way (verified).
// MEASURED on the vetted c151 poly, q=3M, 307 lattices (GPU ms/lat, relations identical in all):
//   BOX (default)          scatter 7.23  resieve 5.31  GPU 15.5
//   EXACT_ENUM (int div)   scatter 8.40  resieve 7.07  GPU 18.3   (+18%)
//   DENSE_ENUM (dbl step)  scatter 15.25 resieve 14.26 GPU 32.3   (+108%)
// Why both lose: the true in-region run is ~1 lattice point per c1 row, so there is no run to
// densify -- the per-c1 range math can never amortize over a single hit, and the box walk's
// filtered-out iterations are nearly free (adds+compares, no memory op). Keep BOX.
#if !defined(EXACT_ENUM) && !defined(DENSE_ENUM)
#define BOX_ENUM 1
#endif
// DENSE_ENUM: the c2 range for a given c1 is bounded by two LINEAR functions of c1
//   i = c1*p1 + c2*p2 in [-I2, I2-1]   ->  c2 between (-I2-c1*p1)/p2 and (I2-1-c1*p1)/p2
//   j = c1*q1 + c2*q2 in [0,   J-1 ]   ->  c2 between (  -c1*q1)/q2 and (J-1-c1*q1)/q2
// so each endpoint moves by a CONSTANT step (-p1/p2, -q1/q2) as c1 advances. Take two
// reciprocals per LINE and then step the four endpoints with plain adds -- no per-c1
// division (that variant, EXACT_ENUM, cost more than the waste it removed).
// Doubles are exact for our magnitudes (<2^27) and drift over <=8192 adds is ~1e-8, so
// widening each endpoint by 1 and KEEPING the in-region test makes this correctness-neutral:
// the test still gates every write, the range only stops us walking iterations that provably
// fail it. Purpose is warp density -- only 21.9% of box-walk iterations pass the test, so the
// atomic issues with ~7/32 lanes active.
struct C2Range { double lo_i,hi_i,lo_j,hi_j,st_i,st_j; int has_i,has_j; };
__device__ __forceinline__ void c2range_init(C2Range* R,int c1,int p1,int q1,int p2,int q2){
  R->has_i=(p2!=0); R->has_j=(q2!=0);
  if(R->has_i){ double inv=1.0/(double)p2, base=(double)c1*(double)p1;
    R->lo_i=(-(double)I2-base)*inv; R->hi_i=((double)(I2-1)-base)*inv; R->st_i=-(double)p1*inv; }
  if(R->has_j){ double inv=1.0/(double)q2, base=(double)c1*(double)q1;
    R->lo_j=(0.0-base)*inv; R->hi_j=((double)(J-1)-base)*inv; R->st_j=-(double)q1*inv; }
}
__device__ __forceinline__ void c2range_get(const C2Range* R,int* lo,int* hi){
  if(R->has_i){ double a=fmin(R->lo_i,R->hi_i), b=fmax(R->lo_i,R->hi_i);
    int x=(int)ceil(a)-1, y=(int)floor(b)+1; if(x>*lo)*lo=x; if(y<*hi)*hi=y; }
  if(R->has_j){ double a=fmin(R->lo_j,R->hi_j), b=fmax(R->lo_j,R->hi_j);
    int x=(int)ceil(a)-1, y=(int)floor(b)+1; if(x>*lo)*lo=x; if(y<*hi)*hi=y; }
}
__device__ __forceinline__ void c2range_step(C2Range* R){
  if(R->has_i){ R->lo_i+=R->st_i; R->hi_i+=R->st_i; }
  if(R->has_j){ R->lo_j+=R->st_j; R->hi_j+=R->st_j; }
}
__device__ __forceinline__ int fdiv_i(int a,int b){ int q=a/b,r=a%b; if(r!=0&&((r<0)!=(b<0)))q--; return q; }
__device__ __forceinline__ int cdiv_i(int a,int b){ int q=a/b,r=a%b; if(r!=0&&((r<0)==(b<0)))q++; return q; }
// EXACT_ENUM: for a fixed c1, solve the two region constraints for c2 directly instead of
// walking the whole (A1..A2)x(B1..B2) bounding box and filtering with an if.
//   -I2 <= i = u + c2*p2 <= I2-1      0 <= j = v + c2*q2 <= J-1
// Dividing by a negative coefficient flips the inequality, hence the sign branches. The
// resulting [lo,hi] is exactly the in-region run, so the inner loop is branch-free and every
// iteration lands a hit -- which is precisely why scat_col (63 G atomics/s) beats the boxed
// scat_lat walk (6.7 G/s). Visits the SAME set of lattice points -> relations are identical.
// All values fit int32: |i|,|j| < 4096, |basis| ~ sqrt(m) <= 2^14, so |c1*p1| < 2^27.
#define EXACT_C2_RANGE(u,v,p2,q2,lo,hi,EMPTY) do{                                  \
  if((p2)>0){ int _a=cdiv_i(-I2-(u),(p2)), _b=fdiv_i(I2-1-(u),(p2));               \
              if(_a>(lo))(lo)=_a; if(_b<(hi))(hi)=_b; }                            \
  else if((p2)<0){ int _a=cdiv_i(I2-1-(u),(p2)), _b=fdiv_i(-I2-(u),(p2));          \
              if(_a>(lo))(lo)=_a; if(_b<(hi))(hi)=_b; }                            \
  else if((u)<-I2||(u)>I2-1){ EMPTY; }                                             \
  if((q2)>0){ int _a=cdiv_i(-(v),(q2)), _b=fdiv_i(J-1-(v),(q2));                   \
              if(_a>(lo))(lo)=_a; if(_b<(hi))(hi)=_b; }                            \
  else if((q2)<0){ int _a=cdiv_i(J-1-(v),(q2)), _b=fdiv_i(-(v),(q2));              \
              if(_a>(lo))(lo)=_a; if(_b<(hi))(hi)=_b; }                            \
  else if((v)<0||(v)>J-1){ EMPTY; }                                                \
}while(0)
#ifdef PROBE_ENUM
// Diagnostic only: count (c1,c2) pairs ENUMERATED vs pairs that actually land in the
// region (= real hits). Ratio = wasted work in the bounding-box walk. Per-thread
// accumulators, one atomic each at kernel exit, so counting barely perturbs timing.
__device__ unsigned long long g_enum_cnt, g_hit_cnt;
__device__ unsigned long long g_vert_lines, g_vert_atomics, g_vert_worst;
#endif
#ifdef PROBE_ITER
// Diagnostic only: Lagrange-reduction trip counts. g_it_sum is the work actually needed
// (sum over lines); g_it_wmax is what the hardware PAYS -- a warp runs until its SLOWEST
// lane breaks, so per grid-stride step the whole warp costs max(trip) not mean(trip).
// g_it_wmax/g_it_sum is therefore the pure divergence waste factor in the reduction.
__device__ unsigned long long g_it_sum, g_it_wmax, g_it_lines;
#endif
__global__ void scat_lat(uint8_t* arr,const uint32_t* M,const uint32_t* R,const uint8_t* L,int n,int* BA,int* BND,unsigned long long* TC,int* vlist,unsigned* vcnt){
#ifdef PROBE_HALF
  // Diagnostic: FB is sorted ascending by prime, so lines [0,n/2) hold nearly all the hits
  // (small p -> many lattice points) while [n/2,n) hold almost none -- but BOTH halves pay
  // identical per-line setup. Comparing the two halves separates setup cost from hit cost
  // without any compiler-DCE games (same code, only the line range differs).
  const int lo=(PROBE_HALF==2)?(n/2):0, hi=(PROBE_HALF==2)?n:(n/2);
#else
  const int lo=0, hi=n;
#endif
#ifdef PROBE_ENUM
  unsigned long long myenum=0,myhit=0;
#endif
#if defined(PROBE_NOATOMIC) || defined(PROBE_NOBOUNDS) || defined(PROBE_NOENUM) || defined(PROBE_NOSTORE)
  unsigned long long sink=0;
#endif
  for(int line=lo+blockIdx.x*blockDim.x+threadIdx.x;line<hi;line+=gridDim.x*blockDim.x){
    long long m=FB_LD(&M[line]); uint32_t rr=FB_LD(&R[line]); unsigned lp=FB_LD(&L[line]); if(rr==0xFFFFFFFFu)continue;
    if(BND){ TC[line]=0; BND[line*4+2]=0; BND[line*4+3]=0; }
    if(rr&0x80000000u){ uint32_t step=rr&0x7FFFFFFFu;                 // vertical line (rare for large m)
      if(BND){ unsigned k=atomicAdd(vcnt,1u); if(k<4096u) vlist[k]=line; }
#ifdef PROBE_ENUM
      { unsigned long long na=0; for(int i=-I2;i<I2;i++) if((((i%(int)step)+(int)step)%(int)step)==0) na+=J;
        atomicAdd(&g_vert_lines,1ULL); atomicAdd(&g_vert_atomics,na); atomicMax(&g_vert_worst,na); }
#endif
      for(int i=-I2;i<I2;i++) if((((i%(int)step)+(int)step)%(int)step)==0) for(int j=0;j<J;j++){ size_t idx=(size_t)(i+I2)*J+j; SIEVE_ADD((unsigned*)&arr[idx&~3ull],(unsigned)lp<<(8*(idx&3))); } continue; }
#ifdef SLOW_REDUCE
    long long r=rr; long long p1=1,q1=r,p2=0,q2=m;
    for(int it=0;it<64;it++){ long long n1=p1*p1+q1*q1,n2=p2*p2+q2*q2; if(n2<n1){long long a=p1;p1=p2;p2=a;a=q1;q1=q2;q2=a;n1=n2;} if(!n1)break;
      long long dot=p1*p2+q1*q2,mu; if(dot>=0)mu=(2*dot+n1)/(2*n1);else mu=-((-2*dot+n1)/(2*n1)); if(!mu)break; p2-=mu*p1;q2-=mu*q1; }
#else
    // Lagrange reduction, int32 basis + double norms. The basis entries stay bounded by ~2m
    // (m<=LIM<2^27) so int32 holds them; only the norms/dot reach ~m^2 (~2^54), and those are
    // needed just to CHOOSE mu. p2-=mu*p1 is unimodular for ANY integer mu, so the lattice --
    // and therefore every relation -- is identical no matter how mu is rounded. That lets the
    // 64-bit software division (per iteration, per line, ~8.2M lines/lattice) become a
    // hardware double divide. Build -DSLOW_REDUCE for the exact-int64 original.
    int p1=1,q1=(int)rr,p2=0,q2=(int)m;
#ifdef FLOAT_REDUCE
    // float norms: mu is unimodular so ANY rounding gives the SAME lattice (relations identical);
    // a slightly-less-reduced basis only enlarges the box. Fewer registers than double.
    for(int it=0;it<64;it++){
      float n1=(float)p1*p1+(float)q1*q1, n2=(float)p2*p2+(float)q2*q2;
      if(n2<n1){ int a=p1;p1=p2;p2=a; a=q1;q1=q2;q2=a; n1=n2; }
      if(n1==0.0f)break;
      float mud=((float)p1*p2+(float)q1*q2)/n1;
      int mu=(int)(mud>=0.0f?floorf(mud+0.5f):ceilf(mud-0.5f));
      if(!mu)break; p2-=mu*p1; q2-=mu*q1;
    }
#else
    // REDUCE_CAP: hard bound on the reduction trip count. Correctness-neutral by the argument
    // above -- every mu is unimodular, so ANY truncation yields the SAME lattice and the SAME
    // relations; a less-reduced basis only widens the (A,B) box, moving work into the walk.
    // Capping cuts the divergence tax: a warp costs max(trip) across its 32 lanes, so the tail
    // of slow-converging lines is paid by every lane. See PROBE_ITER for the measured spread.
#ifndef REDUCE_CAP
#define REDUCE_CAP 64
#endif
#ifdef PROBE_ITER
    int _it=0;
#endif
#ifdef PROBE_NOREDUCE
    // Skip the reduction entirely, keep the loads + the BA store. Basis is NOT reduced, so the
    // box is garbage and results are WRONG -- timing only. Separates the reduction's ARITHMETIC
    // from the cost of merely streaming 6M lines and storing 16B of basis per line.
    goto _ba_store;
#endif
    for(int it=0;it<REDUCE_CAP;it++){
      double n1=(double)p1*(double)p1+(double)q1*(double)q1;
      double n2=(double)p2*(double)p2+(double)q2*(double)q2;
      if(n2<n1){ int a=p1;p1=p2;p2=a; a=q1;q1=q2;q2=a; n1=n2; }
      if(n1==0.0)break;
      double dot=(double)p1*(double)p2+(double)q1*(double)q2;
      double mud=dot/n1;
      int mu=(int)(mud>=0.0?floor(mud+0.5):ceil(mud-0.5));   // round-to-nearest, as the int64 form did
      if(!mu)break;
      p2-=mu*p1; q2-=mu*q1;
#ifdef PROBE_ITER
      _it=it+1;
#endif
    }
#ifdef PROBE_ITER
    { unsigned long long w=_it;
      for(int off=16;off;off>>=1){ unsigned long long o=__shfl_xor_sync(0xFFFFFFFFu,w,off); if(o>w)w=o; }
      atomicAdd(&g_it_sum,(unsigned long long)_it); atomicAdd(&g_it_lines,1ULL);
      if((threadIdx.x&31)==0) atomicAdd(&g_it_wmax,w*32ULL); }
#endif
#endif
#endif
#ifdef PROBE_NOREDUCE
_ba_store:
#endif
#ifdef PROBE_NOSTORE
    // Drop the 16B/line basis store, keep loads + loop. WRONG results, timing only: isolates
    // how much of the 6M-line streaming cost is the BA write-back vs the factor-base reads.
    sink+=(unsigned long long)(p1+q1+p2+q2);
#elif defined(NO_BA_CACHE)
    // NO_BA_CACHE: drop the basis cache; resieve_coop recomputes the reduction from (M,R).
    // *** MEASURED A LOSS -- 8.27 -> 9.55 ms/lat (-15%). Do NOT enable. ***
    //   scatter 5.00 -> 4.92 (store saves only 0.11)   resieve 2.00 -> 3.23 (recompute costs 1.23)
    // The motivating probe was misread: -DPROBE_NOSTORE measures the store at 1.08 ms/lat, but
    // only because that probe also strips the reduction/bounds/walk, leaving the store as the sole
    // bottleneck. In the full kernel the write latency hides under that compute, so the store's
    // MARGINAL cost is 0.11 ms. Isolation probes here give "cost if it were the only work", NOT
    // recoverable time -- only probes that keep the rest of the kernel intact (e.g. PROBE_NOATOMIC)
    // yield a valid marginal cost. The BA cache is doing its job; keep it.
    (void)BA;
#else
#ifdef BA_VEC
    // One 16-byte store instead of four 4-byte stores strided by 16. Consecutive threads take
    // consecutive lines, so the scalar form makes a warp issue 4 strided transactions for what is
    // a single 512-byte contiguous run; the int4 form issues one. BA is cudaMalloc'd (256-byte
    // aligned) and indexed at line*4 ints = line*16 bytes, so the int4 access is always aligned.
    *(int4*)&BA[(size_t)line*4]=make_int4((int)p1,(int)q1,(int)p2,(int)q2);
#else
    FB_ST(&BA[(size_t)line*4],(int)p1);FB_ST(&BA[(size_t)line*4+1],(int)q1);FB_ST(&BA[(size_t)line*4+2],(int)p2);FB_ST(&BA[(size_t)line*4+3],(int)q2);  // cache reduced basis -> resieve_lat reuses it
#endif
#endif
#ifdef PROBE_NOBOUNDS
    sink+=(unsigned long long)(p1+q1+p2+q2);   // stop after reduction+BA: isolates the fdv bounds
    continue;
#endif
    // int32 bounds: reduced basis ~sqrt(m)<2^14, det D=+-m<2^27, corners q2*I-p2*J<2^27 -> fit int32.
    int D=p1*q2-p2*q1; if(!D)continue;
    int cI[2]={-I2,I2-1},cJ[2]={0,J-1}; int a1m=0x3fffffff,a1M=-0x3fffffff,b1m=0x3fffffff,b1M=-0x3fffffff;
    for(int x=0;x<2;x++)for(int y=0;y<2;y++){ int I_=cI[x],J_=cJ[y]; int n1c=q2*I_-p2*J_,n2c=-q1*I_+p1*J_;
      a1m=min(a1m,n1c);a1M=max(a1M,n1c);b1m=min(b1m,n2c);b1M=max(b1M,n2c); }
    int A1,A2,B1,B2; if(D>0){A1=fdiv_i(a1m,D);A2=-fdiv_i(-a1M,D);B1=fdiv_i(b1m,D);B2=-fdiv_i(-b1M,D);}else{A1=fdiv_i(a1M,D);A2=-fdiv_i(-a1m,D);B1=fdiv_i(b1M,D);B2=-fdiv_i(-b1m,D);}
    if(BND&&A2>=A1&&B2>=B1){ BND[line*4]=A1; BND[line*4+1]=B1;
      BND[line*4+2]=(B2-B1+1); BND[line*4+3]=(A2-A1+1);
      TC[line]=(unsigned long long)(A2-A1+1)*(unsigned long long)(B2-B1+1); }
#ifdef PROBE_NOENUM
    if(A2>=A1&&B2>=B1){ sink+=(unsigned long long)(A1+A2+B1+B2); }   // keep setup, skip the walk
    continue;
#endif
#ifdef BOX_ENUM
    for(int c1=A1;c1<=A2;c1++){ int i=c1*p1+B1*p2,j=c1*q1+B1*q2;          // int32: i,j bounded ~region; strength-reduced in c2
#ifdef PROBE_ENUM
      myenum+=(unsigned long long)(B2-B1+1);
#endif
      for(int c2=B1;c2<=B2;c2++,i+=p2,j+=q2)
        if(i>=-I2&&i<I2&&j>=0&&j<J){ size_t off=(size_t)(i+I2)*J+j;
#ifdef PROBE_ENUM
          myhit++;
#endif
#ifdef PROBE_NOATOMIC
          sink+=off;                    // same enumeration+branch, NO memory op
#else
          SIEVE_ADD((unsigned*)&arr[off&~3ull],(unsigned)lp<<(8*(off&3)));
#endif
        } }
#elif defined(DENSE_ENUM)
    { const int ip1=(int)p1,iq1=(int)q1,ip2=(int)p2,iq2=(int)q2;
      C2Range R; c2range_init(&R,(int)A1,ip1,iq1,ip2,iq2);
      for(int c1=(int)A1;c1<=(int)A2;c1++,c2range_step(&R)){
        int u=c1*ip1, v=c1*iq1, lo=(int)B1, hi=(int)B2;
        c2range_get(&R,&lo,&hi);
        if(hi<lo) continue;
        int i=u+lo*ip2, j=v+lo*iq2;
#ifdef PROBE_ENUM
        myenum+=(unsigned long long)(hi-lo+1);
#endif
        for(int c2=lo;c2<=hi;c2++,i+=ip2,j+=iq2)
          if(i>=-I2&&i<I2&&j>=0&&j<J){ size_t off=(size_t)(i+I2)*J+j;
#ifdef PROBE_ENUM
            myhit++;
#endif
            SIEVE_ADD((unsigned*)&arr[off&~3ull],(unsigned)lp<<(8*(off&3)));
          } } }
#else
    { const int ip1=(int)p1,iq1=(int)q1,ip2=(int)p2,iq2=(int)q2;
#ifdef OFF_STRENGTH
      const int doff=ip2*J+iq2;    // per-line constant step through the sieve array
#endif
      for(int c1=(int)A1;c1<=(int)A2;c1++){
        int u=c1*ip1, v=c1*iq1, lo=(int)B1, hi=(int)B2;
        EXACT_C2_RANGE(u,v,ip2,iq2,lo,hi,continue);
        int i=u+lo*ip2, j=v+lo*iq2;
#ifdef PROBE_ENUM
        if(hi>=lo) myenum+=(unsigned long long)(hi-lo+1);
#endif
#ifdef OFF_STRENGTH
        // Strength-reduced offset. EXACT_C2_RANGE guarantees every c2 in [lo,hi] is in-region, so
        // off advances by the per-LINE constant doff = ip2*J + iq2 -- no need to recompute
        // (i+I2)*J+j per point. Also int32: off < 4096*4096 = 2^24, so the size_t (64-bit) multiply
        // and adds were both unnecessary. Same points, same order -> relations identical.
        { int off=(i+I2)*J+j;
          for(int c2=lo;c2<=hi;c2++,off+=doff){
#ifdef PROBE_ENUM
            myhit++;
#endif
#ifdef PROBE_NOATOMIC
            sink+=(unsigned)off;
#else
            SIEVE_ADD((unsigned*)&arr[off&~3],(unsigned)lp<<(8*(off&3)));
#endif
          } } } }
#else
        for(int c2=lo;c2<=hi;c2++,i+=ip2,j+=iq2){ size_t off=(size_t)(i+I2)*J+j;
#ifdef PROBE_ENUM
          myhit++;
#endif
#ifdef PROBE_NOATOMIC
          sink+=off;
#else
          SIEVE_ADD((unsigned*)&arr[off&~3ull],(unsigned)lp<<(8*(off&3)));
#endif
        } } }
#endif
#endif
  }
#ifdef PROBE_ENUM
  atomicAdd(&g_enum_cnt,myenum); atomicAdd(&g_hit_cnt,myhit);
#endif
#if defined(PROBE_NOATOMIC) || defined(PROBE_NOBOUNDS) || defined(PROBE_NOENUM) || defined(PROBE_NOSTORE)
  if(sink==0xDEADBEEFCAFEULL) arr[0]=1;   // consume sink so the loop can't be optimized away
#endif
}
// ---- bucket sieve: lat scatter via shared-memory accumulation -------------------------
// Measured a REGRESSION vs scat_lat (Pipeline.md 3b); compiled only under -DBUCKET_SIEVE.
// NOTE: bucket_flush_lat's static shmem tile is 64KB at STRIPE_H=16, over sm_120's 48KB
// static cap -> that build needs STRIPE_H<=8 (or dynamic shmem). Default build uses scat_lat.
#ifdef BUCKET_SIEVE
// Phase 1: enumerate lattice points per large prime; store (off<<8|logp) in stripe bucket.
__global__ void bucket_fill_lat(
    const uint32_t* M,const uint32_t* R,const uint8_t* L,int n,
    uint32_t* bdata,uint32_t* bcnt,int* BA){
  int line=blockIdx.x*blockDim.x+threadIdx.x; if(line>=n)return;
  uint32_t rr=R[line]; if(rr==0xFFFFFFFFu)return;
  long long m=M[line]; unsigned lp=L[line];
  if(rr&0x80000000u){ uint32_t step=rr&0x7FFFFFFFu;
    for(int i=-I2;i<I2;i++) if((((i%(int)step)+(int)step)%(int)step)==0)
      for(int j=0;j<J;j++){ int row=i+I2; int stripe=row/STRIPE_H;
        uint32_t off=(uint32_t)(row%STRIPE_H)*J+(uint32_t)j;
        uint32_t pos=atomicAdd(&bcnt[stripe],1u);
        if(pos<BUCKET_CAP) bdata[(size_t)stripe*BUCKET_CAP+pos]=(off<<8)|lp; }
    return; }
  long long r=rr,p1=1,q1=r,p2=0,q2=m;
  for(int it=0;it<64;it++){ long long n1=p1*p1+q1*q1,n2=p2*p2+q2*q2;
    if(n2<n1){long long t=p1;p1=p2;p2=t;t=q1;q1=q2;q2=t;n1=n2;} if(!n1)break;
    long long dot=p1*p2+q1*q2,mu;
    if(dot>=0)mu=(2*dot+n1)/(2*n1);else mu=-((-2*dot+n1)/(2*n1));
    if(!mu)break; p2-=mu*p1;q2-=mu*q1; }
  BA[(size_t)line*4]=(int)p1;BA[(size_t)line*4+1]=(int)q1;
  BA[(size_t)line*4+2]=(int)p2;BA[(size_t)line*4+3]=(int)q2;
  long long D=p1*q2-p2*q1; if(!D)return;
  int cI[2]={-I2,I2-1},cJ[2]={0,J-1};
  long long a1m=1LL<<62,a1M=-(1LL<<62),b1m=1LL<<62,b1M=-(1LL<<62);
  for(int x=0;x<2;x++)for(int y=0;y<2;y++){ long long I_=cI[x],J_=cJ[y];
    long long n1c=q2*I_-p2*J_,n2c=-q1*I_+p1*J_;
    a1m=min(a1m,n1c);a1M=max(a1M,n1c);b1m=min(b1m,n2c);b1M=max(b1M,n2c); }
  long long A1,A2,B1,B2;
  if(D>0){A1=fdv(a1m,D);A2=-fdv(-a1M,D);B1=fdv(b1m,D);B2=-fdv(-b1M,D);}
  else{A1=fdv(a1M,D);A2=-fdv(-a1m,D);B1=fdv(b1M,D);B2=-fdv(-b1m,D);}
  for(long long c1=A1;c1<=A2;c1++){ long long i=c1*p1+B1*p2,j=c1*q1+B1*q2;
    for(long long c2=B1;c2<=B2;c2++,i+=p2,j+=q2)
      if(i>=-I2&&i<I2&&j>=0&&j<J){ int row=(int)i+I2; int stripe=row/STRIPE_H;
        uint32_t off=(uint32_t)(row%STRIPE_H)*J+(uint32_t)j;
        uint32_t pos=atomicAdd(&bcnt[stripe],1u);
        if(pos<BUCKET_CAP) bdata[(size_t)stripe*BUCKET_CAP+pos]=(off<<8)|lp; } }
}
// Phase 2: one block per stripe; load bucket into shared-memory tile; write back to arr.
// Shared atomics to tile[]: ~1 cycle vs ~358 cycle L2 atomics. Non-atomic write-back
// is safe: scat_col finished (same stream) and only this block owns this stripe.
__global__ void bucket_flush_lat(uint8_t* arr,const uint32_t* bdata,const uint32_t* bcnt){
  __shared__ uint32_t tile[STRIPE_SIZE/4];  // 16384 uint32 = 64KB shmem per block
  int stripe=blockIdx.x;
  for(int t=threadIdx.x;t<STRIPE_SIZE/4;t+=blockDim.x) tile[t]=0;
  __syncthreads();
  uint32_t cnt=bcnt[stripe]; if(cnt>BUCKET_CAP)cnt=BUCKET_CAP;
  const uint32_t* base=bdata+(size_t)stripe*BUCKET_CAP;
  for(uint32_t t=threadIdx.x;t<cnt;t+=blockDim.x){
    uint32_t e=base[t],off=e>>8,lp=e&0xFF;
    atomicAdd(&tile[off>>2],lp<<((off&3)*8)); }  // packed-byte shmem atomic
  __syncthreads();
  uint32_t* arr32=(uint32_t*)(arr+(size_t)stripe*STRIPE_SIZE);
  for(int t=threadIdx.x;t<STRIPE_SIZE/4;t+=blockDim.x) arr32[t]+=tile[t];
}
#endif  /* BUCKET_SIEVE */
#ifdef PROBE_WARP
// Exact warp-utilisation probe for scat_lat's box walk. Per line compute the total box
// iteration count T=(A2-A1+1)*(B2-B1+1) (0 if the line is skipped), then reduce ACROSS THE
// WARP: a warp costs max(T) iterations (it runs until its slowest lane finishes) while doing
// only sum(T) useful ones. utilisation = sum/(32*max). All 32 lanes reach the reduction --
// no `continue` -- so the shuffle is well-formed.
__device__ unsigned long long g_warp_max, g_warp_sum;
__global__ void probe_warp_util(const uint32_t* M,const uint32_t* R,int n){
  for(int line=blockIdx.x*blockDim.x+threadIdx.x;line<n;line+=gridDim.x*blockDim.x){
    unsigned long long T=0;
    long long m=M[line]; uint32_t rr=R[line];
    if(rr!=0xFFFFFFFFu && !(rr&0x80000000u)){
      long long r=rr,p1=1,q1=r,p2=0,q2=m;
      for(int it=0;it<64;it++){ long long n1=p1*p1+q1*q1,n2=p2*p2+q2*q2;
        if(n2<n1){long long a=p1;p1=p2;p2=a;a=q1;q1=q2;q2=a;n1=n2;} if(!n1)break;
        long long dot=p1*p2+q1*q2,mu; if(dot>=0)mu=(2*dot+n1)/(2*n1);else mu=-((-2*dot+n1)/(2*n1));
        if(!mu)break; p2-=mu*p1;q2-=mu*q1; }
      long long D=p1*q2-p2*q1;
      if(D){
        int cI[2]={-I2,I2-1},cJ[2]={0,J-1};
        long long a1m=1LL<<62,a1M=-(1LL<<62),b1m=1LL<<62,b1M=-(1LL<<62);
        for(int x=0;x<2;x++)for(int y=0;y<2;y++){ long long I_=cI[x],J_=cJ[y];
          long long n1c=q2*I_-p2*J_,n2c=-q1*I_+p1*J_;
          a1m=min(a1m,n1c);a1M=max(a1M,n1c);b1m=min(b1m,n2c);b1M=max(b1M,n2c); }
        long long A1,A2,B1,B2;
        if(D>0){A1=fdv(a1m,D);A2=-fdv(-a1M,D);B1=fdv(b1m,D);B2=-fdv(-b1M,D);}
        else   {A1=fdv(a1M,D);A2=-fdv(-a1m,D);B1=fdv(b1M,D);B2=-fdv(-b1m,D);}
        if(A2>=A1&&B2>=B1) T=(unsigned long long)(A2-A1+1)*(unsigned long long)(B2-B1+1);
      }
    }
    unsigned long long mx=T,sm=T;
    for(int o=16;o;o>>=1){
      unsigned long long om=__shfl_xor_sync(0xffffffffu,mx,o), os=__shfl_xor_sync(0xffffffffu,sm,o);
      if(om>mx)mx=om; sm+=os; }
    if((threadIdx.x&31)==0){ atomicAdd(&g_warp_max,mx); atomicAdd(&g_warp_sum,sm); }
  }
}
#endif
// ---- COEFFICIENT SCALING (2026-08-08). !! DO NOT REMOVE -- THIS IS A ZERO-RELATION BUG FIX !! ----
// The Horner evaluation below IS overflow-safe in t (t in [-1,1] cannot grow the value), which is
// what the old "FP32-safe, no overflow" comment meant. What it missed: the COEFFICIENTS THEMSELVES
// are cast to float by the caller, and for a high-skew c151 poly |c0| ~ c5*skew^5 reaches 5-6e39 --
// past FLT_MAX (3.4e38). c0 becomes +-inf BEFORE any arithmetic runs, h becomes inf, this returns
// inf, and the survivor gate `nla-la[k] <= 58+slack+SCATOFF` is then false for EVERY cell: the
// lattice yields ZERO survivors and the sieve reports 0 relations for an otherwise fine polynomial.
// Measured 2026-08-08 on a real 500-bit challenge: 4 of the 6 bake-off candidates overflowed and
// scored exactly 0.00 rel/lat, including the SECOND-BEST candidate by Murphy E. The bake-off cannot
// tell that from a genuine zero, so it silently selected from 2 candidates instead of 6.
// Fix: the host scales every coefficient by an exact power of two (2^-CSCALE, mantissa untouched, so
// no precision is lost) and this adds CSCALE back to the log. For any poly that did NOT overflow the
// host sets CSCALE=0 and this path is byte-identical to before.
__device__ __constant__ float d_clog2off = 0.f;   // = CSCALE, added back to every log2 result
static float h_clog2off = 0.f;                    // host mirror (fflog2 is also called host-side)
// ---- scan: survivors. log|F_f| via factoring out the dominant variable (FP32-safe, no overflow) ----
__host__ __device__ __forceinline__ float fflog2(float c0,float c1,float c2,float c3,float c4,float c5,float a,float b){
  // log2|c5 a^5 + ... + c0 b^5| = 5 log2|u| + log2|poly(t)|, u=max(|a|,|b|), t=min/max in [-1,1]
#ifdef __CUDA_ARCH__
  const float OFF = d_clog2off;
#else
  const float OFF = h_clog2off;
#endif
  float A=fabsf(a),B=fabsf(b); float u=A>B?A:B; if(u==0)return -1e30f;
  if(A>=B){ float t=b/a; float h=c5+t*(c4+t*(c3+t*(c2+t*(c1+t*c0)))); return 5.f*log2f(A)+log2f(fabsf(h))+OFF; }
  else    { float t=a/b; float h=c0+t*(c1+t*(c2+t*(c3+t*(c4+t*c5)))); return 5.f*log2f(B)+log2f(fabsf(h))+OFF; }
}
__global__ void scan(const uint8_t* la,const uint8_t* lr,unsigned* nsurv,
                     float a0,float b0,float a1,float b1,float logq,
                     float c0,float c1,float c2,float c3,float c4,float c5,float Y0,float Y1){
  for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<NCELL;k+=(size_t)gridDim.x*blockDim.x){
    int i=(int)(k/J)-I2, j=(int)(k%J); float a=a0*i+a1*j,b=b0*i+b1*j; if(b==0)continue;
    float nla=fflog2(c0,c1,c2,c3,c4,c5,a,b)-logq*SQ_ADJ_A;
    float Ga=Y1*a+Y0*b; float nlr=log2f(fabsf(Ga))-logq*SQ_ADJ_R;
    if(nla-la[k]<=78.f && nlr-lr[k]<=77.f) atomicAdd(nsurv,1u);
  }
}
// scan that ASSIGNS a survivor index per cell (-1 if not survivor)
__global__ void scan_idx(const uint8_t* la,const uint8_t* lr,int* sidx,unsigned* cnt,int slack,
                     float a0,float b0,float a1,float b1,float logq,
                     float c0,float c1,float c2,float c3,float c4,float c5,float Y0,float Y1){
  for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<NCELL;k+=(size_t)gridDim.x*blockDim.x){
    int i=(int)(k/J)-I2,j=(int)(k%J); float a=a0*i+a1*j,b=b0*i+b1*j; int keep=0;
    if(b!=0){ float nla=fflog2(c0,c1,c2,c3,c4,c5,a,b)-logq*SQ_ADJ_A, Ga=Y1*a+Y0*b, nlr=log2f(fabsf(Ga))-logq*SQ_ADJ_R;
      if(nla-la[k]<=(float)(58+slack+SCATOFF) && nlr-lr[k]<=(float)(57+slack+SCATOFF+RAT_XSLACK)) keep=1; }
    sidx[k]= keep? (int)atomicAdd(cnt,1u) : -1;
  }
}
// survivor gate: bit-packed mask (4.2MB, L2-resident) -> 99.7% non-survivor hits caught in L2,
// only real survivors pay the 134MB sidx[] DRAM lookup. Built once after scan_idx.
__global__ void mkmask(const int* sidx,uint32_t* bits){
  for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<NCELL;k+=(size_t)gridDim.x*blockDim.x)
    if(sidx[k]>=0) atomicOr(&bits[k>>5],1u<<(k&31)); }
__device__ __forceinline__ void rec(const uint32_t* bits,const int* sidx,uint32_t* plist,unsigned* pcnt,int i,int j,uint32_t p){
  size_t c=(size_t)(i+I2)*J+j;
  if(!((bits[c>>5]>>(c&31))&1u))return;                                 // L2-resident gate
  int s=sidx[c]; if(s<0)return; unsigned pos=atomicAdd(&pcnt[s],1u); if(pos<MAXP) plist[(size_t)s*MAXP+pos]=p; }

// ================= FLAT_WALK: uniform-work decomposition of the lat scatter/resieve =========
// MEASURED PROBLEM: scat_lat runs one thread per line. The box trip count T=(A2-A1+1)*(B2-B1+1)
// depends on the reduced-basis SHAPE, which varies chaotically with the root -- so 32 lanes of a
// warp get 32 unrelated trip counts and the warp runs until its slowest lane finishes.
// Probe measured 187.6M useful lane-iters vs 541.7M warp-cost => WARP UTILISATION 34.6%.
// The hardware runs scat_lat's exact access+branch pattern at 105 G atomics/s (divroof.cu) while
// scat_lat achieves only 11.7 G/s -- so this is inefficiency, not physics.
//
// FIX: flatten. Phase A computes basis+bounds+T per line. Phase B prefix-sums T. Phase C gives
// every thread FLAT_CHUNK consecutive iterations of the global (line,c1,c2) space, located by one
// binary search -- every lane then does identical work. resieve reuses Phase A/B output unchanged.
// Visits exactly the same (line,c1,c2) triples as the box walk -> relations identical.
#ifndef FLAT_CHUNK
#define FLAT_CHUNK 32
#endif
// BND[line*4] = {A1, B1, ncol=(B2-B1+1), nrow=(A2-A1+1)}
__global__ void setup_lat(const uint32_t* M,const uint32_t* R,int n,int* BA,int* BND,
                          unsigned long long* TC,int* vlist,unsigned* vcnt){
  for(int line=blockIdx.x*blockDim.x+threadIdx.x;line<n;line+=gridDim.x*blockDim.x){
    TC[line]=0; BND[line*4+2]=0; BND[line*4+3]=0;
    long long m=M[line]; uint32_t rr=R[line];
    if(rr==0xFFFFFFFFu) continue;
    if(rr&0x80000000u){ unsigned k=atomicAdd(vcnt,1u); if(k<4096u) vlist[k]=line; continue; }
    long long r=rr,p1=1,q1=r,p2=0,q2=m;
    for(int it=0;it<64;it++){ long long n1=p1*p1+q1*q1,n2=p2*p2+q2*q2;
      if(n2<n1){long long a=p1;p1=p2;p2=a;a=q1;q1=q2;q2=a;n1=n2;} if(!n1)break;
      long long dot=p1*p2+q1*q2,mu; if(dot>=0)mu=(2*dot+n1)/(2*n1);else mu=-((-2*dot+n1)/(2*n1));
      if(!mu)break; p2-=mu*p1;q2-=mu*q1; }
    BA[line*4]=(int)p1;BA[line*4+1]=(int)q1;BA[line*4+2]=(int)p2;BA[line*4+3]=(int)q2;
    long long D=(long long)p1*q2-(long long)p2*q1; if(!D)continue;
    int cI[2]={-I2,I2-1},cJ[2]={0,J-1};
    long long a1m=1LL<<62,a1M=-(1LL<<62),b1m=1LL<<62,b1M=-(1LL<<62);
    for(int x=0;x<2;x++)for(int y=0;y<2;y++){ long long I_=cI[x],J_=cJ[y];
      long long n1c=q2*I_-p2*J_,n2c=-q1*I_+p1*J_;
      a1m=min(a1m,n1c);a1M=max(a1M,n1c);b1m=min(b1m,n2c);b1M=max(b1M,n2c); }
    long long A1,A2,B1,B2;
    if(D>0){A1=fdv(a1m,D);A2=-fdv(-a1M,D);B1=fdv(b1m,D);B2=-fdv(-b1M,D);}
    else   {A1=fdv(a1M,D);A2=-fdv(-a1m,D);B1=fdv(b1M,D);B2=-fdv(-b1m,D);}
    if(A2<A1||B2<B1) continue;
    long long nrow=A2-A1+1, ncol=B2-B1+1;
    BND[line*4]=(int)A1; BND[line*4+1]=(int)B1; BND[line*4+2]=(int)ncol; BND[line*4+3]=(int)nrow;
    TC[line]=(unsigned long long)nrow*(unsigned long long)ncol;
  }
}
__global__ void fin_off(unsigned long long* Off,const unsigned long long* TC,int n){ Off[n]=Off[n-1]+TC[n-1]; }
__global__ void vert_scat(uint8_t* arr,const uint32_t* R,const uint8_t* L,const int* vlist,const unsigned* vcnt){
  unsigned nv=*vcnt; if(nv>4096u)nv=4096u;
  for(unsigned k=blockIdx.x*blockDim.x+threadIdx.x;k<nv;k+=gridDim.x*blockDim.x){
    int line=vlist[k]; uint32_t step=R[line]&0x7FFFFFFFu; unsigned lp=L[line];
    for(int i=-I2;i<I2;i++) if((((i%(int)step)+(int)step)%(int)step)==0)
      for(int j=0;j<J;j++){ size_t idx=(size_t)(i+I2)*J+j;
        SIEVE_ADD((unsigned*)&arr[idx&~3ull],(unsigned)lp<<(8*(idx&3))); } }
}
__device__ __forceinline__ int flat_find_line(const unsigned long long* Off,int n,unsigned long long g){
  int lo=0,hi=n-1,res=0;
  while(lo<=hi){ int mid=(lo+hi)>>1; if(Off[mid]<=g){res=mid;lo=mid+1;} else hi=mid-1; }
  return res;
}
__global__ void walk_scat(uint8_t* arr,const uint8_t* L,const int* BA,const int* BND,
                          const unsigned long long* Off,int n){
  const unsigned long long total=Off[n];
  for(unsigned long long base=(unsigned long long)(blockIdx.x*blockDim.x+threadIdx.x)*FLAT_CHUNK;
      base<total; base+=(unsigned long long)gridDim.x*blockDim.x*FLAT_CHUNK){
    unsigned long long g=base, end=min(base+(unsigned long long)FLAT_CHUNK,total);
    int line=flat_find_line(Off,n,g);
    while(g<end){
      while(line+1<n && Off[line+1]<=g) line++;
      unsigned long long le=Off[line+1]; if(le>end)le=end;
      unsigned long long take=le-g; if(!take){ g++; continue; }
      int A1=BND[line*4],B1=BND[line*4+1],ncol=BND[line*4+2];
      if(ncol<=0){ g+=take; continue; }
      int p1=BA[line*4],q1=BA[line*4+1],p2=BA[line*4+2],q2=BA[line*4+3];
      unsigned lp=L[line];
      unsigned long long loc=g-Off[line];
      int c1=A1+(int)(loc/(unsigned)ncol), c2=B1+(int)(loc%(unsigned)ncol);
      int i=c1*p1+c2*p2, j=c1*q1+c2*q2, cend=B1+ncol-1;
      for(unsigned long long k=0;k<take;k++){
        if(i>=-I2&&i<I2&&j>=0&&j<J){ size_t off=(size_t)(i+I2)*J+j;
          SIEVE_ADD((unsigned*)&arr[off&~3ull],(unsigned)lp<<(8*(off&3))); }
        if(c2<cend){ c2++; i+=p2; j+=q2; }
        else { c2=B1; c1++; i=c1*p1+B1*p2; j=c1*q1+B1*q2; }
      }
      g+=take;
    }
  }
}
__global__ void vert_resieve(const uint32_t* M,const uint32_t* P,const uint32_t* R,const int* vlist,
                             const unsigned* vcnt,const uint32_t* bits,const int* sidx,uint32_t* plist,unsigned* pcnt){
  unsigned nv=*vcnt; if(nv>4096u)nv=4096u;
  for(unsigned k=blockIdx.x*blockDim.x+threadIdx.x;k<nv;k+=gridDim.x*blockDim.x){
    int line=vlist[k]; uint32_t m=M[line],pb=P[line]; if(m!=pb)continue;
    uint32_t st=R[line]&0x7FFFFFFFu;
    for(int i=-I2;i<I2;i++) if((((i%(int)st)+(int)st)%(int)st)==0)
      for(int j=0;j<J;j++) rec(bits,sidx,plist,pcnt,i,j,pb); }
}
__global__ void walk_resieve(const uint32_t* M,const uint32_t* P,const int* BA,const int* BND,
                             const unsigned long long* Off,int n,const uint32_t* bits,const int* sidx,
                             uint32_t* plist,unsigned* pcnt){
  const unsigned long long total=Off[n];
  for(unsigned long long base=(unsigned long long)(blockIdx.x*blockDim.x+threadIdx.x)*FLAT_CHUNK;
      base<total; base+=(unsigned long long)gridDim.x*blockDim.x*FLAT_CHUNK){
    unsigned long long g=base, end=min(base+(unsigned long long)FLAT_CHUNK,total);
    int line=flat_find_line(Off,n,g);
    while(g<end){
      while(line+1<n && Off[line+1]<=g) line++;
      unsigned long long le=Off[line+1]; if(le>end)le=end;
      unsigned long long take=le-g; if(!take){ g++; continue; }
      uint32_t m=M[line],pb=P[line];
      if(m!=pb){ g+=take; continue; }
      int A1=BND[line*4],B1=BND[line*4+1],ncol=BND[line*4+2];
      if(ncol<=0){ g+=take; continue; }
      int p1=BA[line*4],q1=BA[line*4+1],p2=BA[line*4+2],q2=BA[line*4+3];
      unsigned long long loc=g-Off[line];
      int c1=A1+(int)(loc/(unsigned)ncol), c2=B1+(int)(loc%(unsigned)ncol);
      int i=c1*p1+c2*p2, j=c1*q1+c2*q2, cend=B1+ncol-1;
      for(unsigned long long k=0;k<take;k++){
        if(i>=-I2&&i<I2&&j>=0&&j<J) rec(bits,sidx,plist,pcnt,i,j,pb);
        if(c2<cend){ c2++; i+=p2; j+=q2; }
        else { c2=B1; c1++; i=c1*p1+B1*p2; j=c1*q1+B1*q2; }
      }
      g+=take;
    }
  }
}

// ============ FLAT_COOP: per-warp cooperative box walk for the lat scatter =================
// Fixes scat_lat's 34.6% warp utilisation WITHOUT losing coalescing. Each lane still owns one
// line (coalesced M/R/L loads + reduction), then the 32 lanes publish their box params to shared
// memory and COOPERATIVELY walk the warp's combined (line,c1,c2) space: lane L processes the
// contiguous slice [L*tot/32,(L+1)*tot/32), reading whichever line owns each point from shared.
// Every lane does tot/32 iterations -> uniform work; loads stay coalesced; redistribution is
// shared-memory (no global scatter like walk_scat). Also emits BA/BND/TC so resieve reuses them.
// Same (line,c1,c2) triples as the box walk -> relations identical.
// COOP_REDUCE_CAP bounds scat_lat_coop's Lagrange trip count, the same divergence tax REDUCE_CAP
// caps in scat_lat: a warp pays max(trip) across its 32 lanes, so the slow-converging tail is
// billed to every lane. Kept as its OWN knob (defaulting to the historical hardcoded 64) so the
// reduction rewrite and the cap can be measured apart -- REDUCE_CAP's measured value of 32 was
// tuned against scat_lat's double-norm loop, not this one.
#ifndef COOP_REDUCE_CAP
#define COOP_REDUCE_CAP 64
#endif
__global__ void scat_lat_coop(uint8_t* arr,const uint32_t* M,const uint32_t* R,const uint8_t* L,int n,
                              int* BA,int* BND,unsigned long long* TC,int* vlist,unsigned* vcnt){
  __shared__ int sA1[256],sB1[256],sNc[256],sP1[256],sQ1[256],sP2[256],sQ2[256];
  __shared__ unsigned sLp[256];
  __shared__ unsigned long long sOff[256];
  int tIdx=threadIdx.x, lane=tIdx&31, wbase=tIdx&~31;
  int gwarp=(blockIdx.x*blockDim.x+tIdx)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  for(int lineBase=gwarp*32; lineBase<n; lineBase+=nwarp*32){
    int line=lineBase+lane;
    unsigned long long T=0; int A1=0,B1=0,ncol=0,p1=0,q1=0,p2=0,q2=0; unsigned lp=0;
    if(line<n){
      long long m=M[line]; uint32_t rr=R[line]; lp=L[line];
      if(BND){ TC[line]=0; BND[line*4+2]=0; BND[line*4+3]=0; }
      if(rr==0xFFFFFFFFu){}
      else if(rr&0x80000000u){ uint32_t step=rr&0x7FFFFFFFu;
        // BND set -> defer to vert_scat (the original SCAT_COOP contract). BND NULL -> scatter the
        // vertical line HERE, exactly as scat_lat does, so the coop kernel is a drop-in for
        // scat_lat's (BA-only) call signature and needs NO vert_scat launch and NO vlist/TC/BND
        // writes. That overhead is why the coop scatter measured 6.8% faster on the kernel yet
        // 1.1% SLOWER end-to-end (8.95 -> 9.05 ms/lat); removing it should net the gain.
        if(BND){ unsigned k=atomicAdd(vcnt,1u); if(k<4096u) vlist[k]=line; }
        else { for(int i=-I2;i<I2;i++) if((((i%(int)step)+(int)step)%(int)step)==0) for(int j=0;j<J;j++){ size_t idx=(size_t)(i+I2)*J+j; SIEVE_ADD((unsigned*)&arr[idx&~3ull],(unsigned)lp<<(8*(idx&3))); } }
      }
      else{
#ifdef PROBE_COOP_NORED
        p1=1;q1=0;p2=0;q2=1; ncol=0; T=0;   // skip reduction+bounds entirely (measures loads+publish+prefixsum)
        // BA IS NOT OPTIONAL: resieve_coop recomputes the lattice from it, so gating it on BND
        // made a BA-only (scat_lat-shaped) call emit a garbage basis -> 0.1 rel/lattice.
        BA[line*4]=1;BA[line*4+1]=0;BA[line*4+2]=0;BA[line*4+3]=1;
      }
#else
#ifdef COOP_DBLRED
        // all-double Lagrange reduction. Norms reach ~m^2 (~2^52 < 2^53) so doubles are exact for
        // the comparisons; mu is unimodular for ANY integer, so rounding choice can't change the
        // lattice -> relations identical. Replaces the int64 software division (per iter, per line).
        int P1=1,Q1=(int)rr,P2=0,Q2=(int)m;
        for(int it=0;it<64;it++){
          double n1=(double)P1*P1+(double)Q1*Q1, n2=(double)P2*P2+(double)Q2*Q2;
          if(n2<n1){int a=P1;P1=P2;P2=a;a=Q1;Q1=Q2;Q2=a; n1=n2;}
          if(n1==0.0)break;
          double dot=(double)P1*P2+(double)Q1*Q2;
          double mud=dot/n1; int mu=(int)(mud>=0?floor(mud+0.5):ceil(mud-0.5));
          if(!mu)break; P2-=mu*P1;Q2-=mu*Q1;
        }
        p1=P1;q1=Q1;p2=P2;q2=Q2;
#elif defined(COOP_FASTRED)
        // ---- 2026-08-10. scat_lat_coop is the DEFAULT scatter kernel (FLAT_COOP2 +
        // SCAT_COOP_LEAN) yet it still runs the ORIGINAL int64 reduction below, with a 64-bit
        // SOFTWARE division per iteration per line and a trip count hardcoded to 64 -- neither
        // the double-norm rewrite nor REDUCE_CAP, both of which scat_lat received, was ever
        // ported here. That is why the `NOWALK` probe still reports basis-reduction setup at
        // 2.08 ms/lattice / 26% of GPU time (Pipeline.md §14e).
        //
        // int32 basis + EXACT int64 norms + ONE FLOAT divide:
        //   * |P|,|Q| stay <= m <= LIM < 2^26, so the basis fits int32 (same bound scat_lat
        //     already relies on) and every norm/dot fits int64 EXACTLY -- and `(long long)P1*P1`
        //     on 32-bit operands is a single IMAD.WIDE, not a 64-bit multiply sequence.
        //   * only mu needs a division, and mu is UNIMODULAR FOR ANY INTEGER VALUE: p2-=mu*p1 is
        //     determinant-preserving however mu is rounded, so the lattice -- and therefore every
        //     relation -- is identical no matter how imprecise the quotient is. That is what lets
        //     the int64 software divide become a full-rate float divide. Note this argument does
        //     NOT extend to the norm comparison, which is why the norms stay exact int64 rather
        //     than going float: `n2<n1` picking the wrong vector would change the trip count.
        //   * FP64 is 1:64 on sm_120, so COOP_DBLRED above is the wrong destination -- a double
        //     divide here is slower than the float one, not faster.
        // Relation-neutral by construction; verify with the sorted-set md5 A/B (Pipeline.md §5).
        {
          int P1=1,Q1=(int)rr,P2=0,Q2=(int)m;
          for(int it=0;it<COOP_REDUCE_CAP;it++){
            long long n1=(long long)P1*P1+(long long)Q1*Q1;
            long long n2=(long long)P2*P2+(long long)Q2*Q2;
            if(n2<n1){ int a=P1;P1=P2;P2=a; a=Q1;Q1=Q2;Q2=a; n1=n2; }
            if(!n1)break;
            long long dot=(long long)P1*P2+(long long)Q1*Q2;
            float mud=(float)dot/(float)n1;
            int mu=(int)(mud>=0.0f?floorf(mud+0.5f):ceilf(mud-0.5f));
            if(!mu)break; P2-=mu*P1; Q2-=mu*Q1;
          }
          p1=P1;q1=Q1;p2=P2;q2=Q2;
        }
#else
        long long r=rr,P1=1,Q1=r,P2=0,Q2=m;
        for(int it=0;it<64;it++){ long long n1=P1*P1+Q1*Q1,n2=P2*P2+Q2*Q2;
          if(n2<n1){long long a=P1;P1=P2;P2=a;a=Q1;Q1=Q2;Q2=a;n1=n2;} if(!n1)break;
          long long dot=P1*P2+Q1*Q2,mu; if(dot>=0)mu=(2*dot+n1)/(2*n1);else mu=-((-2*dot+n1)/(2*n1));
          if(!mu)break; P2-=mu*P1;Q2-=mu*Q1; }
        p1=(int)P1;q1=(int)Q1;p2=(int)P2;q2=(int)Q2;
#endif
        BA[line*4]=p1;BA[line*4+1]=q1;BA[line*4+2]=p2;BA[line*4+3]=q2;   // unconditional: see above
#ifdef COOP_I32BND
        // ---- 2026-08-10. The int32-bounds rewrite that scat_lat (line ~698) and resieve_coop
        // (line ~1640) both carry was never ported to scat_lat_coop, which is the kernel the
        // DEFAULT path actually launches. The int64 form below issues four `fdv` calls, and each
        // one is a 64-bit SOFTWARE div PLUS a software mod -- up to eight per line, ~6M lines per
        // lattice, on top of the reduction. Same bound as the two kernels that already do this:
        // reduced basis ~sqrt(m) < 2^14, det D = +-m < 2^27, corners q2*I-p2*J < 2^27, so every
        // intermediate fits int32 and the divides become hardware ones.
        // Arithmetically identical to the int64 form (no value comes near 2^31), hence
        // relation-identical -- verify with the sorted-set md5 A/B anyway.
        int D=p1*q2-p2*q1;
        if(D){
          int cI[2]={-I2,I2-1},cJ[2]={0,J-1};
          int a1m=0x3fffffff,a1M=-0x3fffffff,b1m=0x3fffffff,b1M=-0x3fffffff;
          for(int x=0;x<2;x++)for(int y=0;y<2;y++){ int I_=cI[x],J_=cJ[y];
            int n1c=q2*I_-p2*J_,n2c=-q1*I_+p1*J_;
            a1m=min(a1m,n1c);a1M=max(a1M,n1c);b1m=min(b1m,n2c);b1M=max(b1M,n2c); }
          int AA1,AA2,BB1,BB2;
          if(D>0){AA1=fdiv_i(a1m,D);AA2=-fdiv_i(-a1M,D);BB1=fdiv_i(b1m,D);BB2=-fdiv_i(-b1M,D);}
          else   {AA1=fdiv_i(a1M,D);AA2=-fdiv_i(-a1m,D);BB1=fdiv_i(b1M,D);BB2=-fdiv_i(-b1m,D);}
          if(AA2>=AA1&&BB2>=BB1){ A1=AA1;B1=BB1;ncol=BB2-BB1+1; int nrow=AA2-AA1+1;
            T=(unsigned long long)nrow*(unsigned long long)ncol;
            if(BND){ BND[line*4]=A1;BND[line*4+1]=B1;BND[line*4+2]=ncol;BND[line*4+3]=nrow; TC[line]=T; } }
        }
#else
        long long D=(long long)p1*q2-(long long)p2*q1;
        if(D){
          int cI[2]={-I2,I2-1},cJ[2]={0,J-1};
          long long a1m=1LL<<62,a1M=-(1LL<<62),b1m=1LL<<62,b1M=-(1LL<<62);
          for(int x=0;x<2;x++)for(int y=0;y<2;y++){ long long I_=cI[x],J_=cJ[y];
            long long n1c=q2*I_-p2*J_,n2c=-q1*I_+p1*J_;
            a1m=min(a1m,n1c);a1M=max(a1M,n1c);b1m=min(b1m,n2c);b1M=max(b1M,n2c); }
          long long AA1,AA2,BB1,BB2;
          if(D>0){AA1=fdv(a1m,D);AA2=-fdv(-a1M,D);BB1=fdv(b1m,D);BB2=-fdv(-b1M,D);}
          else   {AA1=fdv(a1M,D);AA2=-fdv(-a1m,D);BB1=fdv(b1M,D);BB2=-fdv(-b1m,D);}
          if(AA2>=AA1&&BB2>=BB1){ A1=(int)AA1;B1=(int)BB1;ncol=(int)(BB2-BB1+1); int nrow=(int)(AA2-AA1+1);
            T=(unsigned long long)nrow*(unsigned long long)ncol;
            if(BND){ BND[line*4]=A1;BND[line*4+1]=B1;BND[line*4+2]=ncol;BND[line*4+3]=nrow; TC[line]=T; } }
        }
#endif
      }
#endif
    }
    sA1[tIdx]=A1;sB1[tIdx]=B1;sNc[tIdx]=ncol;sP1[tIdx]=p1;sQ1[tIdx]=q1;sP2[tIdx]=p2;sQ2[tIdx]=q2;sLp[tIdx]=lp;
    unsigned long long off=T;                                       // inclusive prefix sum over the warp
    for(int d=1;d<32;d<<=1){ unsigned long long v=__shfl_up_sync(0xffffffffu,off,d); if(lane>=(unsigned)d) off+=v; }
    unsigned long long tot=__shfl_sync(0xffffffffu,off,31);
    off-=T; sOff[tIdx]=off;                                         // exclusive
    __syncwarp();
#ifdef PROBE_COOP_NOWALK
    if(tot==0xFFFFFFFFFFFFFFFFULL) arr[0]=1;   // reduction+publish+prefixsum only, skip walk
    continue;
#endif
    unsigned long long g=(unsigned long long)lane*tot/32, g1=(unsigned long long)(lane+1)*tot/32;
    int owner=0; { int lo=0,hi=31; while(lo<=hi){int mid=(lo+hi)>>1; if(sOff[wbase+mid]<=g){owner=mid;lo=mid+1;}else hi=mid-1;} }
    while(g<g1){
      while(owner+1<32 && sOff[wbase+owner+1]<=g) owner++;
      unsigned long long oe=(owner+1<32)?sOff[wbase+owner+1]:tot; if(oe>g1)oe=g1;
      int oc=wbase+owner, nc=sNc[oc];
      if(nc<=0){ g=oe; continue; }
      int oA1=sA1[oc],oB1=sB1[oc],oP1=sP1[oc],oQ1=sQ1[oc],oP2=sP2[oc],oQ2=sQ2[oc]; unsigned olp=sLp[oc];
      unsigned long long loc=g-sOff[oc];
      int c1=oA1+(int)(loc/(unsigned)nc), c2=oB1+(int)(loc%(unsigned)nc);
      int i=c1*oP1+c2*oP2, j=c1*oQ1+c2*oQ2, cend=oB1+nc-1;
      unsigned long long take=oe-g;
      for(unsigned long long k=0;k<take;k++){
        if(i>=-I2&&i<I2&&j>=0&&j<J
#ifdef PRIM_SCATTER
           && !prim_skip(i,j)                     // skip non-primitive cell: saves the L2 scatter transaction
#endif
          ){ size_t o2=(size_t)(i+I2)*J+j;
#ifdef PROBE_COOP_NOATOMIC
          arr[o2&~3ull]+=(uint8_t)olp;   // non-atomic: isolate atomic-unit cost from walk
#else
          atomicAdd((unsigned*)&arr[o2&~3ull],(unsigned)olp<<(8*(o2&3)));
#endif
        }
        if(c2<cend){c2++;i+=oP2;j+=oQ2;} else {c2=oB1;c1++;i=c1*oP1+oB1*oP2;j=c1*oQ1+oB1*oQ2;}
      }
      g=oe;
    }
    __syncwarp();
  }
}
// resieve (record base prime per survivor); only p^1 lines (m==P). side->its plist.
// ---- Franke-Kleinjung "sieving by vectors" for col-group primes with m > W ----------------
// The column method costs W scans per line regardless of hit count; for m > W only J/m of columns
// contain a hit (3% at m=131072), so ~97% of those scans are wasted. FK enumerates the hits
// directly: O(log m) basis reduction per line, then O(1) per hit, in increasing j.
// Lattice here is {(i,j) : j == r*i (mod m)}; substituting rinv = r^-1 mod m gives
// i == rinv*j (mod m), the standard FK form with the CONSTRAINED coordinate i of width W.
// Verified against brute force on CPU (320 cases, 0 count mismatches) before porting.
__device__ __forceinline__ int fk_modinv(int a,int m){
  int g=m,x=0,x1=1,a1=a%m; if(a1<0)a1+=m;
  while(a1){ int q=g/a1,t=g-q*a1; g=a1;a1=t; t=x-q*x1; x=x1;x1=t; }
  if(g!=1)return 0; x%=m; if(x<0)x+=m; return x;
}
// Reduce to a0<0<=a1 with a1-a0>=Iw and |a0|,a1 minimal. Returns false if no such basis (m<=Iw).
__device__ __forceinline__ bool fk_reduce(int m,int rinv,int Iw,int*A0,int*B0,int*A1,int*B1){
  int a0=-m,b0=0,a1=rinv,b1=1;
  if(rinv<=0) return false;
  for(;;){
    int span=a1-a0;
    if(-a0>a1){ if(a1<=0)break; int kmax=(span-Iw)/a1; if(kmax<=0)break;
                int k=(-a0)/a1; if(k>kmax)k=kmax; if(k<=0)break; a0+=k*a1; b0+=k*b1; }
    else      { if(-a0<=0)break; int kmax=(span-Iw)/(-a0); if(kmax<=0)break;
                int k=a1/(-a0); if(k>kmax)k=kmax; if(k<=0)break; a1+=k*a0; b1+=k*b0; }
  }
  *A0=a0;*B0=b0;*A1=a1;*B1=b1;
  return (a1-a0>=Iw)&&b0>0&&b1>0&&a0<0&&a1>=0;
}
#define FK_NEXT(i,j,a0,b0,a1,b1) do{                                   \
  if((i) < I2-(a1)){ (i)+=(a1); (j)+=(b1); }                           \
  else if((i) >= -I2-(a0)){ (i)+=(a0); (j)+=(b0); }                    \
  else { (i)+=(a0)+(a1); (j)+=(b0)+(b1); }                             \
}while(0)
// scatter for col-group lines [lo,n) (those with m > W), one thread per line
__global__ void scat_fk(uint8_t* arr,const uint32_t* M,const uint32_t* R,const uint8_t* L,int lo,int n){
  for(int line=lo+blockIdx.x*blockDim.x+threadIdx.x;line<n;line+=gridDim.x*blockDim.x){
    uint32_t m=M[line],r=R[line]; unsigned lp=L[line];
    if(r==0xFFFFFFFFu||(r&0x80000000u))continue;      // invalid / vertical: left to scat_col
    if(m<(uint32_t)SCAT_SKIP)continue;
    int rinv=fk_modinv((int)r,(int)m); if(!rinv)continue;
    int a0,b0,a1,b1; if(!fk_reduce((int)m,rinv,W,&a0,&b0,&a1,&b1))continue;
    { size_t off=(size_t)I2*J; SIEVE_ADD((unsigned*)&arr[off&~3ull],(unsigned)lp<<(8*(off&3))); } // (i,j)=(0,0)
    int i=0,j=0;
    for(;;){ FK_NEXT(i,j,a0,b0,a1,b1); if(j>=J)break;
      size_t off=(size_t)(i+I2)*J+j;
      SIEVE_ADD((unsigned*)&arr[off&~3ull],(unsigned)lp<<(8*(off&3))); }
  }
}
__global__ void resieve_fk(const uint32_t* M,const uint32_t* P,const uint32_t* R,int lo,int n,const uint32_t* bits,const int* sidx,uint32_t* plist,unsigned* pcnt){
  for(int line=lo+blockIdx.x*blockDim.x+threadIdx.x;line<n;line+=gridDim.x*blockDim.x){
    uint32_t m=M[line],r=R[line],pb=P[line];
    if(r==0xFFFFFFFFu||m!=pb||(r&0x80000000u))continue;
    if(m<(uint32_t)SPB)continue;
    int rinv=fk_modinv((int)r,(int)m); if(!rinv)continue;
    int a0,b0,a1,b1; if(!fk_reduce((int)m,rinv,W,&a0,&b0,&a1,&b1))continue;
    rec(bits,sidx,plist,pcnt,0,0,pb);
    int i=0,j=0;
    for(;;){ FK_NEXT(i,j,a0,b0,a1,b1); if(j>=J)break; rec(bits,sidx,plist,pcnt,i,j,pb); }
  }
}
__global__ void resieve_col(const uint32_t* M,const uint32_t* P,const uint32_t* R,int n,const uint32_t* bits,const int* sidx,uint32_t* plist,unsigned* pcnt){
  for(long long w=(long long)blockIdx.x*blockDim.x+threadIdx.x; w<(long long)n*W; w+=(long long)gridDim.x*blockDim.x){
    int line=(int)(w/W),col=(int)(w%W),i=col-I2; uint32_t m=M[line],r=R[line],pb=P[line]; if(r==0xFFFFFFFFu||m!=pb)continue;
    if(m<(uint32_t)SPB)continue;     // small primes trial-divided in cofactor/CPU instead of resieved
    if(r&0x80000000u){ uint32_t st=r&0x7FFFFFFFu; if((((i%(int)st)+(int)st)%(int)st)==0) for(int j=0;j<J;j++) rec(bits,sidx,plist,pcnt,i,j,pb); continue; }
    uint32_t im=(uint32_t)(((i%(int)m)+(int)m)%(int)m), j0=(uint32_t)(((uint64_t)r*im)%m);
    for(uint32_t j=j0;j<J;j+=m) rec(bits,sidx,plist,pcnt,i,(int)j,pb);
  }
}
// TILE_RESIEVE: same column-recurrence as scat_tile, applied to the dense-group resieve.
// resieve_col pays the identical 64-bit mulmod per (line,column) -- 96 M per side for band R1 --
// and, unlike the scatter, it has no writes to hide it behind (its stores are gated by the
// survivor bitmask, so <0.1% of visits store anything). One work item now covers RCOLS
// consecutive columns of a line: one mulmod, then j0 += r (mod m) per column.
// rec() appends via atomicAdd on pcnt[s], so per-survivor prime order is already unordered and
// the host qsorts the factor lists -- changing traversal order is relation-neutral.
#ifndef RCOLS
#define RCOLS 8
#endif
static_assert(W%RCOLS==0,"RCOLS must divide W (else the tail columns are never resieved)");
__global__ void resieve_colr(const uint32_t* M,const uint32_t* P,const uint32_t* R,int n,const uint32_t* bits,const int* sidx,uint32_t* plist,unsigned* pcnt){
  const int NCG=W/RCOLS;
  for(long long w=(long long)blockIdx.x*blockDim.x+threadIdx.x; w<(long long)n*NCG; w+=(long long)gridDim.x*blockDim.x){
    int line=(int)(w/NCG),cg=(int)(w%NCG),col0=cg*RCOLS,i0=col0-I2;
    uint32_t m=M[line],r=R[line],pb=P[line]; if(r==0xFFFFFFFFu||m!=pb)continue;
    if(m<(uint32_t)SPB)continue;
    if(r&0x80000000u){ uint32_t st=r&0x7FFFFFFFu;
      for(int c=0;c<RCOLS;c++){ int i=col0+c-I2;
        if((((i%(int)st)+(int)st)%(int)st)==0) for(int j=0;j<J;j++) rec(bits,sidx,plist,pcnt,i,j,pb); }
      continue; }
    uint32_t j0=(uint32_t)(((uint64_t)r*(uint32_t)(((i0%(int)m)+(int)m)%(int)m))%m);
    for(int c=0;c<RCOLS;c++){ int i=col0+c-I2;
      for(uint32_t j=j0;j<J;j+=m) rec(bits,sidx,plist,pcnt,i,(int)j,pb);
      j0+=r; if(j0>=m)j0-=m; }
  }
}
__global__ void resieve_lat(const uint32_t* M,const uint32_t* P,const uint32_t* R,int n,const uint32_t* bits,const int* sidx,uint32_t* plist,unsigned* pcnt,const int* BA){
  for(int line=blockIdx.x*blockDim.x+threadIdx.x;line<n;line+=gridDim.x*blockDim.x){
    long long m=M[line]; uint32_t rr=R[line],pb=P[line]; if(rr==0xFFFFFFFFu||(uint32_t)m!=pb)continue;
    if(rr&0x80000000u){ uint32_t st=rr&0x7FFFFFFFu; for(int i=-I2;i<I2;i++) if((((i%(int)st)+(int)st)%(int)st)==0) for(int j=0;j<J;j++) rec(bits,sidx,plist,pcnt,i,j,pb); continue; }
    long long p1=BA[(size_t)line*4],q1=BA[(size_t)line*4+1],p2=BA[(size_t)line*4+2],q2=BA[(size_t)line*4+3];  // reuse cached basis (skip redundant Gauss reduction)
    long long D=p1*q2-p2*q1; if(!D)continue;
    int cI[2]={-I2,I2-1},cJ[2]={0,J-1}; long long a1m=1LL<<62,a1M=-(1LL<<62),b1m=1LL<<62,b1M=-(1LL<<62);
    for(int x=0;x<2;x++)for(int y=0;y<2;y++){ long long I_=cI[x],J_=cJ[y]; long long n1c=q2*I_-p2*J_,n2c=-q1*I_+p1*J_; a1m=min(a1m,n1c);a1M=max(a1M,n1c);b1m=min(b1m,n2c);b1M=max(b1M,n2c); }
    long long A1,A2,B1,B2; if(D>0){A1=fdv(a1m,D);A2=-fdv(-a1M,D);B1=fdv(b1m,D);B2=-fdv(-b1M,D);}else{A1=fdv(a1M,D);A2=-fdv(-a1m,D);B1=fdv(b1M,D);B2=-fdv(-b1m,D);}
#ifdef BOX_ENUM
    for(long long c1=A1;c1<=A2;c1++){ long long i=c1*p1+B1*p2,j=c1*q1+B1*q2;          // strength-reduced: i,j incremental in c2 (md5-identical, ~2% faster)
      for(long long c2=B1;c2<=B2;c2++,i+=p2,j+=q2)
        if(i>=-I2&&i<I2&&j>=0&&j<J) rec(bits,sidx,plist,pcnt,(int)i,(int)j,pb); }
#elif defined(DENSE_ENUM)
    { const int ip1=(int)p1,iq1=(int)q1,ip2=(int)p2,iq2=(int)q2;
      C2Range R; c2range_init(&R,(int)A1,ip1,iq1,ip2,iq2);
      for(int c1=(int)A1;c1<=(int)A2;c1++,c2range_step(&R)){
        int u=c1*ip1, v=c1*iq1, lo=(int)B1, hi=(int)B2;
        c2range_get(&R,&lo,&hi);
        if(hi<lo) continue;
        int i=u+lo*ip2, j=v+lo*iq2;
        for(int c2=lo;c2<=hi;c2++,i+=ip2,j+=iq2)
          if(i>=-I2&&i<I2&&j>=0&&j<J) rec(bits,sidx,plist,pcnt,i,j,pb);
      } }
#else
    { const int ip1=(int)p1,iq1=(int)q1,ip2=(int)p2,iq2=(int)q2;
      for(int c1=(int)A1;c1<=(int)A2;c1++){
        int u=c1*ip1, v=c1*iq1, lo=(int)B1, hi=(int)B2;
        EXACT_C2_RANGE(u,v,ip2,iq2,lo,hi,continue);
        int i=u+lo*ip2, j=v+lo*iq2;
        for(int c2=lo;c2<=hi;c2++,i+=ip2,j+=iq2) rec(bits,sidx,plist,pcnt,i,j,pb);
      } }
#endif
  }
}
// CPU cofactor: divide norm by recorded primes (completely), check residual smooth in (lim,2^lpb], coprime
static int cof_smooth(mpz_t c,unsigned long lpb){
  mpz_t stk[64]; int sp=0; mpz_init_set(stk[sp++],c); int ok=1;
  while(sp>0&&ok){ mpz_t v; mpz_init_set(v,stk[--sp]); mpz_clear(stk[sp]);
    if(mpz_cmp_ui(v,1)<=0){mpz_clear(v);continue;}
    if(mpz_probab_prime_p(v,25)){ if(mpz_cmp_ui(v,LIM)<=0||mpz_sizeinbase(v,2)>lpb) ok=0; mpz_clear(v); continue; }
    if(mpz_sizeinbase(v,2)<=lpb){ ok=0; mpz_clear(v); continue; }
    mpz_t x,y,d; mpz_inits(x,y,d,NULL); mpz_set_ui(x,2);mpz_set_ui(y,2);mpz_set_ui(d,1); unsigned long cn=1; long it=0;
    while(mpz_cmp_ui(d,1)==0&&it++<(1<<22)){ mpz_mul(x,x,x);mpz_add_ui(x,x,cn);mpz_mod(x,x,v);mpz_mul(y,y,y);mpz_add_ui(y,y,cn);mpz_mod(y,y,v);mpz_mul(y,y,y);mpz_add_ui(y,y,cn);mpz_mod(y,y,v);mpz_sub(d,x,y);mpz_abs(d,d);mpz_gcd(d,d,v); if(mpz_cmp(d,v)==0){cn++;mpz_set_ui(x,2);mpz_set_ui(y,2);mpz_set_ui(d,1);} }
    if(it>=(1<<22)){ok=0;mpz_clears(x,y,d,NULL);mpz_clear(v);break;}
    if(sp<62){mpz_init_set(stk[sp++],d);mpz_divexact(v,v,d);mpz_init_set(stk[sp++],v);} mpz_clears(x,y,d,NULL);mpz_clear(v);
  }
  while(sp>0)mpz_clear(stk[--sp]); return ok;
}
// ---- GPU COFACTOR: norm -> divide Q + resieve primes (completely) -> size filter -> SMOOTHNESS ----
// All fixed-width int arithmetic (256-bit norms, u64 cofactoring), 100k survivors in parallel.
__device__ __forceinline__ void udiv_complete(u256* x,unsigned long long p){
  for(;;){ u256 t=*x; unsigned long long r=u_divmod_u64(&t,p); if(r)break; *x=t; }
}
#ifdef FAST_FINAL
// Like udiv_complete, but RECORDS each division (with multiplicity) into L[] so the CPU
// finalizer can format the relation WITHOUT re-deriving the degree-5 norm in GMP (step 2).
// Q / small / resieved primes all fit in uint32 (<2^28). Stops storing at 48 but keeps
// counting, so an overflowed candidate (*n>48) is detectable and dropped by the caller.
__device__ __forceinline__ void udiv_rec(u256* x,unsigned long long p,uint32_t* L,int* n){
  for(;;){ u256 t=*x; unsigned long long r=u_divmod_u64(&t,p); if(r)break; *x=t; if(*n<48)L[*n]=(uint32_t)p; (*n)++; }
}
#endif
__device__ __forceinline__ unsigned long long mulmod64(unsigned long long a,unsigned long long b,unsigned long long m){
  return (unsigned long long)((unsigned __int128)a*b%m); }
__device__ __forceinline__ unsigned long long powmod64(unsigned long long a,unsigned long long e,unsigned long long m){
  unsigned long long r=1; a%=m; while(e){ if(e&1)r=mulmod64(r,a,m); a=mulmod64(a,a,m); e>>=1; } return r; }
__device__ __forceinline__ unsigned long long gcd64(unsigned long long a,unsigned long long b){ while(b){unsigned long long t=a%b;a=b;b=t;} return a; }
// Montgomery modmul (no 128-bit division) for the MR hot path. modulus odd, < 2^63.
__device__ __forceinline__ unsigned long long d_mont_n0(unsigned long long m){ unsigned long long inv=m; for(int i=0;i<6;i++) inv*=2-m*inv; return ~inv+1; }
__device__ __forceinline__ unsigned long long d_montmul(unsigned long long a,unsigned long long b,unsigned long long m,unsigned long long n0){
  unsigned __int128 T=(unsigned __int128)a*b; unsigned long long u=(unsigned long long)T*n0;
  unsigned __int128 res=(T+(unsigned __int128)u*m)>>64; return res>=m?(unsigned long long)(res-m):(unsigned long long)res; }
__device__ bool isprime64(unsigned long long n){              // 9 bases: deterministic for n<3.3e18 (>2^61)
  if(n<2)return false; const unsigned long long B[9]={2,3,5,7,11,13,17,19,23};
  for(int k=0;k<9;k++){ if(n%B[k]==0)return n==B[k]; }
  // Montgomery domain: one__int128 division for Rmod, then all modmuls are division-free.
  unsigned long long n0=d_mont_n0(n),Rmod=(unsigned long long)(((unsigned __int128)1<<64)%n),
    R2=(unsigned long long)(((unsigned __int128)Rmod*Rmod)%n),one=Rmod,nm1=n-Rmod;
  unsigned long long d=n-1; int s=0; while(!(d&1)){d>>=1;s++;}
  for(int k=0;k<9;k++){ unsigned long long b=d_montmul(B[k]%n,R2,n,n0),x=one,e=d;
    while(e){ if(e&1)x=d_montmul(x,b,n,n0); b=d_montmul(b,b,n,n0); e>>=1; }
    if(x==one||x==nm1)continue;
    bool ok=false; for(int r=1;r<s;r++){ x=d_montmul(x,x,n,n0); if(x==nm1){ok=true;break;} } if(!ok)return false; }
  return true; }
#ifdef PROBE_3LP
// --- 3-large-prime opportunity probe (measurement only; not a correctness path) ---
// gpu_loop is hard-capped at TWO large primes per side: classify() kills any composite
// cofactor > LPB^2, and the shipped residual is a u64. This probe counts how many
// survivors that cap is throwing away, so a 3LP upgrade can be valued BEFORE building it.
//   g_p3[0] survivors reaching the size filter
//   g_p3[1] pass CURRENT 2LP size filter        (ba<=2*lpb && br<=2*lpb)
//   g_p3[2] pass 3LP on the algebraic side only (ba<=3*lpb && br<=2*lpb)
//   g_p3[3] pass 3LP on the rational side only  (ba<=2*lpb && br<=3*lpb)
//   g_p3[4] pass 3LP on BOTH sides              (ba<=3*lpb && br<=3*lpb)
//   g_p3[5] still dead even with 3LP both sides
__device__ unsigned long long g_p3[6];
#endif
#ifdef DUMP_RNORM
// Dump the FULL rational norms (u256, 4 words LSW-first) of candidates that pass FULL algebraic
// validation — i.e. exactly the set a one-sided (RAT_NONE) design would ship to the CPU rational
// batch-smoothness stage. Feeds the standalone gating harness (Pipeline.md §3.7). Build with
// -DRAT_NONE -DDUMP_RNORM; env DUMP_RNORM_FILE sets the output path (default /tmp/rnorms.bin).
#ifndef DUMP_CAP
#define DUMP_CAP 1200000u
#endif
__device__ unsigned long long g_rndump[4*DUMP_CAP];
__device__ unsigned int g_rndump_cnt;
#endif
#ifdef DUMP_CANDS
// Like DUMP_RNORM but dumps the FULL candidate {a, b, Nr[4]} so a standalone CPU stage
// (cpu_batch/recover.c) can batch-detect rational smoothness and emit real relations for the
// one-sided design. -DRAT_NONE -DDUMP_CANDS; env DUMP_CANDS_FILE (default /tmp/cands.bin).
#ifndef CAND_CAP
#define CAND_CAP 1500000u
#endif
// cudaMalloc'd (not a static array) so CAND_CAP can be large without hitting the 2GB static-data
// link limit — needed for the streaming ρ_eff pipeline (big chunks -> few gpu_loop restarts).
__device__ unsigned long long* g_cand;
__device__ unsigned int g_cand_cnt;
#endif
#ifdef ALG_PASS_PROBE
// Algebraic-side pass rate among survivors — THE gating number for the one-sided design (§3.6 #1).
// Counts, per survivor: [0] entered, [1] passed the algebraic SIZE test (u_bits(Na)<=MFB1),
// [2] passed size AND algebraic classify (full algebraic validation). Under RAT_NOLAT the algebraic
// side is still fully sieved+resieved, so Na is exact and [2]/[0] IS the fraction that would be
// shipped to a CPU rational batch stage. NOTE: declared OUTSIDE the PROBE_3LP guard on purpose.
__device__ unsigned long long g_algp[4];
#endif
__device__ unsigned long long pollard64(unsigned long long n){  // capped: cofactor<=2^58 -> a factor <=2^29 -> rho splits in <~2^15
  if((n&1)==0)return 2;
  for(unsigned long long c=1;c<8;c++){ unsigned long long x=2,y=2,d=1; int it=0;
    while(d==1){ x=(mulmod64(x,x,n)+c)%n; y=(mulmod64(y,y,n)+c)%n; y=(mulmod64(y,y,n)+c)%n; unsigned long long t=x>y?x-y:y-x; d=gcd64(t,n); if(++it>200000)break; }
    if(d!=1&&d!=n)return d; }
  return 0; }
// classify a cofactor cheaply (no rho): 0=DEAD(prime>lpb), 1=GOOD(1 or prime in (lim,lpb]), 2=COMPOSITE(needs factoring)
__device__ __forceinline__ int classify(unsigned long long c,unsigned long long LPB){
  if(c==1)return 1;
  if(isprime64(c)) return (c>LIM&&c<=LPB)?1:0;
  return (c<=LPB*LPB)?2:0;                              // composite >2^(2lpb) can't be 2 primes<=2^lpb -> dead
}
// GPU does cheap massively-parallel filtering only (norm+trialdiv+MR). Composite cofactors are SHIPPED to CPU
// (rho is divergence-bound on GPU, fine on CPU). Both-sides-prime relations are emitted directly here.
__global__ void cofactor(const int* SI,const int* SJ,const unsigned* dnsurv,
    const u256* cf,u256 Y0,u256 Y1,unsigned long long Q,
    const uint32_t* PA,const unsigned* CA,const uint32_t* PR,const unsigned* CR,
    long long a0,long long b0,long long a1,long long b1,int MFB1,int MFB0,
    long long* shipA,long long* shipB,unsigned long long* shipCA,unsigned long long* shipCR,
    uint32_t* shipPA,unsigned* shipNA,uint32_t* shipPR,unsigned* shipNR,unsigned* scnt,unsigned shipcap){
  const unsigned long long LPB=LPB_VAL;          // large-prime bound (LPB_VAL; default 2^30)
  int nsurv=(int)*dnsurv; if(nsurv>(int)MAXSURV_VAL)nsurv=MAXSURV_VAL;   // see scan_idx_fused clamp                                // read survivor count on device (no host sync needed)
  for(int s=blockIdx.x*blockDim.x+threadIdx.x;s<nsurv;s+=gridDim.x*blockDim.x){
    int i=SI[s],j=SJ[s]; long long a=a0*i+a1*j,b=b0*i+b1*j;
    if(b<0){a=-a;b=-b;} if(b==0)continue; unsigned long long ub=(unsigned long long)b;
    u256 Na=u_abs(norm_alg(cf,a,ub)), Nr=u_abs(norm_rat(Y0,Y1,a,ub));
#ifdef SQSIDE_RAT
    udiv_complete(&Nr,Q);
#else
    udiv_complete(&Na,Q);
#endif
    for(int t=0;t<cNSP;t++){ udiv_complete(&Na,cSP[t]); udiv_complete(&Nr,cSP[t]); }  // small primes (not resieved)
    unsigned na=CA[s]; if(na>MAXP)na=MAXP; for(unsigned t=0;t<na;t++) udiv_complete(&Na,PA[(size_t)s*MAXP+t]);
    unsigned nr=CR[s]; if(nr>MAXP)nr=MAXP; for(unsigned t=0;t<nr;t++) udiv_complete(&Nr,PR[(size_t)s*MAXP+t]);
    if(u_bits(Na)>MFB1||u_bits(Nr)>MFB0)continue;          // size filter (residual now <=64 bits)
    if(classify(Nr.w[0],LPB0_VAL)==0)continue;             // rational(smaller) side first
    if(classify(Na.w[0],LPB1_VAL)==0)continue;
    unsigned long long ua=a<0?-a:a; if(gcd64(ua,(unsigned long long)b)!=1)continue;    // coprime
    // ship every valid candidate (both-prime + composite) with its resieved prime list -> CPU finalizes + formats
    unsigned pos=atomicAdd(scnt,1u); if(pos>=shipcap)continue;
    shipA[pos]=a; shipB[pos]=b; shipCA[pos]=Na.w[0]; shipCR[pos]=Nr.w[0];
    shipNA[pos]=na; for(unsigned t=0;t<na;t++) shipPA[(size_t)pos*MAXP+t]=PA[(size_t)s*MAXP+t];
    shipNR[pos]=nr; for(unsigned t=0;t<nr;t++) shipPR[(size_t)pos*MAXP+t]=PR[(size_t)s*MAXP+t];
  }
}
#ifdef FAST_FINAL
// FAST_FINAL variant of cofactor(): identical filtering, but RECORDS the complete
// Q+small+resieved factorization (with multiplicity) per side via udiv_rec and ships it,
// so the CPU finalizer skips the GMP norm recompute entirely (step 2). The residual
// cofactors (shipCA/shipCR, <=64 bits) are still factored on the CPU (one u64 rho each).
// shipFA/shipFR are 96-wide per candidate. NOTE: the local FA/FR[96] arrays raise per-thread
// local-memory pressure vs cofactor() — A/B this build against the default before trusting it.
__global__ void cofactor_ff(const int* SI,const int* SJ,const unsigned* dnsurv,
    const u256* cf,u256 Y0,u256 Y1,unsigned long long Q,
    const uint32_t* PA,const unsigned* CA,const uint32_t* PR,const unsigned* CR,
    long long a0,long long b0,long long a1,long long b1,int MFB1,int MFB0,
    long long* shipA,long long* shipB,unsigned long long* shipCA,unsigned long long* shipCR,
    uint32_t* shipFA,unsigned* shipNFA,uint32_t* shipFR,unsigned* shipNFR,unsigned* scnt,unsigned shipcap){
  const unsigned long long LPB=LPB_VAL;
  int nsurv=(int)*dnsurv; if(nsurv>(int)MAXSURV_VAL)nsurv=MAXSURV_VAL;   // see scan_idx_fused clamp
  for(int s=blockIdx.x*blockDim.x+threadIdx.x;s<nsurv;s+=gridDim.x*blockDim.x){
    int i=SI[s],j=SJ[s]; long long a=a0*i+a1*j,b=b0*i+b1*j;
    if(b<0){a=-a;b=-b;} if(b==0)continue; unsigned long long ub=(unsigned long long)b;
    u256 Na=u_abs(norm_alg(cf,a,ub)), Nr=u_abs(norm_rat(Y0,Y1,a,ub));
    uint32_t FA[48],FR[48]; int nfa=0,nfr=0;
    // Divide the special-q out of ITS OWN side and record it in that side's factor list.
    // (FAST_FINAL path -- this is the live one; the mpz formatter below is the #else fallback.)
#ifdef SQSIDE_RAT
    udiv_rec(&Nr,Q,FR,&nfr);
#else
    udiv_rec(&Na,Q,FA,&nfa);
#endif
    for(int t=0;t<cNSP;t++){ udiv_rec(&Na,cSP[t],FA,&nfa); udiv_rec(&Nr,cSP[t],FR,&nfr); }
#ifndef PROBE_COF_NODIV
    unsigned na=CA[s]; if(na>MAXP)na=MAXP; for(unsigned t=0;t<na;t++) udiv_rec(&Na,PA[(size_t)s*MAXP+t],FA,&nfa);
    unsigned nr=CR[s]; if(nr>MAXP)nr=MAXP; for(unsigned t=0;t<nr;t++) udiv_rec(&Nr,PR[(size_t)s*MAXP+t],FR,&nfr);
#endif
#ifdef PROBE_3LP
    { // measure what the 2-large-prime cap ACTUALLY discards.
      // Every large prime must exceed LIM, so with lpb=LB bits and lim=LIMB bits:
      //   <=2 large primes  =>  residual <= 2^(2*LB)
      //    3 large primes   =>  residual >= 3*LIMB bits  AND  <= 2^(3*LB)
      // Residuals between those two ranges are DEAD either way (too big for 2 LPs,
      // too small to be 3 primes that each exceed LIM). A probe that bins all of
      // (2^2LB, 2^3LB] as "3LP" overcounts by exactly that dead zone.
      const int LB=__ffsll((long long)LPB)-1;
      const int LIMB=32-__clz((int)LIM);                  // ceil-ish bits of LIM
      const int LO3=3*LIMB-2;                             // min bits for 3 primes > LIM (slack 2)
      int ba=(int)u_bits(Na), br=(int)u_bits(Nr);
      #define SIDECLS(x) ((x)<=2*LB ? 0 : ((x)>=LO3 && (x)<=3*LB ? 1 : 2))
      int ca=SIDECLS(ba), cr=SIDECLS(br);
      #undef SIDECLS
      atomicAdd(&g_p3[0],1ULL);
      if(ca==2||cr==2)      atomicAdd(&g_p3[5],1ULL);     // dead even with 3LP
      else if(ca==0&&cr==0) atomicAdd(&g_p3[1],1ULL);     // current 2LP
      else if(ca==1&&cr==0) atomicAdd(&g_p3[2],1ULL);     // 3LP alg only
      else if(ca==0&&cr==1) atomicAdd(&g_p3[3],1ULL);     // 3LP rat only
      else                  atomicAdd(&g_p3[4],1ULL);     // 3LP both sides
    }
#endif
#ifdef ALG_PASS_PROBE
    { atomicAdd(&g_algp[0],1ULL);
      if(u_bits(Na)<=MFB1){ atomicAdd(&g_algp[1],1ULL);
        if(classify(Na.w[0],LPB1_VAL)!=0) atomicAdd(&g_algp[2],1ULL); } }
#endif
#ifdef DUMP_RNORM
    // Candidate passes full algebraic validation -> its (unsieved, under RAT_NONE) rational norm Nr
    // is what the CPU batch stage must smoothness-test. Record the full 256-bit value.
    if(u_bits(Na)<=MFB1 && classify(Na.w[0],LPB1_VAL)!=0){
      unsigned pd=atomicAdd(&g_rndump_cnt,1u);
      if(pd<DUMP_CAP){ g_rndump[4*pd]=Nr.w[0]; g_rndump[4*pd+1]=Nr.w[1];
                       g_rndump[4*pd+2]=Nr.w[2]; g_rndump[4*pd+3]=Nr.w[3]; }
    }
#endif
#ifdef DUMP_CANDS
    if(u_bits(Na)<=MFB1 && classify(Na.w[0],LPB1_VAL)!=0){
      unsigned pc=atomicAdd(&g_cand_cnt,1u);
      if(pc<CAND_CAP){ unsigned long long* o=&g_cand[6*pc];
        o[0]=(unsigned long long)a; o[1]=(unsigned long long)b;
        o[2]=Nr.w[0]; o[3]=Nr.w[1]; o[4]=Nr.w[2]; o[5]=Nr.w[3]; }
    }
#endif
    if(u_bits(Na)>MFB1||u_bits(Nr)>MFB0)continue;
#ifdef PROBE_COF_NOMR
    if(Nr.w[0]!=1 && (Nr.w[0]<=LIM))continue;             // size-only, skip MR (timing probe, not correct)
    if(Na.w[0]!=1 && (Na.w[0]<=LIM))continue;
#else
    if(classify(Nr.w[0],LPB0_VAL)==0)continue;
    if(classify(Na.w[0],LPB1_VAL)==0)continue;
#endif
    if(nfa>48||nfr>48)continue;                            // factor list overflowed -> drop
    unsigned long long ua=a<0?-a:a; if(gcd64(ua,(unsigned long long)b)!=1)continue;
    unsigned pos=atomicAdd(scnt,1u); if(pos>=shipcap)continue;
    shipA[pos]=a; shipB[pos]=b; shipCA[pos]=Na.w[0]; shipCR[pos]=Nr.w[0];
    shipNFA[pos]=(unsigned)nfa; for(int t=0;t<nfa;t++) shipFA[(size_t)pos*48+t]=FA[t];
    shipNFR[pos]=(unsigned)nfr; for(int t=0;t<nfr;t++) shipFR[(size_t)pos*48+t]=FR[t];
  }
}
#endif
// ---- host u64 cofactoring (composites are <=2^58 -> native u64, ~100x faster than GMP rho) ----
static unsigned long long h_mulmod(unsigned long long a,unsigned long long b,unsigned long long m){ return (unsigned long long)((unsigned __int128)a*b%m); }
static unsigned long long h_powmod(unsigned long long a,unsigned long long e,unsigned long long m){ unsigned long long r=1;a%=m; while(e){if(e&1)r=h_mulmod(r,a,m);a=h_mulmod(a,a,m);e>>=1;} return r; }
static unsigned long long h_gcd(unsigned long long a,unsigned long long b){ while(b){unsigned long long t=a%b;a=b;b=t;} return a; }
// ---- Montgomery arithmetic (modulus < 2^63; our cofactors <= 2^58). Replaces __int128%% division in the rho/MR hot path. ----
static inline unsigned long long mont_n0(unsigned long long m){ unsigned long long inv=m; for(int i=0;i<6;i++) inv*=2-m*inv; return ~inv+1; } // -m^{-1} mod 2^64
static inline unsigned long long montmul(unsigned long long a,unsigned long long b,unsigned long long m,unsigned long long n0){
  unsigned __int128 T=(unsigned __int128)a*b; unsigned long long u=(unsigned long long)T*n0;
  unsigned __int128 res=(T+(unsigned __int128)u*m)>>64; return res>=m?(unsigned long long)(res-m):(unsigned long long)res; }
static inline unsigned long long to_mont(unsigned long long a,unsigned long long m,unsigned long long n0,unsigned long long R2){ return montmul(a,R2,m,n0); }
static bool h_isprime(unsigned long long n){ if(n<2)return false; const unsigned long long B[9]={2,3,5,7,11,13,17,19,23};
  for(int k=0;k<9;k++){if(n%B[k]==0)return n==B[k];}
  unsigned long long n0=mont_n0(n),Rmod=(unsigned long long)(((unsigned __int128)1<<64)%n),R2=(unsigned long long)(((unsigned __int128)Rmod*Rmod)%n),one=Rmod,nm1=n-Rmod;
  unsigned long long d=n-1;int s=0;while(!(d&1)){d>>=1;s++;}
  for(int k=0;k<9;k++){ unsigned long long b=to_mont(B[k]%n,n,n0,R2),x=one,e=d; while(e){if(e&1)x=montmul(x,b,n,n0);b=montmul(b,b,n,n0);e>>=1;}
    if(x==one||x==nm1)continue; bool ok=false; for(int r=1;r<s;r++){x=montmul(x,x,n,n0); if(x==nm1){ok=true;break;}} if(!ok)return false; } return true; }
static unsigned long long h_pollard(unsigned long long n){ if(!(n&1))return 2;   // Brent's rho in Montgomery domain (batched gcd)
  unsigned long long n0=mont_n0(n),Rmod=(unsigned long long)(((unsigned __int128)1<<64)%n),R2=(unsigned long long)(((unsigned __int128)Rmod*Rmod)%n),one=Rmod;
  for(unsigned long long c=1;c<20;c++){ unsigned long long cm=to_mont(c%n,n,n0,R2);
    unsigned long long y=to_mont(2%n,n,n0,R2),d=1,r=1,q=one,x=0,ys=0;
    while(d==1){ x=y; for(unsigned long long i=0;i<r;i++){ y=montmul(y,y,n,n0); y+=cm; if(y>=n)y-=n; }
      unsigned long long k=0;
      while(k<r&&d==1){ ys=y; unsigned long long m=r-k; if(m>128)m=128;
        for(unsigned long long i=0;i<m;i++){ y=montmul(y,y,n,n0); y+=cm; if(y>=n)y-=n; unsigned long long t=x>y?x-y:y-x; q=montmul(q,t,n,n0); }
        d=h_gcd(q,n); k+=m; }
      r<<=1; }
    if(d!=1&&d!=n)return d;
    if(d==n){ do{ ys=montmul(ys,ys,n,n0); ys+=cm; if(ys>=n)ys-=n; unsigned long long t=x>ys?x-ys:ys-x; d=h_gcd(t,n);}while(d==1); if(d!=1&&d!=n)return d; }
  } return 0; }
static bool h_smooth(unsigned long long c,unsigned long long LPB){ if(c==1)return true; if(h_isprime(c))return c>LIM&&c<=LPB;
  if(c>LPB*LPB)return false; unsigned long long d=h_pollard(c);if(!d||d==c)return false;unsigned long long e=c/d;
  return d>LIM&&d<=LPB&&h_isprime(d)&&e>LIM&&e<=LPB&&h_isprime(e); }
// resieve_coop: the resieve twin of scat_lat_coop. Reads the basis (BA) and bounds (BND) that
// scat_lat_coop already emitted -- coalesced, thread==line -- so it needs NO reduction, NO bounds
// recompute, and NO global scan (the per-warp prefix sum replaces it). Cooperative uniform walk.
#ifdef PROBE_RS_ENUM
// resieve walks the FULL bounding box with an in-region reject, unlike scat_lat which uses
// EXACT_C2_RANGE and lands every iteration. Ratio = wasted work in the resieve walk.
__device__ unsigned long long g_rs_enum, g_rs_hit;
#endif
__global__ void resieve_coop(const uint32_t* M,const uint32_t* P,const int* BA,int n,
                             const uint32_t* bits,const int* sidx,uint32_t* plist,unsigned* pcnt,
                             const uint32_t* R){
#ifdef PROBE_RS_ENUM
  unsigned long long myrsenum=0,myrshit=0;
#endif
  __shared__ int sA1[256],sB1[256],sNc[256],sP1[256],sQ1[256],sP2[256],sQ2[256],sPb[256];
  __shared__ unsigned long long sOff[256];
  int tIdx=threadIdx.x, lane=tIdx&31, wbase=tIdx&~31;
  int gwarp=(blockIdx.x*blockDim.x+tIdx)>>5, nwarp=(gridDim.x*blockDim.x)>>5;
  for(int lineBase=gwarp*32; lineBase<n; lineBase+=nwarp*32){
    int line=lineBase+lane;
    unsigned long long T=0; int A1=0,B1=0,ncol=0,p1=0,q1=0,p2=0,q2=0,pb=0;
    if(line<n){
      uint32_t m=FB_LD(&M[line]),pp=FB_LD(&P[line]);
      if(m==pp){                                        // p^1 lines only (matches resieve_lat)
#ifdef NO_BA_CACHE
        // Recompute the reduced basis instead of reading the 16B/line BA cache (see scat_lat).
        // Identical Lagrange reduction on identical inputs -> identical basis, so relations are
        // unchanged. Lines scat_lat skips (invalid root, or a vertical line) leave the basis at
        // 0 -> D==0 -> the line is dropped, which is exactly what scat_lat does with them.
        { uint32_t rr=R[line];
          if(rr!=0xFFFFFFFFu && !(rr&0x80000000u)){
            p1=1;q1=(int)rr;p2=0;q2=(int)m;
            for(int it=0;it<64;it++){
              double n1=(double)p1*(double)p1+(double)q1*(double)q1;
              double n2=(double)p2*(double)p2+(double)q2*(double)q2;
              if(n2<n1){ int a=p1;p1=p2;p2=a; a=q1;q1=q2;q2=a; n1=n2; }
              if(n1==0.0)break;
              double dot=(double)p1*(double)p2+(double)q1*(double)q2;
              int mu=(int)((dot/n1)>=0.0?floor(dot/n1+0.5):ceil(dot/n1-0.5));
              if(!mu)break;
              p2-=mu*p1; q2-=mu*q1;
            }
          } }
#else
#ifdef BA_VEC
        { int4 b=*(const int4*)&BA[line*4]; p1=b.x;q1=b.y;p2=b.z;q2=b.w; }
#else
        p1=FB_LD(&BA[line*4]);q1=FB_LD(&BA[line*4+1]);p2=FB_LD(&BA[line*4+2]);q2=FB_LD(&BA[line*4+3]);
#endif
#endif
        // int32 bounds: reduced-basis entries ~sqrt(m) (<~8400 for m<70M), det D=+-m (<2^27),
        // corners q2*I-p2*J < 2^27 -> all fit int32; hardware int div instead of int64 software.
        int D=p1*q2-p2*q1;
        if(D){
          int cI[2]={-I2,I2-1},cJ[2]={0,J-1};
          int a1m=0x3fffffff,a1M=-0x3fffffff,b1m=0x3fffffff,b1M=-0x3fffffff;
          for(int x=0;x<2;x++)for(int y=0;y<2;y++){ int I_=cI[x],J_=cJ[y];
            int n1c=q2*I_-p2*J_,n2c=-q1*I_+p1*J_;
            a1m=min(a1m,n1c);a1M=max(a1M,n1c);b1m=min(b1m,n2c);b1M=max(b1M,n2c); }
          int AA1,AA2,BB1,BB2;
          if(D>0){AA1=fdiv_i(a1m,D);AA2=-fdiv_i(-a1M,D);BB1=fdiv_i(b1m,D);BB2=-fdiv_i(-b1M,D);}
          else   {AA1=fdiv_i(a1M,D);AA2=-fdiv_i(-a1m,D);BB1=fdiv_i(b1M,D);BB2=-fdiv_i(-b1m,D);}
          if(AA2>=AA1&&BB2>=BB1){ A1=AA1;B1=BB1;ncol=BB2-BB1+1;
            int nrow=AA2-AA1+1; T=(unsigned long long)nrow*(unsigned long long)ncol; pb=(int)pp; }
        }
      }
    }
    sA1[tIdx]=A1;sB1[tIdx]=B1;sNc[tIdx]=ncol;sP1[tIdx]=p1;sQ1[tIdx]=q1;sP2[tIdx]=p2;sQ2[tIdx]=q2;sPb[tIdx]=pb;
    unsigned long long off=T;
    for(int d=1;d<32;d<<=1){ unsigned long long v=__shfl_up_sync(0xffffffffu,off,d); if(lane>=(unsigned)d) off+=v; }
    unsigned long long tot=__shfl_sync(0xffffffffu,off,31);
    off-=T; sOff[tIdx]=off;
    __syncwarp();
    unsigned long long g=(unsigned long long)lane*tot/32, g1=(unsigned long long)(lane+1)*tot/32;
    int owner=0; { int lo=0,hi=31; while(lo<=hi){int mid=(lo+hi)>>1; if(sOff[wbase+mid]<=g){owner=mid;lo=mid+1;}else hi=mid-1;} }
    while(g<g1){
      while(owner+1<32 && sOff[wbase+owner+1]<=g) owner++;
      unsigned long long oe=(owner+1<32)?sOff[wbase+owner+1]:tot; if(oe>g1)oe=g1;
      int oc=wbase+owner, nc=sNc[oc];
      if(nc<=0){ g=oe; continue; }
      int oA1=sA1[oc],oB1=sB1[oc],oP1=sP1[oc],oQ1=sQ1[oc],oP2=sP2[oc],oQ2=sQ2[oc]; uint32_t opb=(uint32_t)sPb[oc];
      unsigned long long loc=g-sOff[oc];
      int c1=oA1+(int)(loc/(unsigned)nc), c2=oB1+(int)(loc%(unsigned)nc);
      int i=c1*oP1+c2*oP2, j=c1*oQ1+c2*oQ2, cend=oB1+nc-1;
      unsigned long long take=oe-g;
#ifdef PROBE_RS_ENUM
      myrsenum+=take;
#endif
#ifdef RS_EXACT
      // Exact-range resieve walk. The stock loop enumerates the FULL bounding box and rejects
      // per point (measured 4.30x waste); scat_lat already avoids this via EXACT_C2_RANGE. Here we
      // keep the SAME linear box index -> the cooperative work split is untouched -- but per row we
      // clamp c2 to the exact in-region interval and skip the rest, trading ~4 int divisions per row
      // for the discarded points. Visits exactly the same set of in-region points, so relations are
      // identical.
      { unsigned long long rem=take;
        while(rem){
          int span=(int)((unsigned long long)(cend-c2)<rem-1?(unsigned long long)(cend-c2):rem-1);
          int seg_end=c2+span;
          int u=c1*oP1, v=c1*oQ1;
          int lo=c2, hi=seg_end;
          EXACT_C2_RANGE(u,v,oP2,oQ2,lo,hi,{lo=1;hi=0;});
          if(lo<c2)lo=c2; if(hi>seg_end)hi=seg_end;
          for(int cc=lo;cc<=hi;cc++){
#ifdef PROBE_RS_ENUM
            myrshit++;
#endif
            rec(bits,sidx,plist,pcnt,u+cc*oP2,v+cc*oQ2,opb);
          }
          rem-=(unsigned long long)(seg_end-c2+1);
          if(seg_end==cend){ c1++; c2=oB1; } else { c2=seg_end+1; }
        } }
#else
      for(unsigned long long k=0;k<take;k++){
        if(i>=-I2&&i<I2&&j>=0&&j<J){
#ifdef PROBE_RS_ENUM
          myrshit++;
#endif
          rec(bits,sidx,plist,pcnt,i,j,opb);
        }
        if(c2<cend){c2++;i+=oP2;j+=oQ2;} else {c2=oB1;c1++;i=c1*oP1+oB1*oP2;j=c1*oQ1+oB1*oQ2;}
      }
#endif
      g=oe;
    }
    __syncwarp();
  }
#ifdef PROBE_RS_ENUM
  atomicAdd(&g_rs_enum,myrsenum); atomicAdd(&g_rs_hit,myrshit);
#endif
}

// scat_box_pre: box scatter (thread==line, coalesced BA/BND reads) but reduction already done by
// setup_lat -> this kernel has no reduction state -> fewer registers -> higher occupancy. Tests
// whether splitting reduction out lets the atomic walk hide latency better.
__global__ void scat_box_pre(uint8_t* arr,const uint8_t* L,const int* BA,const int* BND,int n){
  for(int line=blockIdx.x*blockDim.x+threadIdx.x;line<n;line+=gridDim.x*blockDim.x){
    int ncol=BND[line*4+2], nrow=BND[line*4+3]; if(ncol<=0||nrow<=0)continue;
    int A1=BND[line*4],B1=BND[line*4+1];
    int p1=BA[line*4],q1=BA[line*4+1],p2=BA[line*4+2],q2=BA[line*4+3];
    unsigned lp=L[line];
    for(int c1=A1;c1<A1+nrow;c1++){ int i=c1*p1+B1*p2,j=c1*q1+B1*q2;
      for(int c2=0;c2<ncol;c2++,i+=p2,j+=q2)
        if(i>=-I2&&i<I2&&j>=0&&j<J){ size_t o=(size_t)(i+I2)*J+j; atomicAdd((unsigned*)&arr[o&~3ull],(unsigned)lp<<(8*(o&3))); } }
  }
}
__global__ void mksij(const int* sidx,int* SI,int* SJ){     // build survivor (i,j) on GPU (avoid host NCELL scan/sq)
  for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<NCELL;k+=(size_t)gridDim.x*blockDim.x){
    int s=sidx[k]; if(s>=0){ SI[s]=(int)(k/J)-I2; SJ[s]=(int)(k%J);} } }
#ifdef FUSED_SCAN
// FUSED scan_idx + mkmask + mksij in ONE pass over NCELL: assigns the survivor index,
// sets the L2-resident gate bit, AND writes (i,j) inline — eliminating the two extra
// full 134MB passes (mkmask reads dSidx, mksij reads dSidx). The inline bits/SI/SJ
// writes touch only survivors (~hundreds/lattice), so they're nearly free.
// Produces IDENTICAL survivors to the unfused path -> validate with the md5 diff.
//
// SCAN_EARLYOUT (optional): skip the 3x log2f for cells whose accumulated sieve value
// can't reach the survivor threshold. A survivor needs la[k] >= nla(i,j)-thresh; since
// nla >= (min norm log over the region), cells with la[k] < la_min (= min_nla - thresh
// - margin, computed conservatively on the host) provably can't survive. la_min/lr_min
// are passed in; margin makes it safe against host-sampling miss (md5 must still match).
__global__ void scan_idx_fused(const uint8_t* la,const uint8_t* lr,int* sidx,unsigned* cnt,int slack,
                     float a0,float b0,float a1,float b1,float logq,
                     float c0,float c1,float c2,float c3,float c4,float c5,float Y0,float Y1,
                     uint32_t* bits,int* SI,int* SJ,int la_min,int lr_min,int alg_tight){
  for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<NCELL;k+=(size_t)gridDim.x*blockDim.x){
    int i=(int)(k/J)-I2,j=(int)(k%J); int keep=0;
#ifdef PRIM_SCAN
    if(prim_skip(i,j)){ sidx[k]=-1; continue; }    // non-primitive (i,j) -> non-coprime (a,b) -> never a valid relation
#endif
#ifdef SCAN_EARLYOUT
    if((int)la[k]<la_min || (int)lr[k]<lr_min){ sidx[k]=-1; continue; }   // can't be a survivor -> skip log2f
#endif
    float a=a0*i+a1*j,b=b0*i+b1*j;
    if(b!=0){ float nla=fflog2(c0,c1,c2,c3,c4,c5,a,b)-logq*SQ_ADJ_A, Ga=Y1*a+Y0*b, nlr=log2f(fabsf(Ga))-logq*SQ_ADJ_R;
#ifdef RAT_NONE
      (void)nlr; (void)lr;                                  // rational array neither cleared nor read
      if(nla-la[k]<=(float)(58+slack+SCATOFF-alg_tight)) keep=1; }
#else
      if(nla-la[k]<=(float)(58+slack+SCATOFF-alg_tight) && nlr-lr[k]<=(float)(57+slack+SCATOFF+RAT_XSLACK)) keep=1; }
#endif
    // Clamp: SI/SJ/dCa/dCr are MAXSURV-sized. Past that, writing sidx[k]=s would scribble off the
    // end (the §1.4 "nsurv blows past MAXSURV" corruption). Drop the overflow instead; cnt still
    // counts true demand so the PROF nsurv figure stays honest, and cofactor_ff clamps its loop.
    if(keep){ int s=(int)atomicAdd(cnt,1u);
      if(s<(int)MAXSURV_VAL){ sidx[k]=s; atomicOr(&bits[k>>5],1u<<(k&31)); SI[s]=i; SJ[s]=j; }
      else sidx[k]=-1; }
    else sidx[k]=-1;
  }
}
#endif
#ifdef RAT_PROBE
// ---- one-sided-sieve feasibility probe (see Pipeline.md §3) --------------------------------
// Pairs with -DRAT_NOLAT, which drops the rational large-prime scatter+resieve (group 1). Without
// those primes the rational sieve log `lr` is short by the logs of whatever primes in (T,LIM]
// divide the rational norm, so the rational gate must be relaxed to keep true relations. This
// kernel measures the COST of that relaxation: for one fixed algebraic gate it counts survivors at
// RXN rational relaxations at once, so a single run yields the whole explosion curve instead of
// one rebuild per threshold. Counting only -- it writes no survivor indices and feeds nothing
// downstream, so it cannot overflow MAXSURV (the §1.4 hang).
#define RXN 9
#define RXSTEP 4
__global__ void scan_count_x(const uint8_t* la,const uint8_t* lr,unsigned long long* cnts,int slack,
                     float a0,float b0,float a1,float b1,float logq,
                     float c0,float c1,float c2,float c3,float c4,float c5,float Y0,float Y1){
  __shared__ unsigned long long sh[RXN];
  if(threadIdx.x<RXN) sh[threadIdx.x]=0ull;
  __syncthreads();
  for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<NCELL;k+=(size_t)gridDim.x*blockDim.x){
    int i=(int)(k/J)-I2,j=(int)(k%J); float a=a0*i+a1*j,b=b0*i+b1*j; if(b==0)continue;
    float nla=fflog2(c0,c1,c2,c3,c4,c5,a,b)-logq*SQ_ADJ_A;
    if(nla-la[k]>(float)(58+slack+SCATOFF)) continue;              // algebraic gate: UNCHANGED
    float Ga=Y1*a+Y0*b, nlr=log2f(fabsf(Ga))-logq*SQ_ADJ_R, d=nlr-lr[k];
#pragma unroll
    for(int t=0;t<RXN;t++) if(d<=(float)(57+slack+SCATOFF+RXSTEP*t)) atomicAdd(&sh[t],1ull);
  }
  __syncthreads();
  if(threadIdx.x<RXN) atomicAdd(&cnts[threadIdx.x],sh[threadIdx.x]);
}
#endif
// skewed Lagrange reduction of the special-q lattice (matches CADO; validated)
static void skew_reduce(long long q,long long s,double S,long long out[4]){
  double rs=sqrt(S); long long v0i=q,v0j=0,v1i=s,v1j=1;
  for(int it=0;it<200;it++){
    double n0=(v0i/rs)*(v0i/rs)+(v0j*rs)*(v0j*rs), n1=(v1i/rs)*(v1i/rs)+(v1j*rs)*(v1j*rs);
    if(n1<n0){ long long t=v0i;v0i=v1i;v1i=t; t=v0j;v0j=v1j;v1j=t; }
    double d0=(double)v0i*v0i/S+(double)v0j*v0j*S; if(d0==0)break;
    double dot=(double)v0i*v1i/S+(double)v0j*v1j*S; long long mu=(long long)llround(dot/d0);
    if(mu==0)break; v1i-=mu*v0i; v1j-=mu*v0j;
  }
  out[0]=v0i;out[1]=v0j;out[2]=v1i;out[3]=v1j;
}
static int isprime_q(long long q){ if(q<2)return 0; for(long long d=2;d*d<=q;d++) if(q%d==0)return 0; return 1; }
// ---------------------------------------------------------------------------
// Sieve-time duplicate detection ("smallest special-q owns the relation").
//
// A relation (a,b) is re-found in EVERY special-q lattice (p,rho_p) with p | F(a,b),
// p in the sieved special-q range, and (a,b) inside that lattice's (i,j) region. That
// is the entire source of the measured 44.5% duplicate rate -- it is not noise, it is
// the lattice sieve re-covering the same (a,b) points from different sublattices.
//
// Because the sweep runs q strictly increasing, the SMALLEST such p is always sieved
// first. So "emit only when QB == smallest owner" is a lossless de-duplication: the
// copy at the smallest p is always kept, every later copy is dropped.
//
// The region test is exact, not heuristic: it reuses the same skew_reduce() and the
// same i in [-I2,I2), j in [0,J) box the kernel sieves, so it answers precisely
// "would lattice p have covered this (a,b)?". The one thing it cannot know is whether
// the earlier lattice's log-threshold actually let the relation through. That residual is
// measured, not assumed: sieve a window with DUPSUP=2 (emits everything) and with DUPSUP=1
// (suppresses), then compare the distinct (a,b) sets. On c151 q=[1.0M,1.1M] it is 27 of
// 379,714 = 0.0071% -- see Report.md and Pipeline.md 1.6.
// ---------------------------------------------------------------------------
// DUPSUP=0 off (default) | 1 suppress duplicates | 2 count only (emit everything, report the rate)
static int g_dupmode=0;
static long long g_dupqmin=0;          // lower edge of the sieved special-q range (DUP_QMIN, default QMIN)
static long long g_dupfound=0, g_dupvalid=0;
// Histogram of the LARGEST covering owner p < QB, in log2 buckets. A relation is a duplicate
// under a hypothetical range start T iff it has a covering owner in [T,QB) -- i.e. iff its
// largest owner is >= T. So this one histogram answers the dup rate for EVERY candidate T,
// which is what makes a single sweep enough to optimise the special-q range start.
static long long g_dupbk[64]={0};
// Cross-sweep ownership. A relation found by THIS sweep may also be reachable by a sweep
// running special-q on the OTHER side over [DUPX_QMIN,DUPX_QMAX). The lattice (p,rho) with
// rho = a/b mod p is defined the same way whichever polynomial p divides, so the identical
// region test answers it. This is what decides whether a second sweep on the other side adds
// genuinely new material or just re-finds what the first sweep already covers.
static long long g_xqmin=0,g_xqmax=0;
static long long g_xowned=0,g_xnew=0;
static long long dup_modinv(long long a,long long m){
  long long g=m,x=0,x1=1,a1=a%m; if(a1<0)a1+=m;
  while(a1){ long long qq=g/a1,t=g-qq*a1; g=a1;a1=t; t=x-qq*x1; x=x1;x1=t; }
  if(g!=1)return -1; x%=m; if(x<0)x+=m; return x;
}
// true iff lattice (p, a/b mod p) covers (a,b) in its sieve region
static bool dup_covered_by(long long a,long long b,long long p,double S){
  long long bm=b%p; if(bm<0)bm+=p;
  if(bm==0) return false;                     // projective root: not a standard (p,rho) lattice
  long long inv=dup_modinv(bm,p); if(inv<0) return false;
  long long am=a%p; if(am<0)am+=p;
  long long rho=(long long)((__int128)am*inv%p);
  long long v[4]; skew_reduce(p,rho,S,v);
  __int128 det=(__int128)v[0]*v[3]-(__int128)v[1]*v[2];
  if(det==0) return false;
  // a = v0i*i + v1i*j ; b = v0j*i + v1j*j   -> invert (det = +-p)
  __int128 ni=(__int128)a*v[3]-(__int128)b*v[2];
  __int128 nj=(__int128)b*v[0]-(__int128)a*v[1];
  if(ni%det || nj%det) return false;          // not on the lattice (cannot happen when p|F, but be safe)
  __int128 i=ni/det, j=nj/det;
  if(j<0){ i=-i; j=-j; }                      // the kernel emits (a,b) and (-a,-b) as the same relation
  return i>=-(__int128)I2 && i<(__int128)I2 && j>=0 && j<(__int128)J;
}
// factor a cofactor residual (<=2^58, all prime factors must be in (LIM, 2^30]) -> append primes to arr; false if invalid
static bool h_factor_cof(unsigned long long c,unsigned long long* arr,int* n,unsigned long long L){
  if(c==1)return true;
  if(h_isprime(c)){ if(c>LIM&&c<=L){arr[(*n)++]=c;return true;} return false; }
  if(c>L*L)return false;
  unsigned long long d=h_pollard(c); if(!d||d==c)return false; unsigned long long e=c/d;
  if(!(d>LIM&&d<=L&&h_isprime(d)))return false; if(!(e>LIM&&e<=L&&h_isprime(e)))return false;
  arr[(*n)++]=d; arr[(*n)++]=e; return true;
}
static int cmp_u64(const void*a,const void*b){ unsigned long long x=*(const unsigned long long*)a,y=*(const unsigned long long*)b; return x<y?-1:x>y?1:0; }
static mpz_t G_cc[6],G_y0,G_y1;          // pre-parsed poly coeffs (parsed ONCE, read-only in parallel format loop)
#ifdef SQSIDE_RAT
// Roots of the RATIONAL polynomial Y1*x + Y0 (mod q) for rational-side special-q. Linear, so
// there is exactly ONE root per q -- unlike the degree-5 algebraic side, which averages one root
// per prime but is distributed 0..5. Returns 0 when q | Y1 (projective root; skip that q).
static long long modinv_ll(long long a,long long m){
  long long g=m,x=0,x1=1,a1=a%m; if(a1<0)a1+=m;
  while(a1){ long long qq=g/a1,t=g-qq*a1; g=a1;a1=t; t=x-qq*x1; x=x1;x1=t; }
  if(g!=1)return 0; x%=m; if(x<0)x+=m; return x;
}
static int rat_roots(long long q,u64* out){
  unsigned long y1=mpz_fdiv_ui(G_y1,(unsigned long)q); if(y1==0)return 0;   // q | Y1 -> projective
  unsigned long y0=mpz_fdiv_ui(G_y0,(unsigned long)q);
  long long inv=modinv_ll((long long)y1,q); if(!inv)return 0;
  long long r=(long long)(((__int128)((q-(long long)(y0%(unsigned long)q))%q)*inv)%q);
  out[0]=(u64)r; return 1;
}
#endif
// ---- relations stream into a RAM buffer (NOT disk): the validator gives 1GB /tmp but 85GB RAM,
// so a deployable GNFS must keep relations resident in memory. Grows geometrically.
static char* g_rambuf=0; static size_t g_ramlen=0, g_ramcap=0;
static inline void rel_emit(const char* s){ size_t n=strlen(s);
  if(g_ramlen+n+1>g_ramcap){ size_t nc=g_ramcap?g_ramcap:(size_t)64<<20; while(nc<g_ramlen+n+1)nc<<=1; g_rambuf=(char*)realloc(g_rambuf,nc); g_ramcap=nc; }
  memcpy(g_rambuf+g_ramlen,s,n); g_ramlen+=n; }
// Incremental crash-safe dump: append newly-accumulated relations to the dump file at every progress
// tick, so a kill / crash / early-stop never loses the whole sieve — the disk file always holds
// everything found so far. (The old code wrote the buffer ONLY at clean exit, so a kill lost it all.)
static FILE* g_dumpf=0; static size_t g_dumpflushed=0;
#ifndef RAMBUF_CAP_BYTES
#define RAMBUF_CAP_BYTES (64ull<<20)   /* recycle the staging buffer once >=64MB is safely on disk */
#endif
static void dump_flush(){ if(g_dumpf && g_ramlen>g_dumpflushed){ fwrite(g_rambuf+g_dumpflushed,1,g_ramlen-g_dumpflushed,g_dumpf); fflush(g_dumpf); g_dumpflushed=g_ramlen;
    // Streaming write-and-discard: everything up to g_ramlen is now on disk (crash-safe), so recycle
    // the RAM staging buffer instead of letting it realloc-double toward ~8GB over a full 64M-relation
    // run. That resident buffer -- not the fwrite/fflush, which measures free even to real disk -- is
    // the software-recoverable part of the realized-vs-benchmark gap (memory pressure at scale). Only
    // when a real dump file is active; the no-dumpfile benchmark path keeps relations resident as before.
    if(g_ramlen>=RAMBUF_CAP_BYTES){ g_ramlen=0; g_dumpflushed=0; }
  } }
int main(int argc,char** argv){
  if(argc<4){ fprintf(stderr,"usage: %s <poly.cado> <qmin> <qmax> [dump_file] [rel_target]\n",argv[0]); return 2; }
  { int procs=omp_get_num_procs(); int nt=procs<24?procs:24;   // CAP threads: per-lattice omp region oversubscribes on big nodes -> thrash
    if(getenv("OMP_NUM_THREADS")==0) omp_set_num_threads(nt);
    fprintf(stderr,"omp threads=%d (procs=%d)\n",getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):nt,procs); }
  load_poly(argv[1]);
  fprintf(stderr,"root-finding factor base (ONCE)...\n"); double t0=wall_s();
  build_fb();
  printf("FB built in %.1fs: ratS=%zu ratL=%zu algS=%zu algL=%zu\n",
    wall_s()-t0,fM[0].size(),fM[1].size(),fM[2].size(),fM[3].size());
  build_smallp();
#ifdef PRIM_WHEEL
  build_wheel();
#endif
  cudaMemcpyToSymbol(cSP,hSP,sizeof(uint32_t)*hNSP); cudaMemcpyToSymbol(cNSP,&hNSP,sizeof(int));
  fprintf(stderr,"small-prime resieve skip: SPB=%d (%d primes trial-divided in cofactor)\n",SPB,hNSP);

  // ---- upload FB ONCE ----
  uint32_t *dM[4],*dRin[4],*dRout[4],*dP[4]; uint8_t *dT[4],*dL[4]; int ng[4];
  int nsm[4]={0,0,0,0};   // FK_COL: first index in the col groups with m > W (groups are ascending)
  for(int g=0;g<4;g++){ ng[g]=fM[g].size();
    CK(cudaMalloc(&dM[g],4*ng[g]));CK(cudaMalloc(&dRin[g],4*ng[g]));CK(cudaMalloc(&dRout[g],4*ng[g]));CK(cudaMalloc(&dT[g],ng[g]));CK(cudaMalloc(&dL[g],ng[g]));CK(cudaMalloc(&dP[g],4*ng[g]));
    // FK_COL needs the col groups ordered by m so [0,nsm) are the m<=W lines. build_fb is
    // OpenMP-parallel and merges per-thread buffers in tid order, which is NOT ascending, so
    // sort here explicitly (col groups are ~12k entries -- negligible, once per run).
    // FB_SORT/FB_BANDS also sort the LARGE-prime groups 1/3 (~3M entries each, ~0.4 s once per
    // run). Correctness-neutral: the scatter is add-only so line order cannot change the sieve
    // array, and the resieve's per-survivor factor lists are qsort'ed on the host before output.
    // FB_BANDS needs it to index-split by prime magnitude; FB_SORT alone tests whether uniform
    // per-warp trip counts (32 similar-sized primes per warp) are worth anything on their own.
#if defined(FB_SORT) || defined(FB_BANDS)
    if(true){
#else
    if(g==0||g==2){
#endif
      std::vector<int> idx(ng[g]); for(int z=0;z<ng[g];z++) idx[z]=z;
      std::sort(idx.begin(),idx.end(),[&](int x,int y){ return fM[g][x]<fM[g][y]; });
      std::vector<uint32_t> tm(ng[g]),tr(ng[g]),tp(ng[g]); std::vector<uint8_t> tt(ng[g]),tl(ng[g]);
      for(int z=0;z<ng[g];z++){ int o=idx[z]; tm[z]=fM[g][o]; tr[z]=fR[g][o]; tp[z]=fP[g][o]; tt[z]=fT[g][o]; tl[z]=fL[g][o]; }
      fM[g]=tm; fR[g]=tr; fP[g]=tp; fT[g]=tt; fL[g]=tl;
    }
    { int lo=0; while(lo<ng[g] && fM[g][lo]<= (uint32_t)W) lo++; nsm[g]=lo; }
    cudaMemcpy(dM[g],fM[g].data(),4*ng[g],cudaMemcpyHostToDevice);cudaMemcpy(dRin[g],fR[g].data(),4*ng[g],cudaMemcpyHostToDevice);
    cudaMemcpy(dT[g],fT[g].data(),ng[g],cudaMemcpyHostToDevice);cudaMemcpy(dL[g],fL[g].data(),ng[g],cudaMemcpyHostToDevice);
    cudaMemcpy(dP[g],fP[g].data(),4*ng[g],cudaMemcpyHostToDevice); }
#ifdef FB_BANDS
  // ---- prime-magnitude bands (profiling only; -DFB_BANDS) --------------------------------
  // Splits each side's scatter and resieve into 4 launches by prime size so cudaEvents can time
  // each band directly. Bands (default edges):
  //   R0 [SCAT_SKIP, 4096]   dense, every column hit many times   (scat_col)
  //   R1 (4096, 131072=T]    small                                 (scat_col)
  //   R2 (T, 2^20]           medium, box walk ~32-256 hits/lattice (scat_lat)
  //   R3 (2^20, LIM]         large sparse, <32 hits/lattice        (scat_lat)
  // The side interleave col(Lr),lat(Lr),col(La),lat(La) is PRESERVED (Pipeline.md 2.4: grouping
  // col,col,lat,lat costs 15%); only each group's line range is chopped, so the array stays hot.
#ifndef FB_BAND_E0
#define FB_BAND_E0 4096u
#endif
#ifndef FB_BAND_E1
#define FB_BAND_E1 131072u
#endif
#ifndef FB_BAND_E2
#define FB_BAND_E2 1048576u
#endif
  int bs[4][5];
  { const uint32_t edge[3]={FB_BAND_E0,FB_BAND_E1,FB_BAND_E2};
    for(int g=0;g<4;g++){ int p=0; bs[g][0]=0;
      for(int b=0;b<3;b++){ while(p<ng[g] && fM[g][p]<=edge[b]) p++; bs[g][b+1]=p; }
      bs[g][4]=ng[g]; }
    fprintf(stderr,"FB_BANDS lines/band  rat: %d %d %d %d   alg: %d %d %d %d\n",
      bs[0][1]-bs[0][0],bs[0][2]-bs[0][1],bs[1][3]-bs[1][2],bs[1][4]-bs[1][3],
      bs[2][1]-bs[2][0],bs[2][2]-bs[2][1],bs[3][3]-bs[3][2],bs[3][4]-bs[3][3]); }
  cudaEvent_t bev[18]; for(int z=0;z<18;z++) cudaEventCreate(&bev[z]);
  double g_bandms[18]={0};
#endif
  // reduced-basis cache for the large-prime (lat) groups: bucket_fill_lat writes, resieve_lat reads.
  int *dBA[4]={0,0,0,0};
  CK(cudaMalloc(&dBA[1],(size_t)4*ng[1]*sizeof(int)));
  CK(cudaMalloc(&dBA[3],(size_t)4*ng[3]*sizeof(int)));
#if defined(FLAT_WALK) || defined(FLAT_RESIEVE) || defined(FLAT_COOP) || defined(FLAT_COOP2)
  int *dBND[4]={0,0,0,0}, *dVL[4]={0,0,0,0}; unsigned *dVC[4]={0,0,0,0};
  unsigned long long *dTC[4]={0,0,0,0}, *dOff[4]={0,0,0,0};
  void* dScanTmp[4]={0,0,0,0}; size_t scanBytes[4]={0,0,0,0};
  for(int g=1;g<4;g+=2){
    CK(cudaMalloc(&dBND[g],(size_t)4*ng[g]*sizeof(int)));
    CK(cudaMalloc(&dTC[g],(size_t)ng[g]*sizeof(unsigned long long)));
    CK(cudaMalloc(&dOff[g],(size_t)(ng[g]+1)*sizeof(unsigned long long)));
    CK(cudaMalloc(&dVL[g],4096*sizeof(int))); CK(cudaMalloc(&dVC[g],4));
    // The FLAT_COOP2 default path passes vcnt=0 to scat_lat, so nothing ever writes dVC[g] -- but
    // vert_resieve still dereferences it (and indexes dVL[g] by it). cudaMalloc does NOT zero, so
    // this was reading uninitialised device memory and only worked because a fresh allocation
    // happens to come back zeroed. Make it explicit.
    CK(cudaMemset(dVC[g],0,4)); CK(cudaMemset(dVL[g],0,4096*sizeof(int)));
    cub::DeviceScan::ExclusiveSum((void*)0,scanBytes[g],dTC[g],dOff[g],ng[g]);
    CK(cudaMalloc(&dScanTmp[g],scanBytes[g]));
    fprintf(stderr,"FLAT_WALK g%d: n=%d scan_tmp=%.1fMB\n",g,ng[g],scanBytes[g]/1048576.0);
  }
#endif
#if defined(RAT_NOLAT) && !defined(FLAT_COOP2)
#error "RAT_NOLAT is only wired for the default FLAT_COOP2 scatter/resieve path"
#endif
#if defined(RAT_NOLAT) && (defined(FK_COL) || defined(BUCKET_SIEVE))
#error "RAT_NOLAT is not wired for the FK_COL / BUCKET_SIEVE variants"
#endif
#ifdef RAT_PROBE
  unsigned long long* dRX; CK(cudaMalloc(&dRX,RXN*sizeof(unsigned long long)));
  CK(cudaMemset(dRX,0,RXN*sizeof(unsigned long long)));
#endif
  // bucket sieve arrays: N_STRIPES=512 stripes * BUCKET_CAP=65536 entries * 4B * 2 sides = 256MB
  // (replaces scat_lat global atomics with shmem accumulation in bucket_flush_lat)
#ifdef BUCKET_SIEVE
  uint32_t *dBucketR,*dBucketA,*dBucketCntR,*dBucketCntA;
  CK(cudaMalloc(&dBucketR,(size_t)N_STRIPES*BUCKET_CAP*4));
  CK(cudaMalloc(&dBucketA,(size_t)N_STRIPES*BUCKET_CAP*4));
  CK(cudaMalloc(&dBucketCntR,(size_t)N_STRIPES*4));
  CK(cudaMalloc(&dBucketCntA,(size_t)N_STRIPES*4));
  fprintf(stderr,"bucket sieve: %zu MB (lat scatter via shmem; N_STRIPES=%d BUCKET_CAP=%d)\n",
    (size_t)N_STRIPES*BUCKET_CAP*4*2/(1<<20),N_STRIPES,BUCKET_CAP);
#else
  fprintf(stderr,"scatter: scat_lat (direct L2 atomics)\n");
#endif
  // ---- allocate ALL working buffers ONCE (max-sized; per-sq cudaMalloc was fixed overhead) ----
  // Survivor cap. NOTE: scan_idx/resieve index plist[] by survivor id with NO bound
  // check, so this MUST exceed the real survivor count or the GPU writes out of bounds.
  // Survivors scale with the sieve-region area (W*J), so a bigger region needs a
  // bigger cap: build -DMAXSURV_VAL=600000 when doubling J. Default fits 4096x4096.
  const unsigned MAXSURV=MAXSURV_VAL;
#ifdef DUMP_CANDS
  unsigned long long* d_cand=0;
  CK(cudaMalloc(&d_cand,(size_t)6*CAND_CAP*sizeof(unsigned long long)));
  cudaMemcpyToSymbol(g_cand,&d_cand,sizeof(d_cand));
  { unsigned int z=0; cudaMemcpyToSymbol(g_cand_cnt,&z,sizeof(z)); }
#endif
#ifdef L2_PERSIST
  // One contiguous 2*NCELL block so a single access-policy window can cover both sieve arrays,
  // and pin it in L2 as "persisting" (see the L2_STREAM/L2_PERSIST note at the top of the file).
  // Pure cache policy: identical addresses, identical values -> relations byte-identical.
  uint8_t *dLa,*dLr,*dL2base; CK(cudaMalloc(&dL2base,2*(size_t)NCELL)); dLa=dL2base; dLr=dL2base+NCELL;
  { size_t want=2*(size_t)NCELL; cudaDeviceProp dp; cudaGetDeviceProperties(&dp,0);
    size_t cap=dp.persistingL2CacheMaxSize; if(want>cap) want=cap;
    CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,want));
    cudaStreamAttrValue av; memset(&av,0,sizeof(av));
    av.accessPolicyWindow.base_ptr=(void*)dL2base;
    av.accessPolicyWindow.num_bytes=(2*(size_t)NCELL<=(size_t)dp.accessPolicyMaxWindowSize)?2*(size_t)NCELL:(size_t)dp.accessPolicyMaxWindowSize;
    av.accessPolicyWindow.hitRatio=1.0f;
    av.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaError_t e=cudaStreamSetAttribute(cudaStreamLegacy,cudaStreamAttributeAccessPolicyWindow,&av);
    fprintf(stderr,"L2_PERSIST: window %.1f MB over sieve arrays, persist limit %.1f MB -> %s\n",
      av.accessPolicyWindow.num_bytes/1048576.0,want/1048576.0,cudaGetErrorString(e)); }
#else
  uint8_t *dLa,*dLr; CK(cudaMalloc(&dLa,NCELL));CK(cudaMalloc(&dLr,NCELL));
#endif
  int* dSidx; CK(cudaMalloc(&dSidx,sizeof(int)*NCELL)); unsigned* dCnt; CK(cudaMalloc(&dCnt,4));
  uint32_t* dBits; CK(cudaMalloc(&dBits,(NCELL+31)/32*4));
  uint32_t *dPa,*dPr; unsigned *dCa,*dCr; int *dSI,*dSJ;
  CK(cudaMalloc(&dPa,(size_t)MAXSURV*MAXP*4));CK(cudaMalloc(&dPr,(size_t)MAXSURV*MAXP*4));
  CK(cudaMalloc(&dCa,(size_t)MAXSURV*4));CK(cudaMalloc(&dCr,(size_t)MAXSURV*4));
  CK(cudaMalloc(&dSI,4*MAXSURV));CK(cudaMalloc(&dSJ,4*MAXSURV));
  const unsigned MAXSHIP=16384;                  // no-dead-side candidates per lattice (~255 typical); cap generously
  long long *dsA,*dsB; unsigned long long *dsCA,*dsCR; unsigned *dSC;
  uint32_t *dsPA,*dsPR; unsigned *dsNA,*dsNR;
  CK(cudaMalloc(&dsA,8*MAXSHIP));CK(cudaMalloc(&dsB,8*MAXSHIP));CK(cudaMalloc(&dsCA,8*MAXSHIP));CK(cudaMalloc(&dsCR,8*MAXSHIP));CK(cudaMalloc(&dSC,4));
  CK(cudaMalloc(&dsPA,(size_t)MAXSHIP*MAXP*4));CK(cudaMalloc(&dsPR,(size_t)MAXSHIP*MAXP*4));CK(cudaMalloc(&dsNA,MAXSHIP*4));CK(cudaMalloc(&dsNR,MAXSHIP*4));
#ifdef FAST_FINAL
  // step-2 fast finalize: complete factor lists (96/side) so the CPU skips the GMP norm.
  uint32_t *dsFA,*dsFR; unsigned *dsNFA,*dsNFR;
  CK(cudaMalloc(&dsFA,(size_t)MAXSHIP*48*4));CK(cudaMalloc(&dsFR,(size_t)MAXSHIP*48*4));
  CK(cudaMalloc(&dsNFA,MAXSHIP*4));CK(cudaMalloc(&dsNFR,MAXSHIP*4));
#endif
  u256 hcf[6]; for(int k=0;k<6;k++)hcf[k]=parse_u256(C[k]); u256 hY0=parse_u256(sY0),hY1=parse_u256(sY1);
  u256* dcf; CK(cudaMalloc(&dcf,sizeof(u256)*6)); cudaMemcpy(dcf,hcf,sizeof(u256)*6,cudaMemcpyHostToDevice);
  // host receive buffers — DOUBLE-BUFFERED so the GPU can copy lattice N+1's shipped candidates into
  // one slot while the CPU finalizes lattice N from the other (GPU/CPU pipeline; see main loop).
  struct ShipBuf{ long long *A,*B; unsigned long long *CA,*CR; uint32_t *PA,*PR; unsigned *NA,*NR; unsigned nship; long long Q; double gpu_ms;
#ifdef FAST_FINAL
    uint32_t *FA,*FR; unsigned *NFA,*NFR;   // step-2: complete factor lists (96/side)
#endif
  };
  ShipBuf sb[2];
  for(int z=0;z<2;z++){ ShipBuf&b=sb[z];
    b.A=(long long*)malloc(8*MAXSHIP); b.B=(long long*)malloc(8*MAXSHIP);
    b.CA=(unsigned long long*)malloc(8*MAXSHIP); b.CR=(unsigned long long*)malloc(8*MAXSHIP);
    b.PA=(uint32_t*)malloc((size_t)MAXSHIP*MAXP*4); b.PR=(uint32_t*)malloc((size_t)MAXSHIP*MAXP*4);
    b.NA=(unsigned*)malloc(MAXSHIP*4); b.NR=(unsigned*)malloc(MAXSHIP*4); b.nship=0; b.Q=0; b.gpu_ms=0;
#ifdef FAST_FINAL
    b.FA=(uint32_t*)malloc((size_t)MAXSHIP*48*4); b.FR=(uint32_t*)malloc((size_t)MAXSHIP*48*4);
    b.NFA=(unsigned*)malloc(MAXSHIP*4); b.NFR=(unsigned*)malloc(MAXSHIP*4);
#endif
  }
  char* ok=(char*)malloc(MAXSHIP);
  unsigned char* dupbk=(unsigned char*)malloc(MAXSHIP);   // log2 bucket of the largest earlier owner (+1), 0 = none
  unsigned char* xown=(unsigned char*)malloc(MAXSHIP);    // 1 = also reachable by the other-side sweep
  char* relbuf=(char*)malloc((size_t)MAXSHIP*640);     // formatted CADO relation per candidate
  double cf[6]; for(int k=0;k<6;k++)cf[k]=strtod(C[k],0); double Y0d=strtod(sY0,0),Y1d=strtod(sY1,0);
  // Keep the float norm gate in range: see d_clog2off above. |c0| for a high-skew c151 poly exceeds
  // FLT_MAX and would silently zero the whole sieve. Scale by 2^-CSCALE (exact) and let fflog2 add
  // CSCALE back. Target 2^100 so the 6-term Horner sum (<= 2^103) stays far inside float's 2^128.
  // Trigger at 2^124, NOT at the 2^100 target: the 6-term Horner sum is bounded by 8*|c|max, so
  // |c|max <= 2^124 can never overflow float's 2^128. Scaling only the polys that would actually
  // overflow keeps every previously-working poly on the EXACT old code path (verified by relation
  // md5 A/B), instead of perturbing it by one rounding of the +OFF add.
  { double cmax=0; for(int k=0;k<6;k++){ double m=fabs(cf[k]); if(m>cmax)cmax=m; }
    int CSCALE=0; if(cmax>0){ int e; frexp(cmax,&e); if(e>124) CSCALE=e-100; }
    if(CSCALE){
      for(int k=0;k<6;k++) cf[k]=ldexp(cf[k],-CSCALE);
      fprintf(stderr,"[poly] |c|max=%.4e exceeds float range; scaling coefficients by 2^-%d for the "
                     "norm gate (exact, log offset restored in fflog2)\n",cmax,CSCALE);
    }
    h_clog2off=(float)CSCALE;
    cudaMemcpyToSymbol(d_clog2off,&h_clog2off,sizeof(float));
  }
  int SLACK=10; double SKEW=G_SKEW;
  // ALG_TIGHT (runtime, env): subtract N bits from the ALGEBRAIC gate only (rational untouched).
  // Runtime rather than compile-time so one build sweeps the whole range. With the rational side
  // removed (§3) the algebraic gate is the ONLY selector and its SCATOFF=10 optimum -- tuned when
  // both sides were sieved -- is no longer necessarily right. Optimize effective_ms = ms/lat / retention,
  // NOT raw ms/lat. 0 = current behaviour, byte-identical.
  int ALG_TIGHT = getenv("ALG_TIGHT") ? atoi(getenv("ALG_TIGHT")) : 0;
  for(int k=0;k<6;k++)mpz_init_set_str(G_cc[k],C[k],10); mpz_init_set_str(G_y0,sY0,10); mpz_init_set_str(G_y1,sY1,10);
  long long QMIN=atoll(argv[2]), QMAX=atoll(argv[3]);
  g_dupmode = getenv("DUPSUP") ? atoi(getenv("DUPSUP")) : 0;
  g_dupqmin = getenv("DUP_QMIN") ? atoll(getenv("DUP_QMIN")) : QMIN;
  g_xqmin = getenv("DUPX_QMIN") ? atoll(getenv("DUPX_QMIN")) : 0;
  g_xqmax = getenv("DUPX_QMAX") ? atoll(getenv("DUPX_QMAX")) : 0;
  if(g_dupmode) fprintf(stderr,"[DUPSUP] mode=%d (%s)  dup_qmin=%lld  region i=[-%d,%d) j=[0,%d)\n",
      g_dupmode, g_dupmode==1?"suppress":"count-only", g_dupqmin, I2, I2, J);
  #pragma omp parallel
  { volatile int w=0; for(int z=0;z<1000;z++)w+=z; }     // warm OMP pool

  // Q initialised to QMIN-1 (was uninitialised): the ZERO_BUDGET path can reach the end-of-run
  // NEXTQ report without the special-q loop ever assigning Q, and an indeterminate value there
  // would be written into <dump>.nextq and corrupt the next resume's start point.
  long long a0,b0,a1,b1,Q=QMIN-1; double logq; u64 rq[8];
  cudaEvent_t ev0,ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
  const int PROF=getenv("PROF")!=0;
  // Launch geometry was hardcoded at <<<GRID,BLK>>> (188 = SM count) and never swept.
  // GRIDMUL/BLK make it tunable so it can be A/B'd; the kernels are all grid-stride loops,
  // so ANY geometry is correctness-neutral -- only the scheduling changes.
  // GRIDMUL default 32 -> 64: Blackwell (RTX PRO 6000, sm_120) has far more SMs than the retired
  // Max-Q dev box the old "32" was tuned on; 32 under-populates it. Measured sweep on target HW:
  // GRIDMUL 32=7.47, 64=7.39, 96=7.42, 128=7.46 ms/lat -> 64 optimal. Correctness-neutral (pure
  // launch geometry); +2.1% rel/s at identical rel/lat, relations byte-identical (sorted-set diff=0).
  const int GRIDMUL=getenv("GRIDMUL")?atoi(getenv("GRIDMUL")):128;
  const int BLK=getenv("BLK")?atoi(getenv("BLK")):256;
  const int GRID=188*GRIDMUL;
  if(getenv("GRIDMUL")||getenv("BLK")) fprintf(stderr,"launch geometry: <<<%d,%d>>>\n",GRID,BLK);
  const int PIPE=getenv("NOPIPE")==0;   // GPU(lattice N+1) overlaps CPU(lattice N); NOPIPE=1 to disable
  cudaEvent_t pe[8]; for(int z=0;z<8;z++) cudaEventCreate(&pe[z]);
#ifdef MEMSET_OVERLAP
  cudaStream_t sms; cudaStreamCreate(&sms);      // side stream: overlap sieve-array clear with project
  cudaEvent_t msev; cudaEventCreate(&msev);
#endif
  // ---- issue the whole GPU kernel stream for one lattice (NO host sync -> GPU runs while host works) ----
  auto issue=[&](long long A0,long long B0,long long A1,long long B1,long long QQ,double LQ){
    a0=A0;b0=B0;a1=A1;b1=B1;Q=QQ;logq=LQ;
    cudaEventRecord(ev0);
#ifdef MEMSET_OVERLAP
    // MEASURED NEGATIVE (2026-07-24, default-off, kept only to document the dead end):
    // Clearing the sieve arrays on a side stream to overlap the 67 MB memset with the project kernels
    // is a REGRESSION of +0.06 ms/lat (7.31 -> 7.37, +0.8%), reproducible across reps: project rises
    // 0.20 -> 0.23 every time. Correctness-neutral (relations byte-identical, same md5), but the async
    // memset and project both pull DRAM, so they contend for the same bus instead of overlapping --
    // there is no spare memory bandwidth to hide the memset in. Independently reconfirms the
    // L2/DRAM-bound floor (Pipeline.md 1.3). Do NOT enable.
  #ifndef RAT_NONE
    cudaMemsetAsync(dLr,0,NCELL,sms);
  #endif
    cudaMemsetAsync(dLa,0,NCELL,sms);
    cudaEventRecord(msev,sms);
    for(int g=0;g<4;g++) project<<<GRID/2,BLK>>>(dM[g],dRin[g],dT[g],dRout[g],ng[g],a0,b0,a1,b1);
    if(PROF)cudaEventRecord(pe[0]);
    cudaStreamWaitEvent(0,msev,0);   // scatter on stream 0 waits for the async clears
#else
    for(int g=0;g<4;g++) project<<<GRID/2,BLK>>>(dM[g],dRin[g],dT[g],dRout[g],ng[g],a0,b0,a1,b1);
    if(PROF)cudaEventRecord(pe[0]);
#ifdef TILE_COL
    // fused init: scat_tile stores (not adds) a zero-initialised shared tile over every cell of
    // both arrays, so the 2 x 33.5 MB clear here is redundant work. See scat_tile.
#else
#ifndef RAT_NONE
    cudaMemset(dLr,0,NCELL);              // skipped under RAT_NONE: dLr is never written nor read
#endif
    cudaMemset(dLa,0,NCELL);
#endif
#endif
#ifdef BUCKET_SIEVE
    cudaMemset(dBucketCntR,0,N_STRIPES*4);cudaMemset(dBucketCntA,0,N_STRIPES*4);
    // Phase 1: scat_col (column primes ≤T) writes directly to arr (unchanged);
    //          bucket_fill_lat (lat primes >T) fills per-stripe buckets — no arr writes.
    scat_col<<<GRID,BLK>>>(dLr,dM[0],dRout[0],dL[0],ng[0]);
    bucket_fill_lat<<<(ng[1]+255)/256,256>>>(dM[1],dRout[1],dL[1],ng[1],dBucketR,dBucketCntR,dBA[1]);
    scat_col<<<GRID,BLK>>>(dLa,dM[2],dRout[2],dL[2],ng[2]);
    bucket_fill_lat<<<(ng[3]+255)/256,256>>>(dM[3],dRout[3],dL[3],ng[3],dBucketA,dBucketCntA,dBA[3]);
    // Phase 2: accumulate buckets into arr via shared-memory tile (scat_col already done).
    bucket_flush_lat<<<N_STRIPES,256>>>(dLr,dBucketR,dBucketCntR);
    bucket_flush_lat<<<N_STRIPES,256>>>(dLa,dBucketA,dBucketCntA);
#else
    // Direct scattered atomics into the L2-resident sieve array. scat_lat also CACHES the
    // Gauss-reduced basis into dBA[] for resieve_lat to reuse (same contract as bucket_fill_lat).
    // NOTE: keep scat_col and scat_lat INTERLEAVED. scat_lat alone cannot fill the GPU
    // (one thread per line, highly variable work); running scat_col concurrently fills the
    // idle slots. Grouping them (col,col,lat,lat) measured +1.6 ms/lat = +10% GPU. Do not
    // "tidy" this into two groups -- and note the PROF scatter split needs the grouped order,
    // so measure it in a throwaway build, not here.
#if defined(FLAT_COOP)
    scat_col<<<GRID,BLK>>>(dLr,dM[0],dRout[0],dL[0],ng[0]);
    scat_col<<<GRID,BLK>>>(dLa,dM[2],dRout[2],dL[2],ng[2]);
    for(int g=1;g<4;g+=2){
      uint8_t* arr=(g==1)?dLr:dLa;
      cudaMemset(dVC[g],0,4);
      scat_lat_coop<<<GRID,BLK>>>(arr,dM[g],dRout[g],dL[g],ng[g],dBA[g],dBND[g],dTC[g],dVL[g],dVC[g]);
      vert_scat<<<1,64>>>(arr,dRout[g],dL[g],dVL[g],dVC[g]);
      cub::DeviceScan::ExclusiveSum(dScanTmp[g],scanBytes[g],dTC[g],dOff[g],ng[g]);
      fin_off<<<1,1>>>(dOff[g],dTC[g],ng[g]);
    }
#elif defined(FLAT_RESIEVE)
    // Box walk for scatter (its BA reads are coalesced: thread==line) but it ALSO emits BND/TC
    // so the resieve can use the uniform-work flat walk, which measured 1.9x faster there.
    scat_col<<<GRID,BLK>>>(dLr,dM[0],dRout[0],dL[0],ng[0]);
    scat_col<<<GRID,BLK>>>(dLa,dM[2],dRout[2],dL[2],ng[2]);
    for(int g=1;g<4;g+=2){
      uint8_t* arr=(g==1)?dLr:dLa;
      cudaMemset(dVC[g],0,4);
      scat_lat<<<GRID,BLK>>>(arr,dM[g],dRout[g],dL[g],ng[g],dBA[g],dBND[g],dTC[g],dVL[g],dVC[g]);
      cub::DeviceScan::ExclusiveSum(dScanTmp[g],scanBytes[g],dTC[g],dOff[g],ng[g]);
      fin_off<<<1,1>>>(dOff[g],dTC[g],ng[g]);
    }
#elif defined(FLAT_WALK)
    scat_col<<<GRID,BLK>>>(dLr,dM[0],dRout[0],dL[0],ng[0]);
    scat_col<<<GRID,BLK>>>(dLa,dM[2],dRout[2],dL[2],ng[2]);
    for(int g=1;g<4;g+=2){
      uint8_t* arr=(g==1)?dLr:dLa;
      cudaMemset(dVC[g],0,4);
      setup_lat<<<GRID,BLK>>>(dM[g],dRout[g],ng[g],dBA[g],dBND[g],dTC[g],dVL[g],dVC[g]);   // A
      cub::DeviceScan::ExclusiveSum(dScanTmp[g],scanBytes[g],dTC[g],dOff[g],ng[g]);          // B
      fin_off<<<1,1>>>(dOff[g],dTC[g],ng[g]);
      walk_scat<<<GRID,BLK>>>(arr,dL[g],dBA[g],dBND[g],dOff[g],ng[g]);                     // C
      vert_scat<<<1,64>>>(arr,dRout[g],dL[g],dVL[g],dVC[g]);
    }
#elif defined(FLAT_COOP2) && defined(SCAT_BOXPRE)
    scat_col<<<GRID,BLK>>>(dLr,dM[0],dRout[0],dL[0],ng[0]);
    scat_col<<<GRID,BLK>>>(dLa,dM[2],dRout[2],dL[2],ng[2]);
    for(int g=1;g<4;g+=2){
      uint8_t* arr=(g==1)?dLr:dLa;
      cudaMemset(dVC[g],0,4);
      setup_lat<<<GRID,BLK>>>(dM[g],dRout[g],ng[g],dBA[g],dBND[g],dTC[g],dVL[g],dVC[g]);
      scat_box_pre<<<GRID,BLK>>>(arr,dL[g],dBA[g],dBND[g],ng[g]);
      vert_scat<<<1,64>>>(arr,dRout[g],dL[g],dVL[g],dVC[g]);
    }
#elif defined(FLAT_COOP2) && defined(SCAT_COOP_LEAN)
    // Cooperative lat scatter with the DEFAULT path's exact call shape (BA only, no BND/TC/vlist,
    // no vert_scat) -- verticals are scattered inline by the kernel above. Keeps scat_tile, which
    // is the sieve array's ONLY initialiser under -DTILE_COL.
    for(int g=1;g<4;g+=2){
      uint8_t* arr=(g==1)?dLr:dLa; int cg=g-1;
      scat_tile<<<W/TCOLS,BLK>>>(arr,dM[cg],dRout[cg],dL[cg],ng[cg]);
      scat_lat_coop<<<GRID,BLK>>>(arr,dM[g],dRout[g],dL[g],ng[g],dBA[g],0,0,0,0);
    }
#elif defined(FLAT_COOP2) && defined(SCAT_COOP)
    // coop scatter + (coop resieve recomputes from BA) -> NO scan, NO BND needed by resieve
    // !! FIXED 2026-08-08: this branch called scat_col for the col groups. Under -DTILE_COL that is
    // WRONG and silently corrupts every lattice: scat_tile is THE ARRAY'S ONLY INITIALISER (the
    // memset is dropped for it -- see the static_assert above scat_tile), so substituting scat_col
    // leaves the sieve array holding the PREVIOUS lattice's log mass. It accumulates, the survivor
    // gate passes almost everything, and the run produces garbage rather than running slowly.
    // Measured before the fix: nsurv 42,746 -> 23,805,896 and 52.9 -> 0.7 rel/lattice.
    // The branch predates -DTILE_COL and was never updated, which is why the cooperative scatter
    // has been dead code. Also restores the col(Lr),lat(Lr),col(La),lat(La) interleave the default
    // path documents as a measured 1.2 ms/lat win (scat_lat wants its side's array hot in L2).
    for(int g=1;g<4;g+=2){
      uint8_t* arr=(g==1)?dLr:dLa; int cg=g-1;
#ifdef TILE_COL
      scat_tile<<<W/TCOLS,BLK>>>(arr,dM[cg],dRout[cg],dL[cg],ng[cg]);
#else
      scat_col<<<GRID,BLK>>>(arr,dM[cg],dRout[cg],dL[cg],ng[cg]);
#endif
      cudaMemset(dVC[g],0,4);
      scat_lat_coop<<<GRID,BLK>>>(arr,dM[g],dRout[g],dL[g],ng[g],dBA[g],dBND[g],dTC[g],dVL[g],dVC[g]);
      vert_scat<<<1,64>>>(arr,dRout[g],dL[g],dVL[g],dVC[g]);
    }
#elif defined(FLAT_COOP2)
    // box scatter (emits BA that coop resieve reuses) -- current default win
    // KEEP THE INTERLEAVE col(Lr),lat(Lr),col(La),lat(La). Grouping it as col,col,lat,lat to let
    // PROF time the two kernels separately measured 8.17 -> 9.36 ms/lat (-15%): scat_lat wants its
    // side's sieve array still hot in L2 from scat_col. Correctness-neutral either way, so this
    // ordering is a real 1.2 ms optimization and must not be "tidied". (Split, measured in the
    // grouped config: scat_col 0.95 | scat_lat 5.28 -- so scat_lat is ~4.0 ms when interleaved.)
#ifdef FK_COL
    scat_col<<<GRID,BLK>>>(dLr,dM[0],dRout[0],dL[0],nsm[0]); scat_fk<<<GRID,BLK>>>(dLr,dM[0],dRout[0],dL[0],nsm[0],ng[0]);
    scat_lat<<<GRID,BLK>>>(dLr,dM[1],dRout[1],dL[1],ng[1],dBA[1],0,0,0,0);
    scat_col<<<GRID,BLK>>>(dLa,dM[2],dRout[2],dL[2],nsm[2]); scat_fk<<<GRID,BLK>>>(dLa,dM[2],dRout[2],dL[2],nsm[2],ng[2]);
    scat_lat<<<GRID,BLK>>>(dLa,dM[3],dRout[3],dL[3],ng[3],dBA[3],0,0,0,0);
#else
    // RAT_NOLAT: drop the RATIONAL large-prime scatter (group 1, p>T=131072). The algebraic side
    // -- the harder side, where the selectivity lives -- stays fully sieved. Keeps the
    // col(Lr),lat(Lr),col(La),lat(La) interleave for the surviving groups; with group 1's lat gone
    // the rational pair degenerates to col only, which is exactly the intent.
#ifdef FB_BANDS
    // one launch per (side, band); events 1..4 = rational R0..R3, 5..8 = algebraic A0..A3
    cudaEventRecord(bev[0]);
    for(int b=0;b<4;b++){
      int g=(b<2)?0:1, s=bs[g][b], nb=bs[g][b+1]-s;
      if(nb>0){ if(g==0) scat_col<<<GRID,BLK>>>(dLr,dM[0]+s,dRout[0]+s,dL[0]+s,nb);
                else     scat_lat<<<GRID,BLK>>>(dLr,dM[1]+s,dRout[1]+s,dL[1]+s,nb,dBA[1]+4*(size_t)s,0,0,0,0); }
      cudaEventRecord(bev[1+b]); }
    for(int b=0;b<4;b++){
      int g=(b<2)?2:3, s=bs[g][b], nb=bs[g][b+1]-s;
      if(nb>0){ if(g==2) scat_col<<<GRID,BLK>>>(dLa,dM[2]+s,dRout[2]+s,dL[2]+s,nb);
                else     scat_lat<<<GRID,BLK>>>(dLa,dM[3]+s,dRout[3]+s,dL[3]+s,nb,dBA[3]+4*(size_t)s,0,0,0,0); }
      cudaEventRecord(bev[5+b]); }
#else
#ifdef TILE_COL
  #ifndef RAT_NONE
    scat_tile<<<W/TCOLS,BLK>>>(dLr,dM[0],dRout[0],dL[0],ng[0]);
  #endif
  #ifndef RAT_NOLAT
    scat_lat<<<GRID,BLK>>>(dLr,dM[1],dRout[1],dL[1],ng[1],dBA[1],0,0,0,0);
  #endif
    scat_tile<<<W/TCOLS,BLK>>>(dLa,dM[2],dRout[2],dL[2],ng[2]);scat_lat<<<GRID,BLK>>>(dLa,dM[3],dRout[3],dL[3],ng[3],dBA[3],0,0,0,0);
#else
#ifndef RAT_NONE
    scat_col<<<GRID,BLK>>>(dLr,dM[0],dRout[0],dL[0],ng[0]);
#endif
#ifndef RAT_NOLAT
    scat_lat<<<GRID,BLK>>>(dLr,dM[1],dRout[1],dL[1],ng[1],dBA[1],0,0,0,0);
#endif
    scat_col<<<GRID,BLK>>>(dLa,dM[2],dRout[2],dL[2],ng[2]);scat_lat<<<GRID,BLK>>>(dLa,dM[3],dRout[3],dL[3],ng[3],dBA[3],0,0,0,0);
#endif
#endif
#endif
#else
    scat_col<<<GRID,BLK>>>(dLr,dM[0],dRout[0],dL[0],ng[0]);scat_lat<<<GRID,BLK>>>(dLr,dM[1],dRout[1],dL[1],ng[1],dBA[1],0,0,0,0);
    scat_col<<<GRID,BLK>>>(dLa,dM[2],dRout[2],dL[2],ng[2]);scat_lat<<<GRID,BLK>>>(dLa,dM[3],dRout[3],dL[3],ng[3],dBA[3],0,0,0,0);
#endif
#endif
    if(PROF)cudaEventRecord(pe[1]);
#ifdef RAT_PROBE
    // counts survivors at RXN rational relaxations against the UNCHANGED algebraic gate.
    // Runs after the scatter and before the real scan; adds time, so never use a RAT_PROBE build
    // for a ms/lat number -- build without it for timing.
    scan_count_x<<<GRID/2,BLK>>>(dLa,dLr,dRX,SLACK,a0,b0,a1,b1,(float)logq,
      (float)cf[0],(float)cf[1],(float)cf[2],(float)cf[3],(float)cf[4],(float)cf[5],(float)Y0d,(float)Y1d);
#endif
    cudaMemset(dCnt,0,4);
#ifdef FUSED_SCAN
    int la_min=0,lr_min=0;
  #ifdef SCAN_EARLYOUT
    { float mnla=1e30f,mnlr=1e30f;                          // conservative per-lattice norm-log floor (sampled grid + margin)
      for(int si=0;si<=32;si++)for(int sj=0;sj<=32;sj++){
        long long ii=-I2+(long long)si*W/32, jj=(long long)sj*J/32;
        float a=(float)(a0*ii+a1*jj),b=(float)(b0*ii+b1*jj); if(b==0)continue;
        float nla=fflog2((float)cf[0],(float)cf[1],(float)cf[2],(float)cf[3],(float)cf[4],(float)cf[5],a,b)-(float)logq*SQ_ADJ_A;
        float Ga=(float)Y1d*a+(float)Y0d*b, nlr=log2f(fabsf(Ga))-(float)logq*SQ_ADJ_R;
        if(nla<mnla)mnla=nla; if(nlr<mnlr)mnlr=nlr; }
      const int MARGIN=12;   // safety vs host-sampling miss; raise if the md5 A/B ever differs
      la_min=(int)mnla-(58+SLACK+SCATOFF)-MARGIN; if(la_min<0)la_min=0; if(la_min>255)la_min=255;
      lr_min=(int)mnlr-(57+SLACK+SCATOFF)-MARGIN; if(lr_min<0)lr_min=0; if(lr_min>255)lr_min=255;
    }
  #endif
    cudaMemset(dBits,0,(NCELL+31)/32*4);                    // fused kernel SETS the bits -> zero first
    scan_idx_fused<<<GRID/2,BLK>>>(dLa,dLr,dSidx,dCnt,SLACK,a0,b0,a1,b1,(float)logq,(float)cf[0],(float)cf[1],(float)cf[2],(float)cf[3],(float)cf[4],(float)cf[5],(float)Y0d,(float)Y1d,dBits,dSI,dSJ,la_min,lr_min,ALG_TIGHT);
    if(PROF)cudaEventRecord(pe[2]);
    cudaMemset(dCa,0,(size_t)MAXSURV*4);cudaMemset(dCr,0,(size_t)MAXSURV*4);
#else
    scan_idx<<<GRID/2,BLK>>>(dLa,dLr,dSidx,dCnt,SLACK,a0,b0,a1,b1,(float)logq,(float)cf[0],(float)cf[1],(float)cf[2],(float)cf[3],(float)cf[4],(float)cf[5],(float)Y0d,(float)Y1d);
    if(PROF)cudaEventRecord(pe[2]);
    cudaMemset(dCa,0,(size_t)MAXSURV*4);cudaMemset(dCr,0,(size_t)MAXSURV*4);   // over-allocate -> no need to read nsurv on host
    cudaMemset(dBits,0,(NCELL+31)/32*4); mkmask<<<GRID/2,BLK>>>(dSidx,dBits);
#endif
#if defined(FLAT_COOP2)
#ifdef FK_COL
    resieve_col<<<GRID,BLK>>>(dM[0],dP[0],dRout[0],nsm[0],dBits,dSidx,dPr,dCr);
    resieve_fk <<<GRID,BLK>>>(dM[0],dP[0],dRout[0],nsm[0],ng[0],dBits,dSidx,dPr,dCr);
    resieve_col<<<GRID,BLK>>>(dM[2],dP[2],dRout[2],nsm[2],dBits,dSidx,dPa,dCa);
    resieve_fk <<<GRID,BLK>>>(dM[2],dP[2],dRout[2],nsm[2],ng[2],dBits,dSidx,dPa,dCa);
    // !! FIXED 2026-08-08: this branch resieved ONLY the col groups (0,2). Groups 1,3 -- the LARGE
    // PRIME lat groups -- were never resieved, so survivors were found normally (nsurv unchanged)
    // but their large-prime factors were never recorded and nothing became a relation: 0.1 rel/lat.
    // Same defect class as the SCAT_COOP dispatch: this branch predates the lat/coop refactor and
    // was never updated, which is why FK_COL has been dead code.
    resieve_coop<<<GRID,BLK>>>(dM[1],dP[1],dBA[1],ng[1],dBits,dSidx,dPr,dCr,dRout[1]);
    resieve_coop<<<GRID,BLK>>>(dM[3],dP[3],dBA[3],ng[3],dBits,dSidx,dPa,dCa,dRout[3]);
#else
#ifdef FB_BANDS
    // events 10,11 = rational col R0,R1 ; 12,13 = algebraic col A0,A1 ; 14,15 / 16,17 = lat bands
    cudaEventRecord(bev[9]);
    for(int b=0;b<2;b++){ int s=bs[0][b],nb=bs[0][b+1]-s;
      if(nb>0) resieve_col<<<GRID,BLK>>>(dM[0]+s,dP[0]+s,dRout[0]+s,nb,dBits,dSidx,dPr,dCr);
      cudaEventRecord(bev[10+b]); }
    for(int b=0;b<2;b++){ int s=bs[2][b],nb=bs[2][b+1]-s;
      if(nb>0) resieve_col<<<GRID,BLK>>>(dM[2]+s,dP[2]+s,dRout[2]+s,nb,dBits,dSidx,dPa,dCa);
      cudaEventRecord(bev[12+b]); }
    for(int b=2;b<4;b++){ int s=bs[1][b],nb=bs[1][b+1]-s;
      if(nb>0) resieve_coop<<<GRID,BLK>>>(dM[1]+s,dP[1]+s,dBA[1]+4*(size_t)s,nb,dBits,dSidx,dPr,dCr,dRout[1]+s);
      cudaEventRecord(bev[12+b]); }
    vert_resieve<<<1,64>>>(dM[1],dP[1],dRout[1],dVL[1],dVC[1],dBits,dSidx,dPr,dCr);
    for(int b=2;b<4;b++){ int s=bs[3][b],nb=bs[3][b+1]-s;
      if(nb>0) resieve_coop<<<GRID,BLK>>>(dM[3]+s,dP[3]+s,dBA[3]+4*(size_t)s,nb,dBits,dSidx,dPa,dCa,dRout[3]+s);
      cudaEventRecord(bev[14+b]); }
    vert_resieve<<<1,64>>>(dM[3],dP[3],dRout[3],dVL[3],dVC[3],dBits,dSidx,dPa,dCa);
#else
#ifdef TILE_RESIEVE
  #ifndef RAT_NONE
    resieve_colr<<<GRID,BLK>>>(dM[0],dP[0],dRout[0],ng[0],dBits,dSidx,dPr,dCr);
  #endif
    resieve_colr<<<GRID,BLK>>>(dM[2],dP[2],dRout[2],ng[2],dBits,dSidx,dPa,dCa);
#else
#ifndef RAT_NONE
    resieve_col<<<GRID,BLK>>>(dM[0],dP[0],dRout[0],ng[0],dBits,dSidx,dPr,dCr);
#endif
    resieve_col<<<GRID,BLK>>>(dM[2],dP[2],dRout[2],ng[2],dBits,dSidx,dPa,dCa);
#endif
#ifndef RAT_NOLAT
    // Skipped under RAT_NOLAT: with group 1 never scattered, dBA[1] is stale and there is no
    // rational large-prime contribution to recover. In the full design these primes come back from
    // the CPU batch/remainder-tree stage instead.
    resieve_coop<<<GRID,BLK>>>(dM[1],dP[1],dBA[1],ng[1],dBits,dSidx,dPr,dCr,dRout[1]);
    vert_resieve<<<1,64>>>(dM[1],dP[1],dRout[1],dVL[1],dVC[1],dBits,dSidx,dPr,dCr);
#endif
    resieve_coop<<<GRID,BLK>>>(dM[3],dP[3],dBA[3],ng[3],dBits,dSidx,dPa,dCa,dRout[3]);
    vert_resieve<<<1,64>>>(dM[3],dP[3],dRout[3],dVL[3],dVC[3],dBits,dSidx,dPa,dCa);
#endif
#endif
#elif defined(FLAT_WALK) || defined(FLAT_RESIEVE) || defined(FLAT_COOP)
    resieve_col<<<GRID,BLK>>>(dM[0],dP[0],dRout[0],ng[0],dBits,dSidx,dPr,dCr);
    resieve_col<<<GRID,BLK>>>(dM[2],dP[2],dRout[2],ng[2],dBits,dSidx,dPa,dCa);
    walk_resieve<<<GRID,BLK>>>(dM[1],dP[1],dBA[1],dBND[1],dOff[1],ng[1],dBits,dSidx,dPr,dCr);
    vert_resieve<<<1,64>>>(dM[1],dP[1],dRout[1],dVL[1],dVC[1],dBits,dSidx,dPr,dCr);
    walk_resieve<<<GRID,BLK>>>(dM[3],dP[3],dBA[3],dBND[3],dOff[3],ng[3],dBits,dSidx,dPa,dCa);
    vert_resieve<<<1,64>>>(dM[3],dP[3],dRout[3],dVL[3],dVC[3],dBits,dSidx,dPa,dCa);
#else
    resieve_col<<<GRID,BLK>>>(dM[0],dP[0],dRout[0],ng[0],dBits,dSidx,dPr,dCr);resieve_lat<<<GRID,BLK>>>(dM[1],dP[1],dRout[1],ng[1],dBits,dSidx,dPr,dCr,dBA[1]);
    resieve_col<<<GRID,BLK>>>(dM[2],dP[2],dRout[2],ng[2],dBits,dSidx,dPa,dCa);resieve_lat<<<GRID,BLK>>>(dM[3],dP[3],dRout[3],ng[3],dBits,dSidx,dPa,dCa,dBA[3]);
#endif
#ifndef FUSED_SCAN
    mksij<<<GRID/2,BLK>>>(dSidx,dSI,dSJ);
#endif
    if(PROF)cudaEventRecord(pe[3]);
    cudaMemset(dSC,0,4);
#ifdef FAST_FINAL
    cofactor_ff<<<GRID/2,BLK>>>(dSI,dSJ,dCnt,dcf,hY0,hY1,(unsigned long long)Q,dPa,dCa,dPr,dCr,a0,b0,a1,b1,CFG_MFB1,CFG_MFB0,dsA,dsB,dsCA,dsCR,dsFA,dsNFA,dsFR,dsNFR,dSC,MAXSHIP);
#else
    cofactor<<<GRID/2,BLK>>>(dSI,dSJ,dCnt,dcf,hY0,hY1,(unsigned long long)Q,dPa,dCa,dPr,dCr,a0,b0,a1,b1,CFG_MFB1,CFG_MFB0,dsA,dsB,dsCA,dsCR,dsPA,dsNA,dsPR,dsNR,dSC,MAXSHIP);
#endif
    if(PROF)cudaEventRecord(pe[4]);
    cudaEventRecord(ev1);
  };
  // ---- wait for the issued GPU stream + copy its shipped candidates into double-buffer slot *b ----
  auto fetch=[&](ShipBuf*b){
    CK(cudaEventSynchronize(ev1));
    float gms; cudaEventElapsedTime(&gms,ev0,ev1); b->gpu_ms=gms;
    if(PROF){ float t; cudaEventElapsedTime(&t,ev0,pe[0]);g_prof[0]+=t; cudaEventElapsedTime(&t,pe[0],pe[1]);g_prof[1]+=t;
      cudaEventElapsedTime(&t,pe[1],pe[2]);g_prof[2]+=t; cudaEventElapsedTime(&t,pe[2],pe[3]);g_prof[3]+=t;
      cudaEventElapsedTime(&t,pe[3],pe[4]);g_prof[4]+=t;
#ifdef FB_BANDS
      for(int z=1;z<=8;z++){ cudaEventElapsedTime(&t,bev[z-1],bev[z]); g_bandms[z]+=t; }
      for(int z=10;z<=17;z++){ cudaEventElapsedTime(&t,bev[z-1],bev[z]); g_bandms[z]+=t; }
#endif
      // NOTE: no scat_col/scat_lat split here. Timing them separately requires grouping the
      // launches, which costs 15% (see the scatter block) -- the split is not worth the L2 loss.
      unsigned nsv; cudaMemcpy(&nsv,dCnt,4,cudaMemcpyDeviceToHost); g_prof[5]+=nsv; }
    unsigned nship; cudaMemcpy(&nship,dSC,4,cudaMemcpyDeviceToHost); if(nship>MAXSHIP)nship=MAXSHIP;
    b->nship=nship; b->Q=Q;                 // store this lattice's special-q with its candidates
    extern long g_nship_sum,g_lat_cnt; g_nship_sum+=nship; g_lat_cnt++;
    cudaMemcpy(b->A,dsA,8*nship,cudaMemcpyDeviceToHost);cudaMemcpy(b->B,dsB,8*nship,cudaMemcpyDeviceToHost);
    cudaMemcpy(b->CA,dsCA,8*nship,cudaMemcpyDeviceToHost);cudaMemcpy(b->CR,dsCR,8*nship,cudaMemcpyDeviceToHost);
#ifdef FAST_FINAL
    cudaMemcpy(b->NFA,dsNFA,4*nship,cudaMemcpyDeviceToHost);cudaMemcpy(b->NFR,dsNFR,4*nship,cudaMemcpyDeviceToHost);
    cudaMemcpy(b->FA,dsFA,(size_t)nship*48*4,cudaMemcpyDeviceToHost);cudaMemcpy(b->FR,dsFR,(size_t)nship*48*4,cudaMemcpyDeviceToHost);
#else
    cudaMemcpy(b->NA,dsNA,4*nship,cudaMemcpyDeviceToHost);cudaMemcpy(b->NR,dsNR,4*nship,cudaMemcpyDeviceToHost);
    cudaMemcpy(b->PA,dsPA,(size_t)nship*MAXP*4,cudaMemcpyDeviceToHost);cudaMemcpy(b->PR,dsPR,(size_t)nship*MAXP*4,cudaMemcpyDeviceToHost);
#endif
  };
  // ---- CPU: build full factorization + CADO relation format from slot *b (uses b->Q) ----
  auto process=[&](ShipBuf*bf,double* cpu_ms,long long* nrel)->void{
    unsigned nship=bf->nship; long long QB=bf->Q;
    long long *hsA=bf->A,*hsB=bf->B; uint32_t *hsPA=bf->PA,*hsPR=bf->PR; unsigned *hsNA=bf->NA,*hsNR=bf->NR;
    unsigned long long *hsCA=bf->CA,*hsCR=bf->CR; (void)hsPA;(void)hsPR;(void)hsNA;(void)hsNR;(void)hsCA;(void)hsCR;
#ifdef FAST_FINAL
    uint32_t *hsFA=bf->FA,*hsFR=bf->FR; unsigned *hsNFA=bf->NFA,*hsNFR=bf->NFR;
#endif
    double tc=wall_s();
    #pragma omp parallel
    { mpz_t F,G,t1,t2; mpz_init(F);mpz_init(G);mpz_init(t1);mpz_init(t2);   // thread-persistent (no per-relation malloc storm)
    #pragma omp for schedule(dynamic,8)
    for(int s=0;s<(int)nship;s++){ ok[s]=0; dupbk[s]=0; xown[s]=0;
      long long a=hsA[s],b=hsB[s];
      unsigned long long af[96],rf[96]; int naf=0,nrf=0; bool good=true;
#ifdef FAST_FINAL
      // step 2: the GPU shipped the complete Q+small+resieved factorization (with
      // multiplicity); copy it and factor only the residual cofactor -> NO GMP norm.
      unsigned nfa=hsNFA[s],nfr=hsNFR[s];
      if(nfa>48u||nfr>48u) good=false;
      else{
        for(unsigned t=0;t<nfa;t++) af[naf++]=hsFA[(size_t)s*48+t];
        for(unsigned t=0;t<nfr;t++) rf[nrf++]=hsFR[(size_t)s*48+t];
        if(!h_factor_cof(hsCA[s],af,&naf,LPB1_VAL)) good=false;
        if(good&&!h_factor_cof(hsCR[s],rf,&nrf,LPB0_VAL)) good=false;
      }
#else
      // |F(a,b)| algebraic (coeffs pre-parsed in G_cc -> no per-relation string parsing)
      mpz_set_ui(F,0); for(int k=0;k<=5;k++){ mpz_set(t1,G_cc[k]); for(int e=0;e<k;e++)mpz_mul_si(t1,t1,a); for(int e=0;e<5-k;e++)mpz_mul_si(t1,t1,b); mpz_add(F,F,t1);} mpz_abs(F,F);
      // |G(a,b)| rational = Y1*a+Y0*b
      mpz_mul_si(t1,G_y1,a); mpz_mul_si(t2,G_y0,b); mpz_add(G,t1,t2); mpz_abs(G,G);
      // special-q first, on whichever side it lives (SQSIDE_RAT -> rational G / rf, else algebraic F / af)
#ifdef SQSIDE_RAT
      while(mpz_divisible_ui_p(G,(unsigned long long)QB)){ rf[nrf++]=(unsigned long long)QB; mpz_divexact_ui(G,G,(unsigned long long)QB); if(nrf>=90){good=false;break;} }
#else
      while(mpz_divisible_ui_p(F,(unsigned long long)QB)){ af[naf++]=(unsigned long long)QB; mpz_divexact_ui(F,F,(unsigned long long)QB); if(naf>=90){good=false;break;} }
#endif
      for(int t=0;t<hNSP&&good;t++){ unsigned long long p=hSP[t]; while(mpz_divisible_ui_p(F,p)){ af[naf++]=p; mpz_divexact_ui(F,F,p); if(naf>=90){good=false;break;} } }
      for(unsigned t=0;t<hsNA[s]&&good;t++){ unsigned long long p=hsPA[(size_t)s*MAXP+t]; while(mpz_divisible_ui_p(F,p)){ af[naf++]=p; mpz_divexact_ui(F,F,p); if(naf>=90){good=false;break;} } }
      if(good){ if(mpz_sizeinbase(F,2)>62)good=false; else good=h_factor_cof(mpz_get_ui(F),af,&naf,LPB1_VAL); }
      // rational: small primes (not resieved), resieved rat primes, then cofactor residual
      for(int t=0;t<hNSP&&good;t++){ unsigned long long p=hSP[t]; while(mpz_divisible_ui_p(G,p)){ rf[nrf++]=p; mpz_divexact_ui(G,G,p); if(nrf>=90){good=false;break;} } }
      for(unsigned t=0;t<hsNR[s]&&good;t++){ unsigned long long p=hsPR[(size_t)s*MAXP+t]; while(mpz_divisible_ui_p(G,p)){ rf[nrf++]=p; mpz_divexact_ui(G,G,p); if(nrf>=90){good=false;break;} } }
      if(good){ if(mpz_sizeinbase(G,2)>62)good=false; else good=h_factor_cof(mpz_get_ui(G),rf,&nrf,LPB0_VAL); }
#endif
      if(good){
        qsort(af,naf,8,cmp_u64); qsort(rf,nrf,8,cmp_u64);
        // ---- duplicate detection: is some SMALLER special-q in range also an owner? ----
        // Runs on the sorted side that carries the special-q, so the scan stops at the
        // first factor >= QB. Cost is one skew_reduce per in-range smaller factor.
        if(g_dupmode){
#ifdef SQSIDE_RAT
          const unsigned long long* qf=rf; int nqf=nrf;
#else
          const unsigned long long* qf=af; int nqf=naf;
#endif
          // Scan DOWNWARD from the largest factor below QB: the first hit is the largest
          // covering owner, which is exactly what the threshold histogram needs.
          int t=nqf-1; while(t>=0 && (long long)qf[t]>=QB) t--;
          for(;t>=0;t--){
            long long p=(long long)qf[t];
            if(p<g_dupqmin) break;                 // sorted ascending: nothing bigger left below
            if(t+1<nqf && qf[t]==qf[t+1]) continue;// repeated factor: same lattice, test once
            if(dup_covered_by(a,b,p,G_SKEW)){
              ok[s]=2; int bk=0; while((1LL<<(bk+1))<=p) bk++; dupbk[s]=(unsigned char)(bk+1);
              break;
            }
          }
          if(ok[s]==2 && g_dupmode==1) goto rel_done;   // mode 1 = suppress; mode 2 = count only
          // cross-sweep: would a special-q sweep on the OTHER side over [g_xqmin,g_xqmax) own this?
          if(g_xqmax>g_xqmin){
#ifdef SQSIDE_RAT
            const unsigned long long* xf=af; int nxf=naf;   // sq on rational -> cross-check algebraic
#else
            const unsigned long long* xf=rf; int nxf=nrf;   // sq on algebraic -> cross-check rational
#endif
            bool owned=false;
            for(int t=0;t<nxf;t++){
              long long p=(long long)xf[t];
              if(p<g_xqmin) continue;
              if(p>=g_xqmax) break;
              if(t&&xf[t]==xf[t-1]) continue;
              if(dup_covered_by(a,b,p,G_SKEW)){ owned=true; break; }
            }
            xown[s]=owned?1:0;
          }
        }
        {
        char* o=relbuf+(size_t)s*640; int p=0;
        long long A=a,B=b; if(B<0){A=-A;B=-B;}
        p+=sprintf(o+p,"%lld,%lld:",A,B);
        for(int t=0;t<nrf;t++)p+=sprintf(o+p,t?",%llx":"%llx",rf[t]);   // rational side (side 0) first
        o[p++]=':';
        for(int t=0;t<naf;t++)p+=sprintf(o+p,t?",%llx":"%llx",af[t]);   // algebraic side (side 1)
        o[p++]='\n'; o[p]=0; ok[s]=(ok[s]==2)?3:1;   // 3 = emitted but flagged dup (count-only mode)
        }
      }
      rel_done: ;
    }
    mpz_clear(F);mpz_clear(G);mpz_clear(t1);mpz_clear(t2);
    }
    long long r=0,d=0; for(unsigned s=0;s<nship;s++){ if(ok[s]==1||ok[s]==3)r++; if(ok[s]==2||ok[s]==3){d++; g_dupbk[dupbk[s]&63]++;}
      if(g_xqmax>g_xqmin && ok[s]!=0 && ok[s]!=2){ if(xown[s]) g_xowned++; else g_xnew++; } }
    *cpu_ms=(wall_s()-tc)*1000; *nrel=r; g_dupfound+=d; g_dupvalid+=r+ (g_dupmode==1?d:0);
    for(unsigned s=0;s<nship;s++) if(ok[s]==1||ok[s]==3) rel_emit(relbuf+(size_t)s*640);   // -> RAM buffer (not disk)
  };
  // warmup: first valid (q,rho) in range for the LOADED poly (then discard its relations)
  { long long bas[4]; u64 wr[8];
    for(long long q=QMIN;q<QMAX;q++){ if(!isprime_q(q))continue;
#ifdef SQSIDE_RAT
      int nr=rat_roots(q,wr);
#else
      int nr=find_roots(C,q,wr);
#endif
      if(!nr)continue;
      skew_reduce(q,(long long)wr[0],SKEW,bas);
      issue(bas[0],bas[1],bas[2],bas[3],q,log2((double)q)); fetch(&sb[0]); double c; long long n; process(&sb[0],&c,&n); break; }
    g_ramlen=0;   // discard warmup relations from the RAM buffer
  }
  // ---- SPECIAL-Q LOOP (relations accumulate in RAM). Pipelined: issue GPU(N+1), finalize CPU(N). ----
  const char* dumpf=argc>4?argv[4]:0;                // optional: also dump RAM buffer to a file for validation
  long long TARGET=argc>5?atoll(argv[5]):0;          // stop after this many relations (0 = whole range)
  // GPULOOP_APPEND=1 opens the dump for APPEND instead of truncate, so a later invocation can
  // extend an existing relation set. Used by factor_msv.sh's adaptive filter ladder: sieve to a
  // checkpoint, test the matrix, and if the filter wants more, resume from the recorded next q
  // instead of restarting the whole sweep (a restart would re-emit the same relations as dups).
  const int APPEND=getenv("GPULOOP_APPEND")!=0;
  if(dumpf && strcmp(dumpf,"/dev/null")!=0){ g_dumpf=fopen(dumpf,APPEND?"a":"w"); g_dumpflushed=0;   // crash-safe incremental dump to disk
    if(!g_dumpf) fprintf(stderr,"WARN: cannot open dump file %s (relations will stay in RAM only)\n",dumpf); }
#ifdef PROBE_WARP
  { unsigned long long z=0; cudaMemcpyToSymbol(g_warp_max,&z,sizeof(z)); cudaMemcpyToSymbol(g_warp_sum,&z,sizeof(z));
    for(int g=1;g<4;g+=2) probe_warp_util<<<GRID,BLK>>>(dM[g],dRout[g],ng[g]);
    cudaDeviceSynchronize();
    unsigned long long wmax=0,wsum=0;
    cudaMemcpyFromSymbol(&wmax,g_warp_max,sizeof(wmax)); cudaMemcpyFromSymbol(&wsum,g_warp_sum,sizeof(wsum));
    printf("  [PROBE-WARP] scat_lat box walk: useful lane-iters %.1fM | warp-cost lane-iters %.1fM | WARP UTILISATION %.1f%%\n",
      (double)wsum/1e6,(double)wmax*32.0/1e6, 100.0*(double)wsum/((double)wmax*32.0)); }
#endif
  double tot_gpu=0,tot_cpu=0; long long tot_rel=0; int nq=0; double t0w=wall_s();
  // Wall-time budget (env GPULOOP_MAX_SECS, 0=none): stop sieving when the time is up so the caller
  // can still run the downstream. Lets a low-yield key USE its full budget (instead of quitting early
  // at QMAX) to gather as many relations as possible. Additive: only ever stops earlier than QMAX.
  // !! FIXED 2026-07-31 -- the old form was `getenv(..)?atof(..):0` with every deadline check
  // guarded by `SIEVE_MAX>0`, which conflated two OPPOSITE meanings: "env absent -> no limit" and
  // "env present but <=0 -> no time left". A caller that computed a budget of 0 or less therefore
  // got an UNLIMITED sieve. Measured in the field: factor_msv.sh passed GPULOOP_MAX_SECS=-12 after
  // a time-bound phase A, and the sieve ran straight through the 4h wall, consumed the downstream
  // reserve, produced no matrix and was SIGKILLed with no factors.
  //   absent        -> unlimited (unchanged)
  //   present, >0   -> deadline  (unchanged)
  //   present, <=0  -> ZERO budget, sieve nothing (new; the only safe reading)
  const char* _smax_env=getenv("GPULOOP_MAX_SECS");
  double SIEVE_MAX=_smax_env?atof(_smax_env):0;
  const bool ZERO_BUDGET=(_smax_env!=NULL && SIEVE_MAX<=0);
  ShipBuf *cur=&sb[0],*nxt=&sb[1]; bool pending=false, stop=ZERO_BUDGET;
  if(ZERO_BUDGET) fprintf(stderr,"[sieve] GPULOOP_MAX_SECS=%s <= 0: ZERO time budget, sieving nothing "
                                 "(previously this meant 'unlimited' and ran past the wall)\n",_smax_env);
  auto progress=[&](){ if((nq%2000)==0){ dump_flush(); double el=wall_s()-t0w;
      fprintf(stderr,"[%.0fs] q=%lld  lattices=%d  relations=%lld  %.1fms/lat (GPU %.1f)  RAM %.2fGB  ETA(71M) %.2fh\n",
        el,Q,nq,tot_rel,el*1000/nq,tot_gpu/nq, g_ramlen/1e9, (el/tot_rel)*71e6/3600.0);
      fprintf(stderr,"        avg nship(cofactors to CPU-factor)/lattice = %.0f\n", (double)g_nship_sum/g_lat_cnt); } };
  // ---- MULTI-WINDOW MODE (env GPULOOP_WINS="qa:qb qa:qb ...") ----------------------------------
  // build_fb() is special-q INDEPENDENT and costs 4.9s (measured, c151 on sm_120), yet the
  // polyselect bake-off probed each candidate with a SEPARATE process per q-window: 6 candidates
  // x 3 windows = 18 factor-base builds where 6 suffice. This lets one process sweep several
  // disjoint windows on one FB, printing a per-window "avg rel/lattice" so the caller still scores
  // each window exactly as before. Measured: 18 probes 233s -> 6 probes 96s.
  // Absent/empty -> exactly one window [QMIN,QMAX), i.e. byte-identical to the previous behaviour.
  std::vector<long long> WQA, WQB;
  { const char* we=getenv("GPULOOP_WINS");
    if(we&&*we){ const char* s=we;
      while(*s){ while(*s==' '||*s==','||*s=='\t')s++; if(!*s)break;
        char* e; long long a=strtoll(s,&e,10); if(e==s)break; s=e;
        if(*s==':'||*s=='-')s++; else break;
        long long b=strtoll(s,&e,10); if(e==s)break; s=e;
        if(b>a){ WQA.push_back(a); WQB.push_back(b); } } }
    if(WQA.empty()){ WQA.push_back(QMIN); WQB.push_back(QMAX); }
    else fprintf(stderr,"[sieve] GPULOOP_WINS: %d window(s) on ONE factor base\n",(int)WQA.size()); }
  const bool MULTIWIN = WQA.size()>1;
  for(size_t wi=0; wi<WQA.size() && !stop; wi++){
  const int   nq_w0=nq; const long long rel_w0=tot_rel; const double tw0=wall_s();
  for(long long q=WQA[wi];q<WQB[wi] && !stop;q++){
    if(!isprime_q(q))continue;
#ifdef SQSIDE_RAT
    int nr=rat_roots(q,rq);
#else
    int nr=find_roots(C,q,rq);
#endif
    for(int k=0;k<nr && !stop;k++){
      long long bas[4]; skew_reduce(q,(long long)rq[k],SKEW,bas);
      if(PIPE){
        issue(bas[0],bas[1],bas[2],bas[3],q,log2((double)q));   // GPU for THIS lattice (async)
        if(pending){ double c; long long n; process(cur,&c,&n);  // CPU for PREVIOUS lattice, overlaps GPU above
          tot_gpu+=cur->gpu_ms; tot_cpu+=c; tot_rel+=n; nq++; progress();
          if(TARGET&&tot_rel>=TARGET)stop=true;
          if(SIEVE_MAX>0&&wall_s()-t0w>SIEVE_MAX)stop=true; }
        fetch(nxt);                                             // sync THIS GPU, copy ships into nxt
        ShipBuf*tmp=cur;cur=nxt;nxt=tmp; pending=true;          // THIS lattice becomes the pending one
      } else {
        issue(bas[0],bas[1],bas[2],bas[3],q,log2((double)q)); fetch(cur);
        double c; long long n; process(cur,&c,&n);
        tot_gpu+=cur->gpu_ms; tot_cpu+=c; tot_rel+=n; nq++; progress();
        if(TARGET&&tot_rel>=TARGET)stop=true;
        if(SIEVE_MAX>0&&wall_s()-t0w>SIEVE_MAX)stop=true;
      }
    }
  }
  // End of ONE window. Only in MULTIWIN do we drain here -- so the in-flight lattice is counted in
  // the window that produced it and cannot leak into the next one. The single-window path keeps
  // the original drain below, untouched, so production behaviour is byte-identical.
  if(MULTIWIN && PIPE && pending && !stop){ double c; long long n; process(cur,&c,&n); tot_gpu+=cur->gpu_ms; tot_cpu+=c; tot_rel+=n; nq++; pending=false; }
  if(MULTIWIN){ int dq=nq-nq_w0; long long dr=tot_rel-rel_w0;
    printf("  [WIN %lld:%lld] lattices %d  relations %lld  avg %.1f rel/lattice  wall %.2f s\n",
           WQA[wi],WQB[wi],dq,dr,dq?(double)dr/dq:0.0,wall_s()-tw0); fflush(stdout); }
  }
  if(SIEVE_MAX>0&&wall_s()-t0w>SIEVE_MAX&&(!TARGET||tot_rel<TARGET))
    fprintf(stderr,"[sieve] WALL-TIME BUDGET %.0fs reached with %lld/%lld relations (q=%lld) -- time-bound, not target\n",SIEVE_MAX,tot_rel,TARGET,Q);
  if(PIPE && pending && !stop){ double c; long long n; process(cur,&c,&n); tot_gpu+=cur->gpu_ms; tot_cpu+=c; tot_rel+=n; nq++; }  // drain last
  double wall_tot=wall_s()-t0w;
  dump_flush(); if(g_dumpf){ fclose(g_dumpf); g_dumpf=0; }   // final flush + close (incremental crash-safe disk dump)
  // Record where the sweep stopped so a resume can continue from the next special-q rather than
  // re-sieving (and re-emitting) the range already covered. Written next to the dump.
  // !ZERO_BUDGET: never overwrite the resume marker when nothing was sieved -- the existing file
  // still holds the correct continuation point, and clobbering it would restart the sweep and
  // re-emit relations already banked.
  if(!ZERO_BUDGET && dumpf && strcmp(dumpf,"/dev/null")!=0){ char nq[4096]; snprintf(nq,sizeof(nq),"%s.nextq",dumpf);
    FILE* f=fopen(nq,"w"); if(f){ fprintf(f,"%lld\n",Q+1); fclose(f); } }
  printf("  NEXTQ=%lld\n",Q+1);
  printf("\n=== SUSTAINED GPU SIEVER over special-q [%lld,%lld] ===\n",QMIN,QMAX);
  printf("  (q,rho) lattices: %d   relations: %lld   (avg %.1f rel/lattice)\n",nq,tot_rel,(double)tot_rel/nq);
  if(g_dupmode){
    long long valid = (g_dupmode==1)? tot_rel+g_dupfound : tot_rel;   // relations before suppression
    printf("  [DUPSUP] valid %lld  dup-of-smaller-q %lld (%.2f%%)  new %lld  -> NEW/lat %.2f  (dup_qmin=%lld)\n",
      valid, g_dupfound, valid?100.0*g_dupfound/valid:0.0, valid-g_dupfound,
      nq?(double)(valid-g_dupfound)/nq:0.0, g_dupqmin);
    // Reading this: "T=x" is the hypothetical special-q range start. A relation only
    // duplicates work already done if its largest covering owner is >= T, so the tail sum
    // of this histogram from bucket(T) upward IS the duplicate rate for a sweep starting
    // at T. NEW/lat is then the marginal unique yield of a lattice at this q.
    printf("  [DUPHIST] largest-earlier-owner distribution and implied yield if the sweep started at T:\n");
    long long tail=0;
    for(int bk=63;bk>=1;bk--){
      if(!g_dupbk[bk]) continue;
      tail+=g_dupbk[bk];
    }
    long long cum=0;
    for(int bk=1;bk<64;bk++){
      if(!g_dupbk[bk]) continue;
      long long ge=tail-cum;                       // owners with p >= 2^(bk-1)
      printf("     T=2^%-2d (%10.3fM):  dup %8lld (%5.2f%%)   NEW/lat %6.2f\n",
        bk-1, (1LL<<(bk-1))/1e6, ge, valid?100.0*ge/valid:0.0,
        nq?(double)(valid-ge)/nq:0.0);
      cum+=g_dupbk[bk];
    }
    if(g_xqmax>g_xqmin){
      long long tot=g_xowned+g_xnew;
      printf("  [DUPX] vs other-side sweep [%lldM,%lldM): of %lld own-new relations, %lld (%.2f%%) also\n"
             "         reachable there, %lld (%.2f%%) are UNIQUE to this sweep -> UNIQUE/lat %.2f\n",
        g_xqmin/1000000,g_xqmax/1000000,tot,g_xowned,tot?100.0*g_xowned/tot:0.0,
        g_xnew,tot?100.0*g_xnew/tot:0.0, nq?(double)g_xnew/nq:0.0);
    }
  }
  printf("  per-lattice: GPU %.2f ms   CPU %.2f ms   wall %.2f ms   (vs CADO ~28 ms)\n",tot_gpu/nq,tot_cpu/nq,wall_tot*1000/nq);
  if(PROF){ printf("  [PROF] ms/lat: project %.2f | scatter %.2f | scan %.2f | resieve %.2f | cofactor %.2f | avg nsurv %.0f\n",
    g_prof[0]/nq,g_prof[1]/nq,g_prof[2]/nq,g_prof[3]/nq,g_prof[4]/nq,g_prof[5]/nq);
    if(g_prof[6]||g_prof[7])
      printf("  [PROF]   scatter split: scat_col %.2f (small p, dense) | scat_lat %.2f (large p, %d lines/side)\n",
        g_prof[6]/nq,g_prof[7]/nq,ng[1]);
#ifdef FB_BANDS
    { const char* bn[4]={"R0 p<=4k    ","R1 4k..T    ","R2 T..1M    ","R3 1M..LIM  "};
      double rs=0,as=0,rr=0,ar=0;
      printf("  [FB_BANDS] per-band ms/lat (band | rat scat | rat resv | rat tot || alg scat | alg resv | alg tot)\n");
      // resieve launch order is rat-col, alg-col, rat-lat, alg-lat -> event ids per band:
      const int RRV[4]={10,11,14,15}, ARV[4]={12,13,16,17};
      for(int b=0;b<4;b++){
        double r1=g_bandms[1+b]/nq, r2=g_bandms[RRV[b]]/nq, a1=g_bandms[5+b]/nq, a2=g_bandms[ARV[b]]/nq;
        rs+=r1; rr+=r2; as+=a1; ar+=a2;
        printf("  [FB_BANDS] %s | %6.3f | %6.3f | %6.3f || %6.3f | %6.3f | %6.3f\n",bn[b],r1,r2,r1+r2,a1,a2,a1+a2); }
      printf("  [FB_BANDS] TOTAL        | %6.3f | %6.3f | %6.3f || %6.3f | %6.3f | %6.3f    (rat side = %.3f ms/lat)\n",
        rs,rr,rs+rr,as,ar,as+ar,rs+rr); }
#endif
  }
  printf("  total wall %.2f s\n",wall_tot);
#ifdef ALG_PASS_PROBE
  { unsigned long long ap[4]={0,0,0,0}; cudaMemcpyFromSymbol(ap,g_algp,sizeof(ap));
    double tot=(double)(ap[0]?ap[0]:1);
    printf("  [ALG PASS PROBE] survivors entering cofactor : %llu (%.0f/lat)\n",ap[0],ap[0]/(double)nq);
    printf("    pass algebraic SIZE test (bits<=MFB1)      : %llu = %.3f%%  (%.0f/lat)\n",ap[1],100.0*ap[1]/tot,ap[1]/(double)nq);
    printf("    pass FULL algebraic validation (+classify) : %llu = %.3f%%  (%.0f/lat)  <-- ships to CPU rational batch\n",ap[2],100.0*ap[2]/tot,ap[2]/(double)nq);
  }
#endif
#ifdef DUMP_RNORM
  { unsigned int cnt=0; cudaMemcpyFromSymbol(&cnt,g_rndump_cnt,sizeof(cnt));
    unsigned int n=cnt<DUMP_CAP?cnt:(unsigned)DUMP_CAP;
    unsigned long long* buf=(unsigned long long*)malloc((size_t)n*4*sizeof(unsigned long long));
    cudaMemcpyFromSymbol(buf,g_rndump,(size_t)n*4*sizeof(unsigned long long),0,cudaMemcpyDeviceToHost);
    const char* fn=getenv("DUMP_RNORM_FILE"); if(!fn)fn="/tmp/rnorms.bin";
    FILE* f=fopen(fn,"wb");
    if(f){ fwrite(buf,sizeof(unsigned long long),(size_t)n*4,f); fclose(f);
      fprintf(stderr,"[DUMP_RNORM] wrote %u rational norms (4 u64 LSW-first each; alg-validated) to %s (total seen %u)\n",n,fn,cnt); }
    else fprintf(stderr,"[DUMP_RNORM] could not open %s for writing\n",fn);
    free(buf);
  }
#endif
#ifdef DUMP_CANDS
  { unsigned int cnt=0; cudaMemcpyFromSymbol(&cnt,g_cand_cnt,sizeof(cnt));
    unsigned int n=cnt<CAND_CAP?cnt:(unsigned)CAND_CAP;
    unsigned long long* buf=(unsigned long long*)malloc((size_t)n*6*sizeof(unsigned long long));
    cudaMemcpy(buf,d_cand,(size_t)n*6*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
    const char* fn=getenv("DUMP_CANDS_FILE"); if(!fn)fn="/tmp/cands.bin";
    FILE* f=fopen(fn,"wb");
    if(f){ fwrite(buf,sizeof(unsigned long long),(size_t)n*6,f); fclose(f);
      extern long g_lat_cnt;
      fprintf(stderr,"[DUMP_CANDS] wrote %u candidates {a,b,Nr[4]} to %s (total seen %u over %ld lattices)\n",
              n,fn,cnt,g_lat_cnt); }
    else fprintf(stderr,"[DUMP_CANDS] could not open %s for writing\n",fn);
    free(buf);
  }
#endif
#ifdef RAT_PROBE
  { unsigned long long rx[RXN]; cudaMemcpy(rx,dRX,RXN*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
    printf("  [RAT PROBE] survivors/lattice vs rational-threshold relaxation"
#ifdef RAT_NOLAT
           "  (RAT_NOLAT: rational large primes NOT sieved)\n");
#else
           "  (baseline: full rational sieve)\n");
#endif
    for(int t=0;t<RXN;t++)
      printf("    +%2d bits : %12.0f /lat   (%.1fx baseline)\n",
             RXSTEP*t,(double)rx[t]/nq,(double)rx[t]/(double)(rx[0]?rx[0]:1));
  }
#endif
#ifdef PROBE_3LP
  { unsigned long long p[6]={0,0,0,0,0,0};
    cudaMemcpyFromSymbol(p,g_p3,sizeof(p));
    double base=(double)p[1];
    printf("  [3LP PROBE] survivors at size filter: %llu\n",p[0]);
    printf("    pass 2LP both sides (CURRENT)    : %llu   (baseline)\n",p[1]);
    printf("    + 3LP alg side only              : %llu   (+%.1f%%)\n",p[2],base?100.0*p[2]/base:0);
    printf("    + 3LP rat side only              : %llu   (+%.1f%%)\n",p[3],base?100.0*p[3]/base:0);
    printf("    + 3LP both sides                 : %llu   (+%.1f%%)\n",p[4],base?100.0*p[4]/base:0);
    printf("    still dead even with 3LP         : %llu\n",p[5]);
    printf("    => size-filter ceiling for 3LP   : %.2fx more candidates\n",
           base?(double)(p[1]+p[2]+p[3]+p[4])/base:0);
    printf("    (upper bound: passing the size filter is necessary, not sufficient --\n");
    printf("     each candidate must still actually split into <=3 primes <= lpb)\n");
  }
#endif
#ifdef PROBE_ENUM
  { unsigned long long he=0,hh=0;
    cudaMemcpyFromSymbol(&he,g_enum_cnt,sizeof(he)); cudaMemcpyFromSymbol(&hh,g_hit_cnt,sizeof(hh));
    printf("  [PROBE] scat_lat enumerated %.1fM (c1,c2)/lat | landed in region %.1fM/lat | WASTE %.2fx\n",
      (double)he/nq/1e6,(double)hh/nq/1e6, hh?(double)he/(double)hh:0.0);
    unsigned long long vl=0,va=0,vw=0;
    cudaMemcpyFromSymbol(&vl,g_vert_lines,sizeof(vl)); cudaMemcpyFromSymbol(&va,g_vert_atomics,sizeof(va));
    cudaMemcpyFromSymbol(&vw,g_vert_worst,sizeof(vw));
    printf("  [PROBE] vertical lines %.1f/lat | their atomics %.1fM/lat (%.1f%% of all) | WORST single thread %llu atomics\n",
      (double)vl/nq,(double)va/nq/1e6, hh+va?100.0*va/(double)(hh+va):0.0, vw); }
#endif
#ifdef PROBE_RS_ENUM
  { unsigned long long e=0,h=0;
    cudaMemcpyFromSymbol(&e,g_rs_enum,sizeof(e)); cudaMemcpyFromSymbol(&h,g_rs_hit,sizeof(h));
    printf("  [PROBE] resieve walk: enumerated %.1fM/lat | in-region %.1fM/lat | WASTE %.2fx\n",
      (double)e/nq/1e6,(double)h/nq/1e6, h?(double)e/(double)h:0.0); }
#endif
#ifdef PROBE_ITER
  { unsigned long long s=0,w=0,l=0;
    cudaMemcpyFromSymbol(&s,g_it_sum,sizeof(s)); cudaMemcpyFromSymbol(&w,g_it_wmax,sizeof(w));
    cudaMemcpyFromSymbol(&l,g_it_lines,sizeof(l));
    printf("  [PROBE] reduction trips: mean %.2f/line | warp-max (what HW pays) %.2f/line | DIVERGENCE WASTE %.2fx | lines %.1fM/lat\n",
      l?(double)s/l:0.0, l?(double)w/l:0.0, s?(double)w/(double)s:0.0, (double)l/nq/1e6); }
#endif
  printf("  relations resident in RAM: %.3f GB  (disk untouched; %.0f bytes/rel)\n",g_ramlen/1e9,tot_rel?(double)g_ramlen/tot_rel:0);
  for(long long N : {430000LL,800000LL}){
    printf("  extrapolate %ldk lattices: GPU-bound %.2f h | wall %.2f h\n",(long)(N/1000),(tot_gpu/nq)*N/3.6e6,(wall_tot*1000/nq)*N/3.6e6);
  }
  return 0;
}
