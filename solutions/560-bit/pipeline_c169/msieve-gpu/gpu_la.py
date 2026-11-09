#!/usr/bin/env python3
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

# GPU linear algebra for NFS (GF(2) block-Lanczos, dense-row split + CSR-vector SpMV).
# Reads a msieve .mat, solves B x = 0 on the GPU, writes the .dep msieve expects.
# ALSO reads a CADO replay X.sparse.bin (B = M^T) and then writes CADO's flat kernel file (characters -ker).
# Uses CuPy + NVRTC (runtime kernel compilation) -> needs only the NVIDIA driver + pip CUDA headers.
import os,glob,time,sys,numpy as np
_T_START=time.time()   # s65: "LA loop starts at +X s" (cold vs warm CuPy kernel cache)
# validator runs the container --read-only; redirect CuPy's kernel cache (default ~/.cupy) and
# HOME to the writable tmpfs so `import cupy` / NVRTC don't crash on the read-only rootfs.
os.environ.setdefault("CUPY_CACHE_DIR","/tmp/.cupy")
os.environ.setdefault("HOME","/tmp")
# !! A CACHE DIR THAT EXISTS BUT IS NOT WRITABLE IS SILENTLY FATAL (2026-08-23). !!
# The old code swallowed OSError from makedirs and carried on. If the directory already exists
# owned by someone else, makedirs raises nothing at all -- the failure surfaces MUCH later as an
# uncaught PermissionError from CuPy's kernel cache write, deep inside the first RawKernel call.
# gpu_la.py then exits non-zero, msvrun_gpu.sh falls back to CPU `msieve -nc` (~11x slower), and
# the run loses the wall with a traceback nobody reads as a permissions problem. Observed for real
# on 2026-08-23. Probe writability and fall back to a private directory rather than trusting it.
def _usable_cache(d):
 try:
  os.makedirs(d, exist_ok=True)
  t=os.path.join(d,".wtest.%d"%os.getpid())
  with open(t,"wb") as f: f.write(b"x")
  os.unlink(t); return True
 except OSError: return False
if not _usable_cache(os.environ["CUPY_CACHE_DIR"]):
 for _alt in ("/tmp/.cupy-%d"%os.getuid(), os.path.join(os.environ.get("TMPDIR","/tmp"),".cupy-%d"%os.getuid())):
  if _usable_cache(_alt):
   print("  [la] WARNING: %s is not writable; using %s for the CuPy kernel cache"
         % (os.environ["CUPY_CACHE_DIR"], _alt), flush=True)
   os.environ["CUPY_CACHE_DIR"]=_alt; break
 else:
  print("  [la] WARNING: no writable CuPy kernel cache dir -- NVRTC will re-JIT every kernel",
        flush=True)
for _d in (os.environ["CUPY_CACHE_DIR"], os.path.join(os.environ["CUPY_CACHE_DIR"],"jitify")):
 try: os.makedirs(_d, exist_ok=True)
 except OSError: pass
def _find_cuda_inc():
 incs=[]
 try:
  import nvidia
  for pth in list(nvidia.__path__): incs+=glob.glob(os.path.join(pth,"*","include"))
 except Exception: pass
 incs+=glob.glob(os.path.expanduser("~/.local/lib/python*/site-packages/nvidia/*/include"))
 incs+=glob.glob("/usr/lib/python*/*-packages/nvidia/*/include")+glob.glob("/usr/local/lib/python*/*-packages/nvidia/*/include")
 if os.path.isdir("/usr/local/cuda/include"): incs.append("/usr/local/cuda/include")
 seen=set(); out=[]
 for d in incs:
  if d not in seen and os.path.isdir(d): seen.add(d); out.append(d)
 return out
inc=_find_cuda_inc()
if inc:
 os.environ.setdefault("CUDA_PATH",os.path.dirname(inc[0])); os.environ["CPATH"]=":".join(inc+([os.environ["CPATH"]] if os.environ.get("CPATH") else []))
import cupy as cp
OPT=tuple(f"-I{d}" for d in inc)
KS=r'''
// CSR counting-sort scatter: group the (row,col) pairs by row. `cur` starts as the row-start
// offsets and is bumped atomically, so each nonzero lands somewhere inside its own row's span.
// The order WITHIN a row is deliberately not defined -- SpMV here XORs a row's entries over
// GF(2), so any permutation inside the row gives the identical result.
extern "C" __global__ void csr_scatter(const int* Ri,const int* Ci,int* cur,int* out,long long nnz){
 for(long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x;k<nnz;k+=(long long)gridDim.x*blockDim.x){
  int p=atomicAdd(&cur[Ri[k]],1); out[p]=Ci[k]; }}
// Row histogram. Hand-written rather than cp.bincount: at this nnz cp.bincount dies with
// cudaErrorIllegalAddress on sm_120 and POISONS THE CONTEXT, so every later CuPy call fails too
// (same failure family as the cp.argsort note below). Reproduced standalone at nnz=398,738,029.
// This kernel is also 51x faster than the host np.bincount it replaces: 0.1s vs 5.1s.
extern "C" __global__ void rcount(const int* idx,int* cnt,long long nnz){
 for(long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x;k<nnz;k+=(long long)gridDim.x*blockDim.x)
  atomicAdd(&cnt[idx[k]],1);}
extern "C" __global__ void spmv(const unsigned long long* v,const int* col,const int* rp,unsigned long long* o,int nr){
 int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=nr)return; unsigned long long a=0;
 for(int k=rp[r];k<rp[r+1];++k)a^=v[col[k]]; o[r]=a;}
// balanced SpMV: each thread handles CH consecutive nonzeros; atomicXor partials into y (y must be pre-zeroed)
extern "C" __global__ void spmv_bal(const unsigned long long* v,const int* col,const int* ridx,
                                    long long nnz,unsigned long long* y,int CH){
 long long start=(long long)(blockIdx.x*blockDim.x+threadIdx.x)*CH; if(start>=nnz)return;
 long long end=start+CH; if(end>nnz)end=nnz;
 unsigned long long acc=0; int cur=ridx[start];
 for(long long k=start;k<end;++k){ int r=ridx[k]; if(r!=cur){atomicXor(&y[cur],acc);acc=0;cur=r;} acc^=v[col[k]]; }
 atomicXor(&y[cur],acc);}
// CSR-vector SpMV: one warp per row, warp-reduce XOR (heavy rows split across 32 lanes). overwrites y.
extern "C" __global__ void spmv_vec(const unsigned long long* v,const int* col,const int* rp,unsigned long long* y,int nr){
 int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(gw>=nr)return;
 int s=rp[gw],e=rp[gw+1]; unsigned long long acc=0;
 for(int k=s+lane;k<e;k+=32) acc^=v[col[k]];
 for(int o=16;o>0;o>>=1) acc^=__shfl_down_sync(0xffffffffu,acc,o);
 if(lane==0) y[gw]=acc;}
// ---- HEAVY-ROW SPLIT (2026-08-16).  WORTH ~14s of 545s -- KNOW THIS BEFORE EXTENDING IT. ----
// B is badly row-skewed and B^T is not, which is why the two SpMVs cost different amounts on
// IDENTICAL nnz.  Measured on key08 (4,339,002 x 4,339,129, sparse_nnz 401,725,072):
//     spmv_B  3.199 ms/it     spmv_Bt 2.648 ms/it     (+21% on the same 401.7M nonzeros)
//     B : 7,346 rows above weight 4096, holding 144,783,082 nnz = 36.04% of the matrix,
//         max row 502,265 against a 92.6 average (5,423x).
//     Bt: NO row above 4096 -- perfectly uniform, so build_heavy() returns None and Btmul
//         keeps the plain kernel.  Do not "symmetrise" this; there is nothing to split.
// warp-per-row gives the heaviest row's warp ~15,700 sequential gather trips while a typical
// warp finishes in ~3.  spmv_skip zeroes rows above the threshold and skips them; spmv_heavy
// re-does those rows as independent CHUNKS, atomicXor-ing partials into y[r].
// BIT-EXACT: GF(2) XOR is associative and commutative, so chunk order cannot matter.  Verified
// end-to-end on key08 -- 68,617 iterations and 60 deps, both IDENTICAL to the unsplit run,
// VERIFY B*x==0 passed, same factor on dep 3.
//
// !! THE MEASURED GAIN IS ONLY 1.6% (Lanczos 545s -> 531s). DO NOT EXPECT MORE FROM THIS KNOB. !!
// The imbalance is real but the scheduler was already hiding nearly all of it: there are 4.34M
// warps in flight, so heavy warps overlap with millions of light ones and the only true
// serialisation is the single longest row.  LA_HEAVY_T/LA_HEAVY_CH tuning has a ceiling of
// ~15s.  The remaining Lanczos headroom is NOT here -- profiling (LA_PROF=1) attributes
// 5.85 ms of the 7.41 ms/it to the two SpMVs, running at only 33-40% of peak HBM bandwidth
// (~1.89 GB of HBM traffic per SpMV; the 34.7 MB gather target fits in Blackwell's 128 MB L2).
// Reaching that headroom needs gather cache-blocking, i.e. a rewrite of the matrix build.
// ---- SUB-WARP SpMV (2026-09-12) -------------------------------------------------------------
// spmv_vec/spmv_skip give each row a FULL WARP and then pay an unconditional 5-step shuffle
// reduction. Average row weight here is ~90, so with 32 lanes each lane does ~3 useful gathers
// and is then reduced by 5 shuffle steps: the reduction costs MORE lane-ops than the work it is
// reducing. Narrowing the group to VW lanes (VW rows per warp) gives 90/VW gathers per lane and
// log2(VW) shuffle steps -- at VW=8 that is ~11 gathers against 3 steps.
// BIT-EXACT: GF(2) XOR is associative and commutative, so regrouping the reduction cannot change
// the result. Verify by comparing deps/iteration counts against the VW=32 path, never by timing.
//
// !! ❌ MEASURED 2026-09-12 AND IT IS A REGRESSION. LA_VW=8 IS 8.31 ms/it AGAINST 8.10 AT VW=32
// !! (+2.6 %), on the identical c155 matrix (4,851,143 x 4,851,268, sparse_nnz 440,711,800).
// The reduction-overhead argument is arithmetically right and still loses: dropping 5 shuffle
// steps to 3 saves less than the narrower group costs. Each row now takes ~11 sequential gather
// trips instead of ~3, and those gathers are the random-access half of the kernel -- so the loop
// carries a longer dependent chain and fewer independent loads are in flight per row, which is
// exactly the latency the wide group was hiding. Shuffles are cheap; gather latency is not.
// Kept behind LA_VW (default 32 = the shipped kernels, unchanged) so the result stays reproducible.
// DO NOT RE-TRY without changing what the gathers COST -- that is the cache-blocking rewrite the
// heavy-row note above points at, not a group-width knob.
// NOTE all threads reach the shuffles (no early return) -- __shfl_down_sync with a full mask and
// exited lanes is undefined, which the VW=32 kernels above get away with only by luck of layout.
extern "C" __global__ void spmv_vecw(const unsigned long long* v,const int* col,const int* rp,
                                     unsigned long long* y,int nr,int VSH){
 int tid=blockIdx.x*blockDim.x+threadIdx.x; int VW=1<<VSH;
 int gw=tid>>VSH, lane=tid&(VW-1);
 unsigned long long acc=0;
 if(gw<nr){ int s=rp[gw],e=rp[gw+1]; for(int k=s+lane;k<e;k+=VW) acc^=v[col[k]]; }
 for(int o=VW>>1;o>0;o>>=1) acc^=__shfl_down_sync(0xffffffffu,acc,o,VW);
 if(gw<nr && lane==0) y[gw]=acc;}
extern "C" __global__ void spmv_skipw(const unsigned long long* v,const int* col,const int* rp,
                                      unsigned long long* y,int nr,int T,int VSH){
 int tid=blockIdx.x*blockDim.x+threadIdx.x; int VW=1<<VSH;
 int gw=tid>>VSH, lane=tid&(VW-1);
 unsigned long long acc=0; int heavy=0;
 if(gw<nr){ int s=rp[gw],e=rp[gw+1];
   if(e-s>T){ heavy=1; } else { for(int k=s+lane;k<e;k+=VW) acc^=v[col[k]]; } }
 for(int o=VW>>1;o>0;o>>=1) acc^=__shfl_down_sync(0xffffffffu,acc,o,VW);
 if(gw<nr && lane==0) y[gw]= heavy?0ULL:acc;}
extern "C" __global__ void spmv_skip(const unsigned long long* v,const int* col,const int* rp,unsigned long long* y,int nr,int T){
 int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(gw>=nr)return;
 int s=rp[gw],e=rp[gw+1];
 if(e-s>T){ if(lane==0) y[gw]=0ULL; return; }          // handled by spmv_heavy
 unsigned long long acc=0;
 for(int k=s+lane;k<e;k+=32) acc^=v[col[k]];
 for(int o=16;o>0;o>>=1) acc^=__shfl_down_sync(0xffffffffu,acc,o);
 if(lane==0) y[gw]=acc;}
extern "C" __global__ void spmv_heavy(const unsigned long long* v,const int* col,
                                      const int* hrow,const int* hs,const int* he,
                                      unsigned long long* y,int nchunk){
 int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(gw>=nchunk)return;
 int r=hrow[gw], s=hs[gw], e=he[gw]; unsigned long long acc=0;
 for(int k=s+lane;k<e;k+=32) acc^=v[col[k]];
 for(int o=16;o>0;o>>=1) acc^=__shfl_down_sync(0xffffffffu,acc,o);
 if(lane==0) atomicXor(&y[r],acc);}
extern "C" __global__ void blockmul_acc(const unsigned long long* V,const unsigned long long* M,unsigned long long* y,int n){
 int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n)return; unsigned long long x=V[r],a=0;
 while(x){int i=__ffsll(x)-1; a^=M[i]; x&=x-1;} y[r]^=a;}
// REPK: the shared accumulators below are REPLICATED REPK ways and each lane uses its own bank,
// because both this kernel and dense_fwd were limited by shared-memory atomic CONTENTION, not by
// memory traffic. Every set bit costs one atomicXor and many lanes in a warp target the same word.
// The tell was dense_bwd: it reads the identical data but accumulates into a REGISTER and costs
// 0.145 ms against dense_fwd's 0.605 ms. Measured sweep (bit-exact against the originals at every K):
//   inner2     0.857 -> K=4 0.555 | K=8 0.424 | K=16 0.336 | K=32 0.341
//   dense_fwd  0.920 -> K=4 0.612 | K=8 0.489 | K=16 0.365 | K=32 0.473
// K=16 is the optimum; K=32 loses to occupancy (32 KB of shared per block) and K=64 exceeds the
// limit outright. Static shared at K=16 is 16 KB here and 16 KB in dense_fwd, well under the 48 KB
// static cap, so the launch signatures are unchanged.
#define REPK 16
extern "C" __global__ void inner2(const unsigned long long* A,const unsigned long long* B,
                                  unsigned long long* o1,unsigned long long* o2,int n){
 __shared__ unsigned long long s1[64*REPK]; __shared__ unsigned long long s2[64*REPK];
 for(int t=threadIdx.x;t<64*REPK;t+=blockDim.x){s1[t]=0;s2[t]=0;}
 __syncthreads();
 const int bank=threadIdx.x&(REPK-1);
 for(int r=blockIdx.x*blockDim.x+threadIdx.x;r<n;r+=gridDim.x*blockDim.x){
   unsigned long long a=A[r],b=B[r],x=a,y=b;
   while(x){int i=__ffsll(x)-1; atomicXor(&s1[i*REPK+bank],b); x&=x-1;}
   while(y){int i=__ffsll(y)-1; atomicXor(&s2[i*REPK+bank],b); y&=y-1;}}
 __syncthreads();
 if(threadIdx.x<64){
   unsigned long long v1=0,v2=0;
   for(int k=0;k<REPK;k++){ v1^=s1[threadIdx.x*REPK+k]; v2^=s2[threadIdx.x*REPK+k]; }
   atomicXor(&o1[threadIdx.x],v1); atomicXor(&o2[threadIdx.x],v2);}}
// dense rows forward: y[j] (j<nd) ^= v[i] for each column i whose dense bitfield (2 words) has bit j set
extern "C" __global__ void dense_fwd(const unsigned long long* dcol,const unsigned long long* v,
                                     unsigned long long* y,int ncols,int nd){
 // same REPK bank-replication as inner2 -- this kernel was contention-bound, not bandwidth-bound
 __shared__ unsigned long long s[128*REPK];
 for(int t=threadIdx.x;t<128*REPK;t+=blockDim.x) s[t]=0;
 __syncthreads();
 const int bank=threadIdx.x&(REPK-1);
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<ncols;i+=gridDim.x*blockDim.x){
   unsigned long long vi=v[i],w0=dcol[2*i],w1=dcol[2*i+1];
   while(w0){int j=__ffsll(w0)-1; atomicXor(&s[j*REPK+bank],vi); w0&=w0-1;}
   while(w1){int j=__ffsll(w1)-1; atomicXor(&s[(64+j)*REPK+bank],vi); w1&=w1-1;}}
 __syncthreads();
 if(threadIdx.x<nd){
   unsigned long long acc=0;
   for(int k=0;k<REPK;k++) acc^=s[threadIdx.x*REPK+k];
   atomicXor(&y[threadIdx.x],acc);}}
// dense rows backward: z[i] ^= XOR_{j: bit j of dcol[i]} u[j]   (u[0..nd-1])
extern "C" __global__ void dense_bwd(const unsigned long long* dcol,const unsigned long long* u,
                                     unsigned long long* z,int ncols){
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<ncols;i+=gridDim.x*blockDim.x){
   unsigned long long w0=dcol[2*i],w1=dcol[2*i+1],a=0;
   while(w0){int j=__ffsll(w0)-1; a^=u[j]; w0&=w0-1;}
   while(w1){int j=__ffsll(w1)-1; a^=u[64+j]; w1&=w1-1;}
   z[i]^=a;}}
'''
spmv=cp.RawKernel(KS,'spmv',options=OPT); spbal=cp.RawKernel(KS,'spmv_bal',options=OPT); spvec=cp.RawKernel(KS,'spmv_vec',options=OPT); bmacc=cp.RawKernel(KS,'blockmul_acc',options=OPT)
csort=cp.RawKernel(KS,'csr_scatter',options=OPT); rcount=cp.RawKernel(KS,'rcount',options=OPT)
spskip=cp.RawKernel(KS,'spmv_skip',options=OPT); spheavy=cp.RawKernel(KS,'spmv_heavy',options=OPT)
spvecw=cp.RawKernel(KS,'spmv_vecw',options=OPT); spskipw=cp.RawKernel(KS,'spmv_skipw',options=OPT)
# LA_VW: lanes per row in the sparse SpMV. 32 = the shipped warp-per-row kernels, byte-for-byte
# the previous code path. 8/16 use the sub-warp kernels above. Power of two, 1..32.
LA_VSH = max(0, min(5, int(os.environ.get("LA_VW","32")).bit_length()-1))
LA_VW  = 1 << LA_VSH
inner2=cp.RawKernel(KS,'inner2',options=OPT); dfwd=cp.RawKernel(KS,'dense_fwd',options=OPT); dbwd=cp.RawKernel(KS,'dense_bwd',options=OPT)
def G(n,t=256):return((n+t-1)//t,)
HEAVY_T  = int(os.environ.get("LA_HEAVY_T","4096"))    # row weight above which a row is split
HEAVY_CH = int(os.environ.get("LA_HEAVY_CH","4096"))   # nonzeros per chunk
def build_heavy(d_rp,name=""):
 """-> (d_hrow,d_hs,d_he,nchunk).  Empty table => caller keeps the plain kernel."""
 w=cp.diff(d_rp)
 idx=cp.nonzero(w>HEAVY_T)[0]
 if idx.size==0:
  print(f"  [heavy] {name}: no row above {HEAVY_T} -- plain SpMV",flush=True); return None
 rows=cp.asnumpy(idx); rp=cp.asnumpy(d_rp); wt=rp[rows+1]-rp[rows]
 hrow=[];hs=[];he=[]
 for r,st,n_ in zip(rows,rp[rows],wt):
  for c0 in range(0,int(n_),HEAVY_CH):
   hrow.append(int(r)); hs.append(int(st)+c0); he.append(int(st)+min(c0+HEAVY_CH,int(n_)))
 print(f"  [heavy] {name}: {len(rows)} rows > {HEAVY_T} (max {int(wt.max())}, "
       f"{int(wt.sum())} nnz = {wt.sum()/ (rp[-1]) *100:.2f}%) -> {len(hrow)} chunks",flush=True)
 return (cp.asarray(np.array(hrow,dtype=np.int32)),cp.asarray(np.array(hs,dtype=np.int32)),
         cp.asarray(np.array(he,dtype=np.int32)),len(hrow))
bm=[1<<i for i in range(64)]; MASK=(1<<64)-1
# 64x64 GF(2) bit-matrix helpers. These are on the Lanczos critical path: profiling the iteration
# (LA_PROF instrumentation, 2000 iters) attributed 2.871 ms of the 11.88 ms/iteration to mul64+tr64
# and a further 0.615 ms to fns() -- 29% of every iteration spent in interpreted Python, running
# AFTER the O.get() sync, so the GPU is idle throughout. Over 73,466 iterations that is ~256s.
# Both ops are a single numpy matmul over GF(2). Validated EXACT against the original scalar code on
# 303 random 64x64 pairs plus zero/identity/all-ones edge cases before being swapped in -- they feed
# the dependency that "VERIFY B*x==0" checks, so an approximation here would be silently wrong.
# Measured per call: tr64 490.4 -> 12.7 us (38x); mul64 435.0 -> 227.7 us (1.9x -- mul64 gains less
# because the Python-list <-> ndarray conversion dominates, not the arithmetic).
def _bits64(x):                 # (64,64) uint8; element [i,j] = bit j of x[i]
 # asarray, not array: callers pass uint64 ndarrays, for which this is a free view. That is the
 # whole point -- profiling showed the conversion, not the arithmetic, dominated mul64.
 return np.unpackbits(np.asarray(x,dtype='<u8').view(np.uint8).reshape(64,8),axis=1,bitorder='little')
def _pack64(B):                 # inverse of _bits64
 return np.packbits(B.astype(np.uint8),axis=1,bitorder='little').view('<u8').reshape(64)
def mul64(a,b):
 # c[i] = XOR over j of { b[j] : bit j of a[i] set }  ==  (Abits @ Bbits) mod 2.
 # int32 accumulator: the row sums cannot exceed 64, so there is no overflow to mask away.
 # Returns a uint64 ndarray; the iteration keeps these blocks as arrays throughout.
 # float64, NOT int32: numpy dispatches integer matmul to a naive C loop while float matmul goes to
 # BLAS. Measured on the 64x64 blocks -- int32 217.4 us vs float64 25.8 us, an 8.4x difference that
 # is pure dispatch, and the results are BIT-EQUAL: every entry is a count of at most 64 set bits,
 # far inside float64's exactly-representable integer range (2**53). This was the real cost of
 # mul64, not the list<->ndarray conversion I first assumed.
 return _pack64((( _bits64(a).astype(np.float64) @ _bits64(b).astype(np.float64) ).astype(np.int64)) & 1)
def tr64(a):
 return _pack64(_bits64(a).T.copy())
def fns(t,ls,ld):
 M0=list(t);M1=list(bm);cols=[0]*64;mask=0
 for i in range(ld):cols[63-i]=ls[i];mask|=bm[ls[i]]
 j=0
 for i in range(64):
  if not(mask&bm[i]):cols[j]=i;j+=1
 dim=0;s=[]
 for i in range(64):
  m=bm[cols[i]];ci=cols[i];p=-1
  for jj in range(i,64):
   if M0[cols[jj]]&m:p=jj;break
  if p>=0:
   b=cols[p];M0[ci],M0[b]=M0[b],M0[ci];M1[ci],M1[b]=M1[b],M1[ci]
   for jj in range(64):
    rj=cols[jj]
    if rj!=ci and(M0[rj]&m):M0[rj]^=M0[ci];M1[rj]^=M1[ci]
   s.append(ci);dim+=1
  else:
   p=-1
   for jj in range(i,64):
    if M1[cols[jj]]&m:p=jj;break
   if p<0:return 0,[],[0]*64
   b=cols[p];M0[ci],M0[b]=M0[b],M0[ci];M1[ci],M1[b]=M1[b],M1[ci]
   for jj in range(64):
    rj=cols[jj]
    if rj!=ci and(M1[rj]&m):M0[rj]^=M0[ci];M1[rj]^=M1[ci]
   M0[ci]=0;M1[ci]=0
 return dim,s,[M1[i] for i in range(64)]
def transpose_to_rows(V,off,M):
 ncols=len(V); cw=(ncols+63)//64
 Vp=np.zeros(cw*64,dtype=np.uint64); Vp[:ncols]=V; Vp=Vp.reshape(cw,64)
 wts=(np.uint64(1)<<np.arange(64,dtype=np.uint64))
 for j in range(64):
  bitj=((Vp>>np.uint64(j))&np.uint64(1)).astype(np.uint64)
  M[off+j]=(bitj*wts).sum(axis=1).astype(np.uint64)
def combine_cols(ncols,x,v,ax,av):
 cw=(ncols+63)//64
 mat=np.zeros((128,cw),dtype=np.uint64); amat=np.zeros((128,cw),dtype=np.uint64)
 transpose_to_rows(x,0,mat); transpose_to_rows(ax,0,amat)
 transpose_to_rows(v,64,mat); transpose_to_rows(av,64,amat)
 i=0; bitpos=0
 while i<128 and bitpos<ncols:
  col=bitpos>>6; mask=np.uint64(1<<(bitpos&63)); piv=-1
  for j in range(i,128):
   if amat[j,col]&mask: piv=j;break
  if piv<0: bitpos+=1; continue
  if piv!=i: amat[[i,piv]]=amat[[piv,i]]; mat[[i,piv]]=mat[[piv,i]]
  sel=(amat[i+1:,col]&mask)!=0; idx=np.nonzero(sel)[0]+i+1
  if len(idx): amat[idx]^=amat[i]; mat[idx]^=mat[i]
  i+=1; bitpos+=1
 # out[jc] gets bit bp set iff column jc is set in reduced row mat[i+bp];
 # i.e. out is the bit-transpose of mat[i:64]. The original pure-Python triple
 # loop over ~100M set bits took ~180s; vectorize per dependency-plane with
 # unpackbits (mat[i+bp] is a little-endian packed bit-vector over the ncols
 # columns, so unpackbits(...,bitorder='little')[jc] == column jc). ~80s, and
 # bit-for-bit identical to the old loop (verified). (~2.3x; a CuPy GPU
 # transpose is ~instant in isolation but fell back to this in-context, so we
 # keep the reliable numpy path.)
 out=np.zeros(ncols,dtype=np.uint64)
 for bp in range(64-i):
  bits=np.unpackbits(mat[i+bp].view(np.uint8),bitorder='little')[:ncols]
  out|=bits.astype(np.uint64)<<np.uint64(bp)
 nd=0 if i>64 else 64-i
 return out,nd

# globals set in __main__: d_colB,d_rpB,d_colBt,d_rpBt,d_dcol,ND,sc(buffer)
def gpu_block_lanczos(d_colB,d_rpB,d_colBt,d_rpBt,d_dcol,ND,n,seg=None):
 tpb=256; gr=G(n); IG=(2048,); DG=G(n); WB=((n*32+tpb-1)//tpb,)
 # Sub-warp SpMV grid: VW lanes per row instead of 32 (see spmv_vecw). LA_VW=32 keeps the
 # shipped kernels and the shipped grid exactly.
 WBW=((n*LA_VW+tpb-1)//tpb,)
 if LA_VW!=32 and seg is None: print("  [gpu-la] sparse SpMV: %d lanes/row (LA_VW=%d)"%(LA_VW,LA_VW),flush=True)
 sc=cp.empty(n,cp.uint64)
 HB=HBt=None
 if seg is None: HB=build_heavy(d_rpB,"B"); HBt=build_heavy(d_rpBt,"Bt")
 HBG =G(HB[3]*32)  if HB  else None
 HBtG=G(HBt[3]*32) if HBt else None
 def Bmul(src,dst):  # dst = B*src  (CSR-vector sparse rows + dense rows)
  if HB is None:
   if LA_VW==32: spvec(WB,(tpb,),(src,d_colB,d_rpB,dst,n))   # warp-per-row (overwrites all rows; dense rows=0)
   else:         spvecw(WBW,(tpb,),(src,d_colB,d_rpB,dst,n,LA_VSH))
  else:
   if LA_VW==32: spskip(WB,(tpb,),(src,d_colB,d_rpB,dst,n,HEAVY_T))   # all rows except the heavy ones
   else:         spskipw(WBW,(tpb,),(src,d_colB,d_rpB,dst,n,HEAVY_T,LA_VSH))
   spheavy(HBG,(tpb,),(src,d_colB,HB[0],HB[1],HB[2],dst,HB[3]))# heavy rows, chunked, atomicXor
  dfwd(DG,(tpb,),(d_dcol,src,dst,n,ND))         # dense rows accumulate into dst[0..ND-1]
 def Btmul(src,dst): # dst = B^T*src
  # WARP per row, not thread per row (2026-07-31). The scalar spmv has lane i walking
  # col[rp[r_i]+k], so consecutive lanes touch unrelated addresses and each 32-byte sector fetch
  # returns 4 useful bytes. B^T's rows are B's columns, averaging 97.99 entries, which suits
  # warp-per-row: the index reads become coalesced. Benchmarked on this matrix, output identical:
  #     spmv     (thread/row) 3.18 ms      spmv_vec (warp/row) 2.29 ms   -28%
  # Over 73,466 iterations that is -65s. spmv_bal was also tried and is 6x worse (18.2 ms).
  # NB Bmul below keeps spvec for the opposite reason -- B has a 541,881-weight row, where
  # thread-per-row collapses to 40.35 ms against spvec's 2.97 ms.
  if HBt is None:
   if LA_VW==32: spvec(WB,(tpb,),(src,d_colBt,d_rpBt,dst,n))  # sparse contribution
   else:         spvecw(WBW,(tpb,),(src,d_colBt,d_rpBt,dst,n,LA_VSH))
  else:
   if LA_VW==32: spskip(WB,(tpb,),(src,d_colBt,d_rpBt,dst,n,HEAVY_T))
   else:         spskipw(WBW,(tpb,),(src,d_colBt,d_rpBt,dst,n,HEAVY_T,LA_VSH))
   spheavy(HBtG,(tpb,),(src,d_colBt,HBt[0],HBt[1],HBt[2],dst,HBt[3]))
  dbwd(DG,(tpb,),(d_dcol,src,dst,n))            # dense contribution z[i]^=...
 if seg is not None:
  # LA_SPMV=seg: segmented hot-ideal SpMV (segspmv.py). Writes every row of dst like spvec/spskip, so the
  # dense-row kernels below are applied exactly as on the production path.
  def Bmul(src,dst):
   seg.B(src,dst); dfwd(DG,(tpb,),(d_dcol,src,dst,n,ND))
  def Btmul(src,dst):
   seg.Bt(src,dst); dbwd(DG,(tpb,),(d_dcol,src,dst,n))
 def AvP(src,dst): Bmul(src,sc); Btmul(sc,dst)  # dst = B^T B src
 p0=cp.empty(n,cp.uint64); p1=cp.zeros(n,cp.uint64); p2=cp.zeros(n,cp.uint64); pn=cp.empty(n,cp.uint64)
 xb=cp.asarray(np.random.default_rng(int(os.environ.get("LA_SEED","5"))).integers(0,1<<64,n,dtype=np.uint64))
 AvP(xb,p0); vsav=p0.copy()
 O=cp.zeros((2,64),cp.uint64); ov=cp.zeros(64,cp.uint64); dbuf=cp.empty(64,cp.uint64); hbuf=np.empty(64,np.uint64)
 def up(lst): hbuf[:]=lst; dbuf.set(hbuf)
 # The 64-word blocks below are kept as uint64 ndarrays, NOT Python lists. Profiling showed mul64
 # spends most of its time in the list<->ndarray conversion rather than the matmul, so passing
 # ndarrays through removes that cost at every call site. Elementwise steps that were 64-iteration
 # list comprehensions become single vectorised ops. NOTE masks must be np.uint64, never Python
 # ints: numpy refuses to mix uint64 with a Python int above 2**63 and would raise here.
 Z64=lambda: np.zeros(64,dtype='<u8')
 vt_a_v=[Z64(),Z64()];vt_a2_v=[Z64(),Z64()];vt_v0=[Z64(),Z64(),Z64()];winv=[[0]*64]*3
 BM=np.array(bm,dtype='<u8')
 s=[[0]*64,list(range(64))];dim1=64;mask1=MASK;it=0;dim0=0; t0=time.time()
 # LA_PROF=1: split each iteration into the GPU-bound half (the two SpMVs + inner2, ending at the
 # O.get() that SYNCS) and the CPU-bound half (fns/mul64/tr64 and the bmacc issues). The loop is
 # strictly serial across that boundary -- the GPU is idle for the whole CPU half -- so the split
 # says directly how much of the stage is reachable by kernel work and how much is not.
 # Asynchronous bmacc work issued in the CPU half is absorbed by the NEXT iteration's sync, so
 # tGPU is "all GPU time" and tCPU is "host time the GPU cannot overlap".
 PROF=int(os.environ.get("LA_PROF","0")); tGPU=0.0; tCPU=0.0; _t1=time.time()
 # LA_PROF=2: also event-time Bmul and Btmul (incl. their dense-row kernels) inside the loop; the events are read
 # after the O.get() sync the loop already has, so this adds no synchronisation.
 _pe=[cp.cuda.Event() for _ in range(3)] if PROF>=2 else None; tBm=0.0; tBtm=0.0
 # HARD ITERATION CAP: block Lanczos needs ~n/63.2 iterations at 64 bits. A corrupted vector (transient
 # device fault) keeps the symmetric recurrence alive without ever converging; without a cap the loop
 # would run to the wall and ds_cado's LA retry would never start. Exit loudly instead (-> 0 deps, rc 3).
 ITMAX=n//56+64
 while True:
  it+=1
  if it>ITMAX:
   print(f"  FATAL: block Lanczos did not converge in {ITMAX} iterations (expected ~{n//63}) -- corrupted state?",flush=True)
   dim0=0; break
  if PROF: _t0=time.time()
  if PROF>=2: _pe[0].record(); Bmul(p0,sc); _pe[1].record(); Btmul(sc,pn); _pe[2].record()
  else: AvP(p0,pn)
  O[:]=0; inner2(IG,(tpb,),(p0,pn,O[0],O[1],n))
  Oh=O.get(); vt_a_v[0]=Oh[0].copy(); vt_a2_v[0]=Oh[1].copy()
  if PROF:
   _t1=time.time(); tGPU+=_t1-_t0
   if PROF>=2: tBm+=cp.cuda.get_elapsed_time(_pe[0],_pe[1]); tBtm+=cp.cuda.get_elapsed_time(_pe[1],_pe[2])
   if it%500==0:
    _el=time.time()-t0
    print("   [LA_PROF] it=%d  %.2f ms/it | GPU %.2f ms (%.0f%%) | CPU %.2f ms (%.0f%%)"%(
      it,_el*1000/it,tGPU*1000/it,100*tGPU/_el,tCPU*1000/it,100*tCPU/_el)+
      ("" if PROF<2 else " | Bmul %.3f ms Btmul %.3f ms (events, incl. dense rows) | xb sum %016x"%(tBm/it,tBtm/it,int(xb.sum())&0xFFFFFFFFFFFFFFFF)),flush=True)
  if not vt_a_v[0].any():break
  # fns() still takes/returns Python lists: it does scalar bit work mixing these words with the
  # Python ints in bm[], and feeding it np.uint64 would risk a silent type promotion. One tolist()
  # per iteration is ~5 us, far below what the conversions inside mul64 were costing.
  dim0,s0,winv[0]=fns(vt_a_v[0].tolist(),s[1],dim1)
  if dim0==0:break
  s[0]=s0+[0]*(64-len(s0));mask0=0
  for i in range(dim0):mask0|=bm[s0[i]]
  M0=np.uint64(mask0)
  if mask0!=MASK: pn&=cp.uint64(mask0)
  if it<4:
   O[:]=0; inner2(IG,(tpb,),(p0,vsav,O[0],O[1],n)); vt_v0[0]=O.get()[0].copy()
  d=(vt_a2_v[0]&M0)^vt_a_v[0]; d=mul64(winv[0],d); d=d^BM
  up(d); bmacc(gr,(tpb,),(p0,dbuf,pn,n))
  vt_v0_next=mul64(tr64(d),vt_v0[0])
  e=mul64(winv[1],vt_a_v[0]); e=e&M0
  up(e); bmacc(gr,(tpb,),(p1,dbuf,pn,n))
  ee=mul64(tr64(e),vt_v0[1]); vt_v0_next=vt_v0_next^ee
  if mask1!=MASK:
   M1=np.uint64(mask1)
   f=mul64(vt_a_v[1],winv[1]); f=f^BM; f=mul64(winv[2],f)
   f2=((vt_a2_v[1]&M1)^vt_a_v[1])&M0; f=mul64(f,f2)
   up(f); bmacc(gr,(tpb,),(p2,dbuf,pn,n))
   ff=mul64(tr64(f),vt_v0[2]); vt_v0_next=vt_v0_next^ff
  d2=mul64(winv[0],vt_v0[0]); up(d2); bmacc(gr,(tpb,),(p0,dbuf,xb,n))
  p0,p1,p2,pn = pn,p0,p1,p2
  winv=[winv[2],winv[0],winv[1]]; vt_v0=[vt_v0_next,vt_v0[0],vt_v0[1]]
  vt_a_v=[vt_a_v[1],vt_a_v[0]];vt_a2_v=[vt_a2_v[1],vt_a2_v[0]];s[1]=list(s[0]);mask1=mask0;dim1=dim0
  if PROF: tCPU+=time.time()-_t1
  if it%2000==0: print(f"   iter {it}  {time.time()-t0:.0f}s ({(time.time()-t0)*1000/it:.2f} ms/it)",flush=True)
 if dim0==0:return None,0,it
 Bmul(xb,pn); ax=cp.asnumpy(pn); Bmul(p0,pn); av=cp.asnumpy(pn)
 outx,nd=combine_cols(n,cp.asnumpy(xb),cp.asnumpy(p0),ax,av)
 return outx,nd,it

# =================================================================================================
# LA_BLOCK=128 (2026-09-27): 128-bit block Lanczos with a COLUMN-SPLIT SpMV.
# -------------------------------------------------------------------------------------------------
# WHY. At c160 the SpMV is L2-gather/sector-bound: the 64-bit vector (N x 8 B = 77-92 MB) sits in
# the 128 MB L2 and a 32 B sector fetch returns one 8 B element, so a 16 B element costs about the
# same gather. A 128-bit block does the same gathers per iteration but needs ~N/127.24 iterations
# instead of ~N/63.24 -- IF the 128-bit vector (155-184 MB) stays L2-resident, which it does not as
# a whole. So the matrix COLUMNS (the gathered index) are split into P parts of <= LA_SPLIT_MB
# (default 100 MB) of vector each; every part is its own CSR over the SAME rows, launched in turn,
# and y accumulates by XOR across parts (part 0 overwrites, parts 1.. XOR in; one warp owns a row
# within a launch and launches serialise on the stream, so no atomics are needed). B^T gets its own
# split over ITS gathered index (B's row index R).
#
# MATH. Montgomery's block Lanczos exactly as the 64-bit loop above (which is msieve's), with
# VBITS=128 as in msieve 1.54 common/lanczos: all 64x64 blocks become 128x128 GF(2) matrices
# (host: (128,2) uint64, row i = 128-bit word i, [i,0]=bits 0-63, [i,1]=bits 64-127), masks are
# 128 bits, find_nonsingular_sub works on 128 columns, combine_cocycles on 2*128 rows. Added
# msieve's "not all columns used" guard (mask0|mask1 must be all ones) -- the 64-bit loop lacks it.
#
# OUTPUT. The kernel file stays the flat CADO/msieve format: nrows x uint64 = 64 vectors. combine128
# yields up to 256-rank candidate vectors; the first 64 that are NON-ZERO and INDEPENDENT (greedy
# GF(2) basis on a 16384-coordinate prefix, which can only under-count independence -> never admits
# a dependent vector) are written, so ds_cado.sh / characters / sqrt are UNCHANGED. With LA_KER2=1 the
# next (up to) 64 independent vectors go to <out>.2 in the same format (a spare kernel for a second
# `characters` pass instead of a second ~2,500 s LA; ds_cado does not use it yet).
#
# VERIFY. Independent of everything new: B*x over the ORIGINAL COO arrays straight from the loader
# (coo_xor64, one atomicXor per nonzero) + the production 64-bit dense_fwd -- NOT the split CSR, so a
# split-build or 128-bit-SpMV bug cannot verify itself.
# =================================================================================================
LA_BLOCK = int(os.environ.get("LA_BLOCK", "64"))
if LA_BLOCK not in (64, 128, 256):
 print(f"  FATAL: LA_BLOCK={LA_BLOCK} (must be 64, 128 or 256)", flush=True); sys.exit(3)
# LA_SPMV=seg (2026-09-27): segmented, hot-ideal-aware SpMV from segspmv.py for BOTH block widths.
# Default "prod" = the kernels above, unchanged. seg is bit-identical to them (XOR of the same words per row;
# checked by dev/spmv/bench.py on the c160 matrix) -- VERIFY below still runs on the production CSR (64) or the
# original COO (128), never on the seg structures.
_LA_MAXIT=int(os.environ.get("LA_MAXIT","0") or 0)   # timing runs: stop after this many iterations
LA_SPMV = os.environ.get("LA_SPMV", "prod")
if LA_SPMV not in ("prod", "seg"):
 print(f"  FATAL: LA_SPMV={LA_SPMV} (must be prod or seg)", flush=True); sys.exit(3)
# LA_DEFER_ROWS (review 8 rank 2, 2026-10-24; default 0 = every path below unchanged): POST-LANCZOS DENSE CONSTRAINTS.
# B = [D; S] with D = the K heaviest ideal rows 0..K-1 (replay numbers ideals by decreasing weight, so cado_mat packs
# exactly them into dcol with loader nd=K). The recurrence runs on Bwork = [0; S] (Awork = Bwork^T Bwork): no dense
# forward product, no fused dense backward, for all ~N/255 iterations. Rows keep their ORIGINAL indices (rows 0..K-1
# are explicitly zero in every Bwork image: the seg plan's part-0 zero list); n, the relation-set order and the
# replay index are unchanged. Only the two final images that feed combineV / combine_fast are the ORIGINAL B
# (seg B + dense_fwd256 over the saved dcol), so the elimination finds combinations of [xb | p0] in the kernel of the
# FULL matrix -- exact, no relation loss; the K extra constraints cost at most K candidate dimensions.
# VERIFY then checks B x == 0 against the ORIGINAL COO + all K dense rows, and the run refuses (exit 3, no kernel file)
# unless 64 output columns with GF(2) rank 64 come out. K in {0, 64, 128} only (cado_mat clamps nd at 128; LA_CADO_ND
# must NOT be set alongside: it is not a deferral knob). Requires LA_BLOCK=256, LA_SPMV=seg, a CADO .sparse.bin input
# and LA_GRAPH=0 (not wired). LA_DEFER_STRICT=1: start-up self-test of Bwork / Btwork / Bfull against the independent
# COO + 64-bit dense_fwd on LA_DEFER_CHECKS (2) random 256-bit vectors (every row, every word); a mismatch exits 5.
def _defer_rows():
 v=os.environ.get("LA_DEFER_ROWS","0").strip() or "0"
 if v not in ("0","64","128"):
  print(f"  FATAL: LA_DEFER_ROWS={v} (must be 0, 64 or 128)",flush=True); sys.exit(3)
 return int(v)
LA_DEFER=_defer_rows()
def _rank64_py(x):
 """GF(2) rank of the 64 bit-columns of a (n,) uint64 kernel array (= rank of its distinct row words).
 Python loop over the sorted distinct words: on a full-rank c169 kernel (16M rows) it visits ~half of them before
 the 64th pivot (the pivot of bit 63 is the last group of the sort) -- 21.3 s on an M1 for 16,052,864 random words
 (sqrt_tail research 2026-10-25); keys 26/27/28 (LA_DEFER_ROWS=64) show the same 22-24 s between "LA done" and the
 kernel write, keys 22-24 (no DEFER, no rank check) ~1 s."""
 basis={}; r=0
 for w in np.unique(np.asarray(x,dtype=np.uint64)).tolist():
  w=int(w)
  while w:
   h=w.bit_length()-1
   if h in basis: w^=basis[h]
   else: basis[h]=w; r+=1; break
  if r>=64: break
 return r
def _rank64_fast(x,sample=4096,seed=20261025):
 """Same value as _rank64_py, without the per-row Python loop (LA_RANK64=fast; sqrt_tail research 2026-10-25).
 The rank is at most 64 and never below the rank of any row subset: a random sample of `sample` rows with rank 64
 proves rank 64 (~0.01 s). Otherwise the exact rank of the distinct words by a vectorized column elimination
 (64 numpy passes over np.unique(x); 1.7 s on 16M rows masked to rank 10, where the sample cannot reach 64)."""
 x=np.asarray(x,dtype=np.uint64)
 def elim(w):
  w=w.copy(); r=0
  for h in range(63,-1,-1):
   idx=np.flatnonzero((w>>np.uint64(h))&np.uint64(1))
   if idx.size==0: continue
   piv=w[idx[0]]; w[idx[1:]]^=piv; w[idx[0]]=0; r+=1
  return r
 if x.size>sample and elim(x[np.random.default_rng(seed).choice(x.size,sample,replace=False)])>=64: return 64
 return elim(np.unique(x))
def _rank64(x):
 """LA_RANK64=py (default: the Python loop) | fast (sampled / vectorized exact rank, same value) | check (both, FATAL on
 a mismatch, prints both times)."""
 m=os.environ.get("LA_RANK64","py")
 if m=="py": return _rank64_py(x)
 if m=="fast": return _rank64_fast(x)
 if m=="check":
  t=time.time(); rf=_rank64_fast(x); tf=time.time()-t; t=time.time(); rp=_rank64_py(x); tp=time.time()-t
  print(f"  [la256] LA_RANK64=check: fast {rf} in {tf:.2f}s, py {rp} in {tp:.2f}s -> {'MATCH' if rf==rp else 'MISMATCH'}",flush=True)
  if rf!=rp: print("  [la256] FATAL: LA_RANK64=check mismatch",flush=True); sys.exit(5)
  return rp
 print(f"  FATAL: LA_RANK64={m} (must be py, fast or check)",flush=True); sys.exit(3)
def _load_seg():
 for d in (os.path.dirname(os.path.abspath(__file__)),):
  if os.path.exists(os.path.join(d, "segspmv.py")):
   if d not in sys.path: sys.path.insert(0, d)
   break
 import segspmv
 segspmv.OPTS = OPT
 return segspmv
# LA_KERN=v2 (2026-09-28, the DEFAULT): table-driven upd128 (32 KB shared nibble tables, fixed 32 lookups per
# matrix per row instead of ~64 data-dependent ffs steps). Same contract; XOR is associative, so the result is
# identical bit for bit: synthetic A/B identical for useF 0 and 1 (1.85 -> 0.81 ms per call at n 10.6M, memory
# floor 0.79), and the Lanczos xb checksum after 2000 iterations of the key-1 matrix equals v1's (49587b04f8e1223e)
# at 14.26 vs 14.9 ms/it. inner128_v2 (2 rows per step) is identical but NOT faster (1.18 vs 1.16 ms): selectable
# as LA_KERN=v2i for A/B only. LA_KERN=v1 = the previous kernels. Both v2 kernels are compiled up front; any
# failure falls back to v1 with a WARNING.
KS128_V2=r'''
extern "C" __global__ void upd128_v2(const ulonglong2* p0,const ulonglong2* p1,const ulonglong2* p2,
                                     ulonglong2* pn,ulonglong2* xb,const ulonglong2* mats,
                                     unsigned long long mlo,unsigned long long mhi,int useF,long long n){
 // tables: T[m][c][v] = XOR of rows 4c+j of matrix m for bits j of v; m: 0=D 1=D2(X) 2=E 3=F
 __shared__ ulonglong2 T[4][32][16];
 for(int e=threadIdx.x;e<4*32*16;e+=blockDim.x){
   int m=e>>9, c=(e>>4)&31, v=e&15;
   if(m==3 && !useF) continue;   // F unused: skip the build (T[3] is never read then)
   /* mats layout is [D | E | F | D2] (see upd128's sD/sE/sF/sX loads); table index m: 0=D 1=D2 2=E 3=F */
   const int moff = (m==0)?0:(m==1)?384:(m==2)?128:256;
   const ulonglong2* M=mats+moff;
   unsigned long long ax=0ULL, ay=0ULL;
   #pragma unroll
   for(int j=0;j<4;j++) if(v&(1<<j)){ ulonglong2 t=M[4*c+j]; ax^=t.x; ay^=t.y; }
   ulonglong2 o; o.x=ax; o.y=ay; T[m][c][v]=o;
 }
 __syncthreads();
 for(long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x;r<n;r+=(long long)gridDim.x*blockDim.x){
   ulonglong2 a=p0[r], b=p1[r], y=pn[r], x=xb[r];
   y.x&=mlo; y.y&=mhi;
   unsigned long long w;
   w=a.x;
   #pragma unroll
   for(int c=0;c<16;c++){ int v=(int)((w>>(4*c))&15ULL); ulonglong2 t=T[0][c][v], s=T[1][c][v]; y.x^=t.x; y.y^=t.y; x.x^=s.x; x.y^=s.y; }
   w=a.y;
   #pragma unroll
   for(int c=0;c<16;c++){ int v=(int)((w>>(4*c))&15ULL); ulonglong2 t=T[0][16+c][v], s=T[1][16+c][v]; y.x^=t.x; y.y^=t.y; x.x^=s.x; x.y^=s.y; }
   w=b.x;
   #pragma unroll
   for(int c=0;c<16;c++){ int v=(int)((w>>(4*c))&15ULL); ulonglong2 t=T[2][c][v]; y.x^=t.x; y.y^=t.y; }
   w=b.y;
   #pragma unroll
   for(int c=0;c<16;c++){ int v=(int)((w>>(4*c))&15ULL); ulonglong2 t=T[2][16+c][v]; y.x^=t.x; y.y^=t.y; }
   if(useF){ ulonglong2 cc=p2[r];
     w=cc.x;
     #pragma unroll
     for(int c=0;c<16;c++){ int v=(int)((w>>(4*c))&15ULL); ulonglong2 t=T[3][c][v]; y.x^=t.x; y.y^=t.y; }
     w=cc.y;
     #pragma unroll
     for(int c=0;c<16;c++){ int v=(int)((w>>(4*c))&15ULL); ulonglong2 t=T[3][16+c][v]; y.x^=t.x; y.y^=t.y; }
   }
   pn[r]=y; xb[r]=x; }}

#define TL2 512
extern "C" __global__ void inner128_v2(const ulonglong2* A,const ulonglong2* B,unsigned long long* o1,
                                       unsigned long long* o2,long long n,int nout1){
 // same contract as inner128 (o1 += A^T B, o2 += B^T B over rows), two rows per inner step
 __shared__ ulonglong2 sa[TL2]; __shared__ ulonglong2 sb[TL2];
 const int t=threadIdx.x, bit=t&127, second=t>>7, w=bit>>6;
 const unsigned long long m=1ULL<<(bit&63);
 unsigned long long c0=0,c1=0;
 for(long long base=(long long)blockIdx.x*TL2; base<n; base+=(long long)gridDim.x*TL2){
   for(int q=t;q<TL2;q+=blockDim.x){
     long long r=base+q; ulonglong2 z; z.x=0ULL; z.y=0ULL;
     sa[q]= r<n? A[r]: z; sb[q]= r<n? B[r]: z; }
   __syncthreads();
   #pragma unroll 4
   for(int k=0;k<TL2;k+=2){
     ulonglong2 bv=sb[k], bw=sb[k+1];
     ulonglong2 av= second? bv: sa[k], aw= second? bw: sa[k+1];
     unsigned long long wd= w? av.y: av.x, we= w? aw.y: aw.x;
     unsigned long long mk=0ULL-(unsigned long long)((wd&m)!=0ULL), mj=0ULL-(unsigned long long)((we&m)!=0ULL);
     c0^=(bv.x&mk)^(bw.x&mj); c1^=(bv.y&mk)^(bw.y&mj); }
   __syncthreads();
 }
 if(second){ atomicXor(&o2[2*bit],c0); atomicXor(&o2[2*bit+1],c1); }
 else if(bit<nout1){ atomicXor(&o1[2*bit],c0); atomicXor(&o1[2*bit+1],c1); }}
'''
KS128=r'''
// 128-bit SpMV over one column part. VW=1<<VSH lanes per row (VSH=5: warp per row, the shipped shape).
// acc==0: y[r] = row sum (rows with weight > T are written 0 and left to spmv_heavy128);
// acc!=0: y[r] ^= row sum (rows with weight > T untouched).  v is already offset to the part's base.
extern "C" __global__ void spmv128(const ulonglong2* v,const int* col,const int* rp,ulonglong2* y,
                                   int nr,int T,int acc,int VSH){
 long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x; int VW=1<<VSH;
 long long gw=tid>>VSH; int lane=(int)(tid&(VW-1));
 unsigned long long a0=0,a1=0; int heavy=0;
 if(gw<nr){ int s=rp[gw],e=rp[gw+1];
   if(e-s>T) heavy=1;
   else for(int k=s+lane;k<e;k+=VW){ ulonglong2 t=v[col[k]]; a0^=t.x; a1^=t.y; } }
 for(int o=VW>>1;o>0;o>>=1){ a0^=__shfl_down_sync(0xffffffffu,a0,o,VW); a1^=__shfl_down_sync(0xffffffffu,a1,o,VW); }
 if(gw<nr && lane==0){
   if(acc){ if(!heavy){ ulonglong2 t=y[gw]; t.x^=a0; t.y^=a1; y[gw]=t; } }
   else { ulonglong2 t; t.x=heavy?0ULL:a0; t.y=heavy?0ULL:a1; y[gw]=t; } }}
// heavy rows of one part, as independent chunks: atomicXor partials into y (both words).
extern "C" __global__ void spmv_heavy128(const ulonglong2* v,const int* col,const int* hrow,const int* hs,
                                         const int* he,unsigned long long* y,int nchunk){
 int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(gw>=nchunk)return;
 int r=hrow[gw], s=hs[gw], e=he[gw]; unsigned long long a0=0,a1=0;
 for(int k=s+lane;k<e;k+=32){ ulonglong2 t=v[col[k]]; a0^=t.x; a1^=t.y; }
 for(int o=16;o>0;o>>=1){ a0^=__shfl_down_sync(0xffffffffu,a0,o); a1^=__shfl_down_sync(0xffffffffu,a1,o); }
 if(lane==0){ atomicXor(&y[2*(long long)r],a0); atomicXor(&y[2*(long long)r+1],a1); }}
// A^T B for N x 128 blocks, THREAD PER OUTPUT BIT (no shared atomics -- the 64-bit inner2 is
// shared-atomic-contention bound and at 128 bits would do 4x the atomics per row).
// A tile of TL rows of A and B is staged in shared memory; every thread then walks the tile reading
// the SAME address (broadcast) and XORs B[r] into its accumulator under a bit mask.
//   thread t <  128: o1[t] ^= XOR_{r: bit t of A[r]} B[r]   (only t < nout1)
//   thread t >= 128: o2[t-128] ^= XOR_{r: bit t-128 of B[r]} B[r]   (blockDim 256 only)
// o1/o2 are 128 x 2 uint64 (row i = 128-bit word). blockDim.x = 64 or 128 (o1 rows < blockDim only) or 256.
#define TL 256
extern "C" __global__ void inner128(const ulonglong2* A,const ulonglong2* B,unsigned long long* o1,
                                    unsigned long long* o2,long long n,int nout1){
 __shared__ ulonglong2 sa[TL]; __shared__ ulonglong2 sb[TL];
 const int t=threadIdx.x, bit=t&127, second=t>>7, w=bit>>6;
 const unsigned long long m=1ULL<<(bit&63);
 unsigned long long c0=0,c1=0;
 for(long long base=(long long)blockIdx.x*TL; base<n; base+=(long long)gridDim.x*TL){
   for(int q=t;q<TL;q+=blockDim.x){
     long long r=base+q; ulonglong2 z; z.x=0ULL; z.y=0ULL;
     sa[q]= r<n? A[r]: z; sb[q]= r<n? B[r]: z; }
   __syncthreads();
   #pragma unroll 8
   for(int k=0;k<TL;k++){
     ulonglong2 bv=sb[k]; ulonglong2 av= second? bv: sa[k];
     unsigned long long wd= w? av.y: av.x;
     unsigned long long mk=0ULL-(unsigned long long)((wd&m)!=0ULL);
     c0^=bv.x&mk; c1^=bv.y&mk; }
   __syncthreads();
 }
 if(second){ atomicXor(&o2[2*bit],c0); atomicXor(&o2[2*bit+1],c1); }
 else if(bit<nout1){ atomicXor(&o1[2*bit],c0); atomicXor(&o1[2*bit+1],c1); }}
// dense rows backward: z[i] ^= XOR_{j: bit j of dcol[i]} u[j]   (u = the vector's rows 0..127)
extern "C" __global__ void dense_bwd128(const unsigned long long* dcol,const ulonglong2* u,ulonglong2* z,long long ncols){
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<ncols;i+=(long long)gridDim.x*blockDim.x){
   unsigned long long w0=dcol[2*i],w1=dcol[2*i+1],a0=0,a1=0;
   while(w0){int j=__ffsll((long long)w0)-1; ulonglong2 t=u[j]; a0^=t.x; a1^=t.y; w0&=w0-1;}
   while(w1){int j=__ffsll((long long)w1)+63; ulonglong2 t=u[j]; a0^=t.x; a1^=t.y; w1&=w1-1;}
   ulonglong2 r=z[i]; r.x^=a0; r.y^=a1; z[i]=r; }}
// fused Lanczos update (replaces the mask pass + 4 blockmul_acc passes of the 64-bit loop):
//   pn = (pn & mask) ^ p0*D ^ p1*E ^ [useF] p2*F ;   xb ^= p0*D2
// mats = [D | E | F | D2], each 128 x ulonglong2 (row i = what bit i of the vector selects).
extern "C" __global__ void upd128(const ulonglong2* p0,const ulonglong2* p1,const ulonglong2* p2,
                                  ulonglong2* pn,ulonglong2* xb,const ulonglong2* mats,
                                  unsigned long long mlo,unsigned long long mhi,int useF,long long n){
 __shared__ ulonglong2 sD[128]; __shared__ ulonglong2 sE[128]; __shared__ ulonglong2 sF[128]; __shared__ ulonglong2 sX[128];
 for(int q=threadIdx.x;q<128;q+=blockDim.x){ sD[q]=mats[q]; sE[q]=mats[128+q]; sF[q]=mats[256+q]; sX[q]=mats[384+q]; }
 __syncthreads();
 for(long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x;r<n;r+=(long long)gridDim.x*blockDim.x){
   ulonglong2 a=p0[r], b=p1[r], y=pn[r], x=xb[r]; unsigned long long w;
   y.x&=mlo; y.y&=mhi;
   w=a.x; while(w){int i=__ffsll((long long)w)-1; ulonglong2 t=sD[i],s=sX[i]; y.x^=t.x; y.y^=t.y; x.x^=s.x; x.y^=s.y; w&=w-1;}
   w=a.y; while(w){int i=__ffsll((long long)w)+63; ulonglong2 t=sD[i],s=sX[i]; y.x^=t.x; y.y^=t.y; x.x^=s.x; x.y^=s.y; w&=w-1;}
   w=b.x; while(w){int i=__ffsll((long long)w)-1; ulonglong2 t=sE[i]; y.x^=t.x; y.y^=t.y; w&=w-1;}
   w=b.y; while(w){int i=__ffsll((long long)w)+63; ulonglong2 t=sE[i]; y.x^=t.x; y.y^=t.y; w&=w-1;}
   if(useF){ ulonglong2 c=p2[r];
     w=c.x; while(w){int i=__ffsll((long long)w)-1; ulonglong2 t=sF[i]; y.x^=t.x; y.y^=t.y; w&=w-1;}
     w=c.y; while(w){int i=__ffsll((long long)w)+63; ulonglong2 t=sF[i]; y.x^=t.x; y.y^=t.y; w&=w-1;} }
   pn[r]=y; xb[r]=x; }}
// VERIFY helper, deliberately independent of the CSR builds: y[R[k]] ^= x[C[k]] over the loader's COO.
extern "C" __global__ void coo_xor64(const int* R,const int* C,const unsigned long long* x,unsigned long long* y,long long nnz){
 for(long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x;k<nnz;k+=(long long)gridDim.x*blockDim.x){
   unsigned long long v=x[C[k]]; if(v) atomicXor(&y[R[k]],v); }}
'''
VB128=128; M64=(1<<64)-1; ALL128=(1<<128)-1
def _bits128(a):            # (128,2) uint64 -> (128,128) uint8, [i,j] = bit j of row i
 return np.unpackbits(np.ascontiguousarray(a,dtype='<u8').view(np.uint8).reshape(128,16),axis=1,bitorder='little')
def _pack128(B):
 return np.packbits(np.ascontiguousarray(B,dtype=np.uint8),axis=1,bitorder='little').view('<u8').reshape(128,2)
def mul128(a,b):
 # c[i] = XOR_{j: bit j of a[i]} b[j]  (same convention as mul64). float32 matmul is exact: counts <= 128.
 return _pack128(((_bits128(a).astype(np.float32) @ _bits128(b).astype(np.float32)).astype(np.int32)) & 1)
def tr128(a): return _pack128(_bits128(a).T)
BM128=_pack128(np.eye(128,dtype=np.uint8))
def rows_to_ints(a): return [l|(h<<64) for l,h in np.asarray(a,dtype=np.uint64).tolist()]
def ints_to_rows(v): return np.array([[x&M64, x>>64] for x in v],dtype=np.uint64).reshape(-1,2)
def mask_arr(m): return np.array([m&M64, m>>64],dtype=np.uint64)
def fnsV(t,ls,ld,VB):
 """msieve find_nonsingular_sub for VB-bit blocks. t: VB Python ints (rows of v'Av). Returns
 (dim, s, winv-as-VB-ints) or (0,[],None). Same pivot order as fns(): columns NOT in the previous
 subspace first, the previous ones (last_s) from the back. Each row of [t | I] is ONE Python int
 (t in the high VB bits, I in the low VB bits) so an elimination step is one XOR."""
 M=[(t[i]<<VB)|(1<<i) for i in range(VB)]
 cols=[0]*VB; mask=0
 for i in range(ld): cols[VB-1-i]=ls[i]; mask|=1<<ls[i]
 j=0
 for i in range(VB):
  if not (mask>>i)&1: cols[j]=i; j+=1
 dim=0; s=[]
 for i in range(VB):
  ci=cols[i]; m0=1<<(ci+VB); m1=1<<ci; p=-1
  for jj in range(i,VB):
   if M[cols[jj]]&m0: p=jj; break
  if p>=0:
   b=cols[p]; M[ci],M[b]=M[b],M[ci]; Mi=M[ci]
   M=[x^Mi if x&m0 else x for x in M]; M[ci]=Mi
   s.append(ci); dim+=1
  else:
   for jj in range(i,VB):
    if M[cols[jj]]&m1: p=jj; break
   if p<0: return 0,[],None
   b=cols[p]; M[ci],M[b]=M[b],M[ci]; Mi=M[ci]
   M=[x^Mi if x&m1 else x for x in M]          # also zeroes M[ci] itself = msieve's "wipe the pivot row"
 low=(1<<VB)-1
 return dim,s,[x&low for x in M]

# LA_HOST_C (default 1): the per-iteration 128x128 GF(2) host algebra (mul128 / tr128 / fnsV) in C
# (la128host.so next to this script, source la128host.c; exact ports, bit-identical on 3000 random + rank-
# deficient cases). Python: ~1.1 ms per Lanczos iteration with the GPU idle; C: ~0.1 ms. Missing .so or
# LA_HOST_C=0 -> the Python versions above.
_LAH=None
if os.environ.get("LA_HOST_C","1")=="1":
 try:
  import ctypes as _ct
  _LAH=_ct.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),"la128host.so"))
  _P64=_ct.POINTER(_ct.c_uint64)
  _mul128_py,_tr128_py=mul128,tr128
  def mul128(a,b):
   a=np.ascontiguousarray(a,dtype=np.uint64); b=np.ascontiguousarray(b,dtype=np.uint64); c=np.empty((128,2),np.uint64)
   _LAH.la_mul128(a.ctypes.data_as(_P64),b.ctypes.data_as(_P64),c.ctypes.data_as(_P64)); return c
  def tr128(a):
   a=np.ascontiguousarray(a,dtype=np.uint64); c=np.empty((128,2),np.uint64)
   _LAH.la_tr128(a.ctypes.data_as(_P64),c.ctypes.data_as(_P64)); return c
  BM128=np.ascontiguousarray(BM128,dtype=np.uint64)
 except OSError as _e:
  print(f"  WARNING: la128host.so unavailable ({_e}); Python host algebra (slower)",flush=True); _LAH=None
def fns128_rows(t_rows,ls,ld):
 """fnsV for the 128-bit loop on (128,2) rows -> (dim, s, winv rows) or (0, [], None)."""
 if _LAH is not None:
  t=np.ascontiguousarray(t_rows,dtype=np.uint64); lsa=(_ct.c_int*128)(*(list(ls[:ld])+[0]*(128-ld))); so=(_ct.c_int*128)()
  w=np.empty((128,2),np.uint64)
  d=_LAH.la_fns128(t.ctypes.data_as(_P64),lsa,_ct.c_int(ld),so,w.ctypes.data_as(_P64))
  return (d,list(so[:d]),w) if d else (0,[],None)
 d,s0,wi=fnsV(rows_to_ints(t_rows),ls,ld,128)
 return (d,s0,ints_to_rows(wi)) if d else (0,[],None)

def planes128(X,n,CH=1<<20):
 """(n,2) uint64 -> (128, cw) uint64 bit-planes: bit c of plane j = bit j of X[c]."""
 cw=(n+63)//64; P=np.zeros((128,cw),dtype=np.uint64); X=np.asarray(X,dtype=np.uint64).reshape(n,2)
 for a in range(0,n,CH):
  b=min(n,a+CH); m=b-a; mp=(m+63)//64*64
  blk=np.zeros((mp,2),dtype=np.uint64); blk[:m]=X[a:b]
  bits=np.unpackbits(blk.view(np.uint8).reshape(mp,16),axis=1,bitorder='little')     # (mp,128)
  pk=np.packbits(bits,axis=0,bitorder='little')                                        # (mp/8,128)
  P[:,a//64:a//64+mp//64]=np.ascontiguousarray(pk.T).view('<u8')
 return P
def rows_to_words64(S,n,CH=1<<20):
 """k<=64 bit-planes (k, cw) -> (n,) uint64 with bit j of out[c] = bit c of plane j."""
 S=np.asarray(S,dtype=np.uint64); k=S.shape[0]; out=np.zeros(n,dtype=np.uint64)
 if k==0: return out
 cw=S.shape[1]; step=CH//64
 for w0 in range(0,cw,step):
  w1=min(cw,w0+step); c0=w0*64; c1=min(n,w1*64)
  bits=np.zeros((64,(w1-w0)*64),dtype=np.uint8)
  bits[:k]=np.unpackbits(np.ascontiguousarray(S[:,w0:w1]).view(np.uint8),axis=1,bitorder='little')
  pk=np.packbits(bits,axis=0,bitorder='little')                                        # (8, cols)
  out[c0:c1]=np.ascontiguousarray(pk.T).view('<u8').reshape(-1)[:c1-c0]
 return out
def combine128(n,x,v,ax,av,nout=64):
 """msieve combine_cocycles for VBITS=128: Gaussian elimination on the 256 bit-planes of [Bx|Bv],
 mirrored on [x|v]; planes whose B-image reduces to zero are kernel vectors. Returns
 (out (n,) uint64 = up to nout independent non-zero kernel vectors, nd, extra planes, info)."""
 t0=time.time()
 mat=np.concatenate([planes128(x,n),planes128(v,n)]); amat=np.concatenate([planes128(ax,n),planes128(av,n)])
 R_=256; cw=mat.shape[1]; i=0
 # Only words where SOME plane of amat is non-zero can hold a pivot (row ops never grow the union
 # support), so skip the rest outright; within a word the next pivot is the lowest set bit of the
 # OR over the not-yet-pivoted rows.
 nzw=np.flatnonzero(np.bitwise_or.reduce(amat,axis=0)) if cw else np.zeros(0,np.int64)
 for w in nzw.tolist():
  while i<R_:
   colw=amat[i:,w]; orv=int(np.bitwise_or.reduce(colw))
   if orv==0: break
   m=np.uint64(orv&-orv)
   rows=np.flatnonzero(colw&m)+i; piv=int(rows[0])
   if piv!=i:
    amat[[i,piv]]=amat[[piv,i]]; mat[[i,piv]]=mat[[piv,i]]
   oth=rows[1:]
   if len(oth):
    amat[oth,w:]^=amat[i,w:]; mat[oth]^=mat[i]
   i+=1
  if i>=R_: break
 rank=i
 cand=mat[rank:]; nzr=cand.any(axis=1) if cw else np.zeros(len(cand),bool)
 PW=min(cw,256); basis={}; chosen=[]; unsure=[]
 for k in range(len(cand)):
  if not nzr[k]: continue
  vv=int.from_bytes(cand[k,:PW].tobytes(),'little')
  if vv==0: unsure.append(k); continue
  while vv:
   h=vv.bit_length()-1
   if h in basis: vv^=basis[h]
   else: basis[h]=vv; chosen.append(k); break
  if len(chosen)>=2*nout: break
 first=chosen[:nout]
 if len(first)<nout and unsure:   # non-zero, but zero on the prefix: cannot vouch for independence
  first+=unsure[:nout-len(first)]
 out=rows_to_words64(cand[first],n)
 extra=cand[chosen[nout:2*nout]] if len(chosen)>nout else None
 info=dict(rank=rank,cand=int(len(cand)),nonzero=int(nzr.sum()),indep=len(chosen),unsure=len(unsure),t=time.time()-t0)
 return out,len(first),extra,info

def _mtake(a,m):
 # a[m] for a CuPy bool mask. CuPy's boolean-mask indexing raises OverflowError ("value too large to convert to
 # int", _prepare_mask_indexing_single) once the mask has >= 2^31 elements -- the TD 220 crash (2.77G weight). Below
 # that size this IS a[m]; above it the same selection is made in 2^30-element chunks and concatenated: same elements,
 # same order, so the result is identical.
 if a.size<2**31-1: return a[m]
 CH=1<<30; return cp.concatenate([a[i:i+CH][m[i:i+CH]] for i in range(0,int(a.size),CH)])
def _rowptr128(d_idx,n):
 nnz=int(d_idx.size)
 d_cnt=cp.zeros(n,cp.int32)
 if nnz: rcount((4096,),(256,),(d_idx,d_cnt,np.int64(nnz)))
 rp=np.zeros(n+1,np.int64); rp[1:]=np.cumsum(cp.asnumpy(d_cnt),dtype=np.int64)
 # spmv128/spmv256 and build_heavy hold row pointers / chunk offsets as int32: one part must stay below 2^31 entries
 if rp[-1]>=2**31: raise OverflowError(f"column-split part has {int(rp[-1])} entries >= 2^31 (int32 row pointers); raise LA_SPLIT")
 return cp.asarray(rp.astype(np.int32))
def build_parts128(d_Ri,d_Ci,n,P):
 """Column-split CSRs. B (row R, gathers src[C]): part p holds the entries with C in [cb[p],cb[p+1]),
 column-relative, rows in order and columns ascending within a row (packed-key sort, as the 64-bit
 build). B^T (row C, gathers src[R]): part p holds R in [rb[p],rb[p+1]), in file order (C is
 non-decreasing, so the mask keeps it grouped by row, exactly like d_colBt=d_Ri)."""
 nnz=int(d_Ri.size); cb=[(p*n)//P for p in range(P+1)]
 partsB=[]; partsBt=[]; totB=0; totBt=0
 for p in range(P):
  c0,c1=cb[p],cb[p+1]
  sel=(d_Ci>=c0)&(d_Ci<c1); Rp=_mtake(d_Ri,sel); Cp=(_mtake(d_Ci,sel)-c0).astype(cp.int32); del sel
  rp=_rowptr128(Rp,n)
  try:
   # Cp stays alive until the sort succeeded: the OOM fallback below needs it (a `del Cp` here made
   # the fallback raise UnboundLocalError instead of falling back)
   key=(Rp.astype(cp.uint64)<<cp.uint64(32))|Cp.astype(cp.uint64)
   key.sort(); col=(key&cp.uint64(0xffffffff)).astype(cp.int32); del key
  except cp.cuda.memory.OutOfMemoryError as e:
   print(f"  [la128] GPU sort OOM ({e}); host argsort for part {p}",flush=True)
   key=None; cp.get_default_memory_pool().free_all_blocks()
   Rh=cp.asnumpy(Rp); Ch=cp.asnumpy(Cp)
   col=cp.asarray(Ch[np.argsort(Rh,kind='stable')]); del Rh,Ch
  del Cp
  del Rp; totB+=int(col.size)
  partsB.append((c0,c1,col,rp,build_heavy(rp,f"B[{p}/{P}]")))
 for p in range(P):
  r0,r1=cb[p],cb[p+1]
  sel=(d_Ri>=r0)&(d_Ri<r1); col=(_mtake(d_Ri,sel)-r0).astype(cp.int32); Cp=_mtake(d_Ci,sel); del sel
  rp=_rowptr128(Cp,n); del Cp; totBt+=int(col.size)
  partsBt.append((r0,r1,col,rp,build_heavy(rp,f"Bt[{p}/{P}]")))
 if totB!=nnz or totBt!=nnz:
  print(f"  FATAL: column split lost entries (B {totB}, Bt {totBt}, want {nnz})",flush=True); sys.exit(3)
 return partsB,partsBt

def run128(R,C,dcol,ND,n):
 """Build the split CSRs, run 128-bit block Lanczos, verify. -> (x (n,) uint64, nd, ok)"""
 global spmv128,spheavy128,inner128,dbwd128,upd128,coo64
 spmv128=cp.RawKernel(KS128,'spmv128',options=OPT); spheavy128=cp.RawKernel(KS128,'spmv_heavy128',options=OPT)
 inner128=cp.RawKernel(KS128,'inner128',options=OPT); dbwd128=cp.RawKernel(KS128,'dense_bwd128',options=OPT)
 upd128=cp.RawKernel(KS128,'upd128',options=OPT); coo64=cp.RawKernel(KS128,'coo_xor64',options=OPT)
 global inner_k,upd_k
 inner_k,upd_k=inner128,upd128
 _lk=os.environ.get("LA_KERN","v2")
 if _lk in ("v2","v2i"):
  try:
   upd_k=cp.RawKernel(KS128_V2,'upd128_v2',options=OPT); upd_k.compile()     # compile NOW: a lazy compile error
   if _lk=="v2i": inner_k=cp.RawKernel(KS128_V2,'inner128_v2',options=OPT); inner_k.compile()   # would hit mid-LA
   print(f"  [la128] LA_KERN={_lk} kernels (table upd128{', 2-row inner128' if _lk=='v2i' else ''})",flush=True)
  except Exception as _e:
   print(f"  WARNING: LA_KERN=v2 kernels unavailable ({type(_e).__name__}: {str(_e)[:120]}) -- v1 kernels",flush=True)
   inner_k,upd_k=inner128,upd128
 t=time.time()
 Ri=R.astype(np.int32,copy=False); Ci=C.astype(np.int32,copy=False)
 d_Ri=cp.asarray(Ri); d_Ci=cp.asarray(Ci); del Ri,Ci
 nnz=int(d_Ri.size)
 SEG=None
 if LA_SPMV=="seg":
  try:
   SEG=_load_seg().SegPlan(d_Ri,d_Ci,n,ND,width=128)
   print(f"  seg SpMV plan built in {time.time()-t:.0f}s: LA_BLOCK=128, nnz {nnz}",flush=True)
  except Exception as _e:   # never let the fast path kill the run: fall back to the column-split SpMV
   print(f"  WARNING: seg SpMV plan failed ({type(_e).__name__}: {str(_e)[:200]}) -- using the column-split SpMV",flush=True)
   SEG=None; cp.get_default_memory_pool().free_all_blocks()
 if SEG is None:
  SPLIT_MB=float(os.environ.get("LA_SPLIT_MB","100"))
  P=int(os.environ.get("LA_SPLIT","0") or 0)
  if P<=0: P=max(1,-(-(n*16)//int(SPLIT_MB*2**20)))
  P=max(1,min(P,max(1,n)))
  partsB,partsBt=build_parts128(d_Ri,d_Ci,n,P)
  print(f"  sparse CSR built in {time.time()-t:.0f}s: LA_BLOCK=128, {P} column part(s) of <= "
        f"{-(-n//P)*16/2**20:.1f} MB of vector (LA_SPLIT_MB={SPLIT_MB:g}), nnz {nnz}",flush=True)
 d_dcol=cp.asarray(dcol)
 VSH=max(0,min(5,int(os.environ.get("LA_VW","32")).bit_length()-1)); VW=1<<VSH
 tpb=256; WBW=((n*VW+tpb-1)//tpb,); IGR=(max(1,min(2048,(n+255)//256)),); UG=(max(1,min(8192,(n+255)//256)),)
 DG=G(n); INTMAX=np.int32(2**31-1)
 heavy_grid=lambda H: G(H[3]*32)
 def spmv_parts(parts,src,dst):
  for p,(c0,c1,col,rp,H) in enumerate(parts):
   sv=src[c0:c1]
   spmv128(WBW,(tpb,),(sv,col,rp,dst,np.int32(n),np.int32(HEAVY_T) if H else INTMAX,np.int32(1 if p else 0),np.int32(VSH)))
   if H: spheavy128(heavy_grid(H),(tpb,),(sv,col,H[0],H[1],H[2],dst,np.int32(H[3])))
 dummy=cp.zeros((128,2),cp.uint64)
 DTB=64 if ND<=64 else 128     # dense_fwd = inner128 with one thread per dense row (bits >= ND are 0)
 def Bmul(src,dst):
  if SEG is not None: SEG.B(src,dst)
  else: spmv_parts(partsB,src,dst)
  if ND: inner128(IGR,(DTB,),(d_dcol,src,dst,dummy,np.int64(n),np.int32(ND)))   # dense rows: dst[j]^=..., j<ND
 def Btmul(src,dst):
  if SEG is not None: SEG.Bt(src,dst)
  else: spmv_parts(partsBt,src,dst)
  if ND: dbwd128(DG,(tpb,),(d_dcol,src,dst,np.int64(n)))
 print("running GPU block-Lanczos (LA_BLOCK=128, %s SpMV) ..."%("segmented" if SEG is not None else "column-split"),flush=True); t=time.time()
 x,nd,it,info,extra=gpu_block_lanczos128(Bmul,Btmul,n,IGR,UG)
 print(f"  LA done: {it} iters, {nd} deps, {time.time()-t:.0f}s  (N/iters = {n/max(it,1):.2f}; combine: rank {info.get('rank')}, "
       f"{info.get('nonzero')} non-zero / {info.get('indep')} independent candidates, {info.get('t',0):.1f}s)",flush=True)
 if x is None: return None,0,False,None
 # VERIFY on the ORIGINAL COO (independent of the split CSR and of the 128-bit kernels)
 xg=cp.asarray(x); bx=cp.zeros(n,cp.uint64)
 coo64((4096,),(256,),(d_Ri,d_Ci,xg,bx,np.int64(nnz))); dfwd(G(n),(256,),(d_dcol,xg,bx,n,ND))
 ok=bool((bx.get()==0).all()) and bool((x!=0).any())
 print(f"  VERIFY B*x==0 & x!=0: {ok}  (COO path, {nnz} nonzeros + {ND} dense rows)",flush=True)
 xe=None
 if extra is not None and os.environ.get("LA_KER2","0")=="1":
  xe=rows_to_words64(extra,n); xg=cp.asarray(xe); bx[:]=0
  coo64((4096,),(256,),(d_Ri,d_Ci,xg,bx,np.int64(nnz))); dfwd(G(n),(256,),(d_dcol,xg,bx,n,ND))
  if not bool((bx.get()==0).all()):
   print("  WARNING: LA_KER2 spare vectors FAIL B*x==0 -- not written",flush=True); xe=None
 return x,nd,ok,xe

def gpu_block_lanczos128(Bmul,Btmul,n,IGR,UG):
 tpb=256
 sc=cp.empty((n,2),cp.uint64)
 def AvP(src,dst): Bmul(src,sc); Btmul(sc,dst)
 p0=cp.empty((n,2),cp.uint64); p1=cp.zeros((n,2),cp.uint64); p2=cp.zeros((n,2),cp.uint64); pn=cp.empty((n,2),cp.uint64)
 xb=cp.asarray(np.random.default_rng(int(os.environ.get("LA_SEED","5"))).integers(0,1<<64,(n,2),dtype=np.uint64))
 AvP(xb,p0); vsav=p0.copy()
 O=cp.zeros((2,128,2),cp.uint64); O2=cp.zeros((2,128,2),cp.uint64)
 dmats=cp.zeros((4,128,2),cp.uint64); hmats=np.zeros((4,128,2),np.uint64)
 Z=lambda: np.zeros((128,2),dtype=np.uint64)
 vt_a_v=[Z(),Z()]; vt_a2_v=[Z(),Z()]; vt_v0=[Z(),Z(),Z()]; winv=[Z(),Z(),Z()]
 s=[[0]*128,list(range(128))]; dim1=128; mask1=ALL128; it=0; dim0=0; t0=time.time()
 PROF=int(os.environ.get("LA_PROF","0")); tGPU=0.0; tCPU=0.0; _t1=time.time()
 # LA_PROF=2: also event-time Bmul and Btmul (incl. their dense-row kernels) inside the loop; the events are read
 # after the O.get() sync the loop already has, so this adds no synchronisation.
 _pe=[cp.cuda.Event() for _ in range(3)] if PROF>=2 else None; tBm=0.0; tBtm=0.0
 LOGI=int(os.environ.get("LA_LOG_EVERY","2000"))
 ITMAX=n//120+64     # ~n/127.2 expected at 128 bits; see the 64-bit loop's HARD ITERATION CAP
 while True:
  it+=1
  if _LA_MAXIT and it>_LA_MAXIT:
   print(f"  [la] LA_MAXIT={_LA_MAXIT} reached after {time.time()-t0:.1f}s = {(time.time()-t0)*1000/(it-1):.2f} ms/it -- timing run, stopping",flush=True)
   dim0=0; break
  if it>ITMAX:
   print(f"  [la128] FATAL: block Lanczos did not converge in {ITMAX} iterations (expected ~{n//127}) -- corrupted state?",flush=True)
   dim0=0; break
  if PROF: _t0=time.time()
  if PROF>=2: _pe[0].record(); Bmul(p0,sc); _pe[1].record(); Btmul(sc,pn); _pe[2].record()
  else: AvP(p0,pn)
  O[:]=0; inner_k(IGR,(256,),(p0,pn,O[0],O[1],np.int64(n),np.int32(128)))
  Oh=O.get(); vt_a_v[0]=Oh[0].copy(); vt_a2_v[0]=Oh[1].copy()
  if PROF:
   _t1=time.time(); tGPU+=_t1-_t0
   if PROF>=2: tBm+=cp.cuda.get_elapsed_time(_pe[0],_pe[1]); tBtm+=cp.cuda.get_elapsed_time(_pe[1],_pe[2])
   if it%500==0:
    _el=time.time()-t0
    print("   [LA_PROF] it=%d  %.2f ms/it | GPU %.2f ms (%.0f%%) | CPU %.2f ms (%.0f%%)"%(
      it,_el*1000/it,tGPU*1000/it,100*tGPU/_el,tCPU*1000/it,100*tCPU/_el)+
      ("" if PROF<2 else " | Bmul %.3f ms Btmul %.3f ms (events, incl. dense rows) | xb sum %016x"%(tBm/it,tBtm/it,int(xb.sum())&0xFFFFFFFFFFFFFFFF)),flush=True)
  if not vt_a_v[0].any(): break
  dim0,s0,wi=fns128_rows(vt_a_v[0],s[1],dim1)
  if dim0==0: break
  winv[0]=wi
  mask0=0
  for i in s0: mask0|=1<<i
  m1c=0
  for i in s[1][:dim1]: m1c|=1<<i
  if (mask0|m1c)!=ALL128:     # msieve: "not all columns used" -- the recurrence would be invalid
   print(f"  [la128] ERROR: not all columns used at iteration {it} (dim0={dim0}, dim1={dim1})",flush=True)
   dim0=0; break
  s[0]=s0+[0]*(128-len(s0)); M0=mask_arr(mask0)
  if it<4:
   O2[:]=0; inner128(IGR,(128,),(p0,vsav,O2[0],O2[1],np.int64(n),np.int32(128))); vt_v0[0]=O2.get()[0].copy()
  d=(vt_a2_v[0]&M0)^vt_a_v[0]; d=mul128(winv[0],d); d=d^BM128
  vt_v0_next=mul128(tr128(d),vt_v0[0])
  e=mul128(winv[1],vt_a_v[0]); e=e&M0
  vt_v0_next=vt_v0_next^mul128(tr128(e),vt_v0[1])
  useF=mask1!=ALL128
  if useF:
   M1=mask_arr(mask1)
   f=mul128(vt_a_v[1],winv[1]); f=f^BM128; f=mul128(winv[2],f)
   f2=((vt_a2_v[1]&M1)^vt_a_v[1])&M0; f=mul128(f,f2)
   vt_v0_next=vt_v0_next^mul128(tr128(f),vt_v0[2])
  else: f=Z()
  d2=mul128(winv[0],vt_v0[0])
  hmats[0]=d; hmats[1]=e; hmats[2]=f; hmats[3]=d2; dmats.set(hmats)
  upd_k(UG,(tpb,),(p0,p1,p2,pn,xb,dmats,np.uint64(mask0&M64),np.uint64(mask0>>64),np.int32(1 if useF else 0),np.int64(n)))
  p0,p1,p2,pn = pn,p0,p1,p2
  winv=[winv[2],winv[0],winv[1]]; vt_v0=[vt_v0_next,vt_v0[0],vt_v0[1]]
  vt_a_v=[vt_a_v[1],vt_a_v[0]]; vt_a2_v=[vt_a2_v[1],vt_a2_v[0]]
  if hostf is None: s[1]=list(s[0]); mask1=mask0
  else: s1h=s0h; M1h=M0
  dim1=dim0
  if PROF: tCPU+=time.time()-_t1
  if it%LOGI==0: print(f"   iter {it}  {time.time()-t0:.0f}s ({(time.time()-t0)*1000/it:.2f} ms/it)",flush=True)
 if dim0==0: return None,0,it,{},None
 ax=cp.empty((n,2),cp.uint64); av=cp.empty((n,2),cp.uint64)
 Bmul(xb,ax); Bmul(p0,av)
 out,nd,extra,info=combine128(n,cp.asnumpy(xb),cp.asnumpy(p0),cp.asnumpy(ax),cp.asnumpy(av))
 return out,nd,it,info,extra

# =================================================================================================
# LA_BLOCK=256 (W6, 2026-10-17): the 128-bit path at 256-bit block width -- half the Lanczos iterations
# (n/254 instead of n/127) for a per-pass SpMV cost measured at 1.12x the 128-bit pass on the RTX PRO 6000
# when the gathered x slice stays in L2 (ledger "W6 gate": gpu/labench/spmv_gf2.cu), i.e. ~0.56x the LA wall.
# Rows are ulonglong4 = (n,4) uint64. Host algebra in la256host.c (same ports as la128host.c), Python
# fallbacks width-generic. Kernels: the 128-bit ones re-typed (inner256 = 512 threads per block: 256 output
# bits for o1 + 256 for o2; upd256 = the ffs variant, 32 KB of shared tables; dense rows via dense_fwd256
# since the dense bitfield stays 2 words per column). SpMV: segspmv width 256 (6 column parts at 16M rows)
# or the column-split spmv256 fallback. Output: the same 64 kernel vectors (combine on 512 bit-planes).
# =================================================================================================
VB256=256; ALL256=(1<<256)-1
def _bitsV(a,VB): return np.unpackbits(np.ascontiguousarray(a,dtype='<u8').view(np.uint8).reshape(VB,VB//8),axis=1,bitorder='little')
def _packV(B,VB): return np.packbits(np.ascontiguousarray(B,dtype=np.uint8),axis=1,bitorder='little').view('<u8').reshape(VB,VB//64)
def mulV_np(a,b,VB): return _packV(((_bitsV(a,VB).astype(np.float32)@_bitsV(b,VB).astype(np.float32)).astype(np.int32))&1,VB)
def trV_np(a,VB): return _packV(_bitsV(a,VB).T,VB)
def rows_to_intsV(a,W): return [sum(int(w)<<(64*k) for k,w in enumerate(r)) for r in np.asarray(a,dtype=np.uint64).reshape(-1,W).tolist()]
def ints_to_rowsV(v,W): return np.array([[(x>>(64*k))&M64 for k in range(W)] for x in v],dtype=np.uint64).reshape(-1,W)
def mask_arrV(m,W): return np.array([(m>>(64*k))&M64 for k in range(W)],dtype=np.uint64)
BM256=_packV(np.eye(256,dtype=np.uint8),256)
_LAH256=None
if os.environ.get("LA_HOST_C","1")=="1":
 try:
  import ctypes as _ct2
  _LAH256=_ct2.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),"la256host.so"))
  _P64b=_ct2.POINTER(_ct2.c_uint64)
 except OSError as _e:
  if LA_BLOCK==256: print(f"  WARNING: la256host.so unavailable ({_e}); Python host algebra (slower)",flush=True)
  _LAH256=None
def mul256(a,b):
 if _LAH256 is None: return mulV_np(a,b,256)
 a=np.ascontiguousarray(a,dtype=np.uint64); b=np.ascontiguousarray(b,dtype=np.uint64); c=np.empty((256,4),np.uint64)
 _LAH256.la_mul256(a.ctypes.data_as(_P64b),b.ctypes.data_as(_P64b),c.ctypes.data_as(_P64b)); return c
def tr256(a):
 if _LAH256 is None: return trV_np(a,256)
 a=np.ascontiguousarray(a,dtype=np.uint64); c=np.empty((256,4),np.uint64)
 _LAH256.la_tr256(a.ctypes.data_as(_P64b),c.ctypes.data_as(_P64b)); return c
def fns256_rows(t_rows,ls,ld):
 if _LAH256 is not None:
  t=np.ascontiguousarray(t_rows,dtype=np.uint64); lsa=(_ct2.c_int*256)(*(list(ls[:ld])+[0]*(256-ld))); so=(_ct2.c_int*256)()
  w=np.empty((256,4),np.uint64)
  d=_LAH256.la_fns256(t.ctypes.data_as(_P64b),lsa,_ct2.c_int(ld),so,w.ctypes.data_as(_P64b))
  return (d,list(so[:d]),w) if d else (0,[],None)
 d,s0,wi=fnsV(rows_to_intsV(t_rows,4),ls,ld,256)
 return (d,s0,ints_to_rowsV(wi,4)) if d else (0,[],None)
# LA_V3 host: la256host.c's la_mul256f / la_tr256f / la_fns256f (Four Russians product, 64x64-block transpose,
# branch-free elimination + the chosen-column mask) behind thin ctypes wrappers (c_void_p arguments, no list->ctypes
# conversions); used by gpu_block_lanczos256 only when LA_V3 has 'host' and _v3_setup's self-test passed.
def _hostf_bind():
 L=_LAH256
 if L is None: raise RuntimeError("la256host.so not loaded")
 vp=_ct2.c_void_p
 for nm,na in (("la_mul256f",3),("la_tr256f",2),("la_fns256f",6)): getattr(L,nm).argtypes=[vp]*na   # AttributeError: old .so
 L.la_fns256f.restype=_ct2.c_int
 def mulf(a,b):
  a=np.ascontiguousarray(a,dtype=np.uint64); b=np.ascontiguousarray(b,dtype=np.uint64); c=np.empty((256,4),np.uint64)
  L.la_mul256f(a.ctypes.data,b.ctypes.data,c.ctypes.data); return c
 def trf(a):
  a=np.ascontiguousarray(a,dtype=np.uint64); c=np.empty((256,4),np.uint64); L.la_tr256f(a.ctypes.data,c.ctypes.data); return c
 def fnsf(t,ls,ld):
  """-> (dim, s_out int32[dim], winv, mask uint64[4]); ls: int32 array (its first ld entries are used)"""
  t=np.ascontiguousarray(t,dtype=np.uint64); ls=np.ascontiguousarray(ls,dtype=np.int32)
  so=np.empty(256,np.int32); w=np.empty((256,4),np.uint64); m=np.empty(4,np.uint64)
  d=L.la_fns256f(t.ctypes.data,ls.ctypes.data,ld,so.ctypes.data,w.ctypes.data,m.ctypes.data)
  return d,so[:d].copy(),w,m
 return mulf,trf,fnsf
def planesV(X,n,VB,CH=1<<20):
 """(n,VB/64) uint64 -> (VB, cw) uint64 bit-planes: bit c of plane j = bit j of X[c]."""
 W=VB//64; cw=(n+63)//64; P=np.zeros((VB,cw),dtype=np.uint64); X=np.asarray(X,dtype=np.uint64).reshape(n,W)
 for a in range(0,n,CH):
  b=min(n,a+CH); m=b-a; mp=(m+63)//64*64
  blk=np.zeros((mp,W),dtype=np.uint64); blk[:m]=X[a:b]
  bits=np.unpackbits(blk.view(np.uint8).reshape(mp,VB//8),axis=1,bitorder='little')
  pk=np.packbits(bits,axis=0,bitorder='little')
  P[:,a//64:a//64+mp//64]=np.ascontiguousarray(pk.T).view('<u8')
 return P
def combineV(n,x,v,ax,av,VB,nout=64,count_all=False):
 """combine128 for any block width: elimination on the 2*VB bit-planes of [Bx|Bv] mirrored on [x|v].
 count_all=True (LA_DEFER_ROWS only) adds info['indep_all']: the UNCAPPED number of head-independent non-zero
 candidates (the selection itself stops at 2*nout); the returned vectors are the same either way."""
 t0=time.time()
 mat=np.concatenate([planesV(x,n,VB),planesV(v,n,VB)]); amat=np.concatenate([planesV(ax,n,VB),planesV(av,n,VB)])
 R_=2*VB; cw=mat.shape[1]; i=0
 nzw=np.flatnonzero(np.bitwise_or.reduce(amat,axis=0)) if cw else np.zeros(0,np.int64)
 for w in nzw.tolist():
  while i<R_:
   colw=amat[i:,w]; orv=int(np.bitwise_or.reduce(colw))
   if orv==0: break
   m=np.uint64(orv&-orv)
   rows=np.flatnonzero(colw&m)+i; piv=int(rows[0])
   if piv!=i:
    amat[[i,piv]]=amat[[piv,i]]; mat[[i,piv]]=mat[[piv,i]]
   oth=rows[1:]
   if len(oth):
    amat[oth,w:]^=amat[i,w:]; mat[oth]^=mat[i]
   i+=1
  if i>=R_: break
 rank=i
 cand=mat[rank:]; nzr=cand.any(axis=1) if cw else np.zeros(len(cand),bool)
 PW=min(cw,R_); basis={}; chosen=[]; unsure=[]
 for k in range(len(cand)):
  if not nzr[k]: continue
  vv=int.from_bytes(cand[k,:PW].tobytes(),'little')
  if vv==0: unsure.append(k); continue
  while vv:
   h=vv.bit_length()-1
   if h in basis: vv^=basis[h]
   else: basis[h]=vv; chosen.append(k); break
  if len(chosen)>=2*nout: break
 first=chosen[:nout]
 if len(first)<nout and unsure: first+=unsure[:nout-len(first)]
 out=rows_to_words64(cand[first],n)
 extra=cand[chosen[nout:2*nout]] if len(chosen)>nout else None
 info=dict(rank=rank,cand=int(len(cand)),nonzero=int(nzr.sum()),indep=len(chosen),unsure=len(unsure),t=time.time()-t0)
 if count_all: info["indep_all"]=_indep_all([int.from_bytes(cand[k,:PW].tobytes(),'little') for k in range(len(cand)) if nzr[k]])
 return out,len(first),extra,info
def _indep_all(heads):
 """GF(2) rank of the candidate head bitsets (python ints): combineV's independence test without its 2*nout cap."""
 basis={}
 for vv in heads:
  while vv:
   h=vv.bit_length()-1
   if h in basis: vv^=basis[h]
   else: basis[h]=vv; break
 return len(basis)

KS256=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"ks256.cu")).read()

# LA_V256 (2026-10-20): faster non-SpMV kernels for the 256-bit loop, in ks256v2.cu (read from this script's REAL
# directory, compiled only when asked, so the default path never depends on it). Comma list:
#   upd   table-driven update upd256_v2 (+ updF256 pass when useF) instead of the ffs upd256   (~9.7 ms/it at TD 100)
#   inner inner256_v2 (word-select, 32-bit mask) instead of inner256                            (~5.6 ms/it)
#   db    dense_bwd256 fused into the inner-product pass (implies inner)                        (saves ~1 GB/it)
#   all   = upd,inner,db       xfix = upd256_v2x: also update bits 0..127 of xb (see ks256v2.cu: ks256.cu's X4(x,s)
#         never did) -- changes xb (and the LA_PROF xb sum), not the recurrence; NOT bit-identical to the old path.
# Without xfix every kernel is bit-identical to the old one; a start-up self-test compares them on random data and
# falls back to the old kernels with a WARNING (LA_V256_STRICT=1: exit 5) on any difference.
def _v256_opts():
 # default "all" since sandbox #22: TD 100 56.78 -> 49.70 ms/it, xb checksums identical over 1,500 iterations; LA_V256=none = old kernels
 o=set(x.strip() for x in os.environ.get("LA_V256","all").split(",") if x.strip() and x.strip()!="none")
 if "all" in o: o|={"upd","inner","db"}; o.discard("all")
 if "db" in o: o.add("inner")
 if "xfix" in o: o.add("upd")
 bad=o-{"upd","inner","db","xfix"}
 if bad: print(f"  FATAL: LA_V256 has unknown option(s) {sorted(bad)} (upd, inner, db, all, xfix)",flush=True); sys.exit(3)
 return o
def _v256_setup(old,n_sm):
 """-> dict of LA_V256 kernels (or None): compile ks256v2.cu, self-test against the old kernels on random data."""
 o=_v256_opts()
 if not o: return None
 strict=os.environ.get("LA_V256_STRICT","0")=="1"
 try:
  src=open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"ks256v2.cu")).read()
  K={nm:cp.RawKernel(src,nm,options=OPT) for nm in ("upd256_v2","upd256_v2x","updF256","inner256_v2","inner256_db")}
  for k in K.values(): k.compile()
  smo=int(cp.cuda.Device().attributes.get('MaxSharedMemoryPerBlockOptin',0))
  if smo<98304: raise RuntimeError(f"opt-in shared memory {smo} B < 98304 B for upd256_v2")
  for nm in ("upd256_v2","upd256_v2x"): K[nm].max_dynamic_shared_size_bytes=98304
  K["updF256"].max_dynamic_shared_size_bytes=32768
  V=dict(opts=o,upd=K["upd256_v2x" if "xfix" in o else "upd256_v2"],updF=K["updF256"],inner=K["inner256_v2"],idb=K["inner256_db"],
         UG=(n_sm,),UT=512)
  # ---- self-test (random data, odd n, both useF; ~ms) ----
  t0=time.time(); g=np.random.default_rng(20261020); nt=200003; W=4
  R=lambda *sh: cp.asarray(g.integers(0,1<<64,sh,dtype=np.uint64))
  p0,p1,p2,pn,xb=(R(nt,W) for _ in range(5)); mats=R(4,256,W)
  msk=cp.asarray(np.array([g.integers(0,1<<64,dtype=np.uint64),~np.uint64(0),g.integers(0,1<<64,dtype=np.uint64),np.uint64(2**63-1)],np.uint64))
  IG=(max(1,min(2048,(nt+255)//256)),); UGo=(max(1,min(8192,(nt+255)//256)),)
  errs=[]
  for useF in (0,1):
   a_pn,a_xb,b_pn,b_xb=pn.copy(),xb.copy(),pn.copy(),xb.copy()
   old["upd"](UGo,(256,),(p0,p1,p2,a_pn,a_xb,mats,msk,np.int32(useF),np.int64(nt)))
   V["upd"](V["UG"],(V["UT"],),(p0,p1,b_pn,b_xb,mats,msk,np.int64(nt)),shared_mem=98304)
   if useF: V["updF"](V["UG"],(V["UT"],),(p2,b_pn,mats,np.int64(nt)),shared_mem=32768)
   if not bool((a_pn==b_pn).all()): errs.append(f"upd pn useF={useF}")
   if "xfix" in o:
    # reference for xb words 0,1: the old kernel (which updates words 2,3 correctly) on D2 with words 0,1 moved to 2,3
    m2=mats.copy(); m2[3,:,2:4]=mats[3,:,0:2]; c_pn,c_xb=pn.copy(),xb.copy(); c_xb[:,2:4]=xb[:,0:2]
    old["upd"](UGo,(256,),(p0,p1,p2,c_pn,c_xb,m2,msk,np.int32(useF),np.int64(nt)))
    if not (bool((b_xb[:,2:4]==a_xb[:,2:4]).all()) and bool((b_xb[:,0:2]==c_xb[:,2:4]).all())): errs.append(f"upd xfix xb useF={useF}")
   elif not bool((a_xb==b_xb).all()): errs.append(f"upd xb useF={useF}")
  for bd,no in ((512,256),(256,256),(256,100)):
   O1=cp.zeros((2,256,W),cp.uint64); O2=cp.zeros((2,256,W),cp.uint64)
   old["inner"](IG,(bd,),(p0,pn,O1[0],O1[1],np.int64(nt),np.int32(no)))
   V["inner"](IG,(bd,),(p0,pn,O2[0],O2[1],np.int64(nt),np.int32(no)))
   if not bool((O1==O2).all()): errs.append(f"inner blockDim {bd} nout1 {no}")
  dc=cp.asarray(g.integers(0,1<<64,(nt,2),dtype=np.uint64)&g.integers(0,1<<64,(nt,2),dtype=np.uint64)&g.integers(0,1<<64,(nt,2),dtype=np.uint64))
  dc[::7]=0; u=R(nt,W); a_z,b_z=pn.copy(),pn.copy()
  O1=cp.zeros((2,256,W),cp.uint64); O2=cp.zeros((2,256,W),cp.uint64)
  old["dbwd"](G(nt),(256,),(dc,u,a_z,np.int64(nt))); old["inner"](IG,(512,),(p0,a_z,O1[0],O1[1],np.int64(nt),np.int32(256)))
  V["idb"](IG,(512,),(p0,b_z,O2[0],O2[1],np.int64(nt),np.int32(256),dc,u))
  if not (bool((O1==O2).all()) and bool((a_z==b_z).all())): errs.append("dense_bwd+inner fused")
  cp.cuda.runtime.deviceSynchronize()
  if errs: raise RuntimeError("LA_V256 SELFTEST FAILED: "+", ".join(errs))
  print(f"  [la256] LA_V256={','.join(sorted(o))}: kernels compiled, self-test identical to the old kernels"
        f"{' (xb fixed: words 0,1 vs the moved-word reference)' if 'xfix' in o else ''} in {time.time()-t0:.2f}s",flush=True)
  del p0,p1,p2,pn,xb,mats,dc,u,a_z,b_z
  return V
 except Exception as _e:
  print(f"  WARNING: LA_V256 unavailable ({type(_e).__name__}: {str(_e)[:300]}) -- old 256-bit kernels",flush=True)
  if strict: sys.exit(5)
  return None

# LA_V3 (2026-10-21; DEFAULT "all" since s32: full TD-130 LA VERIFY True + characters deps, 3.8 ms/it less; LA_V3=none =
# the LA_V256 path): ks256v3.cu kernels for the non-SpMV part of the 256-bit loop (see that file's
# header for the measured causes). Comma list:  upd = upd256_v3 (+ updF256_v3 when useF): conflict-free u64-plane nibble
# tables;  inner = inner256_v3 (inner256_v3db when LA_V256 db is live): Four-Russians inner products (warp bit-transposes + conflict-free tables);  all = both.
# Bit-identical to the LA_V256 / ks256.cu kernels (self-tested on random data at start-up, falls back to them with a
# WARNING, LA_V256_STRICT=1: exit 5). Not combinable with LA_V256 xfix (v3 upd is XF=0 only: upd stays on the v2 kernel).
# host = la256host.c la_mul256f/la_tr256f/la_fns256f for the per-iteration host algebra (the GPU idles meanwhile).
# LA_V3_IG / LA_V3_UG / LA_V3_FG: blocks per SM for inner (3) / upd (1) / updF (3).
def _v3_setup(old,V2,n_sm,ND):
 o=set(x.strip() for x in os.environ.get("LA_V3","all").split(",") if x.strip() and x.strip()!="none")
 if "all" in o: o|={"upd","inner","host"}; o.discard("all")
 bad=o-{"upd","inner","host"}
 if bad: print(f"  FATAL: LA_V3 has unknown option(s) {sorted(bad)} (upd, inner, host, all)",flush=True); sys.exit(3)
 if not o: return None
 if "upd" in o and V2 is not None and "xfix" in V2["opts"]:
  print("  WARNING: LA_V3 upd ignored with LA_V256 xfix (v3 update is XF=0 only)",flush=True); o.discard("upd")
 if not o: return None
 strict=os.environ.get("LA_V256_STRICT","0")=="1"
 try:
  src=open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"ks256v3.cu")).read()
  K={nm:cp.RawKernel(src,nm,options=OPT) for nm in ("upd256_v3","updF256_v3","inner256_v3","inner256_v3db")}
  for k in K.values(): k.compile()
  K["upd256_v3"].max_dynamic_shared_size_bytes=81920
  bpsm=lambda e,d: max(1,int(os.environ.get(e,str(d))))
  V=dict(opts=o,upd=K["upd256_v3"],updF=K["updF256_v3"],inner=K["inner256_v3"],idb=K["inner256_v3db"],
         UG=(n_sm*bpsm("LA_V3_UG",1),),FG=(n_sm*bpsm("LA_V3_FG",3),),IG=(n_sm*bpsm("LA_V3_IG",3),),ND=int(ND))
  # ---- self-test against the kernels the loop would otherwise run (random data, odd n) ----
  t0=time.time(); g=np.random.default_rng(20261021); nt=200003; W=4
  R=lambda *sh: cp.asarray(g.integers(0,1<<64,sh,dtype=np.uint64))
  p0,p1,p2,pn,xb=(R(nt,W) for _ in range(5)); mats=R(4,256,W)
  msk=cp.asarray(np.array([g.integers(0,1<<64,dtype=np.uint64),~np.uint64(0),g.integers(0,1<<64,dtype=np.uint64),np.uint64(2**63-1)],np.uint64))
  IG=(max(1,min(2048,(nt+255)//256)),); UGo=(max(1,min(8192,(nt+255)//256)),)
  errs=[]
  if "upd" in o:
   for useF in (0,1):
    a_pn,a_xb,b_pn,b_xb=pn.copy(),xb.copy(),pn.copy(),xb.copy()
    if V2 is not None and "upd" in V2["opts"]:
     V2["upd"](V2["UG"],(V2["UT"],),(p0,p1,a_pn,a_xb,mats,msk,np.int64(nt)),shared_mem=98304)
     if useF: V2["updF"](V2["UG"],(V2["UT"],),(p2,a_pn,mats,np.int64(nt)),shared_mem=32768)
    else: old["upd"](UGo,(256,),(p0,p1,p2,a_pn,a_xb,mats,msk,np.int32(useF),np.int64(nt)))
    V["upd"](V["UG"],(512,),(p0,p1,b_pn,b_xb,mats,msk,np.int64(nt)),shared_mem=81920)
    if useF: V["updF"](V["FG"],(512,),(p2,b_pn,mats,np.int64(nt)),shared_mem=32768)
    if not bool((a_pn==b_pn).all()): errs.append(f"upd pn useF={useF}")
    if not bool((a_xb==b_xb).all()): errs.append(f"upd xb useF={useF}")
  if "inner" in o:
   O1=cp.zeros((2,256,W),cp.uint64); O2=cp.zeros((2,256,W),cp.uint64)
   old["inner"](IG,(512,),(p0,pn,O1[0],O1[1],np.int64(nt),np.int32(256)))
   V["inner"](V["IG"],(256,),(p0,pn,O2[0],O2[1],np.int64(nt)))
   if not bool((O1==O2).all()): errs.append("inner")
   for nd in (64,128):
    dc=cp.asarray(g.integers(0,1<<64,(nt,2),dtype=np.uint64)&g.integers(0,1<<64,(nt,2),dtype=np.uint64))
    if nd==64: dc[:,1]=0
    dc[::7]=0; u=R(nt,W); a_z,b_z=pn.copy(),pn.copy()
    O1=cp.zeros((2,256,W),cp.uint64); O2=cp.zeros((2,256,W),cp.uint64)
    old["dbwd"](G(nt),(256,),(dc,u,a_z,np.int64(nt))); old["inner"](IG,(512,),(p0,a_z,O1[0],O1[1],np.int64(nt),np.int32(256)))
    V["idb"](V["IG"],(256,),(p0,b_z,O2[0],O2[1],np.int64(nt),dc,u,np.int32(nd)))
    if not (bool((O1==O2).all()) and bool((a_z==b_z).all())): errs.append(f"dense_bwd+inner fused nd={nd}")
  if "host" in o:
   mulf,trf,fnsf=_hostf_bind(); V["host"]=(mulf,trf,fnsf)
   gh=np.random.default_rng(5)
   for k in range(40):
    a=gh.integers(0,1<<64,(256,4),dtype=np.uint64); b=gh.integers(0,1<<64,(256,4),dtype=np.uint64)
    if not (np.array_equal(mulf(a,b),mul256(a,b)) and np.array_equal(trf(a),tr256(a))): errs.append("host mul/tr"); break
    X=gh.integers(0,2,(256,int(gh.integers(120,257)))).astype(np.float32)          # symmetric, rank-deficient t
    tt=_packV(((X@X.T).astype(np.int64)&1).astype(np.uint8),256); ld=int(gh.integers(0,257)); ls=[int(v) for v in gh.permutation(256)[:ld]]
    d1,s1,w1=fns256_rows(tt,ls,ld); d2,s2,w2,m2=fnsf(tt,np.array(ls+[0]*(256-ld),np.int32),ld)
    mm=0
    for i in s1: mm|=1<<i
    if not (d1==d2 and list(s1)==[int(v) for v in s2] and (d1==0 or (np.array_equal(w1,w2) and np.array_equal(m2,mask_arrV(mm,4))))):
     errs.append("host fns"); break
  cp.cuda.runtime.deviceSynchronize()
  if errs: raise RuntimeError("LA_V3 SELFTEST FAILED: "+", ".join(errs))
  print(f"  [la256] LA_V3={','.join(sorted(o))}: kernels compiled, self-test identical to the "
        f"{'LA_V256' if V2 is not None else 'ks256.cu'} kernels in {time.time()-t0:.2f}s (grids upd {V['UG'][0]} updF {V['FG'][0]} inner {V['IG'][0]})",flush=True)
  return V
 except Exception as _e:
  print(f"  WARNING: LA_V3 unavailable ({type(_e).__name__}: {str(_e)[:300]}) -- LA_V256/old 256-bit kernels",flush=True)
  if strict: sys.exit(5)
  return None

# LA_GRAPH (2026-10-23; default 0 = the loop below unchanged): no host round trip per iteration. ks256g.cu's la_alg256 does
# the LA_V3 'host' algebra (find_nonsingular_sub + the D/E/F/D2 products, same pivot order) in ONE GPU block, writing the
# update matrices straight into dmats/dmask and a control record (halt status, useF); updF256_v3g reads useF on the device.
# An iteration is then a fixed launch sequence:  SpMV B, SpMV B^T, memset O, inner, la_alg256, upd256_v3, updF256_v3g.
#   1 = iterations 4.. are replayed as CUDA graphs (one per p0/p1/p2/pn rotation phase, captured once at iteration 4)
#   2 = the same device algebra launched eagerly (no graph; diagnostic)
# The host reads the 32-byte control record LA_GRAPH_LAG (1..3, default 3) iterations late, so the GPU queue never
# drains; a halt (converged / fns failed / columns) makes every later la_alg256 write the IDENTITY update, and with lag
# <= 3 the halted iteration's p0 is not yet overwritten (the 3rd later iteration's SpMV would write it), so the combine
# step sees exactly the state the host loop breaks with. Requires LA_V3 upd. Bit-identical to LA_GRAPH=0 (start-up
# self-test of la_alg256 against la256host's fnsf/mulf/trf chain and of updF256_v3g; LA_HASH on the box).
GS_SIZE=13320; GS_M1=13312
def _graph_mode():
 v=os.environ.get("LA_GRAPH","0").strip() or "0"
 if v not in ("0","1","2"): print(f"  FATAL: LA_GRAPH={v} (must be 0, 1 or 2)",flush=True); sys.exit(3)
 return int(v)
def _alg_init(S,ctl):
 S[:]=0; S[GS_M1:GS_M1+4]=cp.asarray(np.full(4,~np.uint64(0),np.uint64))
 h=np.zeros(272,np.int32); h[1]=256; h[16:272]=np.arange(256,dtype=np.int32); ctl.set(h)
def _alg_ref_init():
 Z=lambda: np.zeros((256,4),np.uint64)
 return dict(wi1=Z(),wi2=Z(),v0=Z(),v1=Z(),v2=Z(),va1=Z(),va21=Z(),s1=np.arange(256,dtype=np.int32),dim1=256,
             M1=np.full(4,~np.uint64(0),np.uint64))
def _alg_ref_step(Rs,A0,A2,v0):
 """the LA_V3 host branch of gpu_block_lanczos256 on a state dict -> (status, (D,E,F,D2,M0,useF,dim0) or None)"""
 ALLW=np.full(4,~np.uint64(0),np.uint64)
 if _LAH256 is not None:
  mul,tr,fns=_hostf_bind()
 else:
  mul,tr=mul256,tr256
  def fns(t,ls,ld):
   d,s0,w=fns256_rows(t,[int(v) for v in ls[:ld]],ld); m=0
   for i in s0: m|=1<<i
   return d,np.array(s0,np.int32),w,mask_arrV(m,4)
 if not A0.any(): return 1,None
 ls=np.zeros(256,np.int32); ls[:Rs["dim1"]]=Rs["s1"][:Rs["dim1"]]
 dim0,s0,wi,M0=fns(A0,ls,Rs["dim1"])
 if dim0==0: return 2,None
 if not bool(((M0|Rs["M1"])==ALLW).all()): return 3,None
 if v0 is not None: Rs["v0"]=v0
 d=(A2&M0)^A0; d=mul(wi,d)^BM256
 nx=mul(tr(d),Rs["v0"])
 e=mul(Rs["wi1"],A0)&M0
 nx=nx^mul(tr(e),Rs["v1"])
 useF=not bool((Rs["M1"]==ALLW).all())
 if useF:
  f=mul(Rs["va1"],Rs["wi1"])^BM256; f=mul(Rs["wi2"],f)
  f2=((Rs["va21"]&Rs["M1"])^Rs["va1"])&M0; f=mul(f,f2); nx=nx^mul(tr(f),Rs["v2"])
 else: f=np.zeros((256,4),np.uint64)
 d2=mul(wi,Rs["v0"])
 Rs.update(wi2=Rs["wi1"],wi1=wi,v2=Rs["v1"],v1=Rs["v0"],v0=nx,va1=A0.copy(),va21=A2.copy(),s1=np.array(s0,np.int32),dim1=int(dim0),M1=M0.copy())
 return 0,(d,e,f,d2,M0,useF,int(dim0))
def _graph_setup(V3):
 """-> dict(alg, updF, mode) or None. Compiles ks256v3.cu + ks256g.cu, self-tests la_alg256 / updF256_v3g."""
 mode=_graph_mode()
 if not mode: return None
 strict=os.environ.get("LA_V256_STRICT","0")=="1"
 try:
  if not V3 or "upd" not in V3.get("opts",set()): raise RuntimeError("needs LA_V3 upd (upd256_v3)")
  here=os.path.dirname(os.path.realpath(__file__))
  src=open(os.path.join(here,"ks256v3.cu")).read()+"\n"+open(os.path.join(here,"ks256g.cu")).read()
  K={nm:cp.RawKernel(src,nm,options=OPT) for nm in ("la_alg256","updF256_v3g")}
  for k in K.values(): k.compile()
  t0=time.time(); g=np.random.default_rng(20261023); errs=[]
  # ---- updF256_v3g == updF256_v3 when useF, untouched pn when not ----
  nt=200003; R_=lambda *sh: cp.asarray(g.integers(0,1<<64,sh,dtype=np.uint64))
  p2,pn=R_(nt,4),R_(nt,4); mats=R_(4,256,4); c1=cp.zeros(272,cp.int32)
  a_pn,b_pn=pn.copy(),pn.copy()
  V3["updF"](V3["FG"],(512,),(p2,a_pn,mats,np.int64(nt)),shared_mem=32768)
  c1[2]=1; K["updF256_v3g"](V3["FG"],(512,),(p2,b_pn,mats,np.int64(nt),c1),shared_mem=32768)
  if not bool((a_pn==b_pn).all()): errs.append("updF256_v3g useF=1")
  b_pn=pn.copy(); c1[2]=0; K["updF256_v3g"](V3["FG"],(512,),(p2,b_pn,mats,np.int64(nt),c1),shared_mem=32768)
  if not bool((pn==b_pn).all()): errs.append("updF256_v3g useF=0")
  del p2,pn,a_pn,b_pn
  # ---- la_alg256 vs the host chain: random symmetric inner products, chained state, every branch compared ----
  S=cp.zeros(GS_SIZE,cp.uint64); ctl=cp.zeros(272,cp.int32); O=cp.zeros((2,256,4),cp.uint64); O2=cp.zeros((2,256,4),cp.uint64)
  dm=cp.zeros((4,256,4),cp.uint64); dk=cp.zeros(4,cp.uint64)
  def sym(lo,hi):
   X=g.integers(0,2,(256,int(g.integers(lo,hi)))).astype(np.float32)
   return _packV(((X@X.T).astype(np.int64)&1).astype(np.uint8),256)
  st=dict(steps=0,useF=0,halts={1:0,2:0,3:0},dims=set()); Rs=None; k=0
  NSTEP=int(os.environ.get("LA_GRAPH_SELFTEST_STEPS","48"))
  for step in range(NSTEP+1):
   if Rs is None: Rs=_alg_ref_init(); _alg_init(S,ctl); k=0
   k+=1
   last=step==NSTEP
   A0=np.zeros((256,4),np.uint64) if last else sym(240,330) if g.random()<0.85 else sym(150,256)
   A2=sym(200,330); v0=g.integers(0,1<<64,(256,4),dtype=np.uint64) if k<4 else None
   O.set(np.stack([A0,A2])); O2[0].set(v0 if v0 is not None else np.zeros((256,4),np.uint64))
   dm[:]=cp.asarray(g.integers(0,1<<64,(4,256,4),dtype=np.uint64))
   K["la_alg256"]((1,),(256,),(O,O2,np.int32(1 if v0 is not None else 0),S,ctl,dm,dk))
   rs,out=_alg_ref_step(Rs,A0,A2,v0)
   ch=ctl.get(); hm=dm.get(); hk=dk.get(); st["steps"]+=1
   if int(ch[0])!=rs: errs.append(f"step {step}: status {int(ch[0])} != host {rs}"); break
   if rs:
    st["halts"][rs]+=1
    if hm.any() or not bool((hk==~np.uint64(0)).all()) or int(ch[2])!=0: errs.append(f"step {step}: halt {rs} without the identity update"); break
    if int(ch[6])!=k: errs.append(f"step {step}: halt iteration {int(ch[6])} != {k}"); break
    # sticky: one more launch must leave the halt and write the identity update again
    dm[:]=1; K["la_alg256"]((1,),(256,),(O,O2,np.int32(0),S,ctl,dm,dk)); ch2=ctl.get()
    if int(ch2[0])!=rs or bool(dm.get().any()) or int(ch2[6])!=k: errs.append(f"step {step}: halt not sticky"); break
    Rs=None; continue
   D,E,F,D2,M0,useF,dim0=out; st["useF"]+=useF; st["dims"].add(dim0)
   for nm_,a_,b_ in (("D",hm[0],D),("E",hm[1],E),("F",hm[2],F),("D2",hm[3],D2),("mask",hk,M0)):
    if not np.array_equal(a_,b_): errs.append(f"step {step}: {nm_} differs"); break
   if int(ch[2])!=int(useF) or int(ch[1])!=dim0 or int(ch[4])!=k: errs.append(f"step {step}: ctl useF/dim/it {list(ch[1:5])} != {int(useF)}/{dim0}/{k}")
   if not np.array_equal(ch[16:16+dim0],Rs["s1"][:dim0]): errs.append(f"step {step}: chosen columns differ")
   if errs: break
  if not errs and (st["useF"]==0 or st["halts"][1]==0): errs.append(f"self-test coverage: useF {st['useF']} converged {st['halts'][1]}")
  cp.cuda.runtime.deviceSynchronize()
  if errs: raise RuntimeError("LA_GRAPH SELFTEST FAILED: "+"; ".join(errs[:4]))
  print(f"  [la256] LA_GRAPH={mode}: la_alg256 + updF256_v3g compiled, self-test identical to the host algebra in "
        f"{time.time()-t0:.2f}s ({st['steps']} launches, useF {st['useF']}, halts {st['halts']}, dims {min(st['dims'])}..{max(st['dims'])})",flush=True)
  return dict(alg=K["la_alg256"],updF=K["updF256_v3g"],mode=mode)
 except Exception as _e:
  print(f"  WARNING: LA_GRAPH unavailable ({type(_e).__name__}: {str(_e)[:300]}) -- per-iteration host algebra",flush=True)
  if strict: sys.exit(5)
  return None

def _popc64():
 return cp.ReductionKernel('uint64 x','int64 y','__popcll(x)','a + b','y = a','0','popc64_sum')
def _defer_selftest(Bwork,Btwork,Bfull,d_Ri,d_Ci,d_dcol,K,n,nnz,checks=2):
 """LA_DEFER_STRICT=1: the deferred operators against an independent reference on `checks` random 256-bit vectors:
 Bwork rows [0,K) zero; Bfull == Bwork on rows >= K; every word of Bfull == coo_xor64 over the ORIGINAL sparse COO +
 the 64-bit dense_fwd over D_saved (= the VERIFY path); every word of Btwork == coo_xor64 of the transposed COO (no
 dense rows). All n rows compared; any difference exits 5."""
 t0=time.time(); bad=[]
 for s_ in range(max(1,checks)):
  rs=cp.random.RandomState(20261024+s_)
  def rnd():
   r=rs.random_sample((2,n,4)).view(cp.uint64); return r[0]^(r[1]<<cp.uint64(12))
  x=rnd(); y=cp.full((n,4),~np.uint64(0),cp.uint64); yf=cp.full((n,4),np.uint64(0x5555555555555555),cp.uint64)
  Bwork(x,y); Bfull(x,yf)
  if bool(y[:K].any()): bad.append(f"seed {s_}: Bwork rows [0,{K}) not zero")
  if not bool((y[K:]==yf[K:]).all()): bad.append(f"seed {s_}: Bfull != Bwork on rows >= {K}")
  ref=cp.zeros(n,cp.uint64)
  for w in range(4):
   xw=cp.ascontiguousarray(x[:,w]); ref[:]=0
   coo64((4096,),(256,),(d_Ri,d_Ci,xw,ref,np.int64(nnz))); dfwd(G(n),(256,),(d_dcol,xw,ref,n,K))
   if not bool((ref==yf[:,w]).all()): bad.append(f"seed {s_}: Bfull word {w} != COO + dense_fwd ({int(cp.count_nonzero(ref!=yf[:,w]))} rows)")
  u=rnd(); z=cp.full((n,4),~np.uint64(0),cp.uint64); Btwork(u,z)
  for w in range(4):
   uw=cp.ascontiguousarray(u[:,w]); ref[:]=0
   coo64((4096,),(256,),(d_Ci,d_Ri,uw,ref,np.int64(nnz)))
   if not bool((ref==z[:,w]).all()): bad.append(f"seed {s_}: Btwork word {w} != transposed COO ({int(cp.count_nonzero(ref!=z[:,w]))} rows)")
  del x,y,yf,u,z,ref
 cp.get_default_memory_pool().free_all_blocks()
 if bad:
  print("  [la256] FATAL: LA_DEFER_STRICT self-test FAILED: "+"; ".join(bad[:6]),flush=True); sys.exit(5)
 print(f"  [la256] LA_DEFER_STRICT self-test OK: Bwork/Btwork/Bfull vs independent COO + dense_fwd, {max(1,checks)} random "
       f"256-bit vectors, all {n:,} rows ({time.time()-t0:.1f}s)",flush=True)

def run256(R,C,dcol,ND,n,defer=0):
 """Build the SpMV (seg width 256 or column-split), run 256-bit block Lanczos, verify. -> (x (n,) uint64, nd, ok, spare)
 defer=K (LA_DEFER_ROWS): the ND=K dense rows are left out of the recurrence and enforced only in the final combine."""
 global coo64,dfwd
 DEF=int(defer)
 if DEF and (ND!=DEF or _graph_mode()!=0 or LA_SPMV!="seg"):
  print(f"  FATAL: LA_DEFER_ROWS={DEF} needs ND == {DEF} (got {ND}), LA_GRAPH=0 and LA_SPMV=seg (got {LA_SPMV})",flush=True); sys.exit(3)
 spmv256=cp.RawKernel(KS256,'spmv256',options=OPT); spheavy256=cp.RawKernel(KS256,'spmv_heavy256',options=OPT)
 inner256=cp.RawKernel(KS256,'inner256',options=OPT); dfwd256=cp.RawKernel(KS256,'dense_fwd256',options=OPT)
 dbwd256=cp.RawKernel(KS256,'dense_bwd256',options=OPT); upd256=cp.RawKernel(KS256,'upd256',options=OPT)
 coo64=cp.RawKernel(KS128,'coo_xor64',options=OPT)
 for k in (spmv256,spheavy256,inner256,dfwd256,dbwd256,upd256): k.compile()
 t=time.time()
 Ri=R.astype(np.int32,copy=False); Ci=C.astype(np.int32,copy=False)
 d_Ri=cp.asarray(Ri); d_Ci=cp.asarray(Ci); del Ri,Ci
 nnz=int(d_Ri.size)
 SEG=None
 if LA_SPMV=="seg":
  try:
   SEG=_load_seg().SegPlan(d_Ri,d_Ci,n,ND,width=256)
   print(f"  seg SpMV plan built in {time.time()-t:.0f}s: LA_BLOCK=256, nnz {nnz}",flush=True)
  except Exception as _e:
   if DEF:   # the deferred operator is defined on the seg plan only (its zero list makes rows [0,K) explicit zeros)
    print(f"  FATAL: seg SpMV plan failed ({type(_e).__name__}: {str(_e)[:400]}) and LA_DEFER_ROWS={DEF}",flush=True); sys.exit(4)
   # LA_SEG_STRICT=1 (timing runs): never fall back silently -- the column-split SpMV is ~2x slower per iteration
   # (TD 170: 96 ms/it) and dies on boolean-mask indexing above 2^31 nonzeros (TD 220).
   if os.environ.get("LA_SEG_STRICT","0")=="1":
    print(f"  FATAL: seg SpMV plan failed ({type(_e).__name__}: {str(_e)[:400]}) and LA_SEG_STRICT=1",flush=True); sys.exit(4)
   print(f"  WARNING: seg SpMV plan failed ({type(_e).__name__}: {str(_e)[:200]}) -- using the column-split SpMV",flush=True)
   SEG=None; cp.get_default_memory_pool().free_all_blocks()
 try:
  _fr,_tot=cp.cuda.runtime.memGetInfo()
  print(f"  [la] GPU memory after the SpMV plan: pool {cp.get_default_memory_pool().used_bytes()/2**30:.1f} GiB used, "
        f"device {_fr/2**30:.1f} of {_tot/2**30:.1f} GiB free",flush=True)
 except Exception: pass
 if SEG is None:
  SPLIT_MB=float(os.environ.get("LA_SPLIT_MB","96"))
  P=int(os.environ.get("LA_SPLIT","0") or 0)
  if P<=0: P=max(1,-(-(n*32)//int(SPLIT_MB*2**20)))
  P=max(1,min(P,max(1,n)))
  partsB,partsBt=build_parts128(d_Ri,d_Ci,n,P)
  print(f"  sparse CSR built in {time.time()-t:.0f}s: LA_BLOCK=256, {P} column part(s) of <= "
        f"{-(-n//P)*32/2**20:.1f} MB of vector (LA_SPLIT_MB={SPLIT_MB:g}), nnz {nnz}",flush=True)
 d_dcol=cp.asarray(dcol)
 VSH=max(0,min(5,int(os.environ.get("LA_VW","32")).bit_length()-1)); VW=1<<VSH
 tpb=256; WBW=((n*VW+tpb-1)//tpb,); IGR=(max(1,min(2048,(n+255)//256)),); UG=(max(1,min(8192,(n+255)//256)),)
 DG=G(n); INTMAX=np.int32(2**31-1)
 heavy_grid=lambda H: G(H[3]*32)
 def spmv_parts(parts,src,dst):
  for p,(c0,c1,col,rp,H) in enumerate(parts):
   sv=src[c0:c1]
   spmv256(WBW,(tpb,),(sv,col,rp,dst,np.int32(n),np.int32(HEAVY_T) if H else INTMAX,np.int32(1 if p else 0),np.int32(VSH)))
   if H: spheavy256(heavy_grid(H),(tpb,),(sv,col,H[0],H[1],H[2],dst,np.int32(H[3])))
 DTB=64 if ND<=64 else 128
 def Bmul(src,dst):
  if SEG is not None: SEG.B(src,dst)
  else: spmv_parts(partsB,src,dst)
  if ND: dfwd256(IGR,(DTB,),(d_dcol,src,dst,np.int64(n),np.int32(ND)))
 def Btmul(src,dst):
  if SEG is not None: SEG.Bt(src,dst)
  else: spmv_parts(partsBt,src,dst)
  if ND: dbwd256(DG,(tpb,),(d_dcol,src,dst,np.int64(n)))
 Bfull=None
 if DEF:
  # sparse entries must all lie in rows >= K (cado_mat put rows < K into dcol); rows [0,K) of a SEG.B image are zeroed
  # by the plan's zero list, so SEG.B alone IS Bwork = [0; S] and SEG.Bt alone IS Btwork (no D contribution)
  _rmin=int(d_Ri.min()) if nnz else n
  if _rmin<DEF or SEG.hlo!=DEF:
   print(f"  FATAL: LA_DEFER_ROWS={DEF}: sparse row min {_rmin}, seg hot window start {SEG.hlo} (must be >= {DEF} / == {DEF})",flush=True); sys.exit(3)
  Bfull=Bmul                                  # SEG.B + dense_fwd256(D_saved): the ORIGINAL B, final images only
  def Bmul(src,dst): SEG.B(src,dst)           # Bwork: rows [0,K) explicitly zero
  def Btmul(src,dst): SEG.Bt(src,dst)         # Btwork: no dense-row contribution
  _dbits=int(_popc64()(d_dcol))
  if os.environ.get("LA_DEFER_STRICT","0")=="1":
   _defer_selftest(Bmul,Btmul,Bfull,d_Ri,d_Ci,d_dcol,DEF,n,nnz,int(os.environ.get("LA_DEFER_CHECKS","2")))
  print(f"  [la256] LA_DEFER_ROWS={DEF} active: recurrence on Bwork=[0;S] (working ND 0, no dense fwd / fused dense bwd), "
        f"dense rows [0,{DEF}) = {_dbits} entries enforced in the final combine (ORIGINAL B images)",flush=True)
 V2=_v256_setup(dict(upd=upd256,inner=inner256,dbwd=dbwd256),int(cp.cuda.Device().attributes['MultiProcessorCount']))
 if V2 is not None and "db" in V2["opts"] and not DEF:
  # the loop's B^T product WITHOUT the dense rows: inner256_db adds them while it reads pn for the inner products
  def Btsp(src,dst):
   if SEG is not None: SEG.Bt(src,dst)
   else: spmv_parts(partsBt,src,dst)
  V2["Btsp"]=Btsp; V2["dcol"]=d_dcol
 V3=_v3_setup(dict(upd=upd256,inner=inner256,dbwd=dbwd256),V2,int(cp.cuda.Device().attributes['MultiProcessorCount']),0 if DEF else ND)
 GM=_graph_setup(V3)
 if GM is not None: V3["graph"]=GM
 print("running GPU block-Lanczos (LA_BLOCK=256, %s SpMV) ... [loop starts at +%.1fs]"%("segmented" if SEG is not None else "column-split",time.time()-_T_START),flush=True); t=time.time()
 x,nd,it,info,extra=gpu_block_lanczos256(Bmul,Btmul,n,IGR,UG,inner256,upd256,V2,V3,Bcheck=Bfull)
 print(f"  LA done: {it} iters, {nd} deps, {time.time()-t:.0f}s  (N/iters = {n/max(it,1):.2f}; combine: rank {info.get('rank')}, "
       f"{info.get('nonzero')} non-zero / {info.get('indep')} independent candidates, {info.get('t',0):.1f}s)",flush=True)
 if x is None: return None,0,False,None
 xg=cp.asarray(x); bx=cp.zeros(n,cp.uint64)
 coo64((4096,),(256,),(d_Ri,d_Ci,xg,bx,np.int64(nnz))); dfwd(G(n),(256,),(d_dcol,xg,bx,n,ND))
 ok=bool((bx.get()==0).all()) and bool((x!=0).any())
 print(f"  VERIFY B*x==0 & x!=0: {ok}  (COO path, {nnz} nonzeros + {ND} dense rows)",flush=True)
 if DEF:
  # the ORIGINAL matrix (COO + all DEF dense rows) was just verified; also demand 64 independent output columns
  r64=_rank64(x)
  print(f"  [la256] LA_DEFER_ROWS={DEF}: {nd} kernel vectors, output column rank {r64}/64, combine rank {info.get('rank')}, "
        f"{info.get('nonzero')} non-zero candidates, {info.get('indep_all')} independent UNCAPPED (head), {info.get('unsure')} unsure, "
        f"VERIFY against the original B incl. all {DEF} deferred rows: {ok}",flush=True)
  if nd<64 or r64<64:
   print(f"  [la256] FATAL: LA_DEFER_ROWS={DEF} left only {nd} vectors / rank {r64} (< 64) -- refusing to write a short kernel",flush=True)
   ok=False
 xe=None
 if extra is not None and os.environ.get("LA_KER2","0")=="1":
  xe=rows_to_words64(extra,n); xg=cp.asarray(xe); bx[:]=0
  coo64((4096,),(256,),(d_Ri,d_Ci,xg,bx,np.int64(nnz))); dfwd(G(n),(256,),(d_dcol,xg,bx,n,ND))
  if not bool((bx.get()==0).all()):
   print("  WARNING: LA_KER2 spare vectors FAIL B*x==0 -- not written",flush=True); xe=None
 return x,nd,ok,xe

def gpu_block_lanczos256(Bmul,Btmul,n,IGR,UG,inner_k,upd_k,V2=None,V3=None,Bcheck=None):
 # Bcheck (LA_DEFER_ROWS): the ORIGINAL B for the two final images only; every iterative / initial product uses Bmul
 VB=256; W=4; tpb=256
 V2=V2 or {}; vo=V2.get("opts",set())
 upd2=V2.get("upd") if "upd" in vo else None
 V3=V3 or {}; v3o=V3.get("opts",set())
 upd3=V3.get("upd") if "upd" in v3o else None
 inner3=V3.get("inner") if "inner" in v3o else None     # the it<4 vsav products keep inner_k (256 threads, o1 only)
 inner_k=V2["inner"] if "inner" in vo else inner_k          # also for the it<4 vsav products (256 threads, o1 only)
 idb=V2.get("idb") if ("db" in vo and V2.get("Btsp") is not None) else None
 if idb is not None:
  Btsp=V2["Btsp"]; dcol_=V2["dcol"]
 sc=cp.empty((n,W),cp.uint64)
 def AvP(src,dst): Bmul(src,sc); Btmul(sc,dst)
 p0=cp.empty((n,W),cp.uint64); p1=cp.zeros((n,W),cp.uint64); p2=cp.zeros((n,W),cp.uint64); pn=cp.empty((n,W),cp.uint64)
 xb=cp.asarray(np.random.default_rng(int(os.environ.get("LA_SEED","5"))).integers(0,1<<64,(n,W),dtype=np.uint64))
 AvP(xb,p0); vsav=p0.copy()
 O=cp.zeros((2,VB,W),cp.uint64); O2=cp.zeros((2,VB,W),cp.uint64)
 dmats=cp.zeros((4,VB,W),cp.uint64); hmats=np.zeros((4,VB,W),np.uint64); dmask=cp.zeros(W,cp.uint64)
 hostf=V3.get("host") if "host" in v3o else None
 _mul,_tr=(mul256,tr256) if hostf is None else hostf[:2]
 if hostf is not None:
  fnsf=hostf[2]; ALLW=np.full(W,~np.uint64(0),np.uint64); M1h=ALLW.copy(); s1h=np.arange(VB,dtype=np.int32)
  dbuf=cp.zeros(4*VB*W+W,cp.uint64); dmats=dbuf[:4*VB*W].reshape(4,VB,W); dmask=dbuf[4*VB*W:]
  hbuf=cp.cuda.alloc_pinned_memory(dbuf.nbytes); hbuf=np.frombuffer(hbuf,np.uint64,dbuf.size)
 Z=lambda: np.zeros((VB,W),dtype=np.uint64)
 vt_a_v=[Z(),Z()]; vt_a2_v=[Z(),Z()]; vt_v0=[Z(),Z(),Z()]; winv=[Z(),Z(),Z()]
 s=[[0]*VB,list(range(VB))]; dim1=VB; mask1=ALL256; it=0; dim0=0; t0=time.time()
 PROF=int(os.environ.get("LA_PROF","0")); tGPU=0.0; tCPU=0.0; _t1=time.time()
 _pe=[cp.cuda.Event() for _ in range(6)] if PROF>=2 else None; tBm=0.0; tBtm=0.0; tIn=0.0; tUp=0.0; tUk=0.0; nF=0   # LA_PROF=3: + inner/D2H, host+H2D+update, update kernel
 LOGI=int(os.environ.get("LA_LOG_EVERY","1000"))
 ITMAX=n//240+64     # ~n/254 expected at 256 bits
 GM=V3.get("graph") if upd3 is not None else None
 if GM is not None:   # ---- LA_GRAPH: device algebra, iterations 4.. replayed as CUDA graphs (see _graph_setup) ----
  alg=GM["alg"]; updFg=GM["updF"]; LAG=max(1,min(3,int(os.environ.get("LA_GRAPH_LAG","3"))))
  cp.cuda.Device().synchronize()            # everything above ran on the legacy stream
  gst=cp.cuda.Stream(non_blocking=True)
  S_=cp.zeros(GS_SIZE,cp.uint64); ctl=cp.zeros(272,cp.int32); _alg_init(S_,ctl)
  bufs=[(p0,p1,p2,pn)]
  for _k in range(3): _b=bufs[-1]; bufs.append((_b[3],_b[0],_b[1],_b[2]))   # p0,p1,p2,pn = pn,p0,p1,p2
  NRG=8; _hp=cp.cuda.alloc_pinned_memory(NRG*8*4); hctl=np.frombuffer(_hp,np.int32,NRG*8).reshape(NRG,8); hctl[:]=0
  evs=[cp.cuda.Event(block=False,disable_timing=True) for _ in range(NRG)]
  def _zero(a): cp.cuda.runtime.memsetAsync(a.data.ptr,0,a.nbytes,gst.ptr)
  def one_it(ph,usev0):     # one iteration on the current stream: exactly the LA_V3 loop's kernels + la_alg256
   q0,q1,q2,qn=bufs[ph]
   if idb is None:
    AvP(q0,qn); _zero(O)
    if inner3 is not None: inner3(V3["IG"],(256,),(q0,qn,O[0],O[1],np.int64(n)))
    else: inner_k(IGR,(512,),(q0,qn,O[0],O[1],np.int64(n),np.int32(VB)))
   else:
    Bmul(q0,sc); Btsp(sc,qn); _zero(O)
    if inner3 is not None: V3["idb"](V3["IG"],(256,),(q0,qn,O[0],O[1],np.int64(n),dcol_,sc,np.int32(min(128,V3["ND"]))))
    else: idb(IGR,(512,),(q0,qn,O[0],O[1],np.int64(n),np.int32(VB),dcol_,sc))
   if usev0:
    _zero(O2); inner_k(IGR,(256,),(q0,vsav,O2[0],O2[1],np.int64(n),np.int32(VB)))
   alg((1,),(256,),(O,O2,np.int32(1 if usev0 else 0),S_,ctl,dmats,dmask))
   upd3(V3["UG"],(512,),(q0,q1,qn,xb,dmats,dmask,np.int64(n)),shared_mem=81920)
   updFg(V3["FG"],(512,),(q2,qn,dmats,np.int64(n),ctl),shared_mem=32768)
  graphs=None; stop=None; tcap=0.0
  with gst:
   while True:
    it+=1
    if _LA_MAXIT and it>_LA_MAXIT: stop="maxit"; break
    if it>ITMAX: stop="itmax"; break
    if it>LAG:              # lagged halt check: the GPU still has LAG-1 iterations queued
     j=(it-LAG)%NRG; evs[j].synchronize()
     if hctl[j,0]: stop="halt"; break
    if it>=4 and graphs is None and GM["mode"]==1:
     _tc=time.time(); graphs=[]
     try:
      for ph in range(4):
       gst.begin_capture()
       try: one_it(ph,False)
       finally: graphs.append(gst.end_capture())
      for g_ in graphs:
       if hasattr(g_,"upload"): g_.upload(gst)
      print(f"  [la256] LA_GRAPH=1: 4 iteration graphs captured at it {it} in {time.time()-_tc:.2f}s (lag {LAG})",flush=True)
     except Exception as _e:
      print(f"  WARNING: LA_GRAPH capture failed ({type(_e).__name__}: {str(_e)[:300]}) -- eager launches",flush=True)
      if os.environ.get("LA_V256_STRICT","0")=="1": sys.exit(5)
      GM["mode"]=2; graphs=None
     tcap=time.time()-_tc
    if graphs is not None: graphs[(it-1)%4].launch(gst)
    else: one_it((it-1)%4,it<4)
    j=it%NRG
    cp.cuda.runtime.memcpyAsync(hctl[j].ctypes.data,ctl.data.ptr,32,cp.cuda.runtime.memcpyDeviceToHost,gst.ptr)
    evs[j].record(gst)
    if PROF and it%250==0:
     _el=time.time()-t0-tcap; print("   [LA_PROF] it=%d  %.2f ms/it | LA_GRAPH=%d (host enqueue+lag-%d wait)"%(it,_el*1000/it,GM["mode"],LAG),flush=True)
    if it%LOGI==0: print(f"   iter {it}  {time.time()-t0:.0f}s ({(time.time()-t0)*1000/it:.2f} ms/it)",flush=True)
   gst.synchronize()
  ch=ctl.get(); status=int(ch[0])
  if status:                # the host loop breaks at iteration ch[6] with that iteration's p0 (not yet overwritten)
   it=int(ch[6]); p0,p1,p2,pn=bufs[(it-1)%4]
   if status==1: dim0=int(ch[1]) if it>1 else 0
   else:
    if status==3: print(f"  [la256] ERROR: not all columns used at iteration {it} (dim0={int(ch[5])}, dim1={int(ch[1])})",flush=True)
    dim0=0
  elif stop=="maxit":
   p0,p1,p2,pn=bufs[(it-1)%4]
   print(f"  [la] LA_MAXIT={_LA_MAXIT} reached after {time.time()-t0:.1f}s = {(time.time()-t0)*1000/(it-1):.2f} ms/it -- timing run, stopping",flush=True)
   if os.environ.get("LA_HASH","0")=="1":
    import hashlib
    Oh=O.get()
    hh=[hashlib.blake2b(cp.asnumpy(v).tobytes(),digest_size=8).hexdigest() for v in (p0,p1,p2,xb)]
    print(f"  [la] LA_HASH it={_LA_MAXIT} p0 {hh[0]} p1 {hh[1]} p2 {hh[2]} xb {hh[3]} vt {hashlib.blake2b(Oh.tobytes(),digest_size=8).hexdigest()}",flush=True)
   dim0=0
  else:
   print(f"  [la256] FATAL: block Lanczos did not converge in {ITMAX} iterations (expected ~{n//254}) -- corrupted state?",flush=True)
   dim0=0
  print(f"  [la256] LA_GRAPH={GM['mode']}: {it} iterations, graph capture {tcap:.2f}s, status {status} ({stop})",flush=True)
 while GM is None:   # the per-iteration host loop (LA_GRAPH=0, unchanged); `while True` when LA_GRAPH is off
  it+=1
  if _LA_MAXIT and it>_LA_MAXIT:
   print(f"  [la] LA_MAXIT={_LA_MAXIT} reached after {time.time()-t0:.1f}s = {(time.time()-t0)*1000/(it-1):.2f} ms/it -- timing run, stopping",flush=True)
   if os.environ.get("LA_HASH","0")=="1":   # s30: exactness across kernel sets -- blake2b of the full Lanczos state
    import hashlib
    hh=[hashlib.blake2b(cp.asnumpy(v).tobytes(),digest_size=8).hexdigest() for v in (p0,p1,p2,xb)]
    print(f"  [la] LA_HASH it={_LA_MAXIT} p0 {hh[0]} p1 {hh[1]} p2 {hh[2]} xb {hh[3]} vt {hashlib.blake2b(Oh.tobytes(),digest_size=8).hexdigest()}",flush=True)
   dim0=0; break
  if it>ITMAX:
   print(f"  [la256] FATAL: block Lanczos did not converge in {ITMAX} iterations (expected ~{n//254}) -- corrupted state?",flush=True)
   dim0=0; break
  if PROF: _t0=time.time()
  if idb is None:
   if PROF>=2: _pe[0].record(); Bmul(p0,sc); _pe[1].record(); Btmul(sc,pn); _pe[2].record()
   else: AvP(p0,pn)
   O[:]=0
   if inner3 is not None: inner3(V3["IG"],(256,),(p0,pn,O[0],O[1],np.int64(n)))
   else: inner_k(IGR,(512,),(p0,pn,O[0],O[1],np.int64(n),np.int32(VB)))
  else:   # LA_V256 db: Btmul without the dense rows; inner256_db adds them (pn += dense_bwd(sc)) in its read pass
   if PROF>=2: _pe[0].record()
   Bmul(p0,sc)
   if PROF>=2: _pe[1].record()
   Btsp(sc,pn)
   if PROF>=2: _pe[2].record()
   O[:]=0
   if inner3 is not None: V3["idb"](V3["IG"],(256,),(p0,pn,O[0],O[1],np.int64(n),dcol_,sc,np.int32(min(128,V3["ND"]))))
   else: idb(IGR,(512,),(p0,pn,O[0],O[1],np.int64(n),np.int32(VB),dcol_,sc))   # ND=0: dcol is all zero, nothing added
  if PROF>=3: _pe[3].record()      # before the D2H copy, which synchronizes the stream (the event must be complete when queried)
  Oh=O.get(); vt_a_v[0]=Oh[0].copy(); vt_a2_v[0]=Oh[1].copy()
  if PROF:
   _t1=time.time(); tGPU+=_t1-_t0
   if PROF>=2: tBm+=cp.cuda.get_elapsed_time(_pe[0],_pe[1]); tBtm+=cp.cuda.get_elapsed_time(_pe[1],_pe[2])
   if PROF>=3 and it>1: tIn+=cp.cuda.get_elapsed_time(_pe[2],_pe[3]); tUp+=cp.cuda.get_elapsed_time(_pe[4],_pe[0]); tUk+=cp.cuda.get_elapsed_time(_pe[5],_pe[4])
   if it%250==0:
    _el=time.time()-t0
    print("   [LA_PROF] it=%d  %.2f ms/it | GPU %.2f ms (%.0f%%) | CPU %.2f ms (%.0f%%)"%(
      it,_el*1000/it,tGPU*1000/it,100*tGPU/_el,tCPU*1000/it,100*tCPU/_el)+
      ("" if PROF<2 else " | Bmul %.3f ms Btmul %.3f ms | xb sum %016x"%(tBm/it,tBtm/it,int(xb.sum())&0xFFFFFFFFFFFFFFFF))+
      ("" if PROF<3 else " | inner+D2H %.3f ms host+H2D+upd %.3f ms | upd kernel %.3f ms, useF %d/%d its"%(tIn/max(1,it-1),tUp/max(1,it-1),tUk/max(1,it-1),nF,it-1)),flush=True)
  if not vt_a_v[0].any(): break
  if hostf is not None:   # LA_V3 host: same algebra, C fast paths; the column masks stay uint64[4] (m1c == mask1)
   dim0,s0h,wi,M0=fnsf(vt_a_v[0],s1h,dim1)
   if dim0==0: break
   winv[0]=wi
   if not bool(((M0|M1h)==ALLW).all()):
    print(f"  [la256] ERROR: not all columns used at iteration {it} (dim0={dim0}, dim1={dim1})",flush=True)
    dim0=0; break
  else:
   dim0,s0,wi=fns256_rows(vt_a_v[0],s[1],dim1)
   if dim0==0: break
   winv[0]=wi
   mask0=0
   for i in s0: mask0|=1<<i
   m1c=0
   for i in s[1][:dim1]: m1c|=1<<i
   if (mask0|m1c)!=ALL256:
    print(f"  [la256] ERROR: not all columns used at iteration {it} (dim0={dim0}, dim1={dim1})",flush=True)
    dim0=0; break
   s[0]=s0+[0]*(VB-len(s0)); M0=mask_arrV(mask0,W)
  if it<4:
   O2[:]=0; inner_k(IGR,(256,),(p0,vsav,O2[0],O2[1],np.int64(n),np.int32(VB))); vt_v0[0]=O2.get()[0].copy()
  d=(vt_a2_v[0]&M0)^vt_a_v[0]; d=_mul(winv[0],d); d=d^BM256
  vt_v0_next=_mul(_tr(d),vt_v0[0])
  e=_mul(winv[1],vt_a_v[0]); e=e&M0
  vt_v0_next=vt_v0_next^_mul(_tr(e),vt_v0[1])
  useF=(mask1!=ALL256) if hostf is None else not bool((M1h==ALLW).all())
  if useF:
   M1=mask_arrV(mask1,W) if hostf is None else M1h
   f=_mul(vt_a_v[1],winv[1]); f=f^BM256; f=_mul(winv[2],f)
   f2=((vt_a2_v[1]&M1)^vt_a_v[1])&M0; f=_mul(f,f2)
   vt_v0_next=vt_v0_next^_mul(_tr(f),vt_v0[2])
  else: f=Z()
  d2=_mul(winv[0],vt_v0[0])
  if hostf is None: hmats[0]=d; hmats[1]=e; hmats[2]=f; hmats[3]=d2; dmats.set(hmats); dmask.set(M0)
  else:   # one H2D copy: mats and mask share a device buffer
   hbuf[0:1024]=d.ravel(); hbuf[1024:2048]=e.ravel(); hbuf[2048:3072]=f.ravel(); hbuf[3072:4096]=d2.ravel(); hbuf[4096:4100]=M0
   dbuf.set(hbuf)
  if PROF>=3: _pe[5].record()
  if upd3 is not None:
   upd3(V3["UG"],(512,),(p0,p1,pn,xb,dmats,dmask,np.int64(n)),shared_mem=81920)
   if useF: V3["updF"](V3["FG"],(512,),(p2,pn,dmats,np.int64(n)),shared_mem=32768)
  elif upd2 is None: upd_k(UG,(tpb,),(p0,p1,p2,pn,xb,dmats,dmask,np.int32(1 if useF else 0),np.int64(n)))
  else:
   upd2(V2["UG"],(V2["UT"],),(p0,p1,pn,xb,dmats,dmask,np.int64(n)),shared_mem=98304)
   if useF: V2["updF"](V2["UG"],(V2["UT"],),(p2,pn,dmats,np.int64(n)),shared_mem=32768)
  nF+=1 if useF else 0
  if PROF>=3: _pe[4].record()
  p0,p1,p2,pn = pn,p0,p1,p2
  winv=[winv[2],winv[0],winv[1]]; vt_v0=[vt_v0_next,vt_v0[0],vt_v0[1]]
  vt_a_v=[vt_a_v[1],vt_a_v[0]]; vt_a2_v=[vt_a2_v[1],vt_a2_v[0]]
  if hostf is None: s[1]=list(s[0]); mask1=mask0
  else: s1h=s0h; M1h=M0
  dim1=dim0
  if PROF: tCPU+=time.time()-_t1
  if it%LOGI==0: print(f"   iter {it}  {time.time()-t0:.0f}s ({(time.time()-t0)*1000/it:.2f} ms/it)",flush=True)
 if dim0==0: return None,0,it,{},None
 ax=cp.empty((n,W),cp.uint64); av=cp.empty((n,W),cp.uint64)
 if Bcheck is None:
  Bmul(xb,ax); Bmul(p0,av)
  out,nd,extra,info=combineV(n,cp.asnumpy(xb),cp.asnumpy(p0),cp.asnumpy(ax),cp.asnumpy(av),VB)
 else:
  Bcheck(xb,ax); Bcheck(p0,av)
  out,nd,extra,info=combineV(n,cp.asnumpy(xb),cp.asnumpy(p0),cp.asnumpy(ax),cp.asnumpy(av),VB,count_all=True)
 return out,nd,it,info,extra

def parse_mat(path):
 a=np.fromfile(path,dtype=np.uint32)
 nrows,num_dense,ncols=int(a[0]),int(a[1]),int(a[2]); drw=(num_dense+31)//32
 weights=np.empty(ncols,dtype=np.int64); offs=np.empty(ncols,dtype=np.int64)
 pos=3
 for i in range(ncols):
  w=int(a[pos]); weights[i]=w; offs[i]=pos+1; pos+=1+w+drw
 total=int(weights.sum())
 if total>0:
  idx=np.ones(total,dtype=np.int64); idx[0]=offs[0]
  cs=np.cumsum(weights)[:-1]; idx[cs]=offs[1:]-(offs[:-1]+weights[:-1])+1
  positions=np.cumsum(idx)
  R_sp=a[positions].astype(np.uint32); C_sp=np.repeat(np.arange(ncols,dtype=np.uint32),weights)
 else:
  R_sp=np.zeros(0,np.uint32); C_sp=np.zeros(0,np.uint32)
 # dense bitfield -> 2 uint64 per column
 dstart=offs+weights
 dw=a[(dstart[:,None]+np.arange(drw)[None,:])].astype(np.uint64)  # (ncols,drw)
 dcol=np.zeros(ncols*2,np.uint64)
 if drw>=1: dcol[0::2]|=dw[:,0]
 if drw>=2: dcol[0::2]|=dw[:,1]<<np.uint64(32)
 if drw>=3: dcol[1::2]|=dw[:,2]
 if drw>=4: dcol[1::2]|=dw[:,3]<<np.uint64(32)
 return nrows,num_dense,ncols,R_sp,C_sp,dcol

if __name__=="__main__":
 matpath=sys.argv[1]; deppath=sys.argv[2]
 print(f"parsing {matpath} ...",flush=True); t=time.time()
 # CADO-native input (2026-09-27, "bypass msieve -nc2"): replay's X.sparse.bin (written WITHOUT
 # --for_msieve, -skip 0) is loaded directly -- no cado2msieve2, no msieve -nc2, no .mat. The
 # output file below is unchanged and is ALREADY the flat kernel format CADO 'characters -ker'
 # reads (one uint64 per relation-set, bit j = kernel vector j). See cado_mat.py for the layout.
 CADO_IN = matpath.endswith(".sparse.bin") or os.environ.get("LA_FMT","") == "cado"
 if CADO_IN:
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  # LA_PARSE=cpu (default) | gpu | check: cado_mat.load_cado. gpu returns R, C, dcol as CuPy arrays (bit-identical to
  # the numpy loader's); every LA path below cp.asarray()s them, so they are used in place.
  from cado_mat import load_cado, expand_kernel
  if LA_DEFER:
   # LA_DEFER_ROWS: the loader packs exactly K dense rows (never LA_CADO_ND, which clamps silently at 128)
   if LA_BLOCK!=256 or LA_SPMV!="seg" or _graph_mode()!=0 or os.environ.get("LA_CADO_ND","")!="":
    print(f"  FATAL: LA_DEFER_ROWS={LA_DEFER} requires LA_BLOCK=256 (got {LA_BLOCK}), LA_SPMV=seg (got {LA_SPMV}), "
          f"LA_GRAPH=0 and LA_CADO_ND unset (got '{os.environ.get('LA_CADO_ND','')}')",flush=True); sys.exit(3)
   nrows,ND,ncols,R,C,dcol,ROWMAP=load_cado(matpath,nd=LA_DEFER,with_map=True)
   if ND!=LA_DEFER:
    print(f"  FATAL: LA_DEFER_ROWS={LA_DEFER} but the loader packed {ND} dense rows (clamped) -- refusing",flush=True); sys.exit(3)
  else:
   nrows,ND,ncols,R,C,dcol,ROWMAP=load_cado(matpath,with_map=True)
 elif LA_DEFER:
  print(f"  FATAL: LA_DEFER_ROWS={LA_DEFER} needs a CADO .sparse.bin input",flush=True); sys.exit(3)
 else:
  nrows,ND,ncols,R,C,dcol=parse_mat(matpath)
 print(f"  matrix {nrows}x{ncols} dense={ND} sparse_nnz={len(R)} parsed in {time.time()-t:.0f}s",flush=True)
 n=ncols; t=time.time()
 # Bounds guard: every sparse row index must be < n (the rcount/scatter kernels have no bounds check
 # and VERIFY only covers rows < n). A matrix with more rows than columns is refused LOUDLY here.
 # (device arrays from LA_PARSE=gpu: cado_mat already checked every column index < ncols_cado <= n on the GPU)
 if len(R) and isinstance(R,np.ndarray) and int(R.max())>=n:
   print(f"  FATAL: matrix row index {int(R.max())} >= n={n} (more rows than columns) -- refusing to solve",flush=True)
   sys.exit(3)
 X2=None
 if LA_BLOCK==256:
  x,nd,ok,X2=run256(R,C,dcol,ND,n,defer=LA_DEFER); del R,C
  if x is None: x=np.zeros(n,dtype=np.uint64)
 elif LA_BLOCK==128:
  # 128-bit block Lanczos + column-split SpMV (see the LA_BLOCK=128 section). Output format unchanged.
  x,nd,ok,X2=run128(R,C,dcol,ND,n); del R,C
  if x is None: x=np.zeros(n,dtype=np.uint64)
 else:
  # CSR build. Two changes vs the original, both needed for c151-size matrices:
  #
  # 1. The sort runs on the HOST. cp.argsort over nnz this large (362M for this
  #    c151 matrix) dies with CUDA_ERROR_ILLEGAL_ADDRESS on sm_120 with current
  #    CuPy -- the "argsort-300M path" install-env.sh mentions, whose check only
  #    exercises 5M elements and so never trips it. numpy's stable argsort on an
  #    integer key is a radix sort, and host RAM is not the scarce resource here.
  # 2. The transpose side needs NO sort at all: parse_mat builds
  #    C_sp = np.repeat(np.arange(ncols), weights), which is non-decreasing by
  #    construction, so argsort(C_sp) is exactly arange(nnz) and R[oic] == R.
  #    Dropping it removes half the work and half the crash surface.
  Ri=R.astype(np.int32,copy=False); Ci=C.astype(np.int32,copy=False); del R,C
  # 3. (2026-07-31) The host argsort is GONE. It was a 398.7M-element stable radix argsort whose
  #    int64 permutation alone is 3.2 GB, followed by a host gather -- measured 91s, single-threaded,
  #    on the critical path. But a STABLE sort was never required: this CSR is only ever consumed by
  #    SpMV kernels that XOR a row's entries over GF(2), so the order within a row cannot affect the
  #    result -- only the GROUPING by row matters. That is a counting sort, done here as one atomic
  #    scatter pass on the GPU. Note this does not reintroduce the sm_120 crash from (1): that is
  #    cp.argsort specifically; bincount + scatter allocates no large temporary and is a different
  #    code path. Verified by the existing "VERIFY B*x==0 & x!=0" gate, which a mis-built CSR fails.
  d_Ri=cp.asarray(Ri); d_Ci=cp.asarray(Ci); del Ri,Ci
  nnz=int(d_Ri.size)
  def _rowptr(d_idx):            # counts on the GPU, prefix sum on the host (n is only ~4.6M)
    d_cnt=cp.zeros(n,cp.int32); rcount((4096,),(256,),(d_idx,d_cnt,np.int64(nnz)))
    rp=np.zeros(n+1,np.int64); rp[1:]=np.cumsum(cp.asnumpy(d_cnt),dtype=np.int64)
    return cp.asarray(rp.astype(np.int32))
  d_rpB=_rowptr(d_Ri); d_rpBt=_rowptr(d_Ci)
  SEG=None
  if LA_SPMV=="seg":
    try:
      SEG=_load_seg().SegPlan(d_Ri,d_Ci,n,ND,width=64)
    except Exception as _e:   # never let the fast path kill the run: production kernels instead
      print(f"  WARNING: seg SpMV plan failed ({type(_e).__name__}: {str(_e)[:200]}) -- using the production SpMV",flush=True)
      SEG=None; cp.get_default_memory_pool().free_all_blocks()
  # Sort the PACKED key (row<<32)|col. One sort yields row-major order AND ascending columns within
  # each row -- byte-for-byte the layout the old host argsort produced -- and the low half is then the
  # CSR column array directly, with no gather.
  #
  # An atomic counting-sort scatter was tried first and is faster to build (7s vs 91s) but leaves each
  # row's columns in nondeterministic order. SpMV over GF(2) is order-insensitive so it was CORRECT
  # (VERIFY B*x==0 passed), but it cost locality in the v[col[k]] gather: measured 13.53 ms/it against
  # 12.80, i.e. +51s over 73,466 iterations, cancelling most of the build saving. Sorting keeps both.
  #
  # cp.sort is fine here even though cp.argsort and cp.bincount both die at this nnz on sm_120 --
  # measured 0.1s for 398.7M uint64, validated against the argsort layout.
  # MEMORY FALLBACK. The packed-key sort needs ~3.2 GB for the key plus radix scratch plus the 1.6 GB
  # output -- roughly 10 GB of device memory at this nnz, where the old host argsort needed none.
  # That is nothing on a 96 GB card but unknown on an arbitrary validator GPU, and running out here
  # is expensive: gpu_la.py exits non-zero, msvrun_gpu.sh falls back to CPU `msieve -nc`, and the run
  # blows the wall. So fall back to the (slow, safe) host path instead of dying.
  try:
    # d_Ci stays alive until the sort succeeded (the OOM fallback reads it; `del` first = NameError)
    key=(d_Ri.astype(cp.uint64)<<cp.uint64(32))|d_Ci.astype(cp.uint64)
    key.sort()
    d_colB=(key&cp.uint64(0xffffffff)).astype(cp.int32); del key
  except cp.cuda.memory.OutOfMemoryError as e:
    print(f"  [la] GPU sort OOM ({e}); falling back to host argsort (slower but safe)",flush=True)
    key=None; cp.get_default_memory_pool().free_all_blocks()
    Rh=cp.asnumpy(d_Ri); Ch=cp.asnumpy(d_Ci)
    oi=np.argsort(Rh,kind='stable'); d_colB=cp.asarray(Ch[oi]); del oi,Rh,Ch
  del d_Ci
  d_colBt=d_Ri                   # C already sorted -> identity permutation
  print(f"  sparse CSR built in {time.time()-t:.0f}s (max row wt={int(cp.diff(d_rpB).max())})",flush=True)
  d_dcol=cp.asarray(dcol)
  print("running GPU block-Lanczos (dense-split + balanced SpMV) ...",flush=True); t=time.time()
  x,nd,it=gpu_block_lanczos(d_colB,d_rpB,d_colBt,d_rpBt,d_dcol,ND,n,seg=SEG)
  print(f"  LA done: {it} iters, {nd} deps, {time.time()-t:.0f}s",flush=True)
  if x is None:   # no dependency (dim0==0 / iteration cap): the FATAL + exit 3 below, not a traceback
   x=np.zeros(n,np.uint64); ok=False
  else:
   # verify B x = 0  (CSR-vector sparse + dense)
   xg=cp.asarray(x); bx=cp.zeros(n,cp.uint64); spvec((((n*32+255)//256),),(256,),(xg,d_colB,d_rpB,bx,n)); dfwd(G(n),(256,),(d_dcol,xg,bx,n,ND))
   ok=bool((bx.get()==0).all()) and bool((x!=0).any())
  print(f"  VERIFY B*x==0 & x!=0: {ok}",flush=True)
 # Fail LOUDLY. Writing a .dep that is not a kernel vector is worse than writing nothing:
 # msvrun_gpu.sh only tests [ -s "$S.dep" ], so a garbage-but-nonempty file looks like success,
 # every sqrt dependency then comes back "not a square", and the run silently falls back to CPU
 # `msieve -nc` -- which restarts the filter and costs hours. Observed 2026-07-20 on a c121 test:
 # "0 deps / VERIFY False" was written out and consumed downstream as if valid.
 if nd==0 or not ok:
   print(f"  FATAL: block-Lanczos produced no usable dependency (deps={nd}, verify={ok}) "
         f"-- refusing to write {deppath}",flush=True)
   sys.exit(3)
 if CADO_IN:
   # Post-hoc quadratic characters (CADO 'characters') cost ~nchar+3 of these vectors, because
   # unlike msieve's .mat the CADO matrix carries NO QCB rows. Say so loudly if the margin is thin.
   _need=int(os.environ.get("LA_CADO_MIN_DEPS","40"))
   if nd<_need:
     print(f"  WARNING: only {nd} kernel vectors (< {_need}); after characters (nchar+3 lost) few or "
           f"no dependencies may remain -- rerun with another LA_SEED if characters reports dim 0",flush=True)
 if CADO_IN and ROWMAP is not None:
   # empty merged rows were dropped from B (cado_mat.py): re-insert them as ZERO kernel bits so the
   # file has one uint64 per CADO row, in index order, as 'characters -ker' expects
   x=expand_kernel(x,ROWMAP)
   print(f"  kernel expanded to {len(x)} CADO rows ({ROWMAP[0]-ncols} empty rows -> 0 bits)",flush=True)
 x.astype(np.uint64).tofile(deppath); print(f"  wrote {deppath} ({len(x)} uint64)",flush=True)
 if X2 is not None:
   # LA_KER2=1 (128-bit only): up to 64 more verified kernel vectors, same flat format, for a spare
   # `characters` pass. Nothing downstream reads it unless told to.
   if CADO_IN and ROWMAP is not None: X2=expand_kernel(X2,ROWMAP)
   X2.astype(np.uint64).tofile(deppath+".2"); print(f"  wrote {deppath}.2 ({len(X2)} uint64, spare kernel vectors)",flush=True)
