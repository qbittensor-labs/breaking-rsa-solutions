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
if LA_BLOCK not in (64, 128):
 print(f"  FATAL: LA_BLOCK={LA_BLOCK} (must be 64 or 128)", flush=True); sys.exit(3)
# LA_SPMV=seg (2026-09-27): segmented, hot-ideal-aware SpMV from segspmv.py for BOTH block widths.
# Default "prod" = the kernels above, unchanged. seg is bit-identical to them (XOR of the same words per row;
# checked by dev/spmv/bench.py on the c160 matrix) -- VERIFY below still runs on the production CSR (64) or the
# original COO (128), never on the seg structures.
LA_SPMV = os.environ.get("LA_SPMV", "prod")
if LA_SPMV not in ("prod", "seg"):
 print(f"  FATAL: LA_SPMV={LA_SPMV} (must be prod or seg)", flush=True); sys.exit(3)
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

def _rowptr128(d_idx,n):
 nnz=int(d_idx.size)
 d_cnt=cp.zeros(n,cp.int32)
 if nnz: rcount((4096,),(256,),(d_idx,d_cnt,np.int64(nnz)))
 rp=np.zeros(n+1,np.int64); rp[1:]=np.cumsum(cp.asnumpy(d_cnt),dtype=np.int64)
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
  sel=(d_Ci>=c0)&(d_Ci<c1); Rp=d_Ri[sel]; Cp=(d_Ci[sel]-c0).astype(cp.int32); del sel
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
  sel=(d_Ri>=r0)&(d_Ri<r1); col=(d_Ri[sel]-r0).astype(cp.int32); Cp=d_Ci[sel]; del sel
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
  vt_a_v=[vt_a_v[1],vt_a_v[0]]; vt_a2_v=[vt_a2_v[1],vt_a2_v[0]]; s[1]=list(s[0]); mask1=mask0; dim1=dim0
  if PROF: tCPU+=time.time()-_t1
  if it%LOGI==0: print(f"   iter {it}  {time.time()-t0:.0f}s ({(time.time()-t0)*1000/it:.2f} ms/it)",flush=True)
 if dim0==0: return None,0,it,{},None
 ax=cp.empty((n,2),cp.uint64); av=cp.empty((n,2),cp.uint64)
 Bmul(xb,ax); Bmul(p0,av)
 out,nd,extra,info=combine128(n,cp.asnumpy(xb),cp.asnumpy(p0),cp.asnumpy(ax),cp.asnumpy(av))
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
  from cado_mat import load_cado_sparse, expand_kernel
  nrows,ND,ncols,R,C,dcol,ROWMAP=load_cado_sparse(matpath,with_map=True)
 else:
  nrows,ND,ncols,R,C,dcol=parse_mat(matpath)
 print(f"  matrix {nrows}x{ncols} dense={ND} sparse_nnz={len(R)} parsed in {time.time()-t:.0f}s",flush=True)
 n=ncols; t=time.time()
 # Bounds guard: every sparse row index must be < n (the rcount/scatter kernels have no bounds check
 # and VERIFY only covers rows < n). A matrix with more rows than columns is refused LOUDLY here.
 if len(R) and int(R.max())>=n:
   print(f"  FATAL: matrix row index {int(R.max())} >= n={n} (more rows than columns) -- refusing to solve",flush=True)
   sys.exit(3)
 X2=None
 if LA_BLOCK==128:
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
