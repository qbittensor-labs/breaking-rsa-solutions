#!/usr/bin/env python3
# Copyright (C) 2026 qBitTensor Labs.
# Original author: Xdev (Enigma / Breaking RSA competition).
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
def gpu_block_lanczos(d_colB,d_rpB,d_colBt,d_rpBt,d_dcol,ND,n):
 tpb=256; gr=G(n); IG=(2048,); DG=G(n); WB=((n*32+tpb-1)//tpb,)
 sc=cp.empty(n,cp.uint64)
 HB=build_heavy(d_rpB,"B"); HBt=build_heavy(d_rpBt,"Bt")
 HBG =G(HB[3]*32)  if HB  else None
 HBtG=G(HBt[3]*32) if HBt else None
 def Bmul(src,dst):  # dst = B*src  (CSR-vector sparse rows + dense rows)
  if HB is None:
   spvec(WB,(tpb,),(src,d_colB,d_rpB,dst,n))    # warp-per-row SpMV (overwrites all rows; dense rows=0)
  else:
   spskip(WB,(tpb,),(src,d_colB,d_rpB,dst,n,HEAVY_T))          # all rows except the heavy ones
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
   spvec(WB,(tpb,),(src,d_colBt,d_rpBt,dst,n))  # sparse contribution
  else:
   spskip(WB,(tpb,),(src,d_colBt,d_rpBt,dst,n,HEAVY_T))
   spheavy(HBtG,(tpb,),(src,d_colBt,HBt[0],HBt[1],HBt[2],dst,HBt[3]))
  dbwd(DG,(tpb,),(d_dcol,src,dst,n))            # dense contribution z[i]^=...
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
 while True:
  it+=1
  AvP(p0,pn)
  O[:]=0; inner2(IG,(tpb,),(p0,pn,O[0],O[1],n))
  Oh=O.get(); vt_a_v[0]=Oh[0].copy(); vt_a2_v[0]=Oh[1].copy()
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
  if it%2000==0: print(f"   iter {it}  {time.time()-t0:.0f}s ({(time.time()-t0)*1000/it:.2f} ms/it)",flush=True)
 if dim0==0:return None,0,it
 Bmul(xb,pn); ax=cp.asnumpy(pn); Bmul(p0,pn); av=cp.asnumpy(pn)
 outx,nd=combine_cols(n,cp.asnumpy(xb),cp.asnumpy(p0),ax,av)
 return outx,nd,it

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
 nrows,ND,ncols,R,C,dcol=parse_mat(matpath)
 print(f"  matrix {nrows}x{ncols} dense={ND} sparse_nnz={len(R)} parsed in {time.time()-t:.0f}s",flush=True)
 n=ncols; t=time.time()
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
   key=(d_Ri.astype(cp.uint64)<<cp.uint64(32))|d_Ci.astype(cp.uint64)
   del d_Ci
   key.sort()
   d_colB=(key&cp.uint64(0xffffffff)).astype(cp.int32); del key
 except cp.cuda.memory.OutOfMemoryError as e:
   print(f"  [la] GPU sort OOM ({e}); falling back to host argsort (slower but safe)",flush=True)
   cp.get_default_memory_pool().free_all_blocks()
   Rh=cp.asnumpy(d_Ri); Ch=cp.asnumpy(d_Ci); del d_Ci
   oi=np.argsort(Rh,kind='stable'); d_colB=cp.asarray(Ch[oi]); del oi,Rh,Ch
 d_colBt=d_Ri                   # C already sorted -> identity permutation
 print(f"  sparse CSR built in {time.time()-t:.0f}s (max row wt={int(cp.diff(d_rpB).max())})",flush=True)
 d_dcol=cp.asarray(dcol)
 print("running GPU block-Lanczos (dense-split + balanced SpMV) ...",flush=True); t=time.time()
 x,nd,it=gpu_block_lanczos(d_colB,d_rpB,d_colBt,d_rpBt,d_dcol,ND,n)
 print(f"  LA done: {it} iters, {nd} deps, {time.time()-t:.0f}s",flush=True)
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
 x.astype(np.uint64).tofile(deppath); print(f"  wrote {deppath} ({ncols} uint64)",flush=True)
