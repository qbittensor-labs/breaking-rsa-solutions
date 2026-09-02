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
# Uses CuPy + NVRTC (runtime kernel compilation) -> needs only the NVIDIA driver + pip CUDA headers.
import os,glob,time,sys,numpy as np
# validator runs the container --read-only; redirect CuPy's kernel cache (default ~/.cupy) and
# HOME to the writable tmpfs so `import cupy` / NVRTC don't crash on the read-only rootfs.
os.environ.setdefault("CUPY_CACHE_DIR","/tmp/.cupy")
os.environ.setdefault("HOME","/tmp")
for _d in (os.environ["CUPY_CACHE_DIR"], "/tmp/.cupy/jitify"):
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
extern "C" __global__ void blockmul_acc(const unsigned long long* V,const unsigned long long* M,unsigned long long* y,int n){
 int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n)return; unsigned long long x=V[r],a=0;
 while(x){int i=__ffsll(x)-1; a^=M[i]; x&=x-1;} y[r]^=a;}
extern "C" __global__ void inner2(const unsigned long long* A,const unsigned long long* B,
                                  unsigned long long* o1,unsigned long long* o2,int n){
 __shared__ unsigned long long s1[64]; __shared__ unsigned long long s2[64];
 if(threadIdx.x<64){s1[threadIdx.x]=0;s2[threadIdx.x]=0;} __syncthreads();
 for(int r=blockIdx.x*blockDim.x+threadIdx.x;r<n;r+=gridDim.x*blockDim.x){
   unsigned long long a=A[r],b=B[r],x=a,y=b;
   while(x){int i=__ffsll(x)-1; atomicXor(&s1[i],b); x&=x-1;}
   while(y){int i=__ffsll(y)-1; atomicXor(&s2[i],b); y&=y-1;}}
 __syncthreads();
 if(threadIdx.x<64){atomicXor(&o1[threadIdx.x],s1[threadIdx.x]);atomicXor(&o2[threadIdx.x],s2[threadIdx.x]);}}
// dense rows forward: y[j] (j<nd) ^= v[i] for each column i whose dense bitfield (2 words) has bit j set
extern "C" __global__ void dense_fwd(const unsigned long long* dcol,const unsigned long long* v,
                                     unsigned long long* y,int ncols,int nd){
 __shared__ unsigned long long s[128]; if(threadIdx.x<128)s[threadIdx.x]=0; __syncthreads();
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<ncols;i+=gridDim.x*blockDim.x){
   unsigned long long vi=v[i],w0=dcol[2*i],w1=dcol[2*i+1];
   while(w0){int j=__ffsll(w0)-1; atomicXor(&s[j],vi); w0&=w0-1;}
   while(w1){int j=__ffsll(w1)-1; atomicXor(&s[64+j],vi); w1&=w1-1;}}
 __syncthreads();
 if(threadIdx.x<nd) atomicXor(&y[threadIdx.x],s[threadIdx.x]);}
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
inner2=cp.RawKernel(KS,'inner2',options=OPT); dfwd=cp.RawKernel(KS,'dense_fwd',options=OPT); dbwd=cp.RawKernel(KS,'dense_bwd',options=OPT)
def G(n,t=256):return((n+t-1)//t,)
bm=[1<<i for i in range(64)]; MASK=(1<<64)-1
def mul64(a,b):
 c=[0]*64
 for i in range(64):
  ai=a[i];acc=0;j=0
  while ai:
   if ai&1:acc^=b[j]
   ai>>=1;j+=1
  c[i]=acc
 return c
def tr64(a):
 t=[0]*64
 for i in range(64):
  w=a[i]
  while w:
   j=(w&-w).bit_length()-1;t[j]|=bm[i];w&=w-1
 return t
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
 out=np.zeros(ncols,dtype=np.uint64)
 for k in range(i,64):
  row=mat[k]; bp=np.uint64(k-i)
  for cwi in range(cw):
   w=int(row[cwi]); base=cwi*64
   while w:
    b=(w&-w).bit_length()-1; jc=base+b
    if jc<ncols: out[jc]|=(np.uint64(1)<<bp)
    w&=w-1
 nd=0 if i>64 else 64-i
 return out,nd

# globals set in __main__: d_colB,d_rpB,d_colBt,d_rpBt,d_dcol,ND,sc(buffer)
def gpu_block_lanczos(d_colB,d_rpB,d_colBt,d_rpBt,d_dcol,ND,n):
 tpb=256; gr=G(n); IG=(2048,); DG=G(n); WB=((n*32+tpb-1)//tpb,)
 sc=cp.empty(n,cp.uint64)
 def Bmul(src,dst):  # dst = B*src  (CSR-vector sparse rows + dense rows)
  spvec(WB,(tpb,),(src,d_colB,d_rpB,dst,n))     # warp-per-row SpMV (overwrites all rows; dense rows=0)
  dfwd(DG,(tpb,),(d_dcol,src,dst,n,ND))         # dense rows accumulate into dst[0..ND-1]
 def Btmul(src,dst): # dst = B^T*src
  spmv(gr,(tpb,),(src,d_colBt,d_rpBt,dst,n))    # sparse contribution
  dbwd(DG,(tpb,),(d_dcol,src,dst,n))            # dense contribution z[i]^=...
 def AvP(src,dst): Bmul(src,sc); Btmul(sc,dst)  # dst = B^T B src
 p0=cp.empty(n,cp.uint64); p1=cp.zeros(n,cp.uint64); p2=cp.zeros(n,cp.uint64); pn=cp.empty(n,cp.uint64)
 xb=cp.asarray(np.random.default_rng(5).integers(0,1<<64,n,dtype=np.uint64))
 AvP(xb,p0); vsav=p0.copy()
 O=cp.zeros((2,64),cp.uint64); ov=cp.zeros(64,cp.uint64); dbuf=cp.empty(64,cp.uint64); hbuf=np.empty(64,np.uint64)
 def up(lst): hbuf[:]=lst; dbuf.set(hbuf)
 vt_a_v=[[0]*64,[0]*64];vt_a2_v=[[0]*64,[0]*64];vt_v0=[[0]*64]*3;winv=[[0]*64]*3
 s=[[0]*64,list(range(64))];dim1=64;mask1=MASK;it=0;dim0=0; t0=time.time()
 while True:
  it+=1
  AvP(p0,pn)
  O[:]=0; inner2(IG,(tpb,),(p0,pn,O[0],O[1],n))
  Oh=O.get(); vt_a_v[0]=Oh[0].tolist(); vt_a2_v[0]=Oh[1].tolist()
  if not any(vt_a_v[0]):break
  dim0,s0,winv[0]=fns(vt_a_v[0],s[1],dim1)
  if dim0==0:break
  s[0]=s0+[0]*(64-len(s0));mask0=0
  for i in range(dim0):mask0|=bm[s0[i]]
  if mask0!=MASK: pn&=cp.uint64(mask0)
  if it<4:
   O[:]=0; inner2(IG,(tpb,),(p0,vsav,O[0],O[1],n)); vt_v0[0]=O.get()[0].tolist()
  d=[((vt_a2_v[0][i]&mask0)^vt_a_v[0][i]) for i in range(64)]; d=mul64(winv[0],d); d=[d[i]^bm[i] for i in range(64)]
  up(d); bmacc(gr,(tpb,),(p0,dbuf,pn,n))
  vt_v0_next=mul64(tr64(d),vt_v0[0])
  e=mul64(winv[1],vt_a_v[0]); e=[e[i]&mask0 for i in range(64)]
  up(e); bmacc(gr,(tpb,),(p1,dbuf,pn,n))
  ee=mul64(tr64(e),vt_v0[1]); vt_v0_next=[vt_v0_next[i]^ee[i] for i in range(64)]
  if mask1!=MASK:
   f=mul64(vt_a_v[1],winv[1]); f=[f[i]^bm[i] for i in range(64)]; f=mul64(winv[2],f)
   f2=[(((vt_a2_v[1][i]&mask1)^vt_a_v[1][i])&mask0) for i in range(64)]; f=mul64(f,f2)
   up(f); bmacc(gr,(tpb,),(p2,dbuf,pn,n))
   ff=mul64(tr64(f),vt_v0[2]); vt_v0_next=[vt_v0_next[i]^ff[i] for i in range(64)]
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
 oi=np.argsort(R,kind='stable'); rpB=np.zeros(n+1,np.int32); np.add.at(rpB,R.astype(np.int64)+1,1); rpB=np.cumsum(rpB).astype(np.int32); colB=C[oi].astype(np.int32)
 oic=np.argsort(C,kind='stable'); rpBt=np.zeros(n+1,np.int32); np.add.at(rpBt,C.astype(np.int64)+1,1); rpBt=np.cumsum(rpBt).astype(np.int32); colBt=R[oic].astype(np.int32)
 print(f"  sparse CSR built in {time.time()-t:.0f}s (max row wt={int(np.diff(rpB).max())})",flush=True)
 d_colB=cp.asarray(colB);d_rpB=cp.asarray(rpB);d_colBt=cp.asarray(colBt);d_rpBt=cp.asarray(rpBt);d_dcol=cp.asarray(dcol)
 print("running GPU block-Lanczos (dense-split + balanced SpMV) ...",flush=True); t=time.time()
 x,nd,it=gpu_block_lanczos(d_colB,d_rpB,d_colBt,d_rpBt,d_dcol,ND,n)
 print(f"  LA done: {it} iters, {nd} deps, {time.time()-t:.0f}s",flush=True)
 # verify B x = 0  (CSR-vector sparse + dense)
 xg=cp.asarray(x); bx=cp.zeros(n,cp.uint64); spvec((((n*32+255)//256),),(256,),(xg,d_colB,d_rpB,bx,n)); dfwd(G(n),(256,),(d_dcol,xg,bx,n,ND))
 ok=bool((bx.get()==0).all()) and bool((x!=0).any())
 print(f"  VERIFY B*x==0 & x!=0: {ok}",flush=True)
 x.astype(np.uint64).tofile(deppath); print(f"  wrote {deppath} ({ncols} uint64)",flush=True)
