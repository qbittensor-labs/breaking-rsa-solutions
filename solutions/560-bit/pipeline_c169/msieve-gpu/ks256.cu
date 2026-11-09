// Copyright (C) 2026 qBitTensor Labs.
// Original author: an anonymous competition participant (Enigma / Breaking RSA competition).
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

// ks256.cu -- the 256-bit kernels of gpu_la.py (LA_BLOCK=256, W6). Rows are ulonglong4 (32 B). Same contracts as
// the 128-bit kernels in gpu_la.py's KS128 (see there), re-typed; inner256 runs 512 threads (o1 bits 0..255 and
// o2 bits 0..255) or <= 256 threads for o1 only; the dense bitfield stays 2 words per column (dense_fwd256).
extern "C" __global__ void spmv256(const ulonglong4* v,const int* col,const int* rp,ulonglong4* y,
                                   int nr,int T,int acc,int VSH){
 long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x; int VW=1<<VSH;
 long long gw=tid>>VSH; int lane=(int)(tid&(VW-1));
 unsigned long long a0=0,a1=0,a2=0,a3=0; int heavy=0;
 if(gw<nr){ int s=rp[gw],e=rp[gw+1];
   if(e-s>T) heavy=1;
   else for(int k=s+lane;k<e;k+=VW){ ulonglong4 t=v[col[k]]; a0^=t.x; a1^=t.y; a2^=t.z; a3^=t.w; } }
 for(int o=VW>>1;o>0;o>>=1){ a0^=__shfl_down_sync(0xffffffffu,a0,o,VW); a1^=__shfl_down_sync(0xffffffffu,a1,o,VW);
   a2^=__shfl_down_sync(0xffffffffu,a2,o,VW); a3^=__shfl_down_sync(0xffffffffu,a3,o,VW); }
 if(gw<nr && lane==0){
   if(acc){ if(!heavy){ ulonglong4 t=y[gw]; t.x^=a0; t.y^=a1; t.z^=a2; t.w^=a3; y[gw]=t; } }
   else { ulonglong4 t; t.x=heavy?0ULL:a0; t.y=heavy?0ULL:a1; t.z=heavy?0ULL:a2; t.w=heavy?0ULL:a3; y[gw]=t; } }}
extern "C" __global__ void spmv_heavy256(const ulonglong4* v,const int* col,const int* hrow,const int* hs,
                                         const int* he,unsigned long long* y,int nchunk){
 int gw=(blockIdx.x*blockDim.x+threadIdx.x)>>5, lane=threadIdx.x&31; if(gw>=nchunk)return;
 int r=hrow[gw], s=hs[gw], e=he[gw]; unsigned long long a0=0,a1=0,a2=0,a3=0;
 for(int k=s+lane;k<e;k+=32){ ulonglong4 t=v[col[k]]; a0^=t.x; a1^=t.y; a2^=t.z; a3^=t.w; }
 for(int o=16;o>0;o>>=1){ a0^=__shfl_down_sync(0xffffffffu,a0,o); a1^=__shfl_down_sync(0xffffffffu,a1,o);
   a2^=__shfl_down_sync(0xffffffffu,a2,o); a3^=__shfl_down_sync(0xffffffffu,a3,o); }
 if(lane==0){ atomicXor(&y[4*(long long)r],a0); atomicXor(&y[4*(long long)r+1],a1); atomicXor(&y[4*(long long)r+2],a2); atomicXor(&y[4*(long long)r+3],a3); }}
#define TL4 256
extern "C" __global__ void inner256(const ulonglong4* A,const ulonglong4* B,unsigned long long* o1,
                                    unsigned long long* o2,long long n,int nout1){
 __shared__ ulonglong4 sa[TL4]; __shared__ ulonglong4 sb[TL4];
 const int t=threadIdx.x, bit=t&255, second=t>>8, w=bit>>6;
 const unsigned long long m=1ULL<<(bit&63);
 unsigned long long c0=0,c1=0,c2=0,c3=0;
 for(long long base=(long long)blockIdx.x*TL4; base<n; base+=(long long)gridDim.x*TL4){
   for(int q=t;q<TL4;q+=blockDim.x){
     long long r=base+q; ulonglong4 z; z.x=0ULL; z.y=0ULL; z.z=0ULL; z.w=0ULL;
     sa[q]= r<n? A[r]: z; sb[q]= r<n? B[r]: z; }
   __syncthreads();
   #pragma unroll 4
   for(int k=0;k<TL4;k++){
     ulonglong4 bv=sb[k]; ulonglong4 av= second? bv: sa[k];
     unsigned long long wd= (w==0)? av.x: (w==1)? av.y: (w==2)? av.z: av.w;
     unsigned long long mk=0ULL-(unsigned long long)((wd&m)!=0ULL);
     c0^=bv.x&mk; c1^=bv.y&mk; c2^=bv.z&mk; c3^=bv.w&mk; }
   __syncthreads();
 }
 if(second){ atomicXor(&o2[4*bit],c0); atomicXor(&o2[4*bit+1],c1); atomicXor(&o2[4*bit+2],c2); atomicXor(&o2[4*bit+3],c3); }
 else if(bit<nout1){ atomicXor(&o1[4*bit],c0); atomicXor(&o1[4*bit+1],c1); atomicXor(&o1[4*bit+2],c2); atomicXor(&o1[4*bit+3],c3); }}
extern "C" __global__ void dense_fwd256(const ulonglong2* dcol,const ulonglong4* B,unsigned long long* o1,long long n,int nout1){
 __shared__ ulonglong2 sa[TL4]; __shared__ ulonglong4 sb[TL4];
 const int t=threadIdx.x, w=t>>6; const unsigned long long m=1ULL<<(t&63);
 unsigned long long c0=0,c1=0,c2=0,c3=0;
 for(long long base=(long long)blockIdx.x*TL4; base<n; base+=(long long)gridDim.x*TL4){
   for(int q=t;q<TL4;q+=blockDim.x){
     long long r=base+q; ulonglong2 z2; z2.x=0ULL; z2.y=0ULL; ulonglong4 z; z.x=0ULL; z.y=0ULL; z.z=0ULL; z.w=0ULL;
     sa[q]= r<n? dcol[r]: z2; sb[q]= r<n? B[r]: z; }
   __syncthreads();
   #pragma unroll 4
   for(int k=0;k<TL4;k++){
     ulonglong2 av=sa[k]; ulonglong4 bv=sb[k];
     unsigned long long wd= w? av.y: av.x;
     unsigned long long mk=0ULL-(unsigned long long)((wd&m)!=0ULL);
     c0^=bv.x&mk; c1^=bv.y&mk; c2^=bv.z&mk; c3^=bv.w&mk; }
   __syncthreads();
 }
 if(t<nout1){ atomicXor(&o1[4*t],c0); atomicXor(&o1[4*t+1],c1); atomicXor(&o1[4*t+2],c2); atomicXor(&o1[4*t+3],c3); }}
extern "C" __global__ void dense_bwd256(const unsigned long long* dcol,const ulonglong4* u,ulonglong4* z,long long ncols){
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<ncols;i+=(long long)gridDim.x*blockDim.x){
   unsigned long long w0=dcol[2*i],w1=dcol[2*i+1],a0=0,a1=0,a2=0,a3=0;
   while(w0){int j=__ffsll((long long)w0)-1; ulonglong4 t=u[j]; a0^=t.x; a1^=t.y; a2^=t.z; a3^=t.w; w0&=w0-1;}
   while(w1){int j=__ffsll((long long)w1)+63; ulonglong4 t=u[j]; a0^=t.x; a1^=t.y; a2^=t.z; a3^=t.w; w1&=w1-1;}
   ulonglong4 r=z[i]; r.x^=a0; r.y^=a1; r.z^=a2; r.w^=a3; z[i]=r; }}
#define X4(y,t) y.x^=t.x; y.y^=t.y; y.z^=t.z; y.w^=t.w
#define WALK_DX(word,off) w=word; base=off; while(w){int i=__ffsll((long long)w)-1+base; ulonglong4 t=sD[i],s=sX[i]; X4(y,t); X4(x,s); w&=w-1;}
#define WALK_M(word,off,M) w=word; base=off; while(w){int i=__ffsll((long long)w)-1+base; ulonglong4 t=M[i]; X4(y,t); w&=w-1;}
extern "C" __global__ void upd256(const ulonglong4* p0,const ulonglong4* p1,const ulonglong4* p2,
                                  ulonglong4* pn,ulonglong4* xb,const ulonglong4* mats,
                                  const unsigned long long* mask,int useF,long long n){
 __shared__ ulonglong4 sD[256]; __shared__ ulonglong4 sE[256]; __shared__ ulonglong4 sF[256]; __shared__ ulonglong4 sX[256];
 for(int q=threadIdx.x;q<256;q+=blockDim.x){ sD[q]=mats[q]; sE[q]=mats[256+q]; sF[q]=mats[512+q]; sX[q]=mats[768+q]; }
 const unsigned long long m0=mask[0],m1=mask[1],m2=mask[2],m3=mask[3];
 __syncthreads();
 for(long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x;r<n;r+=(long long)gridDim.x*blockDim.x){
   ulonglong4 a=p0[r], b=p1[r], y=pn[r], x=xb[r]; unsigned long long w; int base;
   y.x&=m0; y.y&=m1; y.z&=m2; y.w&=m3;
   WALK_DX(a.x,0) WALK_DX(a.y,64) WALK_DX(a.z,128) WALK_DX(a.w,192)
   WALK_M(b.x,0,sE) WALK_M(b.y,64,sE) WALK_M(b.z,128,sE) WALK_M(b.w,192,sE)
   if(useF){ ulonglong4 c=p2[r]; WALK_M(c.x,0,sF) WALK_M(c.y,64,sF) WALK_M(c.z,128,sF) WALK_M(c.w,192,sF) }
   pn[r]=y; xb[r]=x; }}
