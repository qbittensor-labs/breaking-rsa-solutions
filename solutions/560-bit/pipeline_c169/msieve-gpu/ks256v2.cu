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

// ks256v2.cu -- LA_V256 kernels for gpu_la.py's 256-bit block Lanczos (2026-10-20). Compiled ONLY when LA_V256 is set
// (gpu_la.py reads this file from its real directory), so the default path never depends on it. Every kernel computes
// the same GF(2) result as its ks256.cu counterpart -- XOR is associative and commutative and the integer arithmetic is
// exact, so the outputs are equal bit for bit (gpu_la.py's start-up self-test compares them on random data before use).
//
// WHERE THE ~10.6 ms/it OF NON-SpMV WORK GOES (TD 100, n 15.93M, one 256-bit vector = n x 32 B = 510 MB; LA_PROF=3
// 50.55 ms/it = Bmul 16.14 + Btmul 18.19 + inner 5.61 + ~10.6 unattributed; the unattributed part is upd256 (~9.7 ms:
// GPU 49.65 - 39.94 attributed) + 0.9 ms host algebra with the GPU idle):
//   upd256 (ks256.cu)  reads p0 p1 pn xb [p2], writes pn xb: 3.06 GB (3.57 with F) = 1.9-2.2 ms at 1.6 TB/s, takes ~9.7.
//                      It walks every SET bit of a 256-bit word with __ffsll (~128 per word, 2-3 words per row) and does
//                      a random 32 B shared load per bit: ~12 KB of bank-conflicted shared traffic per row, divergent
//                      trip counts. Same cause as the 128-bit ffs upd128 (1.85 ms -> 0.81 with tables, LA_KERN=v2) --
//                      the 256-bit path never got the table version.
//   inner256           reads p0 pn: 1.02 GB = 0.64 ms, takes 5.6: instruction bound (per row and output bit: 4 x 16 B
//                      shared loads, a 3-way word select, a 64-bit mask, 8 LOP3 -- ~23 instructions x 512 threads).
//   dense_bwd256       (inside Btmul) reads dcol (255 MB) + pn, writes pn: 1.28 GB = 0.8 ms; pn is read again by inner.
//   dense_fwd256       (inside Bmul) reads dcol + p0: 0.77 GB.
// KERNELS HERE
//   upd256_v2   (upd256_v2x = the same with the xb fix, see X4 below) nibble tables (Four Russians, 4 bits): T[m][c][v] = XOR of rows 4c+j (bits j of v) of matrix m, for
//               m = D, D2(X), E: 3 x 64 x 16 x 32 B = 96 KB of dynamic shared memory (1 block/SM). Per row a FIXED
//               64 lookups per matrix word-set (192 x 32 B, no divergence) instead of ~384 data-dependent ffs steps.
//   updF256     the F term (pn ^= p2 F) when useF: a second pass with the 32 KB F table (the four tables, 128 KB, do
//               not fit the 99 KB opt-in limit). (pn & mask) ^ p0 D ^ p1 E ^ p2 F is the same sum in either order.
//   inner256_v2 thread per output bit as before, but it loads only the 32-bit word holding its bit (broadcast) and
//               builds a 32-bit mask: per row 1 LDS.32 + 2 LDS.128 + 2 ALU + 8 LOP3 (~13 vs ~23 instructions).
//   inner256_db inner256_v2 with dense_bwd256 fused into the tile load: pn is read once (+dcol), the dense
//               contribution is XORed in, written back (only rows with dense bits) and used for the inner products in
//               the same pass: saves one 510 MB read + 510 MB write and a launch per iteration.
// NB macro parameters must not be named x/y/z/w: ks256.cu's  #define X4(y,t) y.x^=t.x; y.y^=t.y; ...  also substitutes
// the MEMBER token .y, so its X4(x,s) expands to  x.x^=s.x; x.x^=s.x; x.z^=s.z; x.w^=s.w  -- upd256 has never updated
// bits 0..127 of xb (they keep the random start; the recurrence p0/p1/p2/pn uses X4(y,t), which expands correctly).
// XF=0 below reproduces that exactly (bit-identical xb, the default); XF=1 (LA_V256 'xfix') updates all 256 bits.
#define X4(D_,S_) { (D_).x^=(S_).x; (D_).y^=(S_).y; (D_).z^=(S_).z; (D_).w^=(S_).w; }
#define X4HI(D_,S_) { (D_).z^=(S_).z; (D_).w^=(S_).w; }
#define TL4 256

// T is [3][64][16] (m, nibble c, value v): D at 0, X(D2) at 1024, E at 2048.  mats = [D | E | F | D2], 256 rows each.
template<int XF>
__device__ __forceinline__ void nib_dx(const ulonglong4* __restrict__ T, const unsigned long long w, const int c0,
                                       ulonglong4& y, ulonglong4& x){
 #pragma unroll
 for(int c=0;c<16;c++){
   const int v=(int)((w>>(4*c))&15ULL);
   const ulonglong4 t=T[(c0+c)*16+v];
   X4(y,t);
   if(XF){ const ulonglong4 s=T[1024+(c0+c)*16+v]; X4(x,s); } else { const ulonglong4 s=T[1024+(c0+c)*16+v]; X4HI(x,s); }
 }
}
__device__ __forceinline__ void nib_1(const ulonglong4* __restrict__ T, const unsigned long long w, const int c0, ulonglong4& y){
 #pragma unroll
 for(int c=0;c<16;c++){
   const int v=(int)((w>>(4*c))&15ULL);
   const ulonglong4 t=T[(c0+c)*16+v];
   X4(y,t);
 }
}
// table entry e of nm consecutive 1024-entry tables; src[k] = the matrix for table k (offset into mats, in rows)
__device__ __forceinline__ ulonglong4 tab_entry(const ulonglong4* __restrict__ M, const int c, const int v){
 ulonglong4 a=make_ulonglong4(0ULL,0ULL,0ULL,0ULL);
 #pragma unroll
 for(int j=0;j<4;j++) if((v>>j)&1){ const ulonglong4 t=M[4*c+j]; X4(a,t); }
 return a;
}

// pn = (pn & mask) ^ p0*D ^ p1*E ;  xb ^= p0*D2      (useF: updF256 afterwards adds p2*F)
// dynamic shared memory: 3*1024*32 = 98304 B
template<int XF>
__device__ __forceinline__ void upd_body(const ulonglong4* __restrict__ p0,const ulonglong4* __restrict__ p1,
    ulonglong4* __restrict__ pn, ulonglong4* __restrict__ xb, const ulonglong4* __restrict__ mats,
    const unsigned long long* __restrict__ mask, long long n){
 extern __shared__ __align__(16) unsigned char sm_[];
 ulonglong4* T=(ulonglong4*)sm_;
 for(int e=threadIdx.x;e<3*1024;e+=blockDim.x){
   const int m=e>>10, c=(e>>4)&63, v=e&15;
   const ulonglong4* M=mats+(m==0? 0 : (m==1? 768 : 256));      // D, D2, E
   T[e]=tab_entry(M,c,v);
 }
 const unsigned long long m0=mask[0],m1=mask[1],m2=mask[2],m3=mask[3];
 __syncthreads();
 for(long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x;r<n;r+=(long long)gridDim.x*blockDim.x){
   const ulonglong4 a=p0[r], b=p1[r]; ulonglong4 y=pn[r], x=xb[r];
   y.x&=m0; y.y&=m1; y.z&=m2; y.w&=m3;
   nib_dx<XF>(T,a.x,0,y,x); nib_dx<XF>(T,a.y,16,y,x); nib_dx<XF>(T,a.z,32,y,x); nib_dx<XF>(T,a.w,48,y,x);
   nib_1(T+2048,b.x,0,y); nib_1(T+2048,b.y,16,y); nib_1(T+2048,b.z,32,y); nib_1(T+2048,b.w,48,y);
   pn[r]=y; xb[r]=x;
 }
}
extern "C" __global__ void __launch_bounds__(512,1) upd256_v2(const ulonglong4* __restrict__ p0,const ulonglong4* __restrict__ p1,
    ulonglong4* __restrict__ pn, ulonglong4* __restrict__ xb, const ulonglong4* __restrict__ mats,
    const unsigned long long* __restrict__ mask, long long n){ upd_body<0>(p0,p1,pn,xb,mats,mask,n); }
extern "C" __global__ void __launch_bounds__(512,1) upd256_v2x(const ulonglong4* __restrict__ p0,const ulonglong4* __restrict__ p1,
    ulonglong4* __restrict__ pn, ulonglong4* __restrict__ xb, const ulonglong4* __restrict__ mats,
    const unsigned long long* __restrict__ mask, long long n){ upd_body<1>(p0,p1,pn,xb,mats,mask,n); }
// pn ^= p2*F   (dynamic shared memory 32768 B)
extern "C" __global__ void __launch_bounds__(512,1) updF256(const ulonglong4* __restrict__ p2, ulonglong4* __restrict__ pn,
    const ulonglong4* __restrict__ mats, long long n){
 extern __shared__ __align__(16) unsigned char sm_[];
 ulonglong4* T=(ulonglong4*)sm_;
 for(int e=threadIdx.x;e<1024;e+=blockDim.x) T[e]=tab_entry(mats+512,(e>>4)&63,e&15);
 __syncthreads();
 for(long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x;r<n;r+=(long long)gridDim.x*blockDim.x){
   const ulonglong4 c=p2[r]; ulonglong4 y=pn[r];
   nib_1(T,c.x,0,y); nib_1(T,c.y,16,y); nib_1(T,c.z,32,y); nib_1(T,c.w,48,y);
   pn[r]=y;
 }
}

// o1 += A^T B (bits < nout1), o2 += B^T B (blockDim 512 only); same contract as ks256.cu inner256.
// DB=1: B[r] ^= dense_bwd(dcol[r], u) first (written back), i.e. dense_bwd256 + inner256 in one pass.
template<int DB>
__device__ __forceinline__ void inner_body(const ulonglong4* __restrict__ A, ulonglong4* __restrict__ B,
    unsigned long long* __restrict__ o1, unsigned long long* __restrict__ o2, long long n, int nout1,
    const unsigned long long* __restrict__ dcol, const ulonglong4* __restrict__ u){
 __shared__ ulonglong4 sa[TL4]; __shared__ ulonglong4 sb[TL4];
 const int t=threadIdx.x, bit=t&255, second=t>>8, wi=bit>>5, sh=bit&31;
 const unsigned* __restrict__ s32=(const unsigned*)(second? sb : sa);
 const uint4* __restrict__ sq=(const uint4*)sb;
 unsigned c0=0u,c1=0u,c2=0u,c3=0u,c4=0u,c5=0u,c6=0u,c7=0u;
 for(long long base=(long long)blockIdx.x*TL4; base<n; base+=(long long)gridDim.x*TL4){
   for(int q=t;q<TL4;q+=blockDim.x){
     const long long r=base+q; ulonglong4 z=make_ulonglong4(0ULL,0ULL,0ULL,0ULL), a=z;
     if(r<n){
       a=A[r]; z=B[r];
       if(DB){
         unsigned long long w0=dcol[2*r], w1=dcol[2*r+1];
         if(w0|w1){
           while(w0){ const int j=__ffsll((long long)w0)-1; const ulonglong4 tt=u[j]; X4(z,tt); w0&=w0-1; }
           while(w1){ const int j=__ffsll((long long)w1)+63; const ulonglong4 tt=u[j]; X4(z,tt); w1&=w1-1; }
           B[r]=z;
         }
       }
     }
     sa[q]=a; sb[q]=z;
   }
   __syncthreads();
   #pragma unroll 4
   for(int k=0;k<TL4;k++){
     const unsigned mk=0u-((s32[8*k+wi]>>sh)&1u);
     const uint4 b0=sq[2*k], b1=sq[2*k+1];
     c0^=b0.x&mk; c1^=b0.y&mk; c2^=b0.z&mk; c3^=b0.w&mk;
     c4^=b1.x&mk; c5^=b1.y&mk; c6^=b1.z&mk; c7^=b1.w&mk;
   }
   __syncthreads();
 }
 const unsigned long long q0=(unsigned long long)c0|((unsigned long long)c1<<32), q1=(unsigned long long)c2|((unsigned long long)c3<<32),
                          q2=(unsigned long long)c4|((unsigned long long)c5<<32), q3=(unsigned long long)c6|((unsigned long long)c7<<32);
 if(second){ atomicXor(&o2[4*bit],q0); atomicXor(&o2[4*bit+1],q1); atomicXor(&o2[4*bit+2],q2); atomicXor(&o2[4*bit+3],q3); }
 else if(bit<nout1){ atomicXor(&o1[4*bit],q0); atomicXor(&o1[4*bit+1],q1); atomicXor(&o1[4*bit+2],q2); atomicXor(&o1[4*bit+3],q3); }
}
extern "C" __global__ void inner256_v2(const ulonglong4* A, const ulonglong4* B, unsigned long long* o1,
                                       unsigned long long* o2, long long n, int nout1){
 inner_body<0>(A,(ulonglong4*)B,o1,o2,n,nout1,(const unsigned long long*)0,(const ulonglong4*)0);
}
extern "C" __global__ void inner256_db(const ulonglong4* A, ulonglong4* B, unsigned long long* o1,
                                       unsigned long long* o2, long long n, int nout1,
                                       const unsigned long long* dcol, const ulonglong4* u){
 inner_body<1>(A,B,o1,o2,n,nout1,dcol,u);
}
