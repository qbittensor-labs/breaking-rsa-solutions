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

// ks256v3.cu -- LA_V3 kernels for gpu_la.py's 256-bit block Lanczos (2026-10-21, sandbox #32 / s30). Compiled ONLY when
// LA_V3 is set; the default path never reads this file. Same GF(2) results as the LA_V256 kernels (ks256v2.cu) bit for
// bit (XOR is associative/commutative; gpu_la.py's start-up self-test compares them on random data before use).
//
// WHERE THE TIME WENT (s27 t256, TD 130, n 14.35M, one vector = n x 32 B = 459 MB; LA_PROF=3 51.54 ms/it =
// Bmul 21.85 + Btmul 20.18 + inner 4.16 + upd 4.41 + host 0.90):
//  * upd256_v2/updF256 are SHARED-MEMORY bound, not DRAM bound: per row 192 nibble lookups of 32/16 B (5 KB/row), each
//    32-B entry fetched as 2 LDS.128.  A table chunk is 16 entries x 32 B = 512 B, so for a fixed half h all 16 entries
//    sit in only 4 of the 8 16-byte bank groups ((2v+h) mod 8): random nibbles from the 8 lanes of a quarter-warp
//    collide ~2.4x on average.  s26 measured upd0 3.45 ms = ~110 clk/row against ~40 clk/row of conflict-free traffic.
//  * inner256_v2/_db: thread per output BIT, 512 x 8 masked XORs (+ mask, loads) per data row: INTEGER-pipe bound
//    (~1.4 warp-instr/clk/SM measured, ~150 clk per data row; s26: 3.55 ms, 4.16 in the loop with the dense rows);
//    _db also walks the dense bits with dependent global loads in half of the threads of the tile-load phase.
// KERNELS HERE
//  upd256_v3   same Four-Russians nibble tables as upd256_v2 (D, E: all 4 words; D2: only words 2,3 -- XF=0, see
//              ks256v2.cu), stored as u64 planes T[m][chunk][word q][nibble v]: lane loads are LDS.64 and for a fixed q
//              the 16 entries of a chunk occupy 16 DISTINCT 8-byte bank pairs, so any mix of nibbles in a warp is
//              conflict-free (equal nibbles broadcast).  80 KB dynamic shared memory.
//  updF256_v3  the F term (pn ^= p2 F) in the same layout (32 KB), second pass when useF (as updF256).
//  inner256_v3 Four Russians over DATA rows (details at the kernel): warp bit-transposes turn each 32-row subtile into
//              per-output-row 32-bit patterns, B-row groups of 4 become 16-entry XOR tables, and an output row adds one
//              32-B table row per 4 data rows (8 B of conflict-free shared traffic + ~2.5 integer ops per data row
//              and output row).  A register-tiled LOP3 version (thread = 8 output rows, 64 accumulators) was tried
//              first: 1.15x at 1M rows -- inner256_v2 is bound by the INTEGER pipe (4096 masked XORs per data row at
//              ~1.4 warp-instr/clk/SM), not by its shared traffic, and register tiling does not remove a single LOP3.
//  inner256_v3db  the same with dense_bwd256 fused into the tile load (inner256_db's contract: B[r] ^= XOR of u[j]
//              over the set bits of dcol[r], written back), the u rows held in shared memory.
// MEASURED (s30dev kbench, n 14.35M, random data, CUDA events, alone on the GPU; DRAM floors at 1.34 TB/s in brackets):
//   inner+dense 3.76 -> 1.80 ms [0.86]   upd 3.50 -> 2.09 ms [2.06]   updF 1.50 -> 1.05 ms [1.03]
#define X4(D_,S_) { (D_).x^=(S_).x; (D_).y^=(S_).y; (D_).z^=(S_).z; (D_).w^=(S_).w; }
typedef unsigned long long u64;
typedef unsigned int u32;

// ---------------------------------------------------------------- update
// T layout (u64): D at 0 [64 c][4 q][16 v] (4096), E at 4096 (4096), X (D2 words 2,3) at 8192 [64 c][2][16] (2048).
// mats = [D | E | F | D2], 256 rows x 4 u64 each.
__device__ __forceinline__ u64 tab_word(const u64* __restrict__ M, const int c, const int q, const int v){
 u64 a=0ULL;
 #pragma unroll
 for(int j=0;j<4;j++) if((v>>j)&1) a^=M[4*(4*c+j)+q];
 return a;
}
__device__ __forceinline__ void nib4(const u64* __restrict__ T, const u64 w, const int c0, u64& y0, u64& y1, u64& y2, u64& y3){
 #pragma unroll
 for(int j=0;j<16;j++){
   const int v=(int)((w>>(4*j))&15ULL);
   const u64* e=T+(c0+j)*64+v;
   y0^=e[0]; y1^=e[16]; y2^=e[32]; y3^=e[48];
 }
}
__device__ __forceinline__ void nib4x(const u64* __restrict__ T, const u64* __restrict__ X, const u64 w, const int c0,
    u64& y0, u64& y1, u64& y2, u64& y3, u64& x2, u64& x3){
 #pragma unroll
 for(int j=0;j<16;j++){
   const int v=(int)((w>>(4*j))&15ULL);
   const u64* e=T+(c0+j)*64+v; const u64* f=X+(c0+j)*32+v;
   y0^=e[0]; y1^=e[16]; y2^=e[32]; y3^=e[48]; x2^=f[0]; x3^=f[16];
 }
}
// pn = (pn & mask) ^ p0*D ^ p1*E ; xb(words 2,3) ^= p0*D2     dynamic shared memory 10240*8 = 81920 B
extern "C" __global__ void __launch_bounds__(512,1) upd256_v3(const ulonglong4* __restrict__ p0,const ulonglong4* __restrict__ p1,
    ulonglong4* __restrict__ pn, ulonglong4* __restrict__ xb, const u64* __restrict__ mats,
    const u64* __restrict__ mask, long long n){
 extern __shared__ __align__(16) unsigned char sm_[];
 u64* T=(u64*)sm_;
 for(int e=threadIdx.x;e<10240;e+=blockDim.x){
   u64 t;
   if(e<8192){ const int m=e>>12, c=(e>>6)&63, q=(e>>4)&3, v=e&15; t=tab_word(mats+(m? 1024:0),c,q,v); }
   else { const int f=e-8192, c=f>>5, q=2+((f>>4)&1), v=f&15; t=tab_word(mats+3072,c,q,v); }
   T[e]=t;
 }
 const u64 m0=mask[0],m1=mask[1],m2=mask[2],m3=mask[3];
 __syncthreads();
 const u64* TD=T; const u64* TE=T+4096; const u64* TX=T+8192;
 for(long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x;r<n;r+=(long long)gridDim.x*blockDim.x){
   const ulonglong4 a=p0[r], b=p1[r], y=pn[r]; const ulonglong2 xh=((const ulonglong2*)xb)[2*r+1];
   u64 y0=y.x&m0, y1=y.y&m1, y2=y.z&m2, y3=y.w&m3, x2=xh.x, x3=xh.y;
   nib4x(TD,TX,a.x,0,y0,y1,y2,y3,x2,x3); nib4x(TD,TX,a.y,16,y0,y1,y2,y3,x2,x3);
   nib4x(TD,TX,a.z,32,y0,y1,y2,y3,x2,x3); nib4x(TD,TX,a.w,48,y0,y1,y2,y3,x2,x3);
   nib4(TE,b.x,0,y0,y1,y2,y3); nib4(TE,b.y,16,y0,y1,y2,y3); nib4(TE,b.z,32,y0,y1,y2,y3); nib4(TE,b.w,48,y0,y1,y2,y3);
   pn[r]=make_ulonglong4(y0,y1,y2,y3); ((ulonglong2*)xb)[2*r+1]=make_ulonglong2(x2,x3);
 }
}
// pn ^= p2*F     dynamic shared memory 4096*8 = 32768 B
extern "C" __global__ void __launch_bounds__(512) updF256_v3(const ulonglong4* __restrict__ p2, ulonglong4* __restrict__ pn,
    const u64* __restrict__ mats, long long n){
 extern __shared__ __align__(16) unsigned char sm_[];
 u64* T=(u64*)sm_;
 for(int e=threadIdx.x;e<4096;e+=blockDim.x) T[e]=tab_word(mats+2048,(e>>6)&63,(e>>4)&3,e&15);
 __syncthreads();
 for(long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x;r<n;r+=(long long)gridDim.x*blockDim.x){
   const ulonglong4 c=p2[r], y=pn[r]; u64 y0=y.x, y1=y.y, y2=y.z, y3=y.w;
   nib4(T,c.x,0,y0,y1,y2,y3); nib4(T,c.y,16,y0,y1,y2,y3); nib4(T,c.z,32,y0,y1,y2,y3); nib4(T,c.w,48,y0,y1,y2,y3);
   pn[r]=make_ulonglong4(y0,y1,y2,y3);
 }
}

// ---------------------------------------------------------------- inner products (Four Russians)
// o1 ^= A^T B, o2 ^= B^T B (both 256 x 4 u64; the caller zeroes them).  blockDim 256 (fixed), tile = 128 data rows.
// DB=1: B[r] ^= dense_bwd(dcol[r], u) first (written back when dcol[r] != 0), u = the first nd (<= 128) rows of u.
//  load      thread t < 128 holds A row t of the tile, t >= 128 B row t-128 (prefetched one tile ahead; dense rows
//            added to B here).  Warp s (A) / 4+s (B) holds the 32-row subtile s, lane l = row 32s+l.
//  patterns  each warp bit-transposes its 8 u32 words (32x32 shuffle butterflies): lane b of word w then holds the
//            32-bit pattern "bit (32w+b) of rows 32s..32s+31"  -> P[p][w][s][b]   (8 KB)
//  tables    B warps: lane l = row 4g'+k of group g = 8s+g' builds word q=k of the 16 XOR combinations of the group's
//            4 B rows -> T[g][q][v ^ 2g'] (16 KB; the ^2g' spreads the 4 builders of a group over the bank pairs)
//  lookups   thread (warp w, lane b) owns output row i = 32w+b of O1 and O2: per group a nibble of its P word picks one
//            32-B table row (4 LDS.64; for a fixed q the 16 entries sit in 16 distinct bank pairs -> conflict-free)
//            -> per data row and output row 8 B of shared traffic and 2.5 integer ops, vs 8 LOP3 + a mask.
template<int DB>
__device__ __forceinline__ void m4_fetch(const ulonglong4* __restrict__ A, const ulonglong4* __restrict__ B,
    const u64* __restrict__ dcol, long long r, long long n, int isB, ulonglong4& x, u64& d0, u64& d1){
 if(r<n){ x= isB? B[r] : A[r]; if(DB && isB){ d0=dcol[2*r]; d1=dcol[2*r+1]; } else { d0=0ULL; d1=0ULL; } }
 else { x=make_ulonglong4(0ULL,0ULL,0ULL,0ULL); d0=0ULL; d1=0ULL; }
}
__device__ __forceinline__ u32 tr32(u32 x, const int lane){
 #pragma unroll
 for(int j=16;j>0;j>>=1){
   const u32 m= j==16? 0x0000FFFFu : j==8? 0x00FF00FFu : j==4? 0x0F0F0F0Fu : j==2? 0x33333333u : 0x55555555u;
   const u32 y=__shfl_xor_sync(0xffffffffu,x,j);
   x= (lane&j)? ((x&~m)|((y&~m)>>j)) : ((x&m)|((y&m)<<j));
 }
 return x;
}
template<int DB>
__device__ __forceinline__ void m4_body(const ulonglong4* __restrict__ A, ulonglong4* __restrict__ B,
    u64* __restrict__ o1, u64* __restrict__ o2, long long n,
    const u64* __restrict__ dcol, const ulonglong4* __restrict__ u, int nd){
 __shared__ __align__(16) u64 T[32*64];
 __shared__ __align__(16) u32 P[2*8*4*32];
 __shared__ __align__(16) ulonglong4 su[DB? 128 : 1];
 const int t=threadIdx.x, warp=t>>5, lane=t&31, isB=t>>7, rr=t&127, s=(t>>5)&3;
 if(DB){ for(int j=t;j<nd;j+=blockDim.x) su[j]=u[j]; }
 u64 c10=0ULL,c11=0ULL,c12=0ULL,c13=0ULL,c20=0ULL,c21=0ULL,c22=0ULL,c23=0ULL;
 const long long step=(long long)gridDim.x*128;
 long long base=(long long)blockIdx.x*128;
 ulonglong4 nx; u64 nd0,nd1;
 m4_fetch<DB>(A,B,dcol,base+rr,n,isB,nx,nd0,nd1);
 if(DB) __syncthreads();
 for(; base<n; base+=step){
   ulonglong4 x=nx;
   if(DB && isB){
     u64 w0=nd0, w1=nd1;
     if(w0|w1){
       while(w0){ const int j=__ffsll((long long)w0)-1; const ulonglong4 tt=su[j]; X4(x,tt); w0&=w0-1; }
       while(w1){ const int j=__ffsll((long long)w1)+63; const ulonglong4 tt=su[j]; X4(x,tt); w1&=w1-1; }
       B[base+rr]=x;                              // nd0|nd1 != 0 only for rows < n
     }
   }
   if(base+step<n) m4_fetch<DB>(A,B,dcol,base+step+rr,n,isB,nx,nd0,nd1);   // next tile in flight
   __syncthreads();                               // the previous tile's lookups are done with P and T
   {  // patterns
     const u64 wd[4]={x.x,x.y,x.z,x.w};
     u32* Pp=P+((isB*8)*4+s)*32+lane;
     #pragma unroll
     for(int w=0;w<8;w++){
       const u32 v= (w&1)? (u32)(wd[w>>1]>>32) : (u32)wd[w>>1];
       Pp[w*4*32]=tr32(v,lane);
     }
   }
   if(isB){  // tables: word q=k of the group's 16 combinations
     const int k=lane&3, gl=lane>>2, src=lane&~3;
     u64 r0=0ULL,r1=0ULL,r2=0ULL,r3=0ULL;
     #pragma unroll
     for(int q=0;q<4;q++){
       const u64 mine= q==0? x.x : q==1? x.y : q==2? x.z : x.w;
       const u64 a0=__shfl_sync(0xffffffffu,mine,src), a1=__shfl_sync(0xffffffffu,mine,src+1),
                 a2=__shfl_sync(0xffffffffu,mine,src+2), a3=__shfl_sync(0xffffffffu,mine,src+3);
       if(q==k){ r0=a0; r1=a1; r2=a2; r3=a3; }
     }
     u64* Tq=T+(8*s+gl)*64+k*16; const int f=2*gl;
     const u64 e3=r0^r1, e5=r0^r2, e6=r1^r2, e7=e3^r2;
     Tq[0^f]=0ULL; Tq[1^f]=r0; Tq[2^f]=r1; Tq[3^f]=e3; Tq[4^f]=r2; Tq[5^f]=e5; Tq[6^f]=e6; Tq[7^f]=e7;
     Tq[8^f]=r3; Tq[9^f]=r3^r0; Tq[10^f]=r3^r1; Tq[11^f]=r3^e3; Tq[12^f]=r3^r2; Tq[13^f]=r3^e5; Tq[14^f]=r3^e6; Tq[15^f]=r3^e7;
   }
   __syncthreads();
   {  // lookups: output row 32*warp+lane
     const u32* Pa=P+(warp*4)*32+lane; const u32* Pb=P+((8+warp)*4)*32+lane;
     #pragma unroll
     for(int ss=0;ss<4;ss++){
       const u32 pa=Pa[ss*32], pb=Pb[ss*32];
       #pragma unroll
       for(int gg=0;gg<8;gg++){
         const u64* Tg=T+(8*ss+gg)*64;
         const int ia=(int)((pa>>(4*gg))&15u)^(2*gg), ib=(int)((pb>>(4*gg))&15u)^(2*gg);
         c10^=Tg[ia]; c11^=Tg[16+ia]; c12^=Tg[32+ia]; c13^=Tg[48+ia];
         c20^=Tg[ib]; c21^=Tg[16+ib]; c22^=Tg[32+ib]; c23^=Tg[48+ib];
       }
     }
   }
 }
 const int i=32*warp+lane;
 if(c10) atomicXor(&o1[4*i],c10); if(c11) atomicXor(&o1[4*i+1],c11); if(c12) atomicXor(&o1[4*i+2],c12); if(c13) atomicXor(&o1[4*i+3],c13);
 if(c20) atomicXor(&o2[4*i],c20); if(c21) atomicXor(&o2[4*i+1],c21); if(c22) atomicXor(&o2[4*i+2],c22); if(c23) atomicXor(&o2[4*i+3],c23);
}
extern "C" __global__ void __launch_bounds__(256,3) inner256_v3(const ulonglong4* A, const ulonglong4* B, u64* o1, u64* o2, long long n){
 m4_body<0>(A,(ulonglong4*)B,o1,o2,n,(const u64*)0,(const ulonglong4*)0,0);
}
extern "C" __global__ void __launch_bounds__(256,3) inner256_v3db(const ulonglong4* A, ulonglong4* B, u64* o1, u64* o2, long long n,
    const u64* dcol, const ulonglong4* u, int nd){
 m4_body<1>(A,B,o1,o2,n,dcol,u,nd);
}
