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

# segspmv.py -- segmented, hot-ideal-aware GF(2) SpMV for gpu_la (B = M^T and B^T), 64-bit (uint64)
# and 128-bit (ulonglong2) vector elements. CuPy + NVRTC, like gpu_la.py. Used by gpu_la.py when LA_SPMV=seg.
#
# ---- WHAT BOUNDS THIS SpMV (measured on the key-1 c160 matrix, 928,766,355 sparse nnz; floor.py/floor2.py) ----
#   index stream alone (int4 loads)                   2.61 ps/nnz   (1.53 TB/s)
#   gather x[col] in B CSR order (64-bit)             7.52 ps/nnz   <- L2 random-sector bound
#   gather in B^T order (64-bit)                      6.83 ps/nnz
#   gather in B order, 128-bit                       17.0  ps/nnz   (x = 155 MB > 128 MB L2)
#   gather in B order, 128-bit, x in 2 column parts   7.77 ps/nnz   (each part's x slice L2-resident)
#   gather in B^T order, 128-bit                      7.31 ps/nnz
#   gather from an L1-resident 4096-entry x           2.62 ps/nnz   (= the index-stream floor)
#   global atomicXor, random                         23-99 ps/nnz   (so scatter designs accumulate in smem)
# The "2.67 ps/nnz raw gather" of la128/gatherbench.py gathered x[C] with the loader's C array, which is
# non-decreasing -- a streaming read (2.67 ps reproduced here as 'Csorted'), not the SpMV's access pattern.
# So the SpMV is bound by random L2 sector reads; the only levers are fewer L2 gathers (hot ideals from
# shared memory / L1, scatter for the hot rows of B), L2-resident x slices, and zero overhead on top of the
# gathers (coalesced index loads, no idle lanes on B's short rows: median B row = 21 nonzeros).
#
# ---- DESIGN ---------------------------------------------------------------------------------------------
# Ideals (B rows) are numbered by decreasing weight; the ND densest are gpu_la's dense rows (dcol). The next
# H ideals [hlo=ND, ND+H) are HOT: 38.3% of the sparse nonzeros for H=12,672 (64-bit, 99 KB of shared memory)
# and 33.6% for H=6,336 (128-bit, 99 KB).
#   B  (y[ideal] = XOR x[relset]):
#        cold rows = segmented GATHER over B's CSR restricted to cold ideals; at 128-bit in P column (relset)
#        parts, each part's x slice <= LA_SEG_SPLIT_MB (96 MB) so it stays L2-resident; part>0 XOR-accumulates.
#        hot rows = SCATTER over the hot entries in B^T order: x[relset] is read (near-)sequentially and XORed
#        into a per-block shared-memory y_hot with native 32-bit ATOMS.XOR (64-bit shared atomicXor compiles
#        to a CAS spin loop on sm_120); blocks write partials, seg_fix combines them into y[hlo:hlo+H).
#   Bt (y[relset] = XOR x[ideal]): ONE segmented gather over all of B^T; ideals [hlo, hlo+Hbt) are read from a
#        per-block shared-memory copy of x. Hbt = H at 64-bit; 3,584 at 128-bit (56 KB): a window above 63 KB
#        forces the 100 KB shared carve-out and leaves ~28 KB of L1, which costs more than the hits buy.
# SEGMENTED GATHER (seg4_*): row-sorted nonzeros in chunks of 1024 = 8 steps x 32 lanes x 4; lane j owns the
# 4 consecutive elements base+128*step+4j..+3 (one coalesced int4 index load). A head bitmask (1 bit/nonzero:
# "first element of its row") gives each lane its 4 head bits; the lane XORs its elements sequentially (rows
# that start and end inside the lane are stored directly), then ONE warp segmented XOR-scan per 128 elements
# joins the lane partials (7 shuffles of T per 128 elements; the per-step-scan variant, LA_SEG_SIGMA=1,
# needs 6 per 32). Every stream's rows are made
# contiguous (a dummy entry -- index -1 / hot offset 0xFFFF -- for each row without entries), so a row's output
# index is ordinal+rbase: no row map, whose dependent load before every store stalled the warp (-11% on B^T).
# A row that started in an earlier chunk (at most one per chunk: its first row) is written to cin[chunk];
# seg_fix atomically XORs those into y (0.02-0.08 ms). Every other row is plain-stored by the chunk that owns
# its first element. All results are XORs of the same multiset of words -> bit-identical to the production
# kernels (bench.py checks every row for random x; the c111 LA writes a byte-identical kernel file).
#
# ---- MEASURED (c160 key 1, LA_PROF=2 events inside gpu_la's loop, incl. dense-row kernels) ----------------
#   LA_BLOCK=64 : Bmul 7.46 -> 5.34 ms, Btmul 6.31 -> 5.10 ms, 15.76 -> 12.43 ms/iteration
#   LA_BLOCK=128: Bmul 10.46 -> 6.02 ms, Btmul 9.38 -> 6.11 ms, 23.54 -> 15.82 ms/iteration (vs the dev
#                 column-split path); per 64 Lanczos dimensions 15.76 (64, production) -> 7.91 ms (128, seg)
#   (OPENBLAS_NUM_THREADS=1; isolated products: bench.py -- 64-bit B 8.13 -> 5.64, B^T 7.48 -> 5.62 ps/nnz;
#    128-bit B 11.46 -> 6.54, B^T 10.61 -> 6.38 ps/nnz, all rows bit-exact vs the production kernels)
# ---- KNOBS (env) ------------------------------------------------------------------------------------------
#   LA_SEG_H (hot window, default max = 99 KB / element size), LA_SEG_HBT (B^T smem window), LA_SEG_P /
#   LA_SEG_SPLIT_MB (128-bit column parts; default ceil(n*16 B / 96 MB)), LA_SEG_SIGMA (4; 1 = per-step scan
#   variant, slower), LA_SEG_U64/LA_SEG_U128 (unroll), LA_SEG_BPS (gather blocks/SM override).
import os, sys, time
import numpy as np
import cupy as cp

KSEG = r'''
typedef unsigned long long u64;
#define FULLM 0xffffffffu
#define KS 32
#define CH 1024
template<typename T> __device__ __forceinline__ T vz();
template<> __device__ __forceinline__ u64 vz<u64>(){ return 0ULL; }
template<> __device__ __forceinline__ ulonglong2 vz<ulonglong2>(){ return make_ulonglong2(0ULL,0ULL); }
__device__ __forceinline__ void vx(u64& a, const u64 b){ a^=b; }
__device__ __forceinline__ void vx(ulonglong2& a, const ulonglong2 b){ a.x^=b.x; a.y^=b.y; }
__device__ __forceinline__ u64 vup(const u64 v, int d){ return __shfl_up_sync(FULLM, v, d); }
__device__ __forceinline__ ulonglong2 vup(const ulonglong2 v, int d){ return make_ulonglong2(__shfl_up_sync(FULLM, v.x, d), __shfl_up_sync(FULLM, v.y, d)); }
__device__ __forceinline__ u64 vbc(const u64 v, int l){ return __shfl_sync(FULLM, v, l); }
__device__ __forceinline__ ulonglong2 vbc(const ulonglong2 v, int l){ return make_ulonglong2(__shfl_sync(FULLM, v.x, l), __shfl_sync(FULLM, v.y, l)); }
__device__ __forceinline__ bool vnz(const u64 v){ return v!=0ULL; }
__device__ __forceinline__ bool vnz(const ulonglong2 v){ return (v.x|v.y)!=0ULL; }
__device__ __forceinline__ void gatomic(u64* p, const u64 v){ atomicXor(p, v); }
__device__ __forceinline__ void gatomic(ulonglong2* p, const ulonglong2 v){ atomicXor(&p->x, v.x); atomicXor(&p->y, v.y); }
// shared-memory XOR through native 32-bit ATOMS.XOR (XOR is bitwise, so two halves == one 64-bit XOR)
__device__ __forceinline__ void satomic(u64* p, const u64 v){ unsigned* q=(unsigned*)p; atomicXor(q,(unsigned)v); atomicXor(q+1,(unsigned)(v>>32)); }
__device__ __forceinline__ void satomic(ulonglong2* p, const ulonglong2 v){ unsigned* q=(unsigned*)p;
  atomicXor(q,(unsigned)v.x); atomicXor(q+1,(unsigned)(v.x>>32)); atomicXor(q+2,(unsigned)v.y); atomicXor(q+3,(unsigned)(v.y>>32)); }

// write a finished (or chunk-final) row value: continuation rows -> cin[c]; owned rows -> y (store or XOR)
template<typename T> __device__ __forceinline__ void put(T* __restrict__ y, T* __restrict__ cin, int c, bool cont, int ord,
    const int* __restrict__ rmap, int rbase, int acc, const T v){
  if(cont){ cin[c]=v; return; }
  const int r = rmap ? __ldg(rmap+ord) : ord+rbase;
  if(acc){ T t=y[r]; vx(t,v); y[r]=t; } else y[r]=v;
}

// Segmented gather: y[row] = XOR over the row's elements of x[idx[e]] (hot window from shared memory).
// hb: head bitmask, KS words per chunk; crow[c]: ordinal of the row holding element c*CH.
template<typename T, int U, int HOT>
__device__ __forceinline__ void seg_gather_body(const T* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, T* __restrict__ y, T* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, int hlo, int H, const int* __restrict__ zl, int nz, const T* __restrict__ sx)
{
  const int gt=blockIdx.x*blockDim.x+threadIdx.x, gs=gridDim.x*blockDim.x;
  for(int t=gt;t<nz;t+=gs) y[zl[t]]=vz<T>();
  const int lane=threadIdx.x&31;
  const unsigned lmle=FULLM>>(31-lane);                       // lanes 0..lane
  for(int c=gt>>5;c<nchunk;c+=gs>>5){
    const long long base=(long long)c*CH;
    const unsigned hw=__ldcs(hb+(long long)c*KS+lane);        // lane s holds the head word of step s
    const unsigned h0=__shfl_sync(FULLM,hw,0);
    int ro=__ldg(crow+c)-(int)(h0&1u);                        // ordinal of the row open before element base
    bool cont=!(h0&1u);                                       // that row started in an earlier chunk
    T carry=vz<T>();
    #pragma unroll 1
    for(int s0=0;s0<KS;s0+=U){
      int ci[U]; T v[U];
      #pragma unroll
      for(int u=0;u<U;u++){ const long long e=base+(long long)(s0+u)*32+lane; ci[u]=(e<nnz)? __ldcs(idx+e) : -1; }
      #pragma unroll
      for(int u=0;u<U;u++){
        T t=vz<T>(); const int cc=ci[u];
        if(HOT){ const unsigned o=(unsigned)(cc-hlo); if(o<(unsigned)H) t=sx[o]; else if(cc>=0) t=x[cc]; }
        else { if(cc>=0) t=x[cc]; }
        v[u]=t;
      }
      #pragma unroll
      for(int u=0;u<U;u++){
        const unsigned hs=__shfl_sync(FULLM,hw,s0+u);
        T w=v[u];
        if(hs&1u){ if(s0+u>0 && lane==0) put(y,cin,c,cont,ro,rmap,rbase,acc,carry); cont=false; }
        else if(lane==0) vx(w,carry);
        const unsigned hm=hs&lmle;
        const int ss= hm ? 31-__clz(hm) : 0;
        #pragma unroll
        for(int d=1;d<32;d<<=1){ const T t=vup(w,d); if(lane-d>=ss) vx(w,t); }
        if(lane<31 && ((hs>>(lane+1))&1u)) put(y,cin,c,(hm==0u)&&cont,ro+__popc(hm),rmap,rbase,acc,w);
        carry=vbc(w,31);
        const int np=__popc(hs); ro+=np; if(np) cont=false;
      }
    }
    if(lane==0) put(y,cin,c,cont,ro,rmap,rbase,acc,carry);
  }
}
template<typename T, int U>
__global__ void __launch_bounds__(256) seg_gather(const T* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, T* __restrict__ y, T* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, const int* __restrict__ zl, int nz)
{ seg_gather_body<T,U,0>(x,idx,hb,crow,nnz,nchunk,y,cin,rmap,rbase,acc,0,0,zl,nz,(const T*)0); }
template<typename T, int U>
__global__ void __launch_bounds__(1024,1) seg_gather_hot(const T* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, T* __restrict__ y, T* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, int hlo, int H, const int* __restrict__ zl, int nz)
{
  extern __shared__ __align__(16) unsigned char sm_[];
  T* sx=(T*)sm_;
  for(int t=threadIdx.x;t<H;t+=blockDim.x) sx[t]=x[hlo+t];
  __syncthreads();
  seg_gather_body<T,U,1>(x,idx,hb,crow,nnz,nchunk,y,cin,rmap,rbase,acc,hlo,H,zl,nz,sx);
}
// Scatter of the HOT part of B: for every hot entry (ideal hlo+hidx[e], relset rmap[ord(e)]) in B^T order,
// y_hot[hidx[e]] ^= x[relset], accumulated in shared memory; block partials -> part[blockIdx][0..H).
template<typename T>
__global__ void __launch_bounds__(1024,1) seg_scatter(const T* __restrict__ x, const unsigned short* __restrict__ hidx,
    const unsigned* __restrict__ hb, const int* __restrict__ crow, const int* __restrict__ rmap, int rbase,
    long long nnz, int nchunk, int H, T* __restrict__ part)
{
  extern __shared__ __align__(16) unsigned char sm_[];
  T* sy=(T*)sm_;
  for(int t=threadIdx.x;t<H;t+=blockDim.x) sy[t]=vz<T>();
  __syncthreads();
  const int lane=threadIdx.x&31; const unsigned lmle=FULLM>>(31-lane);
  const int gt=blockIdx.x*blockDim.x+threadIdx.x, gs=gridDim.x*blockDim.x;
  for(int c=gt>>5;c<nchunk;c+=gs>>5){
    const long long base=(long long)c*CH;
    const unsigned hw=__ldcs(hb+(long long)c*KS+lane);
    int ro=__ldg(crow+c)-(int)(__shfl_sync(FULLM,hw,0)&1u);
    #pragma unroll 4
    for(int s=0;s<KS;s++){
      const unsigned hs=__shfl_sync(FULLM,hw,s);
      const long long e=base+(long long)s*32+lane;
      if(e<nnz){
        const int o=ro+__popc(hs&lmle);
        const int r= rmap ? __ldg(rmap+o) : o+rbase;
        const unsigned h=__ldcs(hidx+e);
        if(h!=0xFFFFu) satomic(sy+h, __ldg(x+r));             // 0xFFFF = dummy entry of a row with no hot ideal
      }
      ro+=__popc(hs);
    }
  }
  __syncthreads();
  for(int t=threadIdx.x;t<H;t+=blockDim.x) part[(long long)blockIdx.x*H+t]=sy[t];
}
// ============ SIGMA=4: 4 consecutive elements per lane per step (128-element steps, 8 steps per chunk) ============
// Lane j owns elements base+128*st+4j..+3 (one int4 index load). In-lane: sequential XOR with the lane's 4 head
// bits (rows that start AND end inside the lane are stored directly). Cross-lane: one segmented scan per
// 128 elements over s_j = (lane has a head) ? L_j (its last in-lane row) : F_j (all of it), so a head lane m
// closes the row open before it with value scan[m-1] ^ F_m. Same chunk/cin/ordinal scheme as seg_gather;
// a row is sent to cin[c] iff its ordinal is the chunk's continuation ordinal.
template<typename T, int HOT>
__device__ __forceinline__ T gat(const T* __restrict__ x, const T* __restrict__ sx, int cc, int hlo, int H){
  T t=vz<T>();
  if(HOT){ const unsigned o=(unsigned)(cc-hlo); if(o<(unsigned)H) t=sx[o]; else if(cc>=0) t=x[cc]; }
  else { if(cc>=0) t=x[cc]; }
  return t;
}
__device__ __forceinline__ int4 ld4(const int* __restrict__ idx, long long e4, long long nnz){
  if(e4+3<nnz) return __ldcs((const int4*)(idx+e4));
  int4 r; r.x= e4<nnz? idx[e4]:-1; r.y= e4+1<nnz? idx[e4+1]:-1; r.z= e4+2<nnz? idx[e4+2]:-1; r.w= e4+3<nnz? idx[e4+3]:-1; return r;
}
template<typename T> __device__ __forceinline__ void put4(T* __restrict__ y, T* __restrict__ cin, int c, int cont_ord, int ord,
    const int* __restrict__ rmap, int rbase, int acc, const T v){
  if(ord==cont_ord){ cin[c]=v; return; }
  const int r = rmap ? __ldg(rmap+ord) : ord+rbase;
  if(acc){ T t=y[r]; vx(t,v); y[r]=t; } else y[r]=v;
}
template<typename T, int U, int HOT>
__device__ __forceinline__ void seg4_body(const T* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, T* __restrict__ y, T* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, int hlo, int H, const int* __restrict__ zl, int nz, const T* __restrict__ sx)
{
  const int gt=blockIdx.x*blockDim.x+threadIdx.x, gs=gridDim.x*blockDim.x;
  for(int t=gt;t<nz;t+=gs) y[zl[t]]=vz<T>();
  const int lane=threadIdx.x&31;
  const unsigned lmle=FULLM>>(31-lane), lmlt=lmle>>1;
  const int nsh=(lane&7)*4;
  for(int c=gt>>5;c<nchunk;c+=gs>>5){
    const long long base=(long long)c*CH;
    const unsigned hw=__ldcs(hb+(long long)c*KS+lane);
    const unsigned h0=__shfl_sync(FULLM,hw,0)&1u;
    const int c0=__ldg(crow+c);
    const int cont_ord= h0 ? -1 : c0;
    int ro=c0-(int)h0;
    T carry=vz<T>();
    #pragma unroll 1
    for(int s0=0;s0<8;s0+=U){
      int4 ci[U];
      #pragma unroll
      for(int u=0;u<U;u++) ci[u]=ld4(idx,base+(long long)(s0+u)*128+4*lane,nnz);
      T v[U][4];
      #pragma unroll
      for(int u=0;u<U;u++){ v[u][0]=gat<T,HOT>(x,sx,ci[u].x,hlo,H); v[u][1]=gat<T,HOT>(x,sx,ci[u].y,hlo,H);
                            v[u][2]=gat<T,HOT>(x,sx,ci[u].z,hlo,H); v[u][3]=gat<T,HOT>(x,sx,ci[u].w,hlo,H); }
      #pragma unroll
      for(int u=0;u<U;u++){
        const int st=s0+u;
        const unsigned nib=(__shfl_sync(FULLM,hw,4*st+(lane>>3))>>nsh)&15u;
        const unsigned b0=__ballot_sync(FULLM,nib&1u), b1=__ballot_sync(FULLM,nib&2u), b2=__ballot_sync(FULLM,nib&4u), b3=__ballot_sync(FULLM,nib&8u);
        const int hbb=__popc(b0&lmlt)+__popc(b1&lmlt)+__popc(b2&lmlt)+__popc(b3&lmlt);
        const unsigned HL=b0|b1|b2|b3;
        int ordl=ro+hbb; T F=vz<T>(), L=vz<T>();
        #pragma unroll
        for(int k=0;k<4;k++){
          const bool before=(nib&((1u<<k)-1u))!=0u;
          if((nib>>k)&1u){ if(before) put4(y,cin,c,cont_ord,ordl,rmap,rbase,acc,L); ordl++; L=v[u][k]; }
          else if(before) vx(L,v[u][k]); else vx(F,v[u][k]);
        }
        const bool hh=nib!=0u;
        T sv= hh ? L : F;
        if(lane==0 && !hh) vx(sv,carry);
        const unsigned hm=HL&lmle; const int ss= hm ? 31-__clz(hm) : 0;
        #pragma unroll
        for(int d=1;d<32;d<<=1){ const T w=vup(sv,d); if(lane-d>=ss) vx(sv,w); }
        T prev=vup(sv,1); if(lane==0) prev=carry;
        if(hh && !(st==0 && lane==0 && h0)){ vx(prev,F); put4(y,cin,c,cont_ord,ro+hbb,rmap,rbase,acc,prev); }
        carry=vbc(sv,31);
        ro+=__popc(b0)+__popc(b1)+__popc(b2)+__popc(b3);
      }
    }
    if(lane==0) put4(y,cin,c,cont_ord,ro,rmap,rbase,acc,carry);
  }
}
template<typename T, int U>
__global__ void __launch_bounds__(256) seg4_gather(const T* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, T* __restrict__ y, T* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, const int* __restrict__ zl, int nz)
{ seg4_body<T,U,0>(x,idx,hb,crow,nnz,nchunk,y,cin,rmap,rbase,acc,0,0,zl,nz,(const T*)0); }
template<typename T, int U>
__global__ void __launch_bounds__(1024,1) seg4_gather_hot(const T* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, T* __restrict__ y, T* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, int hlo, int H, const int* __restrict__ zl, int nz)
{
  extern __shared__ __align__(16) unsigned char sm_[];
  T* sx=(T*)sm_;
  for(int t=threadIdx.x;t<H;t+=blockDim.x) sx[t]=x[hlo+t];
  __syncthreads();
  seg4_body<T,U,1>(x,idx,hb,crow,nnz,nchunk,y,cin,rmap,rbase,acc,hlo,H,zl,nz,sx);
}
template<typename T>
__global__ void __launch_bounds__(1024,1) seg4_scatter(const T* __restrict__ x, const unsigned short* __restrict__ hidx,
    const unsigned* __restrict__ hb, const int* __restrict__ crow, const int* __restrict__ rmap, int rbase,
    long long nnz, int nchunk, int H, T* __restrict__ part)
{
  extern __shared__ __align__(16) unsigned char sm_[];
  T* sy=(T*)sm_;
  for(int t=threadIdx.x;t<H;t+=blockDim.x) sy[t]=vz<T>();
  __syncthreads();
  const int lane=threadIdx.x&31; const unsigned lmlt=(FULLM>>(31-lane))>>1; const int nsh=(lane&7)*4;
  const int gt=blockIdx.x*blockDim.x+threadIdx.x, gs=gridDim.x*blockDim.x;
  for(int c=gt>>5;c<nchunk;c+=gs>>5){
    const long long base=(long long)c*CH;
    const unsigned hw=__ldcs(hb+(long long)c*KS+lane);
    int ro=__ldg(crow+c)-(int)(__shfl_sync(FULLM,hw,0)&1u);
    #pragma unroll 2
    for(int st=0;st<8;st++){
      const long long e4=base+(long long)st*128+4*lane;
      unsigned hx, hy; int nv;
      if(e4+3<nnz){ const uint2 hv=__ldcs((const uint2*)(hidx+e4)); hx=hv.x; hy=hv.y; nv=4; }
      else { nv= e4<nnz ? (int)(nnz-e4) : 0; hx= (nv>0? hidx[e4]:0u) | ((nv>1? (unsigned)hidx[e4+1]:0u)<<16); hy= (nv>2? hidx[e4+2]:0u) | ((nv>3? (unsigned)hidx[e4+3]:0u)<<16); }
      const unsigned nib=(__shfl_sync(FULLM,hw,4*st+(lane>>3))>>nsh)&15u;
      const unsigned b0=__ballot_sync(FULLM,nib&1u), b1=__ballot_sync(FULLM,nib&2u), b2=__ballot_sync(FULLM,nib&4u), b3=__ballot_sync(FULLM,nib&8u);
      int ord=ro+__popc(b0&lmlt)+__popc(b1&lmlt)+__popc(b2&lmlt)+__popc(b3&lmlt)+(int)(nib&1u);
      if(nv>0){
        T xv=__ldg(x+(rmap ? __ldg(rmap+ord) : ord+rbase));
        if((hx&0xffffu)!=0xFFFFu) satomic(sy+(hx&0xffffu),xv);   // 0xFFFF = dummy entry (row with no hot ideal)
        #pragma unroll
        for(int k=1;k<4;k++){
          if(k<nv){
            if((nib>>k)&1u){ ord++; xv=__ldg(x+(rmap ? __ldg(rmap+ord) : ord+rbase)); }
            const unsigned h=(k==1? (hx>>16) : k==2? (hy&0xffffu) : (hy>>16));
            if(h!=0xFFFFu) satomic(sy+h,xv);
          }
        }
      }
      ro+=__popc(b0)+__popc(b1)+__popc(b2)+__popc(b3);
    }
  }
  __syncthreads();
  for(int t=threadIdx.x;t<H;t+=blockDim.x) part[(long long)blockIdx.x*H+t]=sy[t];
}
// Fix-up: y[hlo+t] = XOR_b part[b][t] (hot combine, nblk>0), then for
// every chunk c with a continuation row: y[tgt[c]] ^= cin[c] (atomic: a long row spans many chunks).
// tgt[c] = actual output row of chunk c's continuation row, or -1 (precomputed by mk_tgt).
template<typename T>
__global__ void seg_fix(T* __restrict__ y, const T* __restrict__ cin, const int* __restrict__ tgt, int nt,
    const T* __restrict__ part, int nblk, int H, int hlo)
{
  const int gt=blockIdx.x*blockDim.x+threadIdx.x, gs=gridDim.x*blockDim.x;
  if(nblk>0) for(int t=gt;t<H;t+=gs){ T a=vz<T>(); for(int b=0;b<nblk;b++) vx(a,part[(long long)b*H+t]); y[hlo+t]=a; }
  for(int c=gt;c<nt;c+=gs){ const int r=tgt[c]; if(r>=0){ const T v=cin[c]; if(vnz(v)) gatomic(y+r,v); } }
}
extern "C" __global__ void mk_tgt(const unsigned* __restrict__ hb, const int* __restrict__ crow, int nch,
    const int* __restrict__ rmap, int rbase, int* __restrict__ tgt){
  for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<nch;c+=gridDim.x*blockDim.x){
    const int o=crow[c]; tgt[c]= (hb[(long long)c*KS]&1u) ? -1 : (rmap ? rmap[o] : o+rbase); } }
// ---- build helpers ----
// mark the rows present in a row-sorted stream: pres[rows[e]-rlo] = 1 at every row start
extern "C" __global__ void mark_rows(const int* __restrict__ rows, long long nnz, int rlo, unsigned char* __restrict__ pres){
  for(long long e=(long long)blockIdx.x*blockDim.x+threadIdx.x;e<nnz;e+=(long long)gridDim.x*blockDim.x)
    if(e==0 || rows[e]!=rows[e-1]) pres[rows[e]-rlo]=1; }
'''
CH = 1024; KS = 32
SMEM_MAX = int(os.environ.get("LA_SEG_SMEM", "0") or 0)    # 0 = the device's opt-in max dynamic smem per block
_NULL = np.uint64(0)
_mods = {}


OPTS = None     # NVRTC options (include dirs); gpu_la.py sets segspmv.OPTS = OPT before building a plan


def _opts():
    if OPTS is not None:
        return tuple(OPTS) + ('-std=c++17',)
    import glob
    incs = []
    try:
        import nvidia
        for p in list(nvidia.__path__): incs += glob.glob(os.path.join(p, "*", "include"))
    except Exception:
        pass
    return tuple("-I" + d for d in incs if os.path.isdir(d)) + ('-std=c++17',)


def _mod(U64, U128):
    key = (U64, U128)
    if key not in _mods:
        names = []
        for T, U in (('u64', U64), ('ulonglong2', U128)):
            names += [f'seg_gather<{T},{U}>', f'seg_gather_hot<{T},{U}>', f'seg_scatter<{T}>', f'seg_fix<{T}>']
            u4 = max(1, min(4, U // 4))
            names += [f'seg4_gather<{T},{u4}>', f'seg4_gather_hot<{T},{u4}>', f'seg4_scatter<{T}>']
        _mods[key] = cp.RawModule(code=KSEG, options=_opts(), name_expressions=names)
    return _mods[key]


def _sms():
    return cp.cuda.Device().attributes['MultiProcessorCount']


SENT32 = 0xFFFFFFFF


def _pad_rows(key, rlo, rhi):
    """key: sorted uint64 (row<<32 | low32), every row in [rlo,rhi). Adds ONE dummy entry (row, SENT32) for each
    row of [rlo,rhi) that has none, so the stream's rows are exactly rlo..rhi-1: row = ordinal + rlo, no row map
    (a map costs a dependent load before every row store, which stalls the warp's in-order issue).
    A dummy gathers index -1 (-> 0) / hot offset 0xFFFF (-> no atomic): it only makes its row get written 0."""
    nnz = int(key.size)
    pres = cp.zeros(max(0, rhi - rlo), dtype=cp.uint8)
    if nnz:
        rows = (key >> cp.uint64(32)).astype(cp.int32)
        _mod(8, 4).get_function('mark_rows')((4096,), (256,), (rows, np.int64(nnz), np.int32(rlo), pres)); del rows
    miss = cp.flatnonzero(pres == 0)
    if miss.size:
        dk = ((miss.astype(cp.uint64) + cp.uint64(rlo)) << cp.uint64(32)) | cp.uint64(SENT32)
        key = cp.concatenate([key, dk]); key.sort()
    return key, int(miss.size)


class Stream:
    """One segmented element stream over rows rlo..rhi-1 (all present; ordinal = row - rlo): idx[e] = gathered
    index (int32, -1 = dummy) or hot offset (uint16, 0xFFFF = dummy); rows non-decreasing."""
    def __init__(self, d_rows, d_idx, rlo, rhi, ndummy=0, name=""):
        t = time.time()
        nnz = int(d_rows.size); self.nnz = nnz; self.name = name; self.ndummy = ndummy
        self.nch = max(1, -(-nnz // CH)); npad = self.nch * CH
        head = cp.zeros(npad, dtype=cp.bool_)
        if nnz:
            head[0] = True
            if nnz > 1: cp.not_equal(d_rows[1:], d_rows[:-1], out=head[1:nnz])
        self.hb = cp.packbits(head, bitorder='little').view(cp.uint32)       # KS words per chunk
        assert self.hb.size == self.nch * KS
        if nnz:
            nrow = int(cp.count_nonzero(head))
            if nrow != rhi - rlo or int(d_rows[0]) != rlo or int(d_rows[-1]) != rhi - 1:
                raise RuntimeError(f"seg stream {name}: rows not exactly [{rlo},{rhi}) ({nrow} rows)")
            self.crow = (d_rows[::CH] - rlo).astype(cp.int32)
        else:
            self.crow = cp.zeros(1, cp.int32)
        del head
        self.nord = rhi - rlo if nnz else 0
        self.rmap = None; self.rbase = rlo
        self.idx = d_idx
        self.cin = None
        self.tgt = cp.empty(self.nch, cp.int32)
        if nnz:
            _mod(8, 4).get_function('mk_tgt')((max(1, min(4096, -(-self.nch // 256))),), (256,),
                                              (self.hb, self.crow, np.int32(self.nch), _NULL, np.int32(self.rbase), self.tgt))
        else:
            self.tgt[:] = -1
        self.bt = time.time() - t

    def rmap_arg(self):
        return self.rmap if self.rmap is not None else _NULL


def _key_stream(key, rlo, rhi, off, kind, name):
    """sorted (row<<32|col) keys of rows in [rlo,rhi) -> padded Stream; idx = col - off (int32, -1 dummy) or
    uint16 hot offset (0xFFFF dummy)."""
    key, nd = _pad_rows(key, rlo, rhi)
    rows = (key >> cp.uint64(32)).astype(cp.int32)
    low = (key & cp.uint64(0xffffffff)).astype(cp.uint32); del key
    dm = low == cp.uint32(SENT32)
    if kind == 'i32':
        idx = cp.where(dm, cp.int32(-1), (low - cp.uint32(off)).astype(cp.int32))
    else:
        idx = cp.where(dm, cp.uint16(0xFFFF), (low - cp.uint32(off)).astype(cp.uint16))
    del low, dm
    s = Stream(rows, idx, rlo, rhi, nd, name); del rows
    return s


def _zero_list(n, covered):
    """rows in [0,n) not in any of the covered [a,b) ranges."""
    pres = cp.zeros(n, dtype=cp.bool_)
    for a_, b_ in covered:
        if b_ > a_: pres[a_:b_] = True
    return cp.flatnonzero(~pres).astype(cp.int32)


class SegPlan:
    """y = B*x and y = B^T*x (sparse part only; all n output rows written) for gpu_la's matrix:
    d_R[k] = B row (ideal) and d_C[k] = B column (relset) of sparse nonzero k, d_C non-decreasing (file order).
    width 64 -> uint64 vectors (n,); width 128 -> (n,2) uint64 vectors (ulonglong2)."""
    def __init__(self, d_R, d_C, n, ND, width=64, H=None, P=None, verbose=True):
        t0 = time.time()
        assert width in (64, 128)
        self.n = n; self.ND = ND; self.width = width; self.T = 'u64' if width == 64 else 'ulonglong2'
        self.esz = 8 if width == 64 else 16
        self.nnz = int(d_R.size)
        U64 = int(os.environ.get("LA_SEG_U64", "8")); U128 = int(os.environ.get("LA_SEG_U128", "4"))
        self.mod = _mod(U64, U128)
        Ug = U64 if width == 64 else U128
        self.sigma = int(os.environ.get("LA_SEG_SIGMA", "4"))
        if self.sigma == 4:
            u4 = max(1, min(4, Ug // 4))
            self.k_gather = self.mod.get_function(f'seg4_gather<{self.T},{u4}>')
            self.k_ghot = self.mod.get_function(f'seg4_gather_hot<{self.T},{u4}>')
            self.k_scatter = self.mod.get_function(f'seg4_scatter<{self.T}>')
        else:
            self.k_gather = self.mod.get_function(f'seg_gather<{self.T},{Ug}>')
            self.k_ghot = self.mod.get_function(f'seg_gather_hot<{self.T},{Ug}>')
            self.k_scatter = self.mod.get_function(f'seg_scatter<{self.T}>')
        self.k_fix = self.mod.get_function(f'seg_fix<{self.T}>')
        smem = SMEM_MAX
        if smem <= 0:
            try: smem = int(cp.cuda.Device().attributes['MaxSharedMemoryPerBlockOptin'])
            except Exception: smem = 101376      # sm_120: 99 KB
        self.smem = smem
        Hmax = smem // self.esz
        if H is None: H = int(os.environ.get("LA_SEG_H", str(Hmax)))
        self.hlo = ND; self.H = max(0, min(int(H), Hmax, 65535, n - ND))
        # B^T shared-memory hot window (the B^T gather kernel's smem also sets the L1/shared carve-out: the unified
        # 128 KB/SM splits into smem + L1, and smem > 63 KB (+1 KB per-block reserve) jumps to the 100 KB carve-out,
        # leaving ~28 KB of L1). Measured on c160, B^T product ms (bench.py, median of 7-9):
        #   64-bit : window 4096 5.65 | 6144 5.44 | 8064 5.41 | 8192 5.70 | 10240 5.42 | 12672 (99 KB) 5.21  -> max
        #   128-bit: none 6.98 | 1024 6.42 | 2048 6.22 | 2560 6.06 | 3168 5.98 | 3584 5.86-5.96 | 4032 5.94 |
        #            4096 (64 KB+1 KB -> 100 KB carve-out) 8.64 | 6336 (99 KB) 8.51   -> 3584 (56 KB), below the cliff
        hbt = os.environ.get("LA_SEG_HBT", "")
        self.Hbt = (self.H if width == 64 else max(0, min(3584, n - ND))) if hbt == "" else max(0, min(int(hbt), Hmax, n - ND))
        if P is None:
            P = int(os.environ.get("LA_SEG_P", "0") or 0)
            if P <= 0:
                split_mb = float(os.environ.get("LA_SEG_SPLIT_MB", "96"))
                P = 1 if width == 64 else max(1, -(-(n * self.esz) // int(split_mb * 2**20)))
        self.P = P
        self.sms = _sms()
        hlo, H = self.hlo, self.H
        if H > 65534: raise ValueError("hot window must be < 65535 (0xFFFF marks a dummy entry)")
        MAXK = cp.uint64(0xFFFFFFFFFFFFFFFF)
        hot = (d_R >= hlo) & (d_R < hlo + H) if H else None
        # ---- Bt: all entries (rows = relsets 0..n-1, gathers ideals). File order already groups by relset; it is
        # kept (no copy of d_R) unless empty relsets need dummy entries, then (relset, ideal) keys are sorted.
        pres = cp.zeros(n, dtype=cp.uint8)
        _mod(8, 4).get_function('mark_rows')((4096,), (256,), (d_C, np.int64(self.nnz), np.int32(0), pres))
        nmiss = int(cp.count_nonzero(pres == 0)); del pres
        if nmiss == 0:
            self.sBt = Stream(d_C, d_R, 0, n, 0, "Bt")
        else:
            key = (d_C.astype(cp.uint64) << cp.uint64(32)) | d_R.astype(cp.uint64); key.sort()
            self.sBt = _key_stream(key, 0, n, 0, 'i32', "Bt"); del key
        self.sBt.zl = cp.zeros(1, cp.int32); self.sBt.nzl = 0
        self.sBt.cin = cp.zeros(self.sBt.nch, cp.uint64) if width == 64 else cp.zeros((self.sBt.nch, 2), cp.uint64)
        self.tgtBt = self.sBt.tgt
        # ---- B hot: hot entries in (relset, ideal) order; rows = relsets 0..n-1 (dummies for relsets w/o hot ideal) ----
        self.sHot = None
        if H:
            nh = int(cp.count_nonzero(hot))
            key = cp.where(hot, (d_C.astype(cp.uint64) << cp.uint64(32)) | d_R.astype(cp.uint64), MAXK)
            key.sort(); key = key[:nh]
            self.sHot = _key_stream(key, 0, n, hlo, 'u16', "Bhot"); del key
            self.nblk_hot = self.sms
            self.part = cp.empty((self.nblk_hot * H * (1 if width == 64 else 2),), cp.uint64)
            self.k_scatter.max_dynamic_shared_size_bytes = H * self.esz
        # ---- B cold: cold entries, P column (relset) parts, each in (ideal, relset) order over rows [rlo, n) ----
        coldmask = ~hot if H else None
        if coldmask is not None:
            nc_all = int(cp.count_nonzero(coldmask))
            rlo = int(cp.min(cp.where(coldmask, d_R, cp.int32(n)))) if nc_all else n
        else:
            nc_all = self.nnz; rlo = int(d_R.min()) if self.nnz else n
        rlo = min(rlo, n)
        self.sB = []
        cb = [(p * n) // self.P for p in range(self.P + 1)]
        if os.environ.get("LA_SEG_CB"):        # experiment override: explicit inner part boundaries, e.g. "5750000"
            cb = [0] + [int(v) for v in os.environ["LA_SEG_CB"].split(",") if v] + [n]; self.P = len(cb) - 1
        for p in range(self.P):
            m = (d_C >= cb[p]) & (d_C < cb[p + 1])
            if coldmask is not None: m &= coldmask
            nc = int(cp.count_nonzero(m))
            key = cp.where(m, (d_R.astype(cp.uint64) << cp.uint64(32)) | d_C.astype(cp.uint64), MAXK)
            del m
            key.sort(); key = key[:nc]
            s_ = _key_stream(key, rlo, n, cb[p], 'i32', f"B[{p}/{self.P}]"); del key
            s_.c0 = cb[p]; s_.c1 = cb[p + 1]
            self.sB.append(s_)
        # one cin/tgt array over all parts -> a single fix-up launch per B product
        offs = np.cumsum([0] + [s_.nch for s_ in self.sB]); self.ntB = int(offs[-1])
        self.cinB = cp.zeros(self.ntB, cp.uint64) if width == 64 else cp.zeros((self.ntB, 2), cp.uint64)
        self.tgtB = cp.concatenate([s_.tgt for s_ in self.sB])
        for p, s_ in enumerate(self.sB):
            s_.cin = self.cinB[int(offs[p]):int(offs[p + 1])]; s_.tgt = None
        # part 0 writes (store) every row of [rlo,n); parts>0 XOR-accumulate; the hot combine stores [hlo,hlo+H);
        # everything else (the ND dense rows) is zeroed from part 0's zero-list
        s0 = self.sB[0]
        s0.zl = _zero_list(n, [(rlo, n), (hlo, hlo + H)]); s0.nzl = int(s0.zl.size)
        for s_ in self.sB[1:]: s_.zl = cp.zeros(1, cp.int32); s_.nzl = 0
        if hot is not None: del hot
        if coldmask is not None: del coldmask
        self.k_ghot.max_dynamic_shared_size_bytes = max(self.Hbt, 1) * self.esz
        # launch geometry
        occ = int(os.environ.get("LA_SEG_BPS", "0") or 0)
        if occ <= 0:
            try:
                occ = int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(self.k_gather.kernel.ptr, 256, 0))
            except Exception as e:
                print(f"  [seg] occupancy query failed ({e}); using 4 blocks/SM", flush=True); occ = 4
        self.grid_g = (self.sms * max(1, occ),)
        self.grid_hot = (self.sms,)
        self.grid_f = (max(1, min(4096, -(-max(H, self.ntB, self.sBt.nch) // 256))),)
        cp.cuda.runtime.deviceSynchronize()
        ncold = sum(s_.nnz - s_.ndummy for s_ in self.sB); nhot = (self.sHot.nnz - self.sHot.ndummy) if self.sHot else 0
        if ncold + nhot != self.nnz or self.sBt.nnz - self.sBt.ndummy != self.nnz:
            raise RuntimeError(f"seg plan lost entries: cold {ncold} + hot {nhot} (Bt {self.sBt.nnz - self.sBt.ndummy}) != nnz {self.nnz}")
        if verbose:
            print(f"  [seg] {width}-bit plan: hot ideals [{hlo},{hlo+H}) = {nhot:,} nnz ({100*nhot/max(1,self.nnz):.1f}%), "
                  f"cold {ncold:,} in {self.P} column part(s) over rows [{rlo},{n}); dummies Bt {self.sBt.ndummy} hot "
                  f"{self.sHot.ndummy if self.sHot else 0} cold {[s_.ndummy for s_ in self.sB]}; Bt smem window {self.Hbt}; "
                  f"zero rows {s0.nzl}; gather grid {self.grid_g[0]} ({occ}/SM); built in {time.time()-t0:.1f}s", flush=True)

    # ---------------- products ----------------
    def _gather(self, s, x, y, acc, hot):
        if hot and self.Hbt:
            self.k_ghot(self.grid_hot, (1024,), (x, s.idx, s.hb, s.crow, np.int64(s.nnz), np.int32(s.nch), y, s.cin,
                        s.rmap_arg(), np.int32(s.rbase), np.int32(acc), np.int32(self.hlo), np.int32(self.Hbt),
                        s.zl, np.int32(s.nzl)), shared_mem=self.Hbt * self.esz)
        else:
            self.k_gather(self.grid_g, (256,), (x, s.idx, s.hb, s.crow, np.int64(s.nnz), np.int32(s.nch), y, s.cin,
                          s.rmap_arg(), np.int32(s.rbase), np.int32(acc), s.zl, np.int32(s.nzl)))

    def B(self, x, y):
        """y[0:n] = sparse part of B*x (every row written)."""
        for p, s in enumerate(self.sB):
            self._gather(s, x[s.c0:s.c1], y, 1 if p else 0, False)
        if self.sHot is not None:
            h = self.sHot
            self.k_scatter(self.grid_hot, (1024,), (x, h.idx, h.hb, h.crow, h.rmap_arg(), np.int32(h.rbase), np.int64(h.nnz),
                           np.int32(h.nch), np.int32(self.H), self.part), shared_mem=self.H * self.esz)
        hotc = self.sHot is not None
        self.k_fix(self.grid_f, (256,), (y, self.cinB, self.tgtB, np.int32(self.ntB),
                   self.part if hotc else _NULL, np.int32(self.nblk_hot if hotc else 0), np.int32(self.H), np.int32(self.hlo)))

    def Bt(self, x, y):
        """y[0:n] = sparse part of B^T*x (every row written)."""
        s = self.sBt
        self._gather(s, x, y, 0, True)
        self.k_fix(self.grid_f, (256,), (y, s.cin, self.tgtBt, np.int32(s.nch), _NULL, np.int32(0), np.int32(0), np.int32(0)))
