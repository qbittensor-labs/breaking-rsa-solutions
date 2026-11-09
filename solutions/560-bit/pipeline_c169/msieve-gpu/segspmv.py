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
#   LA_SEG_G2 (256-bit cold gather: 1 = two lanes per entry (default), 0 = old kernels, 2 = one 32-byte load; see KG2),
#   LA_SEG_BT_BPS (blocks/SM for the non-hot B^T gather; unset/0 = B's grid, -1 = B^T kernel's occupancy; s83_btbps.sh),
#   LA_SEG_G2_NC, LA_SEG_SELFTEST=1 (G2 variant vs default kernels on the loaded matrix, one B + one B^T).
#   LA_SEG_HINT (0 default; 1 = evict-first y/cin/tgt streams, 2 = 1 + evict_last x gathers; see HINT below).
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
// read-only vector load: __ldg has no ulonglong4 overload (CUDA 13), so the 256-bit load is two 16-byte __ldg
__device__ __forceinline__ u64 vld(const u64* p){ return __ldg(p); }
__device__ __forceinline__ ulonglong2 vld(const ulonglong2* p){ return __ldg(p); }
__device__ __forceinline__ ulonglong4 vld(const ulonglong4* p){ const ulonglong2* q=(const ulonglong2*)p; ulonglong2 a=__ldg(q), b=__ldg(q+1); return make_ulonglong4(a.x,a.y,b.x,b.y); }
__device__ __forceinline__ bool vnz(const u64 v){ return v!=0ULL; }
__device__ __forceinline__ bool vnz(const ulonglong2 v){ return (v.x|v.y)!=0ULL; }
__device__ __forceinline__ void gatomic(u64* p, const u64 v){ atomicXor(p, v); }
__device__ __forceinline__ void gatomic(ulonglong2* p, const ulonglong2 v){ atomicXor(&p->x, v.x); atomicXor(&p->y, v.y); }
// shared-memory XOR through native 32-bit ATOMS.XOR (XOR is bitwise, so two halves == one 64-bit XOR)
__device__ __forceinline__ void satomic(u64* p, const u64 v){ unsigned* q=(unsigned*)p; atomicXor(q,(unsigned)v); atomicXor(q+1,(unsigned)(v>>32)); }
__device__ __forceinline__ void satomic(ulonglong2* p, const ulonglong2 v){ unsigned* q=(unsigned*)p;
  atomicXor(q,(unsigned)v.x); atomicXor(q+1,(unsigned)(v.x>>32)); atomicXor(q+2,(unsigned)v.y); atomicXor(q+3,(unsigned)(v.y>>32)); }
// 256-bit rows (LA_BLOCK=256, W6)
template<> __device__ __forceinline__ ulonglong4 vz<ulonglong4>(){ return make_ulonglong4(0ULL,0ULL,0ULL,0ULL); }
__device__ __forceinline__ void vx(ulonglong4& a, const ulonglong4 b){ a.x^=b.x; a.y^=b.y; a.z^=b.z; a.w^=b.w; }
__device__ __forceinline__ ulonglong4 vup(const ulonglong4 v, int d){ return make_ulonglong4(__shfl_up_sync(FULLM, v.x, d), __shfl_up_sync(FULLM, v.y, d), __shfl_up_sync(FULLM, v.z, d), __shfl_up_sync(FULLM, v.w, d)); }
__device__ __forceinline__ ulonglong4 vbc(const ulonglong4 v, int l){ return make_ulonglong4(__shfl_sync(FULLM, v.x, l), __shfl_sync(FULLM, v.y, l), __shfl_sync(FULLM, v.z, l), __shfl_sync(FULLM, v.w, l)); }
__device__ __forceinline__ bool vnz(const ulonglong4 v){ return (v.x|v.y|v.z|v.w)!=0ULL; }
__device__ __forceinline__ void gatomic(ulonglong4* p, const ulonglong4 v){ atomicXor(&p->x, v.x); atomicXor(&p->y, v.y); atomicXor(&p->z, v.z); atomicXor(&p->w, v.w); }
__device__ __forceinline__ void satomic(ulonglong4* p, const ulonglong4 v){ unsigned* q=(unsigned*)p;
  atomicXor(q,(unsigned)v.x); atomicXor(q+1,(unsigned)(v.x>>32)); atomicXor(q+2,(unsigned)v.y); atomicXor(q+3,(unsigned)(v.y>>32));
  atomicXor(q+4,(unsigned)v.z); atomicXor(q+5,(unsigned)(v.z>>32)); atomicXor(q+6,(unsigned)v.w); atomicXor(q+7,(unsigned)(v.w>>32)); }

// write a finished (or chunk-final) row value: continuation rows -> cin[c]; owned rows -> y (store or XOR)
template<typename T> __device__ __forceinline__ void put(T* __restrict__ y, T* __restrict__ cin, int c, bool cont, int ord,
    const int* __restrict__ rmap, int rbase, int acc, const T v){
  if(cont){ cin[c]=v; return; }
  const int r = rmap ? __ldg(rmap+ord) : ord+rbase;
  if(acc){ if(!vnz(v)) return; T t=y[r]; vx(t,v); y[r]=t; } else y[r]=v;
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
        if(h!=0xFFFFu) satomic(sy+h, vld(x+r));             // 0xFFFF = dummy entry of a row with no hot ideal
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
  if(acc){ if(!vnz(v)) return; T t=y[r]; vx(t,v); y[r]=t; } else y[r]=v;
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
        T xv=vld(x+(rmap ? __ldg(rmap+ord) : ord+rbase));
        if((hx&0xffffu)!=0xFFFFu) satomic(sy+(hx&0xffffu),xv);   // 0xFFFF = dummy entry (row with no hot ideal)
        #pragma unroll
        for(int k=1;k<4;k++){
          if(k<nv){
            if((nib>>k)&1u){ ord++; xv=vld(x+(rmap ? __ldg(rmap+ord) : ord+rbase)); }
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
# ---- LA_SEG_G2 (2026-10-22, opt-in, 256-bit only): the cold gather with LPE lanes per entry ----------------------------
# Today (LA_SEG_G2=0) a lane loads its 32-byte x element as two 16-byte __ldg: each warp-wide LDG.128 touches 32 random
# sectors and uses half of each (64 sector requests per 32 entries), and the segmented scan shuffles 4 words per element.
#   LA_SEG_G2=1: LPE=2. Lanes 2v and 2v+1 form virtual lane v and share entry v; lane half h=lane&1 loads 16-byte half h,
#     so one LDG.128 covers 16 entries = 16 FULL sectors (32 requests per 32 entries). Each lane then runs the unchanged
#     algorithm on its 128-bit half: the warp is 16 virtual lanes x 2 independent halves (shuffle distances x2, ballots
#     masked to even lanes), steps of 16 (seg_gather) / 64 (seg4) entries, twice as many steps per chunk.
#   LA_SEG_G2=2: LPE=1, today's algorithm with ONE 32-byte load per entry (PTX ISA 8.8 ld.global.v4.u64, sm_100+;
#     G2NC=1 adds .nc, LA_SEG_G2_NC=0 drops it if ptxas rejects the combination).
# EXACT: the plan (idx, hb, crow, cin/tgt, zero list) is the same; the chunk -> rows -> elements split is the same, only
# the lanes are renumbered: a virtual lane owns exactly the entries a lane of the same algorithm owns at the halved warp
# width, and the head words are re-cut (step q of NV entries = bits (q%LPE)*NV.. of hb word q/LPE). Bits of half h only
# ever meet bits of half h (shuffles move by multiples of LPE lanes; the carry comes from lane (NV-1)*LPE+h), so each
# 128-bit half of every output row is the XOR of the same multiset of words as today. A half is stored by its own lane
# (y/cin rows are owned by one virtual lane, so no race); the acc path skips an all-zero half (XOR with 0 = no-op).
# Hot B^T smem window (part 0 of B^T) shares the body: its hits read the half from shared memory. seg4_scatter (hot B,
# near-sequential x reads) is untouched. LA_SEG_SELFTEST=1 checks one B and one B^T against the default kernels.
KG2 = r'''
#ifndef G2NC
#define G2NC 1
#endif
template<int LPE> struct G2V;
template<> struct G2V<1>{ typedef ulonglong4 V; };
template<> struct G2V<2>{ typedef ulonglong2 V; };
// half h of x[cc] (LPE=2: one 16-byte __ldg) / all of x[cc] (LPE=1: one 32-byte load)
template<int LPE> __device__ __forceinline__ typename G2V<LPE>::V g2ld(const ulonglong4* __restrict__ x, int cc, int h);
template<> __device__ __forceinline__ ulonglong2 g2ld<2>(const ulonglong4* __restrict__ x, int cc, int h){ return __ldg((const ulonglong2*)(x+cc)+h); }
template<> __device__ __forceinline__ ulonglong4 g2ld<1>(const ulonglong4* __restrict__ x, int cc, int h){
  ulonglong4 r;
#if G2NC
  asm("ld.global.nc.v4.u64 {%0,%1,%2,%3}, [%4];" : "=l"(r.x), "=l"(r.y), "=l"(r.z), "=l"(r.w) : "l"(x+cc));
#else
  asm("ld.global.v4.u64 {%0,%1,%2,%3}, [%4];" : "=l"(r.x), "=l"(r.y), "=l"(r.z), "=l"(r.w) : "l"(x+cc));
#endif
  return r; }
template<int LPE> __device__ __forceinline__ typename G2V<LPE>::V g2sm(const ulonglong4* sx, unsigned o, int h);
template<> __device__ __forceinline__ ulonglong2 g2sm<2>(const ulonglong4* sx, unsigned o, int h){ return ((const ulonglong2*)(sx+o))[h]; }
template<> __device__ __forceinline__ ulonglong4 g2sm<1>(const ulonglong4* sx, unsigned o, int h){ return sx[o]; }
template<int LPE, int HOT>
__device__ __forceinline__ typename G2V<LPE>::V g2gat(const ulonglong4* __restrict__ x, const ulonglong4* __restrict__ sx, int cc, int hlo, int H, int h){
  typename G2V<LPE>::V t=vz<typename G2V<LPE>::V>();
  if(HOT){ const unsigned o=(unsigned)(cc-hlo); if(o<(unsigned)H) t=g2sm<LPE>(sx,o,h); else if(cc>=0) t=g2ld<LPE>(x,cc,h); }
  else { if(cc>=0) t=g2ld<LPE>(x,cc,h); }
  return t;
}
// put/put4 on half h of the row (cont: continuation row of chunk c -> cin[c])
template<int LPE> __device__ __forceinline__ void g2put(ulonglong4* __restrict__ y, ulonglong4* __restrict__ cin, int c, bool cont, int ord,
    const int* __restrict__ rmap, int rbase, int acc, int h, const typename G2V<LPE>::V v){
  typedef typename G2V<LPE>::V V;
  if(cont){ ((V*)(cin+c))[h]=v; return; }
  const int r = rmap ? __ldg(rmap+ord) : ord+rbase;
  V* p=(V*)(y+r)+h;
  if(acc){ if(!vnz(v)) return; V t=*p; vx(t,v); *p=t; } else *p=v;
}
// seg_gather_body with LPE lanes per entry: NV=32/LPE virtual lanes, steps of NV entries, CH/NV steps per chunk
template<int LPE, int U, int HOT>
__device__ __forceinline__ void g2_gather_body(const ulonglong4* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, ulonglong4* __restrict__ y, ulonglong4* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, int hlo, int H, const int* __restrict__ zl, int nz, const ulonglong4* __restrict__ sx)
{
  typedef typename G2V<LPE>::V V;
  const int NV=32/LPE, QS=CH/NV;
  const unsigned VM= LPE==1 ? FULLM : 0xFFFFu;                // head bits of one step
  const int gt=blockIdx.x*blockDim.x+threadIdx.x, gs=gridDim.x*blockDim.x;
  for(int t=gt;t<nz;t+=gs) y[zl[t]]=vz<ulonglong4>();
  const int lane=threadIdx.x&31, vl=lane/LPE, h=lane%LPE;
  const unsigned vmle=VM>>(NV-1-vl);                          // virtual lanes 0..vl
  for(int c=gt>>5;c<nchunk;c+=gs>>5){
    const long long base=(long long)c*CH;
    const unsigned hw=__ldcs(hb+(long long)c*KS+lane);        // lane s holds the head word of 32 entries 32s..
    const unsigned h0=__shfl_sync(FULLM,hw,0);
    int ro=__ldg(crow+c)-(int)(h0&1u);
    bool cont=!(h0&1u);
    V carry=vz<V>();
    #pragma unroll 1
    for(int q0=0;q0<QS;q0+=U){
      int ci[U]; V v[U];
      #pragma unroll
      for(int u=0;u<U;u++){ const long long e=base+(long long)(q0+u)*NV+vl; ci[u]=(e<nnz)? __ldcs(idx+e) : -1; }
      #pragma unroll
      for(int u=0;u<U;u++) v[u]=g2gat<LPE,HOT>(x,sx,ci[u],hlo,H,h);
      #pragma unroll
      for(int u=0;u<U;u++){
        const int q=q0+u;
        const unsigned hs=(__shfl_sync(FULLM,hw,q/LPE)>>((q%LPE)*NV))&VM;
        V w=v[u];
        if(hs&1u){ if(q>0 && vl==0) g2put<LPE>(y,cin,c,cont,ro,rmap,rbase,acc,h,carry); cont=false; }
        else if(vl==0) vx(w,carry);
        const unsigned hm=hs&vmle;
        const int ss= hm ? 31-__clz(hm) : 0;
        #pragma unroll
        for(int d=1;d<NV;d<<=1){ const V t=vup(w,d*LPE); if(vl-d>=ss) vx(w,t); }
        if(vl<NV-1 && ((hs>>(vl+1))&1u)) g2put<LPE>(y,cin,c,(hm==0u)&&cont,ro+__popc(hm),rmap,rbase,acc,h,w);
        carry=vbc(w,(NV-1)*LPE+h);
        const int np=__popc(hs); ro+=np; if(np) cont=false;
      }
    }
    if(vl==0) g2put<LPE>(y,cin,c,cont,ro,rmap,rbase,acc,h,carry);
  }
}
// seg4_body with LPE lanes per entry: virtual lane v owns the 4 entries base+4*NV*q+4v..+3; ballots keep one lane per
// virtual lane (LM), lane masks count virtual lanes; CH/(4*NV) steps per chunk
template<int LPE, int U, int HOT>
__device__ __forceinline__ void g2_seg4_body(const ulonglong4* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, ulonglong4* __restrict__ y, ulonglong4* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, int hlo, int H, const int* __restrict__ zl, int nz, const ulonglong4* __restrict__ sx)
{
  typedef typename G2V<LPE>::V V;
  const int NV=32/LPE, QS=CH/(4*NV);
  const unsigned LM= LPE==1 ? FULLM : 0x55555555u;
  const int gt=blockIdx.x*blockDim.x+threadIdx.x, gs=gridDim.x*blockDim.x;
  for(int t=gt;t<nz;t+=gs) y[zl[t]]=vz<ulonglong4>();
  const int lane=threadIdx.x&31, vl=lane/LPE, h=lane%LPE;
  const unsigned lmle=FULLM>>(31-LPE*vl), lmlt=(1u<<(LPE*vl))-1u;   // lanes of virtual lanes <= vl / < vl
  const int nsh=(vl&7)*4;
  for(int c=gt>>5;c<nchunk;c+=gs>>5){
    const long long base=(long long)c*CH;
    const unsigned hw=__ldcs(hb+(long long)c*KS+lane);
    const unsigned h0=__shfl_sync(FULLM,hw,0)&1u;
    const int c0=__ldg(crow+c);
    const int cont_ord= h0 ? -1 : c0;
    int ro=c0-(int)h0;
    V carry=vz<V>();
    #pragma unroll 1
    for(int q0=0;q0<QS;q0+=U){
      int4 ci[U];
      #pragma unroll
      for(int u=0;u<U;u++) ci[u]=ld4(idx,base+(long long)(q0+u)*(4*NV)+4*vl,nnz);
      V v[U][4];
      #pragma unroll
      for(int u=0;u<U;u++){ v[u][0]=g2gat<LPE,HOT>(x,sx,ci[u].x,hlo,H,h); v[u][1]=g2gat<LPE,HOT>(x,sx,ci[u].y,hlo,H,h);
                            v[u][2]=g2gat<LPE,HOT>(x,sx,ci[u].z,hlo,H,h); v[u][3]=g2gat<LPE,HOT>(x,sx,ci[u].w,hlo,H,h); }
      #pragma unroll
      for(int u=0;u<U;u++){
        const int q=q0+u;
        const unsigned nib=(__shfl_sync(FULLM,hw,(q*4*NV+4*vl)>>5)>>nsh)&15u;
        const unsigned b0=__ballot_sync(FULLM,nib&1u)&LM, b1=__ballot_sync(FULLM,nib&2u)&LM, b2=__ballot_sync(FULLM,nib&4u)&LM, b3=__ballot_sync(FULLM,nib&8u)&LM;
        const int hbb=__popc(b0&lmlt)+__popc(b1&lmlt)+__popc(b2&lmlt)+__popc(b3&lmlt);
        const unsigned HL=b0|b1|b2|b3;
        int ordl=ro+hbb; V F=vz<V>(), L=vz<V>();
        #pragma unroll
        for(int k=0;k<4;k++){
          const bool before=(nib&((1u<<k)-1u))!=0u;
          if((nib>>k)&1u){ if(before) g2put<LPE>(y,cin,c,ordl==cont_ord,ordl,rmap,rbase,acc,h,L); ordl++; L=v[u][k]; }
          else if(before) vx(L,v[u][k]); else vx(F,v[u][k]);
        }
        const bool hh=nib!=0u;
        V sv= hh ? L : F;
        if(vl==0 && !hh) vx(sv,carry);
        const unsigned hm=HL&lmle; const int ss= hm ? (31-__clz(hm))/LPE : 0;
        #pragma unroll
        for(int d=1;d<NV;d<<=1){ const V w=vup(sv,d*LPE); if(vl-d>=ss) vx(sv,w); }
        V prev=vup(sv,LPE); if(vl==0) prev=carry;
        if(hh && !(q==0 && vl==0 && h0)){ vx(prev,F); g2put<LPE>(y,cin,c,ro+hbb==cont_ord,ro+hbb,rmap,rbase,acc,h,prev); }
        carry=vbc(sv,(NV-1)*LPE+h);
        ro+=__popc(b0)+__popc(b1)+__popc(b2)+__popc(b3);
      }
    }
    if(vl==0) g2put<LPE>(y,cin,c,ro==cont_ord,ro,rmap,rbase,acc,h,carry);
  }
}
// kernels: same signatures as seg_gather / seg_gather_hot / seg4_gather / seg4_gather_hot <ulonglong4> (drop-in)
template<int LPE, int U>
__global__ void __launch_bounds__(256) g2_gather(const ulonglong4* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, ulonglong4* __restrict__ y, ulonglong4* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, const int* __restrict__ zl, int nz)
{ g2_gather_body<LPE,U,0>(x,idx,hb,crow,nnz,nchunk,y,cin,rmap,rbase,acc,0,0,zl,nz,(const ulonglong4*)0); }
template<int LPE, int U>
__global__ void __launch_bounds__(1024,1) g2_gather_hot(const ulonglong4* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, ulonglong4* __restrict__ y, ulonglong4* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, int hlo, int H, const int* __restrict__ zl, int nz)
{
  extern __shared__ __align__(16) unsigned char sm_[];
  ulonglong4* sx=(ulonglong4*)sm_;
  for(int t=threadIdx.x;t<H;t+=blockDim.x) sx[t]=x[hlo+t];
  __syncthreads();
  g2_gather_body<LPE,U,1>(x,idx,hb,crow,nnz,nchunk,y,cin,rmap,rbase,acc,hlo,H,zl,nz,sx);
}
template<int LPE, int U>
__global__ void __launch_bounds__(256) g2_seg4_gather(const ulonglong4* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, ulonglong4* __restrict__ y, ulonglong4* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, const int* __restrict__ zl, int nz)
{ g2_seg4_body<LPE,U,0>(x,idx,hb,crow,nnz,nchunk,y,cin,rmap,rbase,acc,0,0,zl,nz,(const ulonglong4*)0); }
template<int LPE, int U>
__global__ void __launch_bounds__(1024,1) g2_seg4_gather_hot(const ulonglong4* __restrict__ x, const int* __restrict__ idx, const unsigned* __restrict__ hb,
    const int* __restrict__ crow, long long nnz, int nchunk, ulonglong4* __restrict__ y, ulonglong4* __restrict__ cin,
    const int* __restrict__ rmap, int rbase, int acc, int hlo, int H, const int* __restrict__ zl, int nz)
{
  extern __shared__ __align__(16) unsigned char sm_[];
  ulonglong4* sx=(ulonglong4*)sm_;
  for(int t=threadIdx.x;t<H;t+=blockDim.x) sx[t]=x[hlo+t];
  __syncthreads();
  g2_seg4_body<LPE,U,1>(x,idx,hb,crow,nnz,nchunk,y,cin,rmap,rbase,acc,hlo,H,zl,nz,sx);
}
'''
G2 = int(os.environ.get("LA_SEG_G2", "1") or 0)          # default 1 since s30g1 (sb44, key1b TD 130): 39.87 -> 37.76 ms/it, B^T 18.50 -> 16.74,
                                                          # hashes identical at it 300/1500; 2 gave 38.88; 0 = the old kernels
G2NC = int(os.environ.get("LA_SEG_G2_NC", "1") or 0)      # LA_SEG_G2=2: 1 = ld.global.nc.v4.u64, 0 = ld.global.v4.u64
# ---- LA_SEG_HINT (2026-10-23, opt-in): L2 cache-priority hints, results bit-identical (only cache operators change) ------
# At 256-bit the column parts 1.. of B / B^T read-modify-write y (~8 GB/it of plain loads/stores at normal L2 priority),
# which evicts the ~96 MB x slice that the cold gathers re-read (128 MB L2). 0 (default) = the kernel text is unchanged.
#   1: y / cin RMW and stores in put/put4/g2put, the zero-row stores, and seg_fix's cin/tgt/part reads use
#      __ldcs/__stcs (ld/st.global.cs = evict-first in L1 and L2); 256-bit values as two 16-byte halves.
#   2: 1 + the cold x gathers (t=x[cc] in KSEG, g2ld<2> in KG2) as ld.global.nc.L2::cache_hint with a
#      createpolicy.fractional.L2::evict_last policy (sm_80+), so the slice outlives the streams.
# Applied as source substitutions in _mod (each must match the expected number of times, else ValueError); pair with a
# larger LA_SEG_SPLIT_MB / LA_SEG_BT_SPLIT_MB (fewer parts) -- s67_lahint.sh.
HINT = int(os.environ.get("LA_SEG_HINT", "0") or 0)
_HINT_HDR = r"""
// LA_SEG_HINT >= 1: streaming (evict-first) loads/stores for y / cin / tgt / part
__device__ __forceinline__ u64 hld(const u64* p){ return __ldcs(p); }
__device__ __forceinline__ ulonglong2 hld(const ulonglong2* p){ return __ldcs(p); }
__device__ __forceinline__ ulonglong4 hld(const ulonglong4* p){ const ulonglong2* q=(const ulonglong2*)p; const ulonglong2 a=__ldcs(q), b=__ldcs(q+1); return make_ulonglong4(a.x,a.y,b.x,b.y); }
__device__ __forceinline__ void hst(u64* p, const u64 v){ __stcs(p, v); }
__device__ __forceinline__ void hst(ulonglong2* p, const ulonglong2 v){ __stcs(p, v); }
__device__ __forceinline__ void hst(ulonglong4* p, const ulonglong4 v){ ulonglong2* q=(ulonglong2*)p; __stcs(q, make_ulonglong2(v.x,v.y)); __stcs(q+1, make_ulonglong2(v.z,v.w)); }
"""
_HINT_X = r"""
// LA_SEG_HINT == 2: x gathers keep L2 priority evict_last (read-only: .nc)
__device__ __forceinline__ u64 hpol(){ u64 p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ u64 xld(const u64* p){ u64 r; asm volatile("ld.global.nc.L2::cache_hint.u64 %0, [%1], %2;" : "=l"(r) : "l"(p), "l"(hpol())); return r; }
__device__ __forceinline__ ulonglong2 xld(const ulonglong2* p){ ulonglong2 r;
  asm volatile("ld.global.nc.L2::cache_hint.v2.u64 {%0,%1}, [%2], %3;" : "=l"(r.x), "=l"(r.y) : "l"(p), "l"(hpol())); return r; }
__device__ __forceinline__ ulonglong4 xld(const ulonglong4* p){ const ulonglong2* q=(const ulonglong2*)p; const ulonglong2 a=xld(q), b=xld(q+1); return make_ulonglong4(a.x,a.y,b.x,b.y); }
"""
_HINT_SUB1 = [   # (old, new, expected count) -- KSEG
    ("// write a finished (or chunk-final) row value", "@HDR@// write a finished (or chunk-final) row value", 1),
    ("  if(acc){ if(!vnz(v)) return; T t=y[r]; vx(t,v); y[r]=t; } else y[r]=v;",
     "  if(acc){ if(!vnz(v)) return; T t=hld(y+r); vx(t,v); hst(y+r,t); } else hst(y+r,v);", 2),
    ("if(cont){ cin[c]=v; return; }", "if(cont){ hst(cin+c,v); return; }", 1),
    ("if(ord==cont_ord){ cin[c]=v; return; }", "if(ord==cont_ord){ hst(cin+c,v); return; }", 1),
    ("y[zl[t]]=vz<T>();", "hst(y+zl[t],vz<T>());", 2),
    ("const int r=tgt[c];", "const int r=__ldcs(tgt+c);", 1),
    ("const T v=cin[c];", "const T v=hld(cin+c);", 1),
    ("vx(a,part[(long long)b*H+t])", "vx(a,hld(part+(long long)b*H+t))", 1)]
_HINT_SUB1_G2 = [  # KG2
    ("y[zl[t]]=vz<ulonglong4>();", "hst(y+zl[t],vz<ulonglong4>());", 2),
    ("if(cont){ ((V*)(cin+c))[h]=v; return; }", "if(cont){ hst((V*)(cin+c)+h,v); return; }", 1),
    ("if(acc){ if(!vnz(v)) return; V t=*p; vx(t,v); *p=t; } else *p=v;", "if(acc){ if(!vnz(v)) return; V t=hld(p); vx(t,v); hst(p,t); } else hst(p,v);", 1)]
_HINT_SUB2 = [("t=x[cc];", "t=xld(x+cc);", 4)]
_HINT_SUB2_G2 = [("return __ldg((const ulonglong2*)(x+cc)+h);", "return xld((const ulonglong2*)(x+cc)+h);", 1)]


def _hinted(code, subs):
    for a, b, n in subs:
        if code.count(a) != n: raise ValueError(f"LA_SEG_HINT: '{a}' found {code.count(a)}x, expected {n}")
        code = code.replace(a, b)
    return code


def _hint_src(hint):
    """(KSEG, KG2) with LA_SEG_HINT applied; hint 0 -> the unchanged strings."""
    if not hint: return KSEG, KG2
    if hint not in (1, 2): raise ValueError(f"LA_SEG_HINT={hint} (must be 0, 1 or 2)")
    s1, s2 = _HINT_SUB1 + (_HINT_SUB2 if hint == 2 else []), _HINT_SUB1_G2 + (_HINT_SUB2_G2 if hint == 2 else [])
    k = _hinted(KSEG, s1).replace("@HDR@", _HINT_HDR + (_HINT_X if hint == 2 else ""))
    return k, _hinted(KG2, s2)


# ---- LA_SEG_HOTT (2026-10-24, r8, opt-in, default 0 = kernel text and plan unchanged): word-major hot-B accumulator ----
# seg4_scatter accumulates hot row t of B in shared memory as ONE element at 32-bit words 8t..8t+7 (256-bit). Its
# ATOMS.XOR for word k of 32 lanes with random rows t hit bank (8t+k) mod 32 = 8(t mod 4)+k: only 4 distinct banks,
# i.e. ~8-way replays on every one of the 8 atomics per entry. seg4_scatter_t stores word k of row t at sw[k*H+t]
# (same 99 KB): bank (k*H+t) mod 32 is uniform in t (H=3168 = 99*32), expected worst bank of 32 random rows ~3.5.
# EXACT: each output word is the XOR of the same multiset of words (atomics only reorder XORs); the block partial is
# written to part[] in the original element-major layout, so seg_fix and everything after it are unchanged.
# LA_SEG_SELFTEST=1 compares one B product against seg4_scatter on every row (gpu/r8_panel.sh also checks LA_HASH).
KHOTT = r"""
template<typename T> __device__ __forceinline__ void tatomic(unsigned* sw, const int H, const unsigned h, const T v);
template<> __device__ __forceinline__ void tatomic<u64>(unsigned* sw, const int H, const unsigned h, const u64 v){
  atomicXor(sw+h,(unsigned)v); atomicXor(sw+H+h,(unsigned)(v>>32)); }
template<> __device__ __forceinline__ void tatomic<ulonglong2>(unsigned* sw, const int H, const unsigned h, const ulonglong2 v){
  atomicXor(sw+h,(unsigned)v.x); atomicXor(sw+H+h,(unsigned)(v.x>>32)); atomicXor(sw+2*H+h,(unsigned)v.y); atomicXor(sw+3*H+h,(unsigned)(v.y>>32)); }
template<> __device__ __forceinline__ void tatomic<ulonglong4>(unsigned* sw, const int H, const unsigned h, const ulonglong4 v){
  atomicXor(sw+h,(unsigned)v.x); atomicXor(sw+H+h,(unsigned)(v.x>>32)); atomicXor(sw+2*H+h,(unsigned)v.y); atomicXor(sw+3*H+h,(unsigned)(v.y>>32));
  atomicXor(sw+4*H+h,(unsigned)v.z); atomicXor(sw+5*H+h,(unsigned)(v.z>>32)); atomicXor(sw+6*H+h,(unsigned)v.w); atomicXor(sw+7*H+h,(unsigned)(v.w>>32)); }
__device__ __forceinline__ u64 tw2(const unsigned* sw, const int H, const int k, const int t){ return (u64)sw[k*H+t] | ((u64)sw[(k+1)*H+t]<<32); }
template<typename T> __device__ __forceinline__ T tload(const unsigned* sw, const int H, const int t);
template<> __device__ __forceinline__ u64 tload<u64>(const unsigned* sw, const int H, const int t){ return tw2(sw,H,0,t); }
template<> __device__ __forceinline__ ulonglong2 tload<ulonglong2>(const unsigned* sw, const int H, const int t){ return make_ulonglong2(tw2(sw,H,0,t),tw2(sw,H,2,t)); }
template<> __device__ __forceinline__ ulonglong4 tload<ulonglong4>(const unsigned* sw, const int H, const int t){
  return make_ulonglong4(tw2(sw,H,0,t),tw2(sw,H,2,t),tw2(sw,H,4,t),tw2(sw,H,6,t)); }
// seg4_scatter with the word-major accumulator; same arguments, same part[] output layout
template<typename T>
__global__ void __launch_bounds__(1024,1) seg4_scatter_t(const T* __restrict__ x, const unsigned short* __restrict__ hidx,
    const unsigned* __restrict__ hb, const int* __restrict__ crow, const int* __restrict__ rmap, int rbase,
    long long nnz, int nchunk, int H, T* __restrict__ part)
{
  extern __shared__ __align__(16) unsigned char sm_[];
  unsigned* sw=(unsigned*)sm_;
  const int NWD=(int)(sizeof(T)/4);
  for(int t=threadIdx.x;t<H*NWD;t+=blockDim.x) sw[t]=0u;
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
        T xv=vld(x+(rmap ? __ldg(rmap+ord) : ord+rbase));
        if((hx&0xffffu)!=0xFFFFu) tatomic<T>(sw,H,hx&0xffffu,xv);
        #pragma unroll
        for(int k=1;k<4;k++){
          if(k<nv){
            if((nib>>k)&1u){ ord++; xv=vld(x+(rmap ? __ldg(rmap+ord) : ord+rbase)); }
            const unsigned h=(k==1? (hx>>16) : k==2? (hy&0xffffu) : (hy>>16));
            if(h!=0xFFFFu) tatomic<T>(sw,H,h,xv);
          }
        }
      }
      ro+=__popc(b0)+__popc(b1)+__popc(b2)+__popc(b3);
    }
  }
  __syncthreads();
  for(int t=threadIdx.x;t<H;t+=blockDim.x) part[(long long)blockIdx.x*H+t]=tload<T>(sw,H,t);
}
"""
HOTT = int(os.environ.get("LA_SEG_HOTT", "0") or 0)


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


def _g2_names(g2, U256):
    """LA_SEG_G2 kernel names {1: (cold, hot) for the seg_gather slot, 4: (cold, hot) for the seg4 slot}. Unroll keeps
    today's registers per lane (LPE=2 holds half an element: 2x the steps in flight) and divides the steps per chunk."""
    lpe = 2 if g2 == 1 else 1
    u1 = U256 * lpe if (32 * lpe) % (U256 * lpe) == 0 and U256 * lpe <= 16 else 4 * lpe
    u4 = max(1, min(4, U256 // 4)) * lpe
    return {1: (f'g2_gather<{lpe},{u1}>', f'g2_gather_hot<{lpe},{u1}>'),
            4: (f'g2_seg4_gather<{lpe},{u4}>', f'g2_seg4_gather_hot<{lpe},{u4}>')}


def _mod(U64, U128, U256=1, g2=0, hott=0):
    key = (U64, U128, U256) if not g2 else (U64, U128, U256, g2, G2NC)
    if HINT: key = key + ('hint', HINT)
    if hott: key = key + ('hott',)          # LA_SEG_HOTT: KHOTT appended (default modules unchanged)
    if key not in _mods:
        kseg, kg2 = _hint_src(HINT)     # LA_SEG_HINT=0: KSEG, KG2 unchanged
        names = []
        for T, U in (('u64', U64), ('ulonglong2', U128), ('ulonglong4', U256)):
            names += [f'seg_gather<{T},{U}>', f'seg_gather_hot<{T},{U}>', f'seg_scatter<{T}>', f'seg_fix<{T}>']
            u4 = max(1, min(4, U // 4))
            names += [f'seg4_gather<{T},{u4}>', f'seg4_gather_hot<{T},{u4}>', f'seg4_scatter<{T}>']
        code = kseg
        if g2:      # LA_SEG_G2: KSEG + the variant kernels in one module (the default kernels stay as the self-test reference)
            if g2 not in (1, 2): raise ValueError(f"LA_SEG_G2={g2} (must be 0, 1 or 2)")
            code = kseg + f"#define G2NC {1 if G2NC else 0}\n" + kg2
            for v in _g2_names(g2, U256).values(): names += list(v)
        if hott:
            code = code + KHOTT
            names += [f'seg4_scatter_t<{T}>' for T in ('u64', 'ulonglong2', 'ulonglong4')]
        _mods[key] = cp.RawModule(code=code, options=_opts(), name_expressions=names)
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


# ---- LA_SEG_BUILD (2026-10-20): 64-bit-safe, low-memory selection of a stream's entries ----------------------------
# v1 (the original build) selects each stream's entries with FULL-LENGTH temporaries: a bool mask, then
# where(mask, (hi<<32)|lo, MAXK) over all nnz -- three nnz x 8 B uint64 temporaries live at once (astype, shift, or) plus
# the where output -- and sorts the whole nnz-long key array for every part (Pbt + 1 + P = 11 sorts of nnz at 256 bits).
# Peak ~42 B per sparse nonzero: ~64 GB at TD 100 (1.53G), ~75 GB at TD 130 (1.78G), ~88 GB at TD 170 (~2.1G) on a
# 96 GB card, and every full-length sort/where runs over > 2^31 elements above TD ~190. A failed plan is caught by
# gpu_la.run256 and silently replaced by the column-split SpMV (~2x slower per iteration; its own boolean-mask
# indexing then dies above 2^31 nonzeros: the TD 220 OverflowError).
# v2 selects in chunks of LA_SEG_SEL_CHUNK entries (count pass, then fill an exact-size key array), so no temporary is
# longer than a chunk and no CuPy op ever sees more than 2^31 elements; only the selected keys (nc) are sorted.
# BIT-IDENTICAL BY CONSTRUCTION: the selected multiset of keys is the same and both builds sort it, so the sorted key
# array -- the only input to _key_stream -- is equal element for element (v1 sorted MAXK sentinels to the end and cut
# them off; real keys are < 2^63, R and C being non-negative int32). Peak ~16 B/nnz + 16 B x the largest stream.
# LA_SEG_BUILD = v1 | v2 | auto (default: v2 when nnz >= LA_SEG_BUILD_AUTO, 1.85e9 -- above TD 130's 1.78G, so
# TD 100/130 keep the measured v1 build unless v2 is asked for).
SEL_CH = int(os.environ.get("LA_SEG_SEL_CHUNK", str(1 << 27)) or (1 << 27))


def seg_build_mode(nnz):
    # default v2 since sandbox #22: identical plans/xb at TD 100/130/150, half the build time, 17-20 GB less GPU memory
    m = os.environ.get("LA_SEG_BUILD", "v2") or "v2"
    if m not in ("v1", "v2", "auto"):
        raise ValueError(f"LA_SEG_BUILD={m} (must be v1, v2 or auto)")
    if m == "auto":
        m = "v2" if nnz >= float(os.environ.get("LA_SEG_BUILD_AUTO", "1.85e9")) else "v1"
    return m


def _sel_mask(R, C, rsel, csel, rx):
    """bool mask of one chunk: R in [rsel) and C in [csel) and R not in [rx) (None = no condition); None = all."""
    m = None
    if rsel is not None:
        m = (R >= rsel[0]) & (R < rsel[1])
    if csel is not None:
        t = (C >= csel[0]) & (C < csel[1]); m = t if m is None else (m & t)
    if rx is not None and rx[1] > rx[0]:
        t = (R < rx[0]) | (R >= rx[1]); m = t if m is None else (m & t)
    return m


def _sel_keys(d_R, d_C, rsel=None, csel=None, rx=None, hi_is_c=True, chunk=None):
    """SORTED uint64 keys (hi<<32 | lo) of the entries selected by _sel_mask; hi,lo = (C,R) if hi_is_c else (R,C).
    Chunked: every CuPy op sees <= chunk elements (no full-length temporary, no > 2^31-element mask indexing)."""
    ch = int(chunk or SEL_CH); nnz = int(d_R.size)
    cnt = []
    for a in range(0, nnz, ch):
        b = min(nnz, a + ch)
        m = _sel_mask(d_R[a:b], d_C[a:b], rsel, csel, rx)
        cnt.append(b - a if m is None else int(cp.count_nonzero(m))); del m
    tot = int(sum(cnt))
    key = cp.empty(tot, dtype=cp.uint64)
    pos = 0
    for i, a in enumerate(range(0, nnz, ch)):
        if cnt[i] == 0: continue
        b = min(nnz, a + ch); R = d_R[a:b]; C = d_C[a:b]
        m = _sel_mask(R, C, rsel, csel, rx)
        hi, lo = (C, R) if hi_is_c else (R, C)
        if m is not None:
            hi = hi[m]; lo = lo[m]
        k = cnt[i]
        if int(hi.size) != k:
            raise RuntimeError(f"seg v2 selection: chunk {i} count {k} != selected {int(hi.size)}")
        cp.bitwise_or(cp.left_shift(hi.astype(cp.uint64), cp.uint64(32)), lo.astype(cp.uint64), out=key[pos:pos + k])
        pos += k; del hi, lo, m
    key.sort()
    return key


def _sel_min_R(d_R, rx, n, chunk=None):
    """min R over entries with R not in [rx) (rx None = all entries), or n if there are none (v1's rlo)."""
    ch = int(chunk or SEL_CH); nnz = int(d_R.size); best = n
    for a in range(0, nnz, ch):
        R = d_R[a:min(nnz, a + ch)]
        if rx is not None and rx[1] > rx[0]:
            v = int(cp.min(cp.where((R < rx[0]) | (R >= rx[1]), R, cp.int32(n))))
        else:
            v = int(R.min())
        best = min(best, v)
    return best


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
        assert width in (64, 128, 256)
        self.n = n; self.ND = ND; self.width = width; self.T = {64: 'u64', 128: 'ulonglong2', 256: 'ulonglong4'}[width]
        self.esz = width // 8
        self.nnz = int(d_R.size)
        U64 = int(os.environ.get("LA_SEG_U64", "8")); U128 = int(os.environ.get("LA_SEG_U128", "4"))
        U256 = int(os.environ.get("LA_SEG_U256", "4"))     # 256-bit: v[U] of ulonglong4 = 16 registers per element at U=4
        self.g2 = G2 if width == 256 else 0                 # LA_SEG_G2 (see KG2): 256-bit cold-gather variant
        self.hott = HOTT                                      # LA_SEG_HOTT (see KHOTT): word-major hot-B accumulator
        if self.hott not in (0, 1): raise ValueError(f"LA_SEG_HOTT={self.hott} (must be 0 or 1)")
        self.mod = _mod(U64, U128, U256, self.g2, self.hott)
        Ug = {64: U64, 128: U128, 256: U256}[width]
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
        self.k_scatter_ref = None
        if self.hott:
            if self.sigma != 4: raise ValueError("LA_SEG_HOTT=1 needs LA_SEG_SIGMA=4 (it replaces seg4_scatter)")
            self.k_scatter_ref = self.k_scatter
            self.k_scatter = self.mod.get_function(f'seg4_scatter_t<{self.T}>')
        self.k_fix = self.mod.get_function(f'seg_fix<{self.T}>')
        # B^T may use the other gather kernel (sandbox #12: seg_gather<T,4> 18.3 ms vs seg4 20.8 on B^T at 256 bits,
        # while seg4 stays best on B): LA_SEG_BT_SIGMA (default = LA_SEG_SIGMA)
        sb = int(os.environ.get("LA_SEG_BT_SIGMA", "1" if width == 256 else str(self.sigma)))
        if sb == 4:
            u4 = max(1, min(4, Ug // 4))
            self.k_gather_bt = self.mod.get_function(f'seg4_gather<{self.T},{u4}>')
            self.k_ghot_bt = self.mod.get_function(f'seg4_gather_hot<{self.T},{u4}>')
        else:
            self.k_gather_bt = self.mod.get_function(f'seg_gather<{self.T},{Ug}>')
            self.k_ghot_bt = self.mod.get_function(f'seg_gather_hot<{self.T},{Ug}>')
        self.k_ref = None; self.g2_names = ()
        if self.g2:     # same slots, same arguments: B cold (sigma) and B^T cold + hot window (LA_SEG_BT_SIGMA)
            self.k_ref = (self.k_gather, self.k_ghot, self.k_gather_bt, self.k_ghot_bt)    # LA_SEG_SELFTEST reference
            nm = _g2_names(self.g2, U256); f = self.mod.get_function
            self.g2_names = nm[4 if self.sigma == 4 else 1] + nm[4 if sb == 4 else 1]
            self.k_gather, self.k_ghot, self.k_gather_bt, self.k_ghot_bt = (f(k) for k in self.g2_names)
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
        #   256-bit: 3168 rows (99 KB) measured 0.5 ms/it better than 1792 on key 1 (sandbox #12), no cliff at 256 bits
        self.Hbt = (self.H if width == 64 else max(0, min(3584 if width == 128 else 3168, n - ND))) if hbt == "" else max(0, min(int(hbt), Hmax, n - ND))
        if P is None:
            P = int(os.environ.get("LA_SEG_P", "0") or 0)
            if P <= 0:
                split_mb = float(os.environ.get("LA_SEG_SPLIT_MB", "96"))
                P = 1 if width == 64 else max(1, -(-(n * self.esz) // int(split_mb * 2**20)))
        self.P = P
        # B^T ideal-axis parts (sandbox #12): the gather over x[ideal] spans the whole vector (~486 MB at 256 bits on
        # key 1, 3.8x the 128 MB L2); cut it like the cold-B column parts so each part's x segment stays L2-resident.
        # LA_SEG_BT_SPLIT_MB sets the segment size (default 0 = one part, the pre-#12 behaviour), LA_SEG_BT_P an explicit
        # count, LA_SEG_BT_CB explicit inner boundaries (ideal indices). Part 0 keeps the hot shared-memory window.
        Pbt = int(os.environ.get("LA_SEG_BT_P", "0") or 0)
        if Pbt <= 0:
            bt_mb = float(os.environ.get("LA_SEG_BT_SPLIT_MB", "96" if width == 256 else "0") or 0)   # 256: measured sandbox #12/#13
            Pbt = 1 if width == 64 or bt_mb <= 0 else max(1, -(-(n * self.esz) // int(bt_mb * 2**20)))
        self.Pbt = Pbt
        self.sms = _sms()
        hlo, H = self.hlo, self.H
        if H > 65534: raise ValueError("hot window must be < 65535 (0xFFFF marks a dummy entry)")
        MAXK = cp.uint64(0xFFFFFFFFFFFFFFFF)
        self.build = seg_build_mode(self.nnz); v2 = self.build == "v2"     # see LA_SEG_BUILD above
        hot = (d_R >= hlo) & (d_R < hlo + H) if (H and not v2) else None
        # ---- Bt: all entries (rows = relsets 0..n-1, gathers ideals). File order already groups by relset; it is
        # kept (no copy of d_R) unless empty relsets need dummy entries, then (relset, ideal) keys are sorted.
        rb = [(p * n) // self.Pbt for p in range(self.Pbt + 1)]
        if os.environ.get("LA_SEG_BT_CB"):
            rb = [0] + [int(v) for v in os.environ["LA_SEG_BT_CB"].split(",") if v] + [n]; self.Pbt = len(rb) - 1
        if self.Pbt > 1 and rb[1] < self.hlo + self.Hbt:
            self.Hbt = max(0, rb[1] - self.hlo)    # the hot window must lie inside part 0
        self.sBt = []
        if self.Pbt == 1:
            pres = cp.zeros(n, dtype=cp.uint8)
            _mod(8, 4).get_function('mark_rows')((4096,), (256,), (d_C, np.int64(self.nnz), np.int32(0), pres))
            nmiss = int(cp.count_nonzero(pres == 0)); del pres
            if nmiss == 0:
                s_ = Stream(d_C, d_R, 0, n, 0, "Bt")
            else:
                key = (d_C.astype(cp.uint64) << cp.uint64(32)) | d_R.astype(cp.uint64); key.sort()
                s_ = _key_stream(key, 0, n, 0, 'i32', "Bt"); del key
            s_.c0 = 0; s_.c1 = n; self.sBt.append(s_)
        else:
            for p in range(self.Pbt):      # rows = relsets [0, n) in every part (dummies pad the rest), idx = ideal - rb[p]
                if v2:
                    key = _sel_keys(d_R, d_C, rsel=(rb[p], rb[p + 1]), hi_is_c=True)
                else:
                    m = (d_R >= rb[p]) & (d_R < rb[p + 1])
                    nc = int(cp.count_nonzero(m))
                    key = cp.where(m, (d_C.astype(cp.uint64) << cp.uint64(32)) | d_R.astype(cp.uint64), MAXK)
                    del m
                    key.sort(); key = key[:nc]
                s_ = _key_stream(key, 0, n, rb[p], 'i32', f"Bt[{p}/{self.Pbt}]"); del key
                s_.c0 = rb[p]; s_.c1 = rb[p + 1]
                self.sBt.append(s_)
        offs = np.cumsum([0] + [s_.nch for s_ in self.sBt]); self.ntBt = int(offs[-1])
        self.cinBt = cp.zeros(self.ntBt, cp.uint64) if width == 64 else cp.zeros((self.ntBt, width // 64), cp.uint64)
        self.tgtBt = cp.concatenate([s_.tgt for s_ in self.sBt])
        for p, s_ in enumerate(self.sBt):
            s_.cin = self.cinBt[int(offs[p]):int(offs[p + 1])]; s_.tgt = None
            s_.zl = cp.zeros(1, cp.int32); s_.nzl = 0
        # ---- B hot: hot entries in (relset, ideal) order; rows = relsets 0..n-1 (dummies for relsets w/o hot ideal) ----
        self.sHot = None
        if H:
            if v2:
                key = _sel_keys(d_R, d_C, rsel=(hlo, hlo + H), hi_is_c=True)
            else:
                nh = int(cp.count_nonzero(hot))
                key = cp.where(hot, (d_C.astype(cp.uint64) << cp.uint64(32)) | d_R.astype(cp.uint64), MAXK)
                key.sort(); key = key[:nh]
            self.sHot = _key_stream(key, 0, n, hlo, 'u16', "Bhot"); del key
            self.nblk_hot = self.sms
            self.part = cp.empty((self.nblk_hot * H * (width // 64),), cp.uint64)
            self.k_scatter.max_dynamic_shared_size_bytes = H * self.esz
            if self.k_scatter_ref is not None: self.k_scatter_ref.max_dynamic_shared_size_bytes = H * self.esz
        # ---- B cold: cold entries, P column (relset) parts, each in (ideal, relset) order over rows [rlo, n) ----
        coldmask = ~hot if (H and not v2) else None
        if v2:
            rlo = _sel_min_R(d_R, (hlo, hlo + H) if H else None, n) if self.nnz else n
        elif coldmask is not None:
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
            if v2:
                key = _sel_keys(d_R, d_C, csel=(cb[p], cb[p + 1]), rx=(hlo, hlo + H) if H else None, hi_is_c=False)
            else:
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
        self.cinB = cp.zeros(self.ntB, cp.uint64) if width == 64 else cp.zeros((self.ntB, width // 64), cp.uint64)
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
        self.k_ghot_bt.max_dynamic_shared_size_bytes = max(self.Hbt, 1) * self.esz
        # launch geometry
        occ = int(os.environ.get("LA_SEG_BPS", "0") or 0)
        if occ <= 0:
            try:
                occ = int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(self.k_gather.kernel.ptr, 256, 0))
            except Exception as e:
                print(f"  [seg] occupancy query failed ({e}); using 4 blocks/SM", flush=True); occ = 4
        self.grid_g = (self.sms * max(1, occ),)
        # Bt non-hot gather (proposal 8, 2026-10-23): its kernel differs from B's under G2 (B g2_seg4_gather<2,2>,
        # Bt g2_gather<2,8>), so query its own occupancy. LA_SEG_BT_BPS > 0 sets blocks/SM for Bt; -1 = the Bt kernel's
        # own occupancy; unset/0 = reuse B's grid (pre-#8 behaviour, production default). Launch geometry only.
        self.occ_bt_q = None
        try:
            self.occ_bt_q = int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(self.k_gather_bt.kernel.ptr, 256, 0))
        except Exception as e:
            print(f"  [seg] Bt occupancy query failed ({e})", flush=True)
        bt_bps = int(os.environ.get("LA_SEG_BT_BPS", "0") or 0)
        if bt_bps < 0: bt_bps = self.occ_bt_q or 0
        self.occ_bt = bt_bps if bt_bps > 0 else occ
        self.grid_g_bt = (self.sms * max(1, self.occ_bt),) if bt_bps > 0 else self.grid_g
        self.grid_hot = (self.sms,)
        self.grid_f = (max(1, min(4096, -(-max(H, self.ntB, self.ntBt) // 256))),)
        cp.cuda.runtime.deviceSynchronize()
        ncold = sum(s_.nnz - s_.ndummy for s_ in self.sB); nhot = (self.sHot.nnz - self.sHot.ndummy) if self.sHot else 0
        nbt = sum(s_.nnz - s_.ndummy for s_ in self.sBt)
        if ncold + nhot != self.nnz or nbt != self.nnz:
            raise RuntimeError(f"seg plan lost entries: cold {ncold} + hot {nhot} (Bt {nbt}) != nnz {self.nnz}")
        if verbose:
            print(f"  [seg] {width}-bit plan: hot ideals [{hlo},{hlo+H}) = {nhot:,} nnz ({100*nhot/max(1,self.nnz):.1f}%), "
                  f"cold {ncold:,} in {self.P} column part(s) over rows [{rlo},{n}); Bt in {self.Pbt} ideal part(s); dummies Bt {[s_.ndummy for s_ in self.sBt]} hot "
                  f"{self.sHot.ndummy if self.sHot else 0} cold {[s_.ndummy for s_ in self.sB]}; Bt smem window {self.Hbt}; "
                  f"zero rows {s0.nzl}; gather grid B {self.grid_g[0]} ({occ}/SM) Bt {self.grid_g_bt[0]} ({self.occ_bt}/SM; Bt kernel occupancy {self.occ_bt_q}/SM); build {self.build}; built in {time.time()-t0:.1f}s"
                  + (f"; LA_SEG_G2={self.g2} (B {self.g2_names[0]}, Bt {self.g2_names[2]}/{self.g2_names[3]})" if self.g2 else "") + (f"; LA_SEG_HINT={HINT}" if HINT else "") + ("; LA_SEG_HOTT=1 (seg4_scatter_t)" if self.hott else ""), flush=True)
        if self.g2 and os.environ.get("LA_SEG_SELFTEST", "0") == "1":
            self.g2_selftest()
        if self.hott and self.sHot is not None and os.environ.get("LA_SEG_SELFTEST", "0") == "1":
            self.hott_selftest()

    def hott_selftest(self, seed=11):
        """LA_SEG_HOTT=1 and LA_SEG_SELFTEST=1: one B product (random x) with seg4_scatter_t vs seg4_scatter, every row
        compared (y prefilled differently per side); prints 'HOTT SELFTEST OK' or raises (gpu_la: exit 4 under strict)."""
        t = time.time(); n = self.n; W = self.width // 64
        r = cp.random.RandomState(seed).random_sample((2, n, W)).view(cp.uint64)
        x = r[0] ^ (r[1] << cp.uint64(12)); del r
        if W == 1: x = x.reshape(n)
        ys = []
        for k, fill in ((self.k_scatter_ref, np.uint64(0)), (self.k_scatter, np.uint64(0xFFFFFFFFFFFFFFFF))):
            kk = self.k_scatter; self.k_scatter = k
            y = cp.full(x.shape, fill, dtype=cp.uint64)
            try: self.B(x, y)
            finally: self.k_scatter = kk
            ys.append(y)
        cp.cuda.runtime.deviceSynchronize()
        a, b = (v.reshape(n, -1) for v in ys)
        nd = int(cp.count_nonzero(cp.any(a != b, axis=1))); del ys, a, b, x, y
        if nd: raise RuntimeError(f"HOTT SELFTEST FAILED: B {nd} of {n} rows differ")
        print(f"  [seg] HOTT SELFTEST OK: seg4_scatter_t B identical to seg4_scatter on all {n:,} rows ({time.time()-t:.1f}s)", flush=True)

    def g2_selftest(self, seed=7):
        """LA_SEG_G2 != 0 and LA_SEG_SELFTEST=1: one B and one B^T product (random x) with the variant kernels vs the
        default kernels on this plan, every row compared; prints 'G2 SELFTEST OK' or raises (gpu_la: plan failed ->
        exit 4 under LA_SEG_STRICT=1). y is prefilled differently per side, so a row either side skips is caught too."""
        if not self.g2: return
        t = time.time(); n = self.n
        r = cp.random.RandomState(seed).random_sample((2, n, 4)).view(cp.uint64)    # low 52 bits random per word
        x = r[0] ^ (r[1] << cp.uint64(12)); del r
        var = (self.k_gather, self.k_ghot, self.k_gather_bt, self.k_ghot_bt)
        for k in (self.k_ref[1], self.k_ref[3]): k.max_dynamic_shared_size_bytes = max(self.Hbt, 1) * self.esz
        bad = []
        for nm_, prod in (("B", self.B), ("Bt", self.Bt)):
            ys = []
            for ks, fill in ((self.k_ref, np.uint64(0)), (var, np.uint64(0xFFFFFFFFFFFFFFFF))):
                self.k_gather, self.k_ghot, self.k_gather_bt, self.k_ghot_bt = ks
                y = cp.full((n, 4), fill, dtype=cp.uint64)
                prod(x, y); ys.append(y)
            self.k_gather, self.k_ghot, self.k_gather_bt, self.k_ghot_bt = var
            cp.cuda.runtime.deviceSynchronize()
            nd = int(cp.count_nonzero(cp.any(ys[0] != ys[1], axis=1)))
            if nd: bad.append(f"{nm_}: {nd} of {n} rows differ")
            del ys, y
        del x
        if bad: raise RuntimeError(f"G2 SELFTEST FAILED (LA_SEG_G2={self.g2}): " + "; ".join(bad))
        print(f"  [seg] G2 SELFTEST OK: LA_SEG_G2={self.g2} B and B^T identical to the default kernels on all {n:,} rows "
              f"({time.time()-t:.1f}s)", flush=True)

    # ---------------- products ----------------
    def _gather(self, s, x, y, acc, hot, bt=False):
        kg = self.k_gather_bt if bt else self.k_gather; kh = self.k_ghot_bt if bt else self.k_ghot
        if hot and self.Hbt:
            kh(self.grid_hot, (1024,), (x, s.idx, s.hb, s.crow, np.int64(s.nnz), np.int32(s.nch), y, s.cin,
                        s.rmap_arg(), np.int32(s.rbase), np.int32(acc), np.int32(self.hlo), np.int32(self.Hbt),
                        s.zl, np.int32(s.nzl)), shared_mem=self.Hbt * self.esz)
        else:
            kg(self.grid_g_bt if bt else self.grid_g, (256,), (x, s.idx, s.hb, s.crow, np.int64(s.nnz), np.int32(s.nch), y, s.cin,
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
        for p, s in enumerate(self.sBt):
            self._gather(s, x[s.c0:s.c1], y, 1 if p else 0, p == 0, bt=True)
        self.k_fix(self.grid_f, (256,), (y, self.cinBt, self.tgtBt, np.int32(self.ntBt), _NULL, np.int32(0), np.int32(0), np.int32(0)))
