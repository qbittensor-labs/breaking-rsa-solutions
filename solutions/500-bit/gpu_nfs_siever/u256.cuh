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

// Fixed-width 256-bit signed (two's complement) integer for GPU GNFS cofactoring.
// Norms are <=181 bits at these parameters; operands a (<=33b), b (<=14b) are single-limb.
// Everything the cofactor needs (norm eval, trial division, size) is fixed-width int
// arithmetic -> embarrassingly parallel -> GPU's home turf (GMP was the wrong tool).
#pragma once
#include <cstdint>
typedef struct { unsigned long long w[4]; } u256;   // little-endian two's complement

__host__ __device__ __forceinline__ u256 u_zero(){ u256 r; r.w[0]=r.w[1]=r.w[2]=r.w[3]=0; return r; }
__host__ __device__ __forceinline__ u256 u_from_i64(long long x){
  u256 r; r.w[0]=(unsigned long long)x; unsigned long long s=(x<0)?~0ULL:0ULL; r.w[1]=r.w[2]=r.w[3]=s; return r; }

__host__ __device__ __forceinline__ u256 u_add(u256 a,u256 b){
  u256 r; unsigned long long c=0;
  for(int i=0;i<4;i++){ unsigned long long s=a.w[i]+c; unsigned long long c1=(s<c); s+=b.w[i]; c1+=(s<b.w[i]); r.w[i]=s; c=c1; }
  return r;
}
__host__ __device__ __forceinline__ u256 u_not(u256 a){ u256 r; for(int i=0;i<4;i++)r.w[i]=~a.w[i]; return r; }
__host__ __device__ __forceinline__ u256 u_neg(u256 a){ return u_add(u_not(a),u_from_i64(1)); }

// unsigned multiply (a interpreted as 256-bit) by 64-bit m, mod 2^256
__host__ __device__ __forceinline__ u256 u_mul_u64(u256 a,unsigned long long m){
  u256 r; unsigned long long carry=0;
  for(int i=0;i<4;i++){ unsigned __int128 p=(unsigned __int128)a.w[i]*m+carry; r.w[i]=(unsigned long long)p; carry=(unsigned long long)(p>>64); }
  return r;
}
// signed multiply by 64-bit m: result two's complement mod 2^256 (exact since norms fit in 256b)
__host__ __device__ __forceinline__ u256 u_mul_i64(u256 a,long long m){
  if(m>=0) return u_mul_u64(a,(unsigned long long)m);
  return u_neg(u_mul_u64(a,(unsigned long long)(-m)));
}
__host__ __device__ __forceinline__ int  u_neg_p(u256 a){ return (int)(a.w[3]>>63); }
__host__ __device__ __forceinline__ u256 u_abs(u256 a){ return u_neg_p(a)?u_neg(a):a; }
__host__ __device__ __forceinline__ int  u_is_zero(u256 a){ return !(a.w[0]|a.w[1]|a.w[2]|a.w[3]); }

// in-place divide unsigned magnitude by 64-bit d; returns remainder.
// Fast paths: single-limb values use native u64 (no 128-bit software divide); otherwise
// skip leading zero limbs. As trial division shrinks the norm most calls hit the u64 path.
__host__ __device__ __forceinline__ unsigned long long u_divmod_u64(u256* a,unsigned long long d){
  if(!(a->w[1]|a->w[2]|a->w[3])){ unsigned long long v=a->w[0]; a->w[0]=v/d; return v%d; }   // single-limb fast path
  int top=3; while(top>0 && a->w[top]==0) top--;                                              // skip leading zeros
  unsigned long long rem=0;
  for(int i=top;i>=0;i--){ unsigned __int128 cur=((unsigned __int128)rem<<64)|a->w[i]; a->w[i]=(unsigned long long)(cur/d); rem=(unsigned long long)(cur%d); }
  return rem;
}
__host__ __device__ __forceinline__ unsigned long long u_mod_u64(u256 a,unsigned long long d){
  unsigned long long rem=0;
  for(int i=3;i>=0;i--){ unsigned __int128 cur=((unsigned __int128)rem<<64)|a.w[i]; rem=(unsigned long long)(cur%d); }
  return rem;
}
#ifdef __CUDA_ARCH__
__device__ __forceinline__ int u_bits(u256 a){ for(int i=3;i>=0;i--) if(a.w[i]) return i*64+(64-__clzll(a.w[i])); return 0; }
#else
#include <strings.h>
__host__ __forceinline__ int u_bits(u256 a){ for(int i=3;i>=0;i--) if(a.w[i]) return i*64+(64-__builtin_clzll(a.w[i])); return 0; }
#endif

// fits in a 64-bit value? (magnitude already abs)
__host__ __device__ __forceinline__ int u_fits64(u256 a){ return !(a.w[1]|a.w[2]|a.w[3]); }

// ---- homogeneous Horner: F(a,b)=sum_{k=0..5} cb[k]*a^k, where cb is precomputed c_k*b^(5-k) on host? ----
// Instead we pass raw coeffs (as u256) + a,b and build cb on the fly via repeated *b.
// cf[k] = polynomial coeff c_k as u256 (two's complement). deg=5.
__host__ __device__ __forceinline__ u256 norm_alg(const u256* cf,long long a,unsigned long long b){
  // cb[k] = c_k * b^(5-k); then Horner in a: acc = (((c5*a+c4 b)a + c3 b^2)a ...)
  u256 cb[6];
  for(int k=5;k>=0;k--){ u256 t=cf[k]; for(int e=0;e<5-k;e++) t=u_mul_u64(t,b); cb[k]=t; }
  u256 acc=cb[5];
  for(int k=4;k>=0;k--){ acc=u_mul_i64(acc,a); acc=u_add(acc,cb[k]); }
  return acc;
}
__host__ __device__ __forceinline__ u256 norm_rat(u256 Y0,u256 Y1,long long a,unsigned long long b){
  return u_add(u_mul_i64(Y1,a), u_mul_u64(Y0,b));
}
