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

// ks256g.cu -- LA_GRAPH kernels for gpu_la.py's 256-bit block Lanczos (2026-10-23). Compiled ONLY when LA_GRAPH != 0,
// APPENDED to ks256v3.cu's source (uses its u64 typedef, tab_word and nib4). The default path never reads this file.
//
// WHY: with LA_V3=all every iteration ends with O.get() (a D2H copy that synchronizes the stream), ~0.45 ms of host
// algebra (la256host.c) and ~17 kernel launches issued from Python while the GPU sits idle; the expert review measured
// inner+D2H 2.13 ms and upd 2.42 ms of ~37.8 ms/it (key1b TD 130), part of it pure queue drain + launch latency.
// LA_GRAPH moves the per-iteration 256x256 GF(2) algebra onto the GPU (la_alg256: ONE block of 256 threads), so an
// iteration is a fixed launch sequence with no host dependency -> captured as a CUDA graph and replayed; the host only
// reads a 32-byte control record LAG iterations late (never draining the queue).
//
// la_alg256 is an exact port of gpu_block_lanczos256's LA_V3 'host' branch (fnsf / mulf / trf):
//   - la_fns256f's elimination with the same column order (previous subspace last), the same pivot rule (first
//     position >= i in that order holding the bit, t part first, else the identity part), the same row XORs.
//     Thread x owns row x ([Mh|Ml] = 8 u64 in registers); the pivot search is a block min-reduction over positions.
//   - products / transposes are unique (GF(2) matrix algebra), so any correct evaluation order is bit-identical.
//   - halts: status 1 = vt_a_v[0] == 0 (converged), 2 = find_nonsingular_sub failed (dim0 == 0), 3 = not all columns
//     used. On a halt (and in every launch after it: sticky) the update matrices are written as the IDENTITY update
//     (D = E = F = D2 = 0, mask all ones), so the upd kernels leave pn and xb unchanged, and useF = 0 (updF256_v3g skips).
// gpu_la.py self-tests it against the host functions on random data at start-up (LA_GRAPH unavailable -> WARNING,
// LA_V256_STRICT=1: exit 5).
//
// state S (u64 offsets): matrices are 256 rows x 4 u64 (1024 u64), the (256,4) layout of the host code.
#define GS_VA1  0        /* vt_a_v[1]   (previous O[0]) */
#define GS_VA21 1024     /* vt_a2_v[1]  (previous O[1]) */
#define GS_WI0  2048     /* winv[0] */
#define GS_WI1  3072     /* winv[1] */
#define GS_WI2  4096     /* winv[2] */
#define GS_V00  5120     /* vt_v0[0] */
#define GS_V01  6144     /* vt_v0[1] */
#define GS_V02  7168     /* vt_v0[2] */
#define GS_NX   8192     /* vt_v0_next */
#define GS_T1   9216
#define GS_T2   10240
#define GS_T3   11264
#define GS_T4   12288
#define GS_M1   13312    /* previous chosen-column mask (4 u64) */
// GS_SIZE 13320 (gpu_la.py allocates it)
// ctl (int32): 0 status (0 running | 1 converged | 2 fns failed | 3 columns not all used; sticky)
//              1 dim1 (dim0 of the last completed iteration)   2 useF of this launch   3 launches
//              4 completed iterations   5 dim0 at a halt   6 halt iteration (1-based)   16..271 s1 (chosen columns, in order)
#define SEL4(a0,a1,a2,a3,q) ((q)==0?(a0):(q)==1?(a1):(q)==2?(a2):(a3))

// warp minimum (shuffle butterfly; every lane gets it)
__device__ __forceinline__ unsigned wmin(unsigned v){
 #pragma unroll
 for(int o=16;o>0;o>>=1){ const unsigned u=__shfl_xor_sync(0xffffffffu,v,o); v= u<v? u:v; }
 return v;
}
// c = a*b (row i of c = XOR of b's rows j over the set bits j of a's row i), ^ I when addI, & cm (columns) when cm.
// a may alias c (thread t reads only row t of a before writing row t of c); b is staged in shared memory first.
__device__ __forceinline__ void g_mul(const u64* a, const u64* b, u64* c, u64* sb, const int addI, const u64* cm){
 const int t=threadIdx.x;
 __syncthreads();
 for(int k=t;k<1024;k+=256) sb[k]=b[k];
 __syncthreads();
 u64 c0=0ULL,c1=0ULL,c2=0ULL,c3=0ULL;
 #pragma unroll
 for(int w=0;w<4;w++){
   const u64 x=a[4*t+w];
   #pragma unroll 8
   for(int j=0;j<64;j++){
     const u64 m=0ULL-((x>>j)&1ULL); const u64* e=sb+4*(64*w+j);
     c0^=e[0]&m; c1^=e[1]&m; c2^=e[2]&m; c3^=e[3]&m;
   }
 }
 if(addI){ const u64 bt=1ULL<<(t&63); const int q=t>>6; if(q==0) c0^=bt; else if(q==1) c1^=bt; else if(q==2) c2^=bt; else c3^=bt; }
 if(cm){ c0&=cm[0]; c1&=cm[1]; c2&=cm[2]; c3&=cm[3]; }
 c[4*t]=c0; c[4*t+1]=c1; c[4*t+2]=c2; c[4*t+3]=c3;
 __syncthreads();
}
// c = a^T (row j of c bit i = row i of a bit j)
__device__ __forceinline__ void g_tr(const u64* a, u64* c, u64* sb){
 const int t=threadIdx.x;
 __syncthreads();
 for(int k=t;k<1024;k+=256) sb[k]=a[k];
 __syncthreads();
 const int jw=t>>6, js=t&63;
 #pragma unroll
 for(int w=0;w<4;w++){
   u64 o=0ULL;
   #pragma unroll 8
   for(int k=0;k<64;k++) o|=((sb[4*(64*w+k)+jw]>>js)&1ULL)<<k;
   c[4*t+w]=o;
 }
 __syncthreads();
}

// one Lanczos iteration's host algebra. grid 1, block 256 (fixed). O = [vt_a_v[0] | vt_a2_v[0]] (this iteration's inner
// products), V0in = O2 (it < 4: vt_v0[0] := O2[0] when usev0), mats = [D | E | F | D2] and mask for upd256_v3 / updF256_v3g.
extern "C" __global__ void __launch_bounds__(256,1) la_alg256(const u64* __restrict__ O, const u64* __restrict__ V0in,
    const int usev0, u64* S, int* ctl, u64* mats, u64* mask){
 __shared__ u64 sb[1024];
 __shared__ u64 Pb[2][8], Qb[2][8];
 __shared__ unsigned wm[2][8], wm2[2][8];
 __shared__ int cols[256], pos[256], mk[256], s0[256];
 const int t=threadIdx.x, lane=t&31, wp=t>>5;
 const u64* A0=O; const u64* A2=O+1024;
 const int halt=ctl[0];
 int st=halt;
 if(!st){
   const u64 r=A0[4*t]|A0[4*t+1]|A0[4*t+2]|A0[4*t+3];
   if(!__syncthreads_or(r!=0ULL)) st=1;                        // if not vt_a_v[0].any(): break
 }
 int dim=0; u64 k0=0ULL,k1=0ULL,k2=0ULL,k3=0ULL;               // dim0 and the chosen-column mask M0
 if(!st){
   // ---- find_nonsingular_sub (la_fns256f) ----
   const int ld=ctl[1];
   mk[t]=0;
   __syncthreads();
   if(t<ld){ const int c=ctl[16+t]; cols[255-t]=c; mk[c]=1; }
   __syncthreads();
   if(t==0){ int j=0; for(int i=0;i<256;i++) if(!mk[i]) cols[j++]=i; }
   __syncthreads();
   pos[cols[t]]=t;
   __syncthreads();
   const unsigned mp=(unsigned)pos[t];
   u64 r0=A0[4*t],r1=A0[4*t+1],r2=A0[4*t+2],r3=A0[4*t+3];
   const u64 ib=1ULL<<(t&63); const int iq=t>>6;
   u64 r4= iq==0? ib:0ULL, r5= iq==1? ib:0ULL, r6= iq==2? ib:0ULL, r7= iq==3? ib:0ULL;
   int par=0, fail=0;
   for(int i=0;i<256;i++){
     const int ci=cols[i], wh=ci>>6, sh=ci&63;
     unsigned v=(mp>=(unsigned)i && ((SEL4(r0,r1,r2,r3,wh)>>sh)&1ULL)) ? mp : 1024u;
     v=wmin(v);
     if(lane==0) wm[par][wp]=v;
     __syncthreads();
     unsigned p=wm[par][0];
     #pragma unroll
     for(int k=1;k<8;k++) { const unsigned q_=wm[par][k]; p= q_<p? q_:p; }
     int off=0;
     if(p==1024u){                                             // no pivot in the t part: look in the identity part
       off=4;
       unsigned v2=(mp>=(unsigned)i && ((SEL4(r4,r5,r6,r7,wh)>>sh)&1ULL)) ? mp : 1024u;
       v2=wmin(v2);
       if(lane==0) wm2[par][wp]=v2;
       __syncthreads();
       p=wm2[par][0];
       #pragma unroll
       for(int k=1;k<8;k++) { const unsigned q_=wm2[par][k]; p= q_<p? q_:p; }
       if(p==1024u){ fail=1; break; }                          // uniform (p comes from shared memory)
     }
     const int b=cols[p];
     if(t==b){ Pb[par][0]=r0; Pb[par][1]=r1; Pb[par][2]=r2; Pb[par][3]=r3; Pb[par][4]=r4; Pb[par][5]=r5; Pb[par][6]=r6; Pb[par][7]=r7; }
     if(t==ci){ Qb[par][0]=r0; Qb[par][1]=r1; Qb[par][2]=r2; Qb[par][3]=r3; Qb[par][4]=r4; Qb[par][5]=r5; Qb[par][6]=r6; Qb[par][7]=r7; }
     __syncthreads();
     const u64 P0=Pb[par][0],P1=Pb[par][1],P2=Pb[par][2],P3=Pb[par][3],P4=Pb[par][4],P5=Pb[par][5],P6=Pb[par][6],P7=Pb[par][7];
     if(t==b){ r0=Qb[par][0]; r1=Qb[par][1]; r2=Qb[par][2]; r3=Qb[par][3]; r4=Qb[par][4]; r5=Qb[par][5]; r6=Qb[par][6]; r7=Qb[par][7]; }
     if(t==ci){ r0=P0; r1=P1; r2=P2; r3=P3; r4=P4; r5=P5; r6=P6; r7=P7; }
     const u64 tw= off? SEL4(r4,r5,r6,r7,wh) : SEL4(r0,r1,r2,r3,wh);
     if((tw>>sh)&1ULL){ r0^=P0; r1^=P1; r2^=P2; r3^=P3; r4^=P4; r5^=P5; r6^=P6; r7^=P7; }
     if(!off){
       if(t==ci){ r0=P0; r1=P1; r2=P2; r3=P3; r4=P4; r5=P5; r6=P6; r7=P7; }
       if(t==0) s0[dim]=ci;
       dim++;
       const u64 bb=1ULL<<sh; if(wh==0) k0|=bb; else if(wh==1) k1|=bb; else if(wh==2) k2|=bb; else k3|=bb;
     }
     par^=1;
   }
   __syncthreads();
   if(fail){ st=2; dim=0; }
   else {
     S[GS_WI0+4*t]=r4; S[GS_WI0+4*t+1]=r5; S[GS_WI0+4*t+2]=r6; S[GS_WI0+4*t+3]=r7;      // winv[0] = the identity half
     const u64* M1=S+GS_M1;
     if(((k0|M1[0])&(k1|M1[1])&(k2|M1[2])&(k3|M1[3]))!=~0ULL) st=3;
   }
 }
 if(st){                                                       // halt (now or earlier): identity update, no F pass
   for(int k=t;k<4096;k+=256) mats[k]=0ULL;
   if(t<4) mask[t]=~0ULL;
   if(t==0){ if(!halt){ ctl[0]=st; ctl[5]=dim; ctl[6]=ctl[4]+1; } ctl[2]=0; ctl[3]++; }
   return;
 }
 // ---- the update matrices (gpu_block_lanczos256, LA_V3 host branch) ----
 const u64* M1=S+GS_M1;
 const int useF= (M1[0]&M1[1]&M1[2]&M1[3])!=~0ULL;
 if(t<4) mask[t]=SEL4(k0,k1,k2,k3,t);                           // M0 (also the column mask of E below)
 if(usev0){ for(int q=0;q<4;q++) S[GS_V00+4*t+q]=V0in[4*t+q]; }  // it < 4: vt_v0[0] = O2[0]
 for(int q=0;q<4;q++){ const u64 mq=SEL4(k0,k1,k2,k3,q); S[GS_T1+4*t+q]=(A2[4*t+q]&mq)^A0[4*t+q]; }
 g_mul(S+GS_WI0,S+GS_T1,mats,sb,1,0);                           // D  = winv0 ((vt_a2 & M0) ^ vt_a) ^ I
 g_tr(mats,S+GS_T2,sb);
 g_mul(S+GS_T2,S+GS_V00,S+GS_NX,sb,0,0);                        // next = D^T vt_v0[0]
 g_mul(S+GS_WI1,A0,mats+1024,sb,0,mask);                         // E  = (winv1 vt_a) & M0
 g_tr(mats+1024,S+GS_T2,sb);
 g_mul(S+GS_T2,S+GS_V01,S+GS_T3,sb,0,0);                        // next ^= E^T vt_v0[1]
 for(int q=0;q<4;q++) S[GS_NX+4*t+q]^=S[GS_T3+4*t+q];
 if(useF){
   g_mul(S+GS_VA1,S+GS_WI1,S+GS_T1,sb,1,0);                     // f = vt_a_v[1] winv1 ^ I
   g_mul(S+GS_WI2,S+GS_T1,S+GS_T3,sb,0,0);                      // f = winv2 f
   for(int q=0;q<4;q++){ const u64 m0=SEL4(k0,k1,k2,k3,q);
     S[GS_T4+4*t+q]=((S[GS_VA21+4*t+q]&M1[q])^S[GS_VA1+4*t+q])&m0; }   // f2
   g_mul(S+GS_T3,S+GS_T4,mats+2048,sb,0,0);                     // F = f f2
   g_tr(mats+2048,S+GS_T2,sb);
   g_mul(S+GS_T2,S+GS_V02,S+GS_T3,sb,0,0);                      // next ^= F^T vt_v0[2]
   for(int q=0;q<4;q++) S[GS_NX+4*t+q]^=S[GS_T3+4*t+q];
 } else { for(int q=0;q<4;q++) mats[2048+4*t+q]=0ULL; }
 g_mul(S+GS_WI0,S+GS_V00,mats+3072,sb,0,0);                     // D2 = winv0 vt_v0[0]
 // ---- rotate (each thread its own row; g_mul ended with a barrier) ----
 for(int q=0;q<4;q++){
   const int o=4*t+q;
   S[GS_WI2+o]=S[GS_WI1+o]; S[GS_WI1+o]=S[GS_WI0+o];
   S[GS_V02+o]=S[GS_V01+o]; S[GS_V01+o]=S[GS_V00+o]; S[GS_V00+o]=S[GS_NX+o];
   S[GS_VA1+o]=A0[o]; S[GS_VA21+o]=A2[o];
 }
 if(t<dim) ctl[16+t]=s0[t];
 if(t<4) S[GS_M1+t]=SEL4(k0,k1,k2,k3,t);
 if(t==0){ ctl[1]=dim; ctl[2]=useF; ctl[3]++; ctl[4]++; }
}

// updF256_v3 gated on ctl[2] (useF of this iteration, written by la_alg256): pn ^= p2*F, or nothing.
extern "C" __global__ void __launch_bounds__(512) updF256_v3g(const ulonglong4* __restrict__ p2, ulonglong4* __restrict__ pn,
    const u64* __restrict__ mats, long long n, const int* __restrict__ ctl){
 if(ctl[2]==0) return;
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
