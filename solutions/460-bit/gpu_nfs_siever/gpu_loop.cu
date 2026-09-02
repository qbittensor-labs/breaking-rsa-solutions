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

// Full per-special-q GPU pipeline (Phase 2d): factor base built ONCE (root-finding),
// then per special-q: GPU PROJECTION (roots->lines via basis) + scatter + scan.
// Validates projected lines reproduce CADO's 163 relations and times the whole pipeline.
#include "rootfind.c"
#include <math.h>
#include <vector>
#include <gmp.h>
#include <sys/time.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <omp.h>
#include <cuda_runtime.h>
#include "u256.cuh"
#define MAXP 48
// Small-prime resieve skip: primes < SPB are NOT resieved (their per-cell walk dominates the
// resieve pass: sum(1/p) for p<256 is ~half of all factor-base hits). Instead the cofactor kernel
// and the CPU finalizer trial-divide each survivor's norm by these primes directly (105k survivors
// x ~54 primes is far cheaper than re-walking ~half the sieve array). Relations are IDENTICAL:
// trial division is the ground truth that resieve merely optimizes.  SPB=0 disables (baseline).
#ifndef SPB
#define SPB 256
#endif
#ifndef LPB_VAL
#define LPB_VAL 536870912ULL    /* large-prime bound 2^29 (lowers the unique-relation matrix bar ~5x -> rescues weak-poly keys; build -DLPB_VAL=1073741824ULL for 2^30) */
#endif
__constant__ uint32_t cSP[64]; __constant__ int cNSP;   // small primes < SPB (device, cofactor)
static uint32_t hSP[64]; static int hNSP=0;             // small primes < SPB (host, CPU finalizer)
static void build_smallp(){ for(uint32_t p=2;p<(uint32_t)SPB && hNSP<64;p++){ int pr=1; for(uint32_t d=2;d*d<=p;d++) if(p%d==0){pr=0;break;} if(pr) hSP[hNSP++]=p; } }
static double wall_s(){ struct timeval t; gettimeofday(&t,0); return t.tv_sec+t.tv_usec*1e-6; }
long g_nship_sum=0,g_lat_cnt=0;
double g_prof[6]={0,0,0,0,0,0};   // [PROF] per-group GPU time: project,scatter,scan,resieve,cofactor
static u256 parse_u256(const char* s){ mpz_t z; mpz_init_set_str(z,s,10); int neg=mpz_sgn(z)<0; mpz_t mag; mpz_init(mag); mpz_abs(mag,z);
  u256 r=u_zero(); size_t cnt=0; mpz_export(r.w,&cnt,-1,8,0,0,mag); if(neg)r=u_neg(r); mpz_clear(z);mpz_clear(mag); return r; }
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
// ---- polynomial: loaded at RUNTIME from a CADO .poly (degree-5 alg + linear rat) ----
static char* C[6]={0,0,0,0,0,0};      // algebraic coeffs c0..c5 (decimal strings)
static char* sY0=0; static char* sY1=0;  // rational poly Y0 + Y1*x
static double G_SKEW=0;                // skew (from .poly)
static char* dupstr(const char* s){ size_t n=strlen(s); char* r=(char*)malloc(n+1); memcpy(r,s,n+1); return r; }
static void rstrip(char* s){ size_t n=strlen(s); while(n&&(s[n-1]=='\n'||s[n-1]=='\r'||s[n-1]==' '||s[n-1]=='\t'))s[--n]=0; }
// Parse a CADO .poly: lines "c0: ..".."c5: ..", "Y0: ..", "Y1: ..", "skew: ..". Other lines ignored.
static void load_poly(const char* path){
  FILE* f=fopen(path,"r"); if(!f){ fprintf(stderr,"poly: cannot open %s\n",path); exit(1); }
  char line[8192];
  while(fgets(line,sizeof(line),f)){
    char* p=line; while(*p==' '||*p=='\t')p++;
    char* colon=strchr(p,':'); if(!colon||p[0]=='#') continue;
    *colon=0; char* key=p; rstrip(key); char* val=colon+1; while(*val==' '||*val=='\t')val++; rstrip(val);
    if(!*val) continue;
    if(key[0]=='c'&&key[1]>='0'&&key[1]<='5'&&key[2]==0) C[key[1]-'0']=dupstr(val);
    else if(!strcmp(key,"Y0")) sY0=dupstr(val);
    else if(!strcmp(key,"Y1")) sY1=dupstr(val);
    else if(!strcmp(key,"skew")) G_SKEW=atof(val);
  }
  fclose(f);
  for(int k=0;k<6;k++) if(!C[k]){ fprintf(stderr,"poly: missing c%d (need a degree-5 .poly)\n",k); exit(1); }
  if(!sY0||!sY1){ fprintf(stderr,"poly: missing Y0/Y1\n"); exit(1); }
  if(G_SKEW<=0){ fprintf(stderr,"poly: missing/invalid skew\n"); exit(1); }
  fprintf(stderr,"poly loaded: c5=%s skew=%.3f Y1=%s\n",C[5],G_SKEW,sY1);
}
#define LIM 14000000u
#define I2 4096
#define J  4096
#define W  (2*I2)
static const size_t NCELL=(size_t)W*J;
static const uint32_t T=65536;   // col/lat boundary: balanced column method up to here (medium primes were lat-imbalanced)
static inline long long smod(long long a,long long p){ long long r=a%p; return r<0?r+p:r; }
static u64 ginv2(u64 a,u64 m){ long long t=0,nt=1,r=m,nr=a%m; while(nr){long long q=r/nr,tmp;tmp=t-q*nt;t=nt;nt=tmp;tmp=r-q*nr;r=nr;nr=tmp;} if(r>1)return 0; if(t<0)t+=m; return (u64)t; }
static u64 evf(char*const cc[6],u64 x,u64 m){u64 v=0;for(int k=5;k>=0;k--)v=addmod(mulmod(v,x,m),strmod(cc[k],m),m);return v;}
static u64 evfp(char*const cc[6],u64 x,u64 m){u64 v=0;for(int k=5;k>=1;k--)v=addmod(mulmod(v,x,m),mulmod(strmod(cc[k],m),(u64)k%m,m),m);return v;}

// ---- factor base, built ONCE (special-q independent): (m, r, type, logp) per side/size group ----
// groups: g = side*2 + (m>T?1:0) ; side 0=rational,1=algebraic ; type 0=affine,1=projective
std::vector<uint32_t> fM[4],fR[4],fP[4]; std::vector<uint8_t> fT[4],fL[4];
static u64 g_pbase;   // base prime for current addfb calls (set per prime in build_fb)
static void addfb(int side,u64 m,u64 r,int type,int lp){ int g=side*2+(m>T?1:0);
  fM[g].push_back((uint32_t)m); fR[g].push_back((uint32_t)r); fT[g].push_back((uint8_t)type); fL[g].push_back((uint8_t)lp);
  fP[g].push_back((uint32_t)g_pbase); }
// general prime-power root lifting for algebraic side (handles RAMIFIED primes where Hensel breaks)
static void lift_alg_powers(u64 p,u64* r0,int nr0,int lp){
  if((u64)p*p>LIM) return;
  u64 cur[128]; int nc=0; for(int i=0;i<nr0 && nc<128;i++) cur[nc++]=r0[i]%p;
  u64 pk=p;
  while(pk<=LIM/p){
    u64 pk1=pk*p; u64 nx[128]; int nn=0;
    for(int i=0;i<nc && nn<128;i++){ u64 r=cur[i]; u64 fpp=evfp(C,r,p);
      if(fpp){ u64 fp1=evfp(C,r,pk1), fr=evf(C,r,pk1), iv=ginv2(fp1,pk1);  // unramified: Hensel
        u64 r1=submod(r%pk1,mulmod(fr,iv,pk1),pk1); if(evf(C,r1,pk1)==0){ nx[nn++]=r1; addfb(1,pk1,r1,0,lp); } }
      else for(u64 t=0;t<p && nn<128;t++){ u64 rr=r+t*pk; if(evf(C,rr,pk1)==0){ nx[nn++]=rr; addfb(1,pk1,rr,0,lp); } } // ramified: all lifts
    }
    if(nn==0) break; for(int i=0;i<nn;i++)cur[i]=nx[i]; nc=nn; pk=pk1;
  }
}
static void build_fb(){
  uint8_t* comp=(uint8_t*)calloc(LIM+1,1);
  for(u64 i=2;i*i<=LIM;i++) if(!comp[i]) for(u64 j=i*i;j<=LIM;j+=i) comp[j]=1;
  u64 roots[8];
  for(u64 p=2;p<=LIM;p++){ if(comp[p])continue; int lp=(int)lround(log2((double)p)); g_pbase=p;
    int nr=find_roots(C,p,roots);
    for(int k=0;k<nr;k++) addfb(1,p,roots[k],0,lp);
    lift_alg_powers(p,roots,nr,lp);                                  // prime powers (ramified-safe)
    if(strmod(C[5],p)==0) addfb(1,p,0,1,lp);                        // algebraic projective (p | leading coeff)
    u64 Y1m=strmod(sY1,p);
    if(Y1m){ u64 m=mulmod(strmod(sY0,p),ginv2(Y1m,p),p); m=(p-m)%p; addfb(0,p,m,0,lp);
      if(p*p<=LIM){ u64 pk=p; while(pk<=LIM/p){ u64 pk1=pk*p,Y1k=strmod(sY1,pk1); if(!Y1k)break;
        u64 mk=mulmod(strmod(sY0,pk1),ginv2(Y1k,pk1),pk1); mk=(pk1-mk)%pk1; addfb(0,pk1,mk,0,lp); pk=pk1;} } }
    else addfb(0,p,0,1,lp);                                          // rational projective
  }
  free(comp);
}
// ---- GPU: projection (roots -> lines) ----
__device__ __forceinline__ long long smd(long long a,long long m){ long long r=a%m; return r<0?r+m:r; }
__device__ uint32_t ginv_d(long long a,long long m){ long long t=0,nt=1,r=m,nr=a%m; if(nr<0)nr+=m;
  while(nr){ long long q=r/nr,tmp; tmp=t-q*nt;t=nt;nt=tmp; tmp=r-q*nr;r=nr;nr=tmp; } if(r>1)return 0xFFFFFFFFu; if(t<0)t+=m; return (uint32_t)t; }
__global__ void project(const uint32_t* M,const uint32_t* Rin,const uint8_t* TY,uint32_t* Rout,int n,
                        long long a0,long long b0,long long a1,long long b1){
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
    long long m=M[i],A,B;
    if(TY[i]){ A=smd(-b0,m); B=smd(b1,m); }                          // projective: b0*i+b1*j≡0
    else { long long r=Rin[i]; A=smd(-(a0-r*b0),m); B=smd(a1-r*b1,m); }  // affine: j≡R i, R=-(a0-r b0)/(a1-r b1)
    if(B==0){ if(A==0){Rout[i]=0xFFFFFFFFu;continue;}                // vertical: A*i≡0 mod m
      long long g=A,mm=m; while(mm){long long t=g%mm;g=mm;mm=t;}
      Rout[i]=0x80000000u | (uint32_t)(m/g);  continue; }            // flag + step (i≡0 mod step)
    uint32_t inv=ginv_d(B,m); if(inv==0xFFFFFFFFu){ Rout[i]=0xFFFFFFFFu; continue; }
    Rout[i]=(uint32_t)(( (long long)A*inv)%m);
  }
}
// ---- scatter (skip invalid R) ----
__global__ void scat_col(uint8_t* arr,const uint32_t* M,const uint32_t* R,const uint8_t* L,int n){
  for(long long w=(long long)blockIdx.x*blockDim.x+threadIdx.x; w<(long long)n*W; w+=(long long)gridDim.x*blockDim.x){
    int line=(int)(w/W),col=(int)(w%W),i=col-I2; uint32_t m=M[line],r=R[line],lp=L[line]; if(r==0xFFFFFFFFu)continue;
    if(r&0x80000000u){ uint32_t step=r&0x7FFFFFFFu;                   // vertical line: i≡0 mod step, all j
      if((((i%(int)step)+(int)step)%(int)step)==0) for(int j=0;j<J;j++){ size_t idx=(size_t)(i+I2)*J+j; atomicAdd((unsigned*)&arr[idx&~3ull],(unsigned)lp<<(8*(idx&3))); } continue; }
    uint32_t im=(uint32_t)(((i%(int)m)+(int)m)%(int)m), j0=(uint32_t)(((uint64_t)r*im)%m);
    for(uint32_t j=j0;j<J;j+=m){ size_t idx=(size_t)(i+I2)*J+j; atomicAdd((unsigned*)&arr[idx&~3ull],(unsigned)lp<<(8*(idx&3))); }
  }
}
__device__ __forceinline__ long long fdv(long long a,long long b){ long long q=a/b,r=a%b; if(r!=0&&((r<0)!=(b<0)))q--; return q; }
__global__ void scat_lat(uint8_t* arr,const uint32_t* M,const uint32_t* R,const uint8_t* L,int n){
  for(int line=blockIdx.x*blockDim.x+threadIdx.x;line<n;line+=gridDim.x*blockDim.x){
    long long m=M[line]; uint32_t rr=R[line]; unsigned lp=L[line]; if(rr==0xFFFFFFFFu)continue;
    if(rr&0x80000000u){ uint32_t step=rr&0x7FFFFFFFu;                 // vertical line (rare for large m)
      for(int i=-I2;i<I2;i++) if((((i%(int)step)+(int)step)%(int)step)==0) for(int j=0;j<J;j++){ size_t idx=(size_t)(i+I2)*J+j; atomicAdd((unsigned*)&arr[idx&~3ull],(unsigned)lp<<(8*(idx&3))); } continue; }
    long long r=rr; long long p1=1,q1=r,p2=0,q2=m;
    for(int it=0;it<64;it++){ long long n1=p1*p1+q1*q1,n2=p2*p2+q2*q2; if(n2<n1){long long a=p1;p1=p2;p2=a;a=q1;q1=q2;q2=a;n1=n2;} if(!n1)break;
      long long dot=p1*p2+q1*q2,mu; if(dot>=0)mu=(2*dot+n1)/(2*n1);else mu=-((-2*dot+n1)/(2*n1)); if(!mu)break; p2-=mu*p1;q2-=mu*q1; }
    long long D=p1*q2-p2*q1; if(!D)continue;
    int cI[2]={-I2,I2-1},cJ[2]={0,J-1}; long long a1m=1LL<<62,a1M=-(1LL<<62),b1m=1LL<<62,b1M=-(1LL<<62);
    for(int x=0;x<2;x++)for(int y=0;y<2;y++){ long long I_=cI[x],J_=cJ[y]; long long n1c=q2*I_-p2*J_,n2c=-q1*I_+p1*J_;
      a1m=min(a1m,n1c);a1M=max(a1M,n1c);b1m=min(b1m,n2c);b1M=max(b1M,n2c); }
    long long A1,A2,B1,B2; if(D>0){A1=fdv(a1m,D);A2=-fdv(-a1M,D);B1=fdv(b1m,D);B2=-fdv(-b1M,D);}else{A1=fdv(a1M,D);A2=-fdv(-a1m,D);B1=fdv(b1M,D);B2=-fdv(-b1m,D);}
    for(long long c1=A1;c1<=A2;c1++)for(long long c2=B1;c2<=B2;c2++){ long long i=c1*p1+c2*p2,j=c1*q1+c2*q2;
      if(i>=-I2&&i<I2&&j>=0&&j<J){ size_t off=(size_t)(i+I2)*J+j; atomicAdd((unsigned*)&arr[off&~3ull],(unsigned)lp<<(8*(off&3))); } }
  }
}
// ---- scan: survivors. log|F_f| via factoring out the dominant variable (FP32-safe, no overflow) ----
__device__ __forceinline__ float fflog2(float c0,float c1,float c2,float c3,float c4,float c5,float a,float b){
  // log2|c5 a^5 + ... + c0 b^5| = 5 log2|u| + log2|poly(t)|, u=max(|a|,|b|), t=min/max in [-1,1]
  float A=fabsf(a),B=fabsf(b); float u=A>B?A:B; if(u==0)return -1e30f;
  if(A>=B){ float t=b/a; float h=c5+t*(c4+t*(c3+t*(c2+t*(c1+t*c0)))); return 5.f*log2f(A)+log2f(fabsf(h)); }
  else    { float t=a/b; float h=c0+t*(c1+t*(c2+t*(c3+t*(c4+t*c5)))); return 5.f*log2f(B)+log2f(fabsf(h)); }
}
__global__ void scan(const uint8_t* la,const uint8_t* lr,unsigned* nsurv,
                     float a0,float b0,float a1,float b1,float logq,
                     float c0,float c1,float c2,float c3,float c4,float c5,float Y0,float Y1){
  for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<NCELL;k+=(size_t)gridDim.x*blockDim.x){
    int i=(int)(k/J)-I2, j=(int)(k%J); float a=a0*i+a1*j,b=b0*i+b1*j; if(b==0)continue;
    float nla=fflog2(c0,c1,c2,c3,c4,c5,a,b)-logq;
    float Ga=Y1*a+Y0*b; float nlr=log2f(fabsf(Ga));
    if(nla-la[k]<=78.f && nlr-lr[k]<=77.f) atomicAdd(nsurv,1u);
  }
}
// scan that ASSIGNS a survivor index per cell (-1 if not survivor)
__global__ void scan_idx(const uint8_t* la,const uint8_t* lr,int* sidx,unsigned* cnt,int slack,
                     float a0,float b0,float a1,float b1,float logq,
                     float c0,float c1,float c2,float c3,float c4,float c5,float Y0,float Y1){
  for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<NCELL;k+=(size_t)gridDim.x*blockDim.x){
    int i=(int)(k/J)-I2,j=(int)(k%J); float a=a0*i+a1*j,b=b0*i+b1*j; int keep=0;
    if(b!=0){ float nla=fflog2(c0,c1,c2,c3,c4,c5,a,b)-logq, Ga=Y1*a+Y0*b, nlr=log2f(fabsf(Ga));
      if(nla-la[k]<=(float)(58+slack) && nlr-lr[k]<=(float)(57+slack)) keep=1; }
    sidx[k]= keep? (int)atomicAdd(cnt,1u) : -1;
  }
}
// survivor gate: bit-packed mask (4.2MB, L2-resident) -> 99.7% non-survivor hits caught in L2,
// only real survivors pay the 134MB sidx[] DRAM lookup. Built once after scan_idx.
__global__ void mkmask(const int* sidx,uint32_t* bits){
  for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<NCELL;k+=(size_t)gridDim.x*blockDim.x)
    if(sidx[k]>=0) atomicOr(&bits[k>>5],1u<<(k&31)); }
__device__ __forceinline__ void rec(const uint32_t* bits,const int* sidx,uint32_t* plist,unsigned* pcnt,int i,int j,uint32_t p){
  size_t c=(size_t)(i+I2)*J+j;
  if(!((bits[c>>5]>>(c&31))&1u))return;                                 // L2-resident gate
  int s=sidx[c]; if(s<0)return; unsigned pos=atomicAdd(&pcnt[s],1u); if(pos<MAXP) plist[(size_t)s*MAXP+pos]=p; }
// resieve (record base prime per survivor); only p^1 lines (m==P). side->its plist.
__global__ void resieve_col(const uint32_t* M,const uint32_t* P,const uint32_t* R,int n,const uint32_t* bits,const int* sidx,uint32_t* plist,unsigned* pcnt){
  for(long long w=(long long)blockIdx.x*blockDim.x+threadIdx.x; w<(long long)n*W; w+=(long long)gridDim.x*blockDim.x){
    int line=(int)(w/W),col=(int)(w%W),i=col-I2; uint32_t m=M[line],r=R[line],pb=P[line]; if(r==0xFFFFFFFFu||m!=pb)continue;
    if(m<(uint32_t)SPB)continue;     // small primes trial-divided in cofactor/CPU instead of resieved
    if(r&0x80000000u){ uint32_t st=r&0x7FFFFFFFu; if((((i%(int)st)+(int)st)%(int)st)==0) for(int j=0;j<J;j++) rec(bits,sidx,plist,pcnt,i,j,pb); continue; }
    uint32_t im=(uint32_t)(((i%(int)m)+(int)m)%(int)m), j0=(uint32_t)(((uint64_t)r*im)%m);
    for(uint32_t j=j0;j<J;j+=m) rec(bits,sidx,plist,pcnt,i,(int)j,pb);
  }
}
__global__ void resieve_lat(const uint32_t* M,const uint32_t* P,const uint32_t* R,int n,const uint32_t* bits,const int* sidx,uint32_t* plist,unsigned* pcnt){
  for(int line=blockIdx.x*blockDim.x+threadIdx.x;line<n;line+=gridDim.x*blockDim.x){
    long long m=M[line]; uint32_t rr=R[line],pb=P[line]; if(rr==0xFFFFFFFFu||(uint32_t)m!=pb)continue;
    if(rr&0x80000000u){ uint32_t st=rr&0x7FFFFFFFu; for(int i=-I2;i<I2;i++) if((((i%(int)st)+(int)st)%(int)st)==0) for(int j=0;j<J;j++) rec(bits,sidx,plist,pcnt,i,j,pb); continue; }
    long long r=rr,p1=1,q1=r,p2=0,q2=m;
    for(int it=0;it<64;it++){ long long n1=p1*p1+q1*q1,n2=p2*p2+q2*q2; if(n2<n1){long long a=p1;p1=p2;p2=a;a=q1;q1=q2;q2=a;n1=n2;} if(!n1)break;
      long long dot=p1*p2+q1*q2,mu; if(dot>=0)mu=(2*dot+n1)/(2*n1);else mu=-((-2*dot+n1)/(2*n1)); if(!mu)break; p2-=mu*p1;q2-=mu*q1; }
    long long D=p1*q2-p2*q1; if(!D)continue;
    int cI[2]={-I2,I2-1},cJ[2]={0,J-1}; long long a1m=1LL<<62,a1M=-(1LL<<62),b1m=1LL<<62,b1M=-(1LL<<62);
    for(int x=0;x<2;x++)for(int y=0;y<2;y++){ long long I_=cI[x],J_=cJ[y]; long long n1c=q2*I_-p2*J_,n2c=-q1*I_+p1*J_; a1m=min(a1m,n1c);a1M=max(a1M,n1c);b1m=min(b1m,n2c);b1M=max(b1M,n2c); }
    long long A1,A2,B1,B2; if(D>0){A1=fdv(a1m,D);A2=-fdv(-a1M,D);B1=fdv(b1m,D);B2=-fdv(-b1M,D);}else{A1=fdv(a1M,D);A2=-fdv(-a1m,D);B1=fdv(b1M,D);B2=-fdv(-b1m,D);}
    for(long long c1=A1;c1<=A2;c1++)for(long long c2=B1;c2<=B2;c2++){ long long i=c1*p1+c2*p2,j=c1*q1+c2*q2;
      if(i>=-I2&&i<I2&&j>=0&&j<J) rec(bits,sidx,plist,pcnt,(int)i,(int)j,pb); }
  }
}
// CPU cofactor: divide norm by recorded primes (completely), check residual smooth in (lim,2^lpb], coprime
static int cof_smooth(mpz_t c,unsigned long lpb){
  mpz_t stk[64]; int sp=0; mpz_init_set(stk[sp++],c); int ok=1;
  while(sp>0&&ok){ mpz_t v; mpz_init_set(v,stk[--sp]); mpz_clear(stk[sp]);
    if(mpz_cmp_ui(v,1)<=0){mpz_clear(v);continue;}
    if(mpz_probab_prime_p(v,25)){ if(mpz_cmp_ui(v,LIM)<=0||mpz_sizeinbase(v,2)>lpb) ok=0; mpz_clear(v); continue; }
    if(mpz_sizeinbase(v,2)<=lpb){ ok=0; mpz_clear(v); continue; }
    mpz_t x,y,d; mpz_inits(x,y,d,NULL); mpz_set_ui(x,2);mpz_set_ui(y,2);mpz_set_ui(d,1); unsigned long cn=1; long it=0;
    while(mpz_cmp_ui(d,1)==0&&it++<(1<<22)){ mpz_mul(x,x,x);mpz_add_ui(x,x,cn);mpz_mod(x,x,v);mpz_mul(y,y,y);mpz_add_ui(y,y,cn);mpz_mod(y,y,v);mpz_mul(y,y,y);mpz_add_ui(y,y,cn);mpz_mod(y,y,v);mpz_sub(d,x,y);mpz_abs(d,d);mpz_gcd(d,d,v); if(mpz_cmp(d,v)==0){cn++;mpz_set_ui(x,2);mpz_set_ui(y,2);mpz_set_ui(d,1);} }
    if(it>=(1<<22)){ok=0;mpz_clears(x,y,d,NULL);mpz_clear(v);break;}
    if(sp<62){mpz_init_set(stk[sp++],d);mpz_divexact(v,v,d);mpz_init_set(stk[sp++],v);} mpz_clears(x,y,d,NULL);mpz_clear(v);
  }
  while(sp>0)mpz_clear(stk[--sp]); return ok;
}
// ---- GPU COFACTOR: norm -> divide Q + resieve primes (completely) -> size filter -> SMOOTHNESS ----
// All fixed-width int arithmetic (256-bit norms, u64 cofactoring), 100k survivors in parallel.
__device__ __forceinline__ void udiv_complete(u256* x,unsigned long long p){
  for(;;){ u256 t=*x; unsigned long long r=u_divmod_u64(&t,p); if(r)break; *x=t; }
}
__device__ __forceinline__ unsigned long long mulmod64(unsigned long long a,unsigned long long b,unsigned long long m){
  return (unsigned long long)((unsigned __int128)a*b%m); }
__device__ __forceinline__ unsigned long long powmod64(unsigned long long a,unsigned long long e,unsigned long long m){
  unsigned long long r=1; a%=m; while(e){ if(e&1)r=mulmod64(r,a,m); a=mulmod64(a,a,m); e>>=1; } return r; }
__device__ __forceinline__ unsigned long long gcd64(unsigned long long a,unsigned long long b){ while(b){unsigned long long t=a%b;a=b;b=t;} return a; }
__device__ bool isprime64(unsigned long long n){              // 9 bases: deterministic for n<3.3e18 (>2^61)
  if(n<2)return false; const unsigned long long B[9]={2,3,5,7,11,13,17,19,23};
  for(int k=0;k<9;k++){ if(n%B[k]==0)return n==B[k]; }
  unsigned long long d=n-1; int s=0; while(!(d&1)){d>>=1;s++;}
  for(int k=0;k<9;k++){ unsigned long long x=powmod64(B[k],d,n); if(x==1||x==n-1)continue;
    bool ok=false; for(int r=1;r<s;r++){ x=mulmod64(x,x,n); if(x==n-1){ok=true;break;} } if(!ok)return false; }
  return true; }
__device__ unsigned long long pollard64(unsigned long long n){  // capped: cofactor<=2^58 -> a factor <=2^29 -> rho splits in <~2^15
  if((n&1)==0)return 2;
  for(unsigned long long c=1;c<8;c++){ unsigned long long x=2,y=2,d=1; int it=0;
    while(d==1){ x=(mulmod64(x,x,n)+c)%n; y=(mulmod64(y,y,n)+c)%n; y=(mulmod64(y,y,n)+c)%n; unsigned long long t=x>y?x-y:y-x; d=gcd64(t,n); if(++it>200000)break; }
    if(d!=1&&d!=n)return d; }
  return 0; }
// classify a cofactor cheaply (no rho): 0=DEAD(prime>lpb), 1=GOOD(1 or prime in (lim,lpb]), 2=COMPOSITE(needs factoring)
__device__ __forceinline__ int classify(unsigned long long c,unsigned long long LPB){
  if(c==1)return 1;
  if(isprime64(c)) return (c>LIM&&c<=LPB)?1:0;
  return (c<=LPB*LPB)?2:0;                              // composite >2^(2lpb) can't be 2 primes<=2^lpb -> dead
}
// GPU does cheap massively-parallel filtering only (norm+trialdiv+MR). Composite cofactors are SHIPPED to CPU
// (rho is divergence-bound on GPU, fine on CPU). Both-sides-prime relations are emitted directly here.
__global__ void cofactor(const int* SI,const int* SJ,const unsigned* dnsurv,
    const u256* cf,u256 Y0,u256 Y1,unsigned long long Q,
    const uint32_t* PA,const unsigned* CA,const uint32_t* PR,const unsigned* CR,
    long long a0,long long b0,long long a1,long long b1,int MFB1,int MFB0,
    long long* shipA,long long* shipB,unsigned long long* shipCA,unsigned long long* shipCR,
    uint32_t* shipPA,unsigned* shipNA,uint32_t* shipPR,unsigned* shipNR,unsigned* scnt,unsigned shipcap){
  const unsigned long long LPB=LPB_VAL;          // large-prime bound (LPB_VAL; default 2^30)
  int nsurv=(int)*dnsurv;                                // read survivor count on device (no host sync needed)
  for(int s=blockIdx.x*blockDim.x+threadIdx.x;s<nsurv;s+=gridDim.x*blockDim.x){
    int i=SI[s],j=SJ[s]; long long a=a0*i+a1*j,b=b0*i+b1*j;
    if(b<0){a=-a;b=-b;} if(b==0)continue; unsigned long long ub=(unsigned long long)b;
    u256 Na=u_abs(norm_alg(cf,a,ub)), Nr=u_abs(norm_rat(Y0,Y1,a,ub));
    udiv_complete(&Na,Q);
    for(int t=0;t<cNSP;t++){ udiv_complete(&Na,cSP[t]); udiv_complete(&Nr,cSP[t]); }  // small primes (not resieved)
    unsigned na=CA[s]; if(na>MAXP)na=MAXP; for(unsigned t=0;t<na;t++) udiv_complete(&Na,PA[(size_t)s*MAXP+t]);
    unsigned nr=CR[s]; if(nr>MAXP)nr=MAXP; for(unsigned t=0;t<nr;t++) udiv_complete(&Nr,PR[(size_t)s*MAXP+t]);
    if(u_bits(Na)>MFB1||u_bits(Nr)>MFB0)continue;          // size filter (residual now <=64 bits)
    if(classify(Nr.w[0],LPB)==0)continue;                  // rational(smaller) side first
    if(classify(Na.w[0],LPB)==0)continue;
    unsigned long long ua=a<0?-a:a; if(gcd64(ua,(unsigned long long)b)!=1)continue;    // coprime
    // ship every valid candidate (both-prime + composite) with its resieved prime list -> CPU finalizes + formats
    unsigned pos=atomicAdd(scnt,1u); if(pos>=shipcap)continue;
    shipA[pos]=a; shipB[pos]=b; shipCA[pos]=Na.w[0]; shipCR[pos]=Nr.w[0];
    shipNA[pos]=na; for(unsigned t=0;t<na;t++) shipPA[(size_t)pos*MAXP+t]=PA[(size_t)s*MAXP+t];
    shipNR[pos]=nr; for(unsigned t=0;t<nr;t++) shipPR[(size_t)pos*MAXP+t]=PR[(size_t)s*MAXP+t];
  }
}
// ---- host u64 cofactoring (composites are <=2^58 -> native u64, ~100x faster than GMP rho) ----
static unsigned long long h_mulmod(unsigned long long a,unsigned long long b,unsigned long long m){ return (unsigned long long)((unsigned __int128)a*b%m); }
static unsigned long long h_powmod(unsigned long long a,unsigned long long e,unsigned long long m){ unsigned long long r=1;a%=m; while(e){if(e&1)r=h_mulmod(r,a,m);a=h_mulmod(a,a,m);e>>=1;} return r; }
static unsigned long long h_gcd(unsigned long long a,unsigned long long b){ while(b){unsigned long long t=a%b;a=b;b=t;} return a; }
// ---- Montgomery arithmetic (modulus < 2^63; our cofactors <= 2^58). Replaces __int128%% division in the rho/MR hot path. ----
static inline unsigned long long mont_n0(unsigned long long m){ unsigned long long inv=m; for(int i=0;i<6;i++) inv*=2-m*inv; return ~inv+1; } // -m^{-1} mod 2^64
static inline unsigned long long montmul(unsigned long long a,unsigned long long b,unsigned long long m,unsigned long long n0){
  unsigned __int128 T=(unsigned __int128)a*b; unsigned long long u=(unsigned long long)T*n0;
  unsigned __int128 res=(T+(unsigned __int128)u*m)>>64; return res>=m?(unsigned long long)(res-m):(unsigned long long)res; }
static inline unsigned long long to_mont(unsigned long long a,unsigned long long m,unsigned long long n0,unsigned long long R2){ return montmul(a,R2,m,n0); }
static bool h_isprime(unsigned long long n){ if(n<2)return false; const unsigned long long B[9]={2,3,5,7,11,13,17,19,23};
  for(int k=0;k<9;k++){if(n%B[k]==0)return n==B[k];}
  unsigned long long n0=mont_n0(n),Rmod=(unsigned long long)(((unsigned __int128)1<<64)%n),R2=(unsigned long long)(((unsigned __int128)Rmod*Rmod)%n),one=Rmod,nm1=n-Rmod;
  unsigned long long d=n-1;int s=0;while(!(d&1)){d>>=1;s++;}
  for(int k=0;k<9;k++){ unsigned long long b=to_mont(B[k]%n,n,n0,R2),x=one,e=d; while(e){if(e&1)x=montmul(x,b,n,n0);b=montmul(b,b,n,n0);e>>=1;}
    if(x==one||x==nm1)continue; bool ok=false; for(int r=1;r<s;r++){x=montmul(x,x,n,n0); if(x==nm1){ok=true;break;}} if(!ok)return false; } return true; }
static unsigned long long h_pollard(unsigned long long n){ if(!(n&1))return 2;   // Brent's rho in Montgomery domain (batched gcd)
  unsigned long long n0=mont_n0(n),Rmod=(unsigned long long)(((unsigned __int128)1<<64)%n),R2=(unsigned long long)(((unsigned __int128)Rmod*Rmod)%n),one=Rmod;
  for(unsigned long long c=1;c<20;c++){ unsigned long long cm=to_mont(c%n,n,n0,R2);
    unsigned long long y=to_mont(2%n,n,n0,R2),d=1,r=1,q=one,x=0,ys=0;
    while(d==1){ x=y; for(unsigned long long i=0;i<r;i++){ y=montmul(y,y,n,n0); y+=cm; if(y>=n)y-=n; }
      unsigned long long k=0;
      while(k<r&&d==1){ ys=y; unsigned long long m=r-k; if(m>128)m=128;
        for(unsigned long long i=0;i<m;i++){ y=montmul(y,y,n,n0); y+=cm; if(y>=n)y-=n; unsigned long long t=x>y?x-y:y-x; q=montmul(q,t,n,n0); }
        d=h_gcd(q,n); k+=m; }
      r<<=1; }
    if(d!=1&&d!=n)return d;
    if(d==n){ do{ ys=montmul(ys,ys,n,n0); ys+=cm; if(ys>=n)ys-=n; unsigned long long t=x>ys?x-ys:ys-x; d=h_gcd(t,n);}while(d==1); if(d!=1&&d!=n)return d; }
  } return 0; }
static bool h_smooth(unsigned long long c,unsigned long long LPB){ if(c==1)return true; if(h_isprime(c))return c>LIM&&c<=LPB;
  if(c>LPB*LPB)return false; unsigned long long d=h_pollard(c);if(!d||d==c)return false;unsigned long long e=c/d;
  return d>LIM&&d<=LPB&&h_isprime(d)&&e>LIM&&e<=LPB&&h_isprime(e); }
__global__ void mksij(const int* sidx,int* SI,int* SJ){     // build survivor (i,j) on GPU (avoid host NCELL scan/sq)
  for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<NCELL;k+=(size_t)gridDim.x*blockDim.x){
    int s=sidx[k]; if(s>=0){ SI[s]=(int)(k/J)-I2; SJ[s]=(int)(k%J);} } }
// skewed Lagrange reduction of the special-q lattice (matches CADO; validated)
static void skew_reduce(long long q,long long s,double S,long long out[4]){
  double rs=sqrt(S); long long v0i=q,v0j=0,v1i=s,v1j=1;
  for(int it=0;it<200;it++){
    double n0=(v0i/rs)*(v0i/rs)+(v0j*rs)*(v0j*rs), n1=(v1i/rs)*(v1i/rs)+(v1j*rs)*(v1j*rs);
    if(n1<n0){ long long t=v0i;v0i=v1i;v1i=t; t=v0j;v0j=v1j;v1j=t; }
    double d0=(double)v0i*v0i/S+(double)v0j*v0j*S; if(d0==0)break;
    double dot=(double)v0i*v1i/S+(double)v0j*v1j*S; long long mu=(long long)llround(dot/d0);
    if(mu==0)break; v1i-=mu*v0i; v1j-=mu*v0j;
  }
  out[0]=v0i;out[1]=v0j;out[2]=v1i;out[3]=v1j;
}
static int isprime_q(long long q){ if(q<2)return 0; for(long long d=2;d*d<=q;d++) if(q%d==0)return 0; return 1; }
// factor a cofactor residual (<=2^58, all prime factors must be in (LIM, 2^30]) -> append primes to arr; false if invalid
static bool h_factor_cof(unsigned long long c,unsigned long long* arr,int* n){
  const unsigned long long L=LPB_VAL;
  if(c==1)return true;
  if(h_isprime(c)){ if(c>LIM&&c<=L){arr[(*n)++]=c;return true;} return false; }
  if(c>L*L)return false;
  unsigned long long d=h_pollard(c); if(!d||d==c)return false; unsigned long long e=c/d;
  if(!(d>LIM&&d<=L&&h_isprime(d)))return false; if(!(e>LIM&&e<=L&&h_isprime(e)))return false;
  arr[(*n)++]=d; arr[(*n)++]=e; return true;
}
static int cmp_u64(const void*a,const void*b){ unsigned long long x=*(const unsigned long long*)a,y=*(const unsigned long long*)b; return x<y?-1:x>y?1:0; }
static mpz_t G_cc[6],G_y0,G_y1;          // pre-parsed poly coeffs (parsed ONCE, read-only in parallel format loop)
// ---- relations stream into a RAM buffer (NOT disk): the validator gives 1GB /tmp but 85GB RAM,
// so a deployable GNFS must keep relations resident in memory. Grows geometrically.
static char* g_rambuf=0; static size_t g_ramlen=0, g_ramcap=0;
static inline void rel_emit(const char* s){ size_t n=strlen(s);
  if(g_ramlen+n+1>g_ramcap){ size_t nc=g_ramcap?g_ramcap:(size_t)64<<20; while(nc<g_ramlen+n+1)nc<<=1; g_rambuf=(char*)realloc(g_rambuf,nc); g_ramcap=nc; }
  memcpy(g_rambuf+g_ramlen,s,n); g_ramlen+=n; }
int main(int argc,char** argv){
  if(argc<4){ fprintf(stderr,"usage: %s <poly.cado> <qmin> <qmax> [dump_file] [rel_target]\n",argv[0]); return 2; }
  { int procs=omp_get_num_procs(); int nt=procs<24?procs:24;   // CAP threads: per-lattice omp region oversubscribes on big nodes -> thrash
    if(getenv("OMP_NUM_THREADS")==0) omp_set_num_threads(nt);
    fprintf(stderr,"omp threads=%d (procs=%d)\n",getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):nt,procs); }
  load_poly(argv[1]);
  fprintf(stderr,"root-finding factor base (ONCE)...\n"); clock_t t0=clock();
  build_fb();
  printf("FB built in %.1fs: ratS=%zu ratL=%zu algS=%zu algL=%zu\n",
    (double)(clock()-t0)/CLOCKS_PER_SEC,fM[0].size(),fM[1].size(),fM[2].size(),fM[3].size());
  build_smallp();
  cudaMemcpyToSymbol(cSP,hSP,sizeof(uint32_t)*hNSP); cudaMemcpyToSymbol(cNSP,&hNSP,sizeof(int));
  fprintf(stderr,"small-prime resieve skip: SPB=%d (%d primes trial-divided in cofactor)\n",SPB,hNSP);

  // ---- upload FB ONCE ----
  uint32_t *dM[4],*dRin[4],*dRout[4],*dP[4]; uint8_t *dT[4],*dL[4]; int ng[4];
  for(int g=0;g<4;g++){ ng[g]=fM[g].size();
    CK(cudaMalloc(&dM[g],4*ng[g]));CK(cudaMalloc(&dRin[g],4*ng[g]));CK(cudaMalloc(&dRout[g],4*ng[g]));CK(cudaMalloc(&dT[g],ng[g]));CK(cudaMalloc(&dL[g],ng[g]));CK(cudaMalloc(&dP[g],4*ng[g]));
    cudaMemcpy(dM[g],fM[g].data(),4*ng[g],cudaMemcpyHostToDevice);cudaMemcpy(dRin[g],fR[g].data(),4*ng[g],cudaMemcpyHostToDevice);
    cudaMemcpy(dT[g],fT[g].data(),ng[g],cudaMemcpyHostToDevice);cudaMemcpy(dL[g],fL[g].data(),ng[g],cudaMemcpyHostToDevice);
    cudaMemcpy(dP[g],fP[g].data(),4*ng[g],cudaMemcpyHostToDevice); }
  // ---- allocate ALL working buffers ONCE (max-sized; per-sq cudaMalloc was fixed overhead) ----
  const unsigned MAXSURV=300000;
  uint8_t *dLa,*dLr; CK(cudaMalloc(&dLa,NCELL));CK(cudaMalloc(&dLr,NCELL));
  int* dSidx; CK(cudaMalloc(&dSidx,sizeof(int)*NCELL)); unsigned* dCnt; CK(cudaMalloc(&dCnt,4));
  uint32_t* dBits; CK(cudaMalloc(&dBits,(NCELL+31)/32*4));
  uint32_t *dPa,*dPr; unsigned *dCa,*dCr; int *dSI,*dSJ;
  CK(cudaMalloc(&dPa,(size_t)MAXSURV*MAXP*4));CK(cudaMalloc(&dPr,(size_t)MAXSURV*MAXP*4));
  CK(cudaMalloc(&dCa,(size_t)MAXSURV*4));CK(cudaMalloc(&dCr,(size_t)MAXSURV*4));
  CK(cudaMalloc(&dSI,4*MAXSURV));CK(cudaMalloc(&dSJ,4*MAXSURV));
  const unsigned MAXSHIP=16384;                  // no-dead-side candidates per lattice (~255 typical); cap generously
  long long *dsA,*dsB; unsigned long long *dsCA,*dsCR; unsigned *dSC;
  uint32_t *dsPA,*dsPR; unsigned *dsNA,*dsNR;
  CK(cudaMalloc(&dsA,8*MAXSHIP));CK(cudaMalloc(&dsB,8*MAXSHIP));CK(cudaMalloc(&dsCA,8*MAXSHIP));CK(cudaMalloc(&dsCR,8*MAXSHIP));CK(cudaMalloc(&dSC,4));
  CK(cudaMalloc(&dsPA,(size_t)MAXSHIP*MAXP*4));CK(cudaMalloc(&dsPR,(size_t)MAXSHIP*MAXP*4));CK(cudaMalloc(&dsNA,MAXSHIP*4));CK(cudaMalloc(&dsNR,MAXSHIP*4));
  u256 hcf[6]; for(int k=0;k<6;k++)hcf[k]=parse_u256(C[k]); u256 hY0=parse_u256(sY0),hY1=parse_u256(sY1);
  u256* dcf; CK(cudaMalloc(&dcf,sizeof(u256)*6)); cudaMemcpy(dcf,hcf,sizeof(u256)*6,cudaMemcpyHostToDevice);
  // host receive buffers — DOUBLE-BUFFERED so the GPU can copy lattice N+1's shipped candidates into
  // one slot while the CPU finalizes lattice N from the other (GPU/CPU pipeline; see main loop).
  struct ShipBuf{ long long *A,*B; unsigned long long *CA,*CR; uint32_t *PA,*PR; unsigned *NA,*NR; unsigned nship; long long Q; double gpu_ms; };
  ShipBuf sb[2];
  for(int z=0;z<2;z++){ ShipBuf&b=sb[z];
    b.A=(long long*)malloc(8*MAXSHIP); b.B=(long long*)malloc(8*MAXSHIP);
    b.CA=(unsigned long long*)malloc(8*MAXSHIP); b.CR=(unsigned long long*)malloc(8*MAXSHIP);
    b.PA=(uint32_t*)malloc((size_t)MAXSHIP*MAXP*4); b.PR=(uint32_t*)malloc((size_t)MAXSHIP*MAXP*4);
    b.NA=(unsigned*)malloc(MAXSHIP*4); b.NR=(unsigned*)malloc(MAXSHIP*4); b.nship=0; b.Q=0; b.gpu_ms=0; }
  char* ok=(char*)malloc(MAXSHIP);
  char* relbuf=(char*)malloc((size_t)MAXSHIP*640);     // formatted CADO relation per candidate
  double cf[6]; for(int k=0;k<6;k++)cf[k]=strtod(C[k],0); double Y0d=strtod(sY0,0),Y1d=strtod(sY1,0);
  int SLACK=10; double SKEW=G_SKEW;
  for(int k=0;k<6;k++)mpz_init_set_str(G_cc[k],C[k],10); mpz_init_set_str(G_y0,sY0,10); mpz_init_set_str(G_y1,sY1,10);
  long long QMIN=atoll(argv[2]), QMAX=atoll(argv[3]);
  #pragma omp parallel
  { volatile int w=0; for(int z=0;z<1000;z++)w+=z; }     // warm OMP pool

  long long a0,b0,a1,b1,Q; double logq; u64 rq[8];
  cudaEvent_t ev0,ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
  const int PROF=getenv("PROF")!=0;
  const int PIPE=getenv("NOPIPE")==0;   // GPU(lattice N+1) overlaps CPU(lattice N); NOPIPE=1 to disable
  cudaEvent_t pe[6]; for(int z=0;z<6;z++) cudaEventCreate(&pe[z]);
  // ---- issue the whole GPU kernel stream for one lattice (NO host sync -> GPU runs while host works) ----
  auto issue=[&](long long A0,long long B0,long long A1,long long B1,long long QQ,double LQ){
    a0=A0;b0=B0;a1=A1;b1=B1;Q=QQ;logq=LQ;
    cudaEventRecord(ev0);
    for(int g=0;g<4;g++) project<<<188*16,256>>>(dM[g],dRin[g],dT[g],dRout[g],ng[g],a0,b0,a1,b1);
    if(PROF)cudaEventRecord(pe[0]);
    cudaMemset(dLr,0,NCELL);cudaMemset(dLa,0,NCELL);
    scat_col<<<188*32,256>>>(dLr,dM[0],dRout[0],dL[0],ng[0]);scat_lat<<<188*32,256>>>(dLr,dM[1],dRout[1],dL[1],ng[1]);
    scat_col<<<188*32,256>>>(dLa,dM[2],dRout[2],dL[2],ng[2]);scat_lat<<<188*32,256>>>(dLa,dM[3],dRout[3],dL[3],ng[3]);
    if(PROF)cudaEventRecord(pe[1]);
    cudaMemset(dCnt,0,4);
    scan_idx<<<188*16,256>>>(dLa,dLr,dSidx,dCnt,SLACK,a0,b0,a1,b1,(float)logq,(float)cf[0],(float)cf[1],(float)cf[2],(float)cf[3],(float)cf[4],(float)cf[5],(float)Y0d,(float)Y1d);
    if(PROF)cudaEventRecord(pe[2]);
    cudaMemset(dCa,0,(size_t)MAXSURV*4);cudaMemset(dCr,0,(size_t)MAXSURV*4);   // over-allocate -> no need to read nsurv on host
    cudaMemset(dBits,0,(NCELL+31)/32*4); mkmask<<<188*16,256>>>(dSidx,dBits);
    resieve_col<<<188*32,256>>>(dM[0],dP[0],dRout[0],ng[0],dBits,dSidx,dPr,dCr);resieve_lat<<<188*32,256>>>(dM[1],dP[1],dRout[1],ng[1],dBits,dSidx,dPr,dCr);
    resieve_col<<<188*32,256>>>(dM[2],dP[2],dRout[2],ng[2],dBits,dSidx,dPa,dCa);resieve_lat<<<188*32,256>>>(dM[3],dP[3],dRout[3],ng[3],dBits,dSidx,dPa,dCa);
    mksij<<<188*16,256>>>(dSidx,dSI,dSJ);
    if(PROF)cudaEventRecord(pe[3]);
    cudaMemset(dSC,0,4);
    cofactor<<<188*16,256>>>(dSI,dSJ,dCnt,dcf,hY0,hY1,(unsigned long long)Q,dPa,dCa,dPr,dCr,a0,b0,a1,b1,58,57,dsA,dsB,dsCA,dsCR,dsPA,dsNA,dsPR,dsNR,dSC,MAXSHIP);
    if(PROF)cudaEventRecord(pe[4]);
    cudaEventRecord(ev1);
  };
  // ---- wait for the issued GPU stream + copy its shipped candidates into double-buffer slot *b ----
  auto fetch=[&](ShipBuf*b){
    CK(cudaEventSynchronize(ev1));
    float gms; cudaEventElapsedTime(&gms,ev0,ev1); b->gpu_ms=gms;
    if(PROF){ float t; cudaEventElapsedTime(&t,ev0,pe[0]);g_prof[0]+=t; cudaEventElapsedTime(&t,pe[0],pe[1]);g_prof[1]+=t;
      cudaEventElapsedTime(&t,pe[1],pe[2]);g_prof[2]+=t; cudaEventElapsedTime(&t,pe[2],pe[3]);g_prof[3]+=t;
      cudaEventElapsedTime(&t,pe[3],pe[4]);g_prof[4]+=t;
      unsigned nsv; cudaMemcpy(&nsv,dCnt,4,cudaMemcpyDeviceToHost); g_prof[5]+=nsv; }
    unsigned nship; cudaMemcpy(&nship,dSC,4,cudaMemcpyDeviceToHost); if(nship>MAXSHIP)nship=MAXSHIP;
    b->nship=nship; b->Q=Q;                 // store this lattice's special-q with its candidates
    extern long g_nship_sum,g_lat_cnt; g_nship_sum+=nship; g_lat_cnt++;
    cudaMemcpy(b->A,dsA,8*nship,cudaMemcpyDeviceToHost);cudaMemcpy(b->B,dsB,8*nship,cudaMemcpyDeviceToHost);
    cudaMemcpy(b->CA,dsCA,8*nship,cudaMemcpyDeviceToHost);cudaMemcpy(b->CR,dsCR,8*nship,cudaMemcpyDeviceToHost);
    cudaMemcpy(b->NA,dsNA,4*nship,cudaMemcpyDeviceToHost);cudaMemcpy(b->NR,dsNR,4*nship,cudaMemcpyDeviceToHost);
    cudaMemcpy(b->PA,dsPA,(size_t)nship*MAXP*4,cudaMemcpyDeviceToHost);cudaMemcpy(b->PR,dsPR,(size_t)nship*MAXP*4,cudaMemcpyDeviceToHost);
  };
  // ---- CPU: build full factorization + CADO relation format from slot *b (uses b->Q) ----
  auto process=[&](ShipBuf*bf,double* cpu_ms,long long* nrel)->void{
    unsigned nship=bf->nship; long long QB=bf->Q;
    long long *hsA=bf->A,*hsB=bf->B; uint32_t *hsPA=bf->PA,*hsPR=bf->PR; unsigned *hsNA=bf->NA,*hsNR=bf->NR;
    double tc=wall_s();
    #pragma omp parallel
    { mpz_t F,G,t1,t2; mpz_init(F);mpz_init(G);mpz_init(t1);mpz_init(t2);   // thread-persistent (no per-relation malloc storm)
    #pragma omp for schedule(dynamic,8)
    for(int s=0;s<(int)nship;s++){ ok[s]=0;
      long long a=hsA[s],b=hsB[s];
      // |F(a,b)| algebraic (coeffs pre-parsed in G_cc -> no per-relation string parsing)
      mpz_set_ui(F,0); for(int k=0;k<=5;k++){ mpz_set(t1,G_cc[k]); for(int e=0;e<k;e++)mpz_mul_si(t1,t1,a); for(int e=0;e<5-k;e++)mpz_mul_si(t1,t1,b); mpz_add(F,F,t1);} mpz_abs(F,F);
      // |G(a,b)| rational = Y1*a+Y0*b
      mpz_mul_si(t1,G_y1,a); mpz_mul_si(t2,G_y0,b); mpz_add(G,t1,t2); mpz_abs(G,G);
      unsigned long long af[96],rf[96]; int naf=0,nrf=0; bool good=true;
      // algebraic: special-q, small primes (not resieved), then resieved alg primes, then cofactor residual
      while(mpz_divisible_ui_p(F,(unsigned long long)QB)){ af[naf++]=(unsigned long long)QB; mpz_divexact_ui(F,F,(unsigned long long)QB); if(naf>=90){good=false;break;} }
      for(int t=0;t<hNSP&&good;t++){ unsigned long long p=hSP[t]; while(mpz_divisible_ui_p(F,p)){ af[naf++]=p; mpz_divexact_ui(F,F,p); if(naf>=90){good=false;break;} } }
      for(unsigned t=0;t<hsNA[s]&&good;t++){ unsigned long long p=hsPA[(size_t)s*MAXP+t]; while(mpz_divisible_ui_p(F,p)){ af[naf++]=p; mpz_divexact_ui(F,F,p); if(naf>=90){good=false;break;} } }
      if(good){ if(mpz_sizeinbase(F,2)>62)good=false; else good=h_factor_cof(mpz_get_ui(F),af,&naf); }
      // rational: small primes (not resieved), resieved rat primes, then cofactor residual
      for(int t=0;t<hNSP&&good;t++){ unsigned long long p=hSP[t]; while(mpz_divisible_ui_p(G,p)){ rf[nrf++]=p; mpz_divexact_ui(G,G,p); if(nrf>=90){good=false;break;} } }
      for(unsigned t=0;t<hsNR[s]&&good;t++){ unsigned long long p=hsPR[(size_t)s*MAXP+t]; while(mpz_divisible_ui_p(G,p)){ rf[nrf++]=p; mpz_divexact_ui(G,G,p); if(nrf>=90){good=false;break;} } }
      if(good){ if(mpz_sizeinbase(G,2)>62)good=false; else good=h_factor_cof(mpz_get_ui(G),rf,&nrf); }
      if(good){
        qsort(af,naf,8,cmp_u64); qsort(rf,nrf,8,cmp_u64);
        char* o=relbuf+(size_t)s*640; int p=0;
        long long A=a,B=b; if(B<0){A=-A;B=-B;}
        p+=sprintf(o+p,"%lld,%lld:",A,B);
        for(int t=0;t<nrf;t++)p+=sprintf(o+p,t?",%llx":"%llx",rf[t]);   // rational side (side 0) first
        o[p++]=':';
        for(int t=0;t<naf;t++)p+=sprintf(o+p,t?",%llx":"%llx",af[t]);   // algebraic side (side 1)
        o[p++]='\n'; o[p]=0; ok[s]=1;
      }
    }
    mpz_clear(F);mpz_clear(G);mpz_clear(t1);mpz_clear(t2);
    }
    long long r=0; for(unsigned s=0;s<nship;s++) if(ok[s])r++;
    *cpu_ms=(wall_s()-tc)*1000; *nrel=r;
    for(unsigned s=0;s<nship;s++) if(ok[s]) rel_emit(relbuf+(size_t)s*640);   // -> RAM buffer (not disk)
  };
  // warmup: first valid (q,rho) in range for the LOADED poly (then discard its relations)
  { long long bas[4]; u64 wr[8];
    for(long long q=QMIN;q<QMAX;q++){ if(!isprime_q(q))continue; int nr=find_roots(C,q,wr); if(!nr)continue;
      skew_reduce(q,(long long)wr[0],SKEW,bas);
      issue(bas[0],bas[1],bas[2],bas[3],q,log2((double)q)); fetch(&sb[0]); double c; long long n; process(&sb[0],&c,&n); break; }
    g_ramlen=0;   // discard warmup relations from the RAM buffer
  }
  // ---- SPECIAL-Q LOOP (relations accumulate in RAM). Pipelined: issue GPU(N+1), finalize CPU(N). ----
  const char* dumpf=argc>4?argv[4]:0;                // optional: also dump RAM buffer to a file for validation
  long long TARGET=argc>5?atoll(argv[5]):0;          // stop after this many relations (0 = whole range)
  double tot_gpu=0,tot_cpu=0; long long tot_rel=0; int nq=0; double t0w=wall_s();
  // Wall-time budget (env GPULOOP_MAX_SECS, 0=none): stop sieving when the time is up so the caller
  // can still run the downstream. Lets a low-yield key USE its full budget (instead of quitting early
  // at QMAX) to gather as many relations as possible. Additive: only ever stops earlier than QMAX.
  double SIEVE_MAX=getenv("GPULOOP_MAX_SECS")?atof(getenv("GPULOOP_MAX_SECS")):0;
  ShipBuf *cur=&sb[0],*nxt=&sb[1]; bool pending=false, stop=false;
  auto progress=[&](){ if((nq%2000)==0){ double el=wall_s()-t0w;
      fprintf(stderr,"[%.0fs] q=%lld  lattices=%d  relations=%lld  %.1fms/lat (GPU %.1f)  RAM %.2fGB  ETA(71M) %.2fh\n",
        el,Q,nq,tot_rel,el*1000/nq,tot_gpu/nq, g_ramlen/1e9, (el/tot_rel)*71e6/3600.0);
      fprintf(stderr,"        avg nship(cofactors to CPU-factor)/lattice = %.0f\n", (double)g_nship_sum/g_lat_cnt); } };
  for(long long q=QMIN;q<QMAX && !stop;q++){
    if(!isprime_q(q))continue;
    int nr=find_roots(C,q,rq);
    for(int k=0;k<nr && !stop;k++){
      long long bas[4]; skew_reduce(q,(long long)rq[k],SKEW,bas);
      if(PIPE){
        issue(bas[0],bas[1],bas[2],bas[3],q,log2((double)q));   // GPU for THIS lattice (async)
        if(pending){ double c; long long n; process(cur,&c,&n);  // CPU for PREVIOUS lattice, overlaps GPU above
          tot_gpu+=cur->gpu_ms; tot_cpu+=c; tot_rel+=n; nq++; progress();
          if(TARGET&&tot_rel>=TARGET)stop=true;
          if(SIEVE_MAX>0&&wall_s()-t0w>SIEVE_MAX)stop=true; }
        fetch(nxt);                                             // sync THIS GPU, copy ships into nxt
        ShipBuf*tmp=cur;cur=nxt;nxt=tmp; pending=true;          // THIS lattice becomes the pending one
      } else {
        issue(bas[0],bas[1],bas[2],bas[3],q,log2((double)q)); fetch(cur);
        double c; long long n; process(cur,&c,&n);
        tot_gpu+=cur->gpu_ms; tot_cpu+=c; tot_rel+=n; nq++; progress();
        if(TARGET&&tot_rel>=TARGET)stop=true;
        if(SIEVE_MAX>0&&wall_s()-t0w>SIEVE_MAX)stop=true;
      }
    }
  }
  if(SIEVE_MAX>0&&wall_s()-t0w>SIEVE_MAX&&(!TARGET||tot_rel<TARGET))
    fprintf(stderr,"[sieve] WALL-TIME BUDGET %.0fs reached with %lld/%lld relations (q=%lld) -- time-bound, not target\n",SIEVE_MAX,tot_rel,TARGET,Q);
  if(PIPE && pending && !stop){ double c; long long n; process(cur,&c,&n); tot_gpu+=cur->gpu_ms; tot_cpu+=c; tot_rel+=n; nq++; }  // drain last
  double wall_tot=wall_s()-t0w;
  if(dumpf){ FILE* df=fopen(dumpf,"w"); if(df){ fwrite(g_rambuf,1,g_ramlen,df); fclose(df); } }
  printf("\n=== SUSTAINED GPU SIEVER over special-q [%lld,%lld] ===\n",QMIN,QMAX);
  printf("  (q,rho) lattices: %d   relations: %lld   (avg %.1f rel/lattice)\n",nq,tot_rel,(double)tot_rel/nq);
  printf("  per-lattice: GPU %.2f ms   CPU %.2f ms   wall %.2f ms   (vs CADO ~28 ms)\n",tot_gpu/nq,tot_cpu/nq,wall_tot*1000/nq);
  if(PROF) printf("  [PROF] ms/lat: project %.2f | scatter %.2f | scan %.2f | resieve %.2f | cofactor %.2f | avg nsurv %.0f\n",
    g_prof[0]/nq,g_prof[1]/nq,g_prof[2]/nq,g_prof[3]/nq,g_prof[4]/nq,g_prof[5]/nq);
  printf("  total wall %.2f s\n",wall_tot);
  printf("  relations resident in RAM: %.3f GB  (disk untouched; %.0f bytes/rel)\n",g_ramlen/1e9,tot_rel?(double)g_ramlen/tot_rel:0);
  for(long long N : {430000LL,800000LL}){
    printf("  extrapolate %ldk lattices: GPU-bound %.2f h | wall %.2f h\n",(long)(N/1000),(tot_gpu/nq)*N/3.6e6,(wall_tot*1000/nq)*N/3.6e6);
  }
  return 0;
}
