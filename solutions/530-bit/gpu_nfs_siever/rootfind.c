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

// Modular root-finding for the algebraic factor base: roots of f(x) mod p, degree 5.
// g = gcd(x^p - x, f) = product of (x - root); then equal-degree split to extract roots.
// Coeffs given as decimal strings, reduced mod p via digit-Horner (no bignum needed).
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef uint64_t u64; typedef __int128 i128;
static inline u64 mulmod(u64 a,u64 b,u64 p){ return (u64)(((i128)a*b)%p); }
static inline u64 addmod(u64 a,u64 b,u64 p){ a+=b; return a>=p?a-p:a; }
static inline u64 submod(u64 a,u64 b,u64 p){ return a>=b?a-b:a+p-b; }
static u64 powmod(u64 a,u64 e,u64 p){ u64 r=1%p; a%=p; while(e){ if(e&1)r=mulmod(r,a,p); a=mulmod(a,a,p); e>>=1;} return r; }
static u64 invmod(u64 a,u64 p){ return powmod(a%p,p-2,p); }
static u64 strmod(const char*s,u64 p){ u64 r=0; int i=0,neg=0; if(s[0]=='-'){neg=1;i=1;}
  for(;s[i];i++) r=(u64)(((i128)r*10 + (s[i]-'0'))%p); if(neg&&r) r=p-r; return r; }

#define MAXD 16
typedef struct{ u64 c[MAXD]; int d; } poly;
static void pnorm(poly*A){ while(A->d>=0 && A->c[A->d]==0) A->d--; }
static void pmod(poly*A,const poly*F,u64 p){ u64 il=invmod(F->c[F->d],p);
  while(A->d>=F->d){ u64 f=mulmod(A->c[A->d],il,p); int sh=A->d-F->d;
    for(int i=0;i<=F->d;i++) A->c[i+sh]=submod(A->c[i+sh],mulmod(f,F->c[i],p),p); pnorm(A);} }
static void pmulmod(const poly*A,const poly*B,const poly*F,u64 p,poly*R){
  poly T; memset(&T,0,sizeof T); T.d=-1;
  if(A->d>=0&&B->d>=0){ T.d=A->d+B->d; for(int i=0;i<=T.d;i++)T.c[i]=0;
    for(int i=0;i<=A->d;i++) if(A->c[i]) for(int j=0;j<=B->d;j++)
      T.c[i+j]=addmod(T.c[i+j],mulmod(A->c[i],B->c[j],p),p); }
  pmod(&T,F,p); *R=T; }
static void pgcd(poly A,poly B,u64 p,poly*G){ pnorm(&A); pnorm(&B);
  while(B.d>=0){ pmod(&A,&B,p); poly t=A;A=B;B=t; }
  if(A.d>=0){ u64 il=invmod(A.c[A.d],p); for(int i=0;i<=A.d;i++)A.c[i]=mulmod(A.c[i],il,p);} *G=A; }
static void pdiv(poly N,const poly*D,u64 p,poly*Q){ memset(Q,0,sizeof*Q); Q->d=N.d-D->d;
  if(Q->d<0){Q->d=-1;return;} for(int i=0;i<=Q->d;i++)Q->c[i]=0; u64 il=invmod(D->c[D->d],p);
  while(N.d>=D->d){ u64 f=mulmod(N.c[N.d],il,p); int sh=N.d-D->d; Q->c[sh]=f;
    for(int i=0;i<=D->d;i++) N.c[i+sh]=submod(N.c[i+sh],mulmod(f,D->c[i],p),p); pnorm(&N);} }

static int split(poly g,u64 p,u64*out){
  if(g.d<=0) return 0;
  if(g.d==1){ out[0]=submod(0,g.c[0],p); return 1; }
  for(u64 d=1;;d++){
    poly base; memset(&base,0,sizeof base); base.d=1; base.c[1]=1; base.c[0]=d%p; pmod(&base,&g,p);
    poly res; memset(&res,0,sizeof res); res.d=0; res.c[0]=1%p; u64 e=(p-1)/2;
    while(e){ if(e&1)pmulmod(&res,&base,&g,p,&res); pmulmod(&base,&base,&g,p,&base); e>>=1; }
    res.c[0]=submod(res.c[0],1%p,p); pnorm(&res);
    if(res.d<0) continue;
    poly G; pgcd(res,g,p,&G);
    if(G.d>=1 && G.d<g.d){ poly Q; pdiv(g,&G,p,&Q);
      int n=split(G,p,out); n+=split(Q,p,out+n); return n; }
  }
}
// public: roots of f (string coeffs c0..c5) mod p
int find_roots(char*const cc[6],u64 p,u64*out){
  poly f; memset(&f,0,sizeof f); f.d=5; for(int i=0;i<=5;i++) f.c[i]=strmod(cc[i],p); pnorm(&f);
  if(f.d<0) return -1; if(f.d==0) return 0;
  if(p<=7){ int n=0; for(u64 x=0;x<p;x++){ u64 v=0; for(int i=f.d;i>=0;i--)v=addmod(mulmod(v,x,p),f.c[i],p); if(!v)out[n++]=x;} return n; }
  poly xp; { poly x;memset(&x,0,sizeof x);x.d=1;x.c[1]=1; poly res;memset(&res,0,sizeof res);res.d=0;res.c[0]=1%p;
    poly base=x; u64 e=p; while(e){ if(e&1)pmulmod(&res,&base,&f,p,&res); pmulmod(&base,&base,&f,p,&base); e>>=1;} xp=res; }
  poly xpx=xp; if(xpx.d<1){xpx.d=1;xpx.c[1]=0;} xpx.c[1]=submod(xpx.c[1],1,p); pnorm(&xpx);
  poly g; pgcd(xpx,f,p,&g); if(g.d<1) return 0; return split(g,p,out);
}

#ifdef TEST
int main(){
  char* c[6]={"159833281045781154978212868605280","6906567572864272983429482096",
              "50768942603145318312392","-73408380947302513","-78411953550","88200"};
  u64 out[8];
  u64 tests[]={2400019,1009,10007,100003,999983,2,3,5,7,13};
  for(int t=0;t<10;t++){ u64 p=tests[t]; int n=find_roots(c,p,out);
    printf("p=%-8llu roots(%d):",(unsigned long long)p,n);
    for(int i=0;i<n;i++) printf(" %llu",(unsigned long long)out[i]);
    // verify
    int okall=1; for(int i=0;i<n;i++){ u64 x=out[i],v=0; char* cs[6]; (void)cs;
      u64 cf[6]; for(int k=0;k<6;k++) cf[k]=strmod(c[k],p);
      for(int k=5;k>=0;k--) v=addmod(mulmod(v,x,p),cf[k],p); if(v)okall=0; }
    printf("   [verify f(root)=0: %s]\n", okall?"OK":"FAIL");
  }
  return 0;
}
#endif
