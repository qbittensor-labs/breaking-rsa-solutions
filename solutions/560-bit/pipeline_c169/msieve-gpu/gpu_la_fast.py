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

# gpu_la_fast.py -- gpu_la.py unchanged, except that the 256-bit COMBINE step (combineV: Gaussian elimination on
# the 512 bit-planes of [B xb | B p0] mirrored on [xb | p0], ~194 s of single-threaded host numpy on the c169
# matrix: "combine: ... 193.8s" in every key-1b LA log) runs on the GPU. Same algorithm, same pivot order, same
# candidate selection: the kernel file is bit-identical (checked by `--selftest`, and LA_CMB=check runs both).
#
# usage: exactly as gpu_la.py (ds_cado.sh: GPU_LA=<dir>/gpu_la_fast.py; it must sit next to gpu_la.py/cado_mat.py)
#        python3 gpu_la_fast.py --selftest <n> [rank]   CPU combineV vs GPU on random inputs of n rows (rank of the
#                                                        B-image planes, default 128 as in the key-1b LA): times + equality
#        python3 gpu_la_fast.py --warm                    compile every kernel the LA uses into the CuPy cache (tiny synthetic
#                                                        CADO matrix through the production path; see _warm). Exact: no
#                                                        result can change, only cache files are written
# env:   LA_CMB = gpu (default) | cpu (gpu_la.py's own combineV) | check (both; prints MATCH/MISMATCH, returns CPU's)
#        GPU_LA_BASE = path of gpu_la.py (default: next to this file, symlinks resolved)
# Any GPU-side exception (e.g. out of memory beside the LA's matrix) falls back to the CPU combineV with a WARNING.
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
BASE = os.environ.get("GPU_LA_BASE") or os.path.join(HERE, "gpu_la.py")

_CU = r'''
extern "C" __global__ void planes(const unsigned long long* X, unsigned long long* P, long long n, int W, long long cw, long long row0) {
  // P[row0 + j][w] bit k = bit j of X[64 w + k]  (gpu_la.planesV)
  long long w = blockIdx.x; int j = threadIdx.x; long long base = w * 64; int wd = j >> 6, sh = j & 63;
  unsigned long long acc = 0;
  for (int k = 0; k < 64; k++) { long long c = base + k; if (c >= n) break; acc |= ((X[c * W + wd] >> sh) & 1ULL) << k; }
  P[(row0 + j) * cw + w] = acc;
}
extern "C" __global__ void xorrows(unsigned long long* M, const long long* rows, long long nr, long long cw, long long w0, long long src) {
  // M[rows[r]][c] ^= M[src][c] for c >= w0  (rows never contains src)
  long long t = blockIdx.x * (long long) blockDim.x + threadIdx.x; long long span = cw - w0;
  if (t >= span * nr) return;
  long long r = t / span, c = w0 + t % span;
  M[rows[r] * cw + c] ^= M[src * cw + c];
}
extern "C" __global__ void words(const unsigned long long* S, const long long* sel, int k, long long cw, long long n, unsigned long long* out) {
  // out[c] bit j = bit c of plane sel[j]  (gpu_la.rows_to_words64)
  long long c = blockIdx.x * (long long) blockDim.x + threadIdx.x; if (c >= n) return;
  long long wi = c >> 6; int sh = c & 63; unsigned long long v = 0;
  for (int j = 0; j < k; j++) v |= ((S[sel[j] * cw + wi] >> sh) & 1ULL) << j;
  out[c] = v;
}
'''

class _Cupy:
    def __init__(self):
        import cupy as cp
        self.cp = cp; m = cp.RawModule(code=_CU)
        self.k_planes = m.get_function("planes"); self.k_xor = m.get_function("xorrows"); self.k_words = m.get_function("words")
    def asarray(self, a): return self.cp.asarray(a)
    def get(self, a): return self.cp.asnumpy(a)
    def zeros(self, shape): return self.cp.zeros(shape, dtype=self.cp.uint64)
    def planes(self, X, n, VB, P, row0):
        cp = self.cp; Xd = cp.ascontiguousarray(cp.asarray(X, dtype=cp.uint64).reshape(n, VB // 64)); cw = P.shape[1]
        self.k_planes((cw,), (VB,), (Xd, P, np.int64(n), np.int32(VB // 64), np.int64(cw), np.int64(row0)))
    def or_rows(self, M):   # cupy has no bitwise_or.reduce (NotImplementedError, sandbox -3): explicit reduction kernel
        if not hasattr(self, 'k_or'):
            self.k_or = self.cp.ReductionKernel('uint64 x', 'uint64 y', 'x', 'a | b', 'y = a', '0', 'or_reduce_u64')
        return self.k_or(M, axis=0)
    def swap(self, M, i, j): t = M[i].copy(); M[i] = M[j]; M[j] = t
    def xorrows(self, M, rows, w0, src):
        cp = self.cp; cw = M.shape[1]; nr = len(rows); tot = (cw - w0) * nr
        if tot <= 0: return
        rd = cp.asarray(np.asarray(rows, dtype=np.int64))
        self.k_xor(((tot + 255) // 256,), (256,), (M, rd, np.int64(nr), np.int64(cw), np.int64(w0), np.int64(src)))
    def any_rows(self, M): return self.cp.asnumpy(M.any(axis=1))
    def words(self, M, sel, n):
        cp = self.cp; out = cp.zeros(n, dtype=cp.uint64); k = len(sel)
        if k:
            sd = cp.asarray(np.asarray(sel, dtype=np.int64))
            self.k_words(((n + 255) // 256,), (256,), (M, sd, np.int32(k), np.int64(M.shape[1]), np.int64(n), out))
        return cp.asnumpy(out)
    def sync(self): self.cp.cuda.Device().synchronize()

class _Numpy:   # the same operations in numpy: for testing combine_fast's control flow without a GPU
    def asarray(self, a): return np.asarray(a)
    def get(self, a): return np.asarray(a)
    def zeros(self, shape): return np.zeros(shape, dtype=np.uint64)
    def planes(self, X, n, VB, P, row0):
        W = VB // 64; X = np.asarray(X, dtype=np.uint64).reshape(n, W); cw = P.shape[1]; mp = cw * 64
        blk = np.zeros((mp, W), dtype=np.uint64); blk[:n] = X
        bits = np.unpackbits(blk.view(np.uint8).reshape(mp, VB // 8), axis=1, bitorder='little')
        pk = np.packbits(bits, axis=0, bitorder='little')
        P[row0:row0 + VB] = np.ascontiguousarray(pk.T).view('<u8')
    def or_rows(self, M): return np.bitwise_or.reduce(M, axis=0)
    def swap(self, M, i, j): M[[i, j]] = M[[j, i]]
    def xorrows(self, M, rows, w0, src): M[np.asarray(rows), w0:] ^= M[src, w0:]
    def any_rows(self, M): return M.any(axis=1)
    def words(self, M, sel, n):
        out = np.zeros(n, dtype=np.uint64)
        for j, r in enumerate(sel):
            bits = np.unpackbits(np.ascontiguousarray(M[r]).view(np.uint8), bitorder='little')[:n].astype(np.uint64)
            out |= bits << np.uint64(j)
        return out
    def sync(self): pass

def combine_fast(n, x, v, ax, av, VB, nout=64, be=None, count_all=False):
    """gpu_la.combineV(n,x,v,ax,av,VB,nout) with the bit-plane work on the device `be`. Same return value.
    count_all (LA_DEFER_ROWS): info['indep_all'] = uncapped head-independent candidate count, as combineV's."""
    t0 = time.time(); be = be or _Cupy()
    R_ = 2 * VB; cw = (n + 63) // 64
    mat = be.zeros((R_, cw)); amat = be.zeros((R_, cw))
    be.planes(x, n, VB, mat, 0); be.planes(v, n, VB, mat, VB); be.planes(ax, n, VB, amat, 0); be.planes(av, n, VB, amat, VB)
    i = 0
    nzw = np.flatnonzero(be.get(be.or_rows(amat))) if cw else np.zeros(0, np.int64)
    # combineV visits every word of nzw in order and pivots while the column (rows >= i) is non-zero; a word whose
    # column is zero is a no-op. Scan BLK words at a time on the host for the next non-zero column, then run that
    # word's pivot loop exactly as combineV does (one small column read per pivot).
    p = 0; L = len(nzw); BLK = 8192
    while p < L and i < R_:
        ws = nzw[p:p + BLK]
        blk = be.get(amat[i:, be.asarray(ws)])
        hit = np.flatnonzero(np.bitwise_or.reduce(blk, axis=0))
        if len(hit) == 0: p += len(ws); continue
        p += int(hit[0]); w = int(nzw[p])
        while i < R_:
            colw = be.get(amat[i:, w]); orv = int(np.bitwise_or.reduce(colw))
            if orv == 0: break
            m = np.uint64(orv & -orv)
            rows = np.flatnonzero(colw & m) + i; piv = int(rows[0])
            if piv != i: be.swap(amat, i, piv); be.swap(mat, i, piv)
            oth = rows[1:]
            if len(oth): be.xorrows(amat, oth, w, i); be.xorrows(mat, oth, 0, i)
            i += 1
        p += 1
    rank = i
    nzr = be.any_rows(mat[rank:]) if cw else np.zeros(R_ - rank, bool)
    PW = min(cw, R_); head = be.get(mat[rank:, :PW]); basis = {}; chosen = []; unsure = []
    for k in range(R_ - rank):          # verbatim combineV from here (cand[k,:PW] == head[k])
        if not nzr[k]: continue
        vv = int.from_bytes(np.ascontiguousarray(head[k]).tobytes(), 'little')
        if vv == 0: unsure.append(k); continue
        while vv:
            h = vv.bit_length() - 1
            if h in basis: vv ^= basis[h]
            else: basis[h] = vv; chosen.append(k); break
        if len(chosen) >= 2 * nout: break
    first = chosen[:nout]
    if len(first) < nout and unsure: first += unsure[:nout - len(first)]
    out = be.words(mat, [rank + k for k in first], n)
    extra = be.get(mat[be.asarray(np.asarray([rank + k for k in chosen[nout:2 * nout]], dtype=np.int64))]) if len(chosen) > nout else None
    be.sync()
    info = dict(rank=rank, cand=int(R_ - rank), nonzero=int(np.asarray(nzr).sum()), indep=len(chosen), unsure=len(unsure), t=time.time() - t0)
    if count_all:
        basis = {}
        for k in range(R_ - rank):
            if not nzr[k]: continue
            vv = int.from_bytes(np.ascontiguousarray(head[k]).tobytes(), 'little')
            while vv:
                h = vv.bit_length() - 1
                if h in basis: vv ^= basis[h]
                else: basis[h] = vv; break
        info["indep_all"] = len(basis)
    return out, len(first), extra, info

def _same(a, b):
    (o1, n1, e1, i1), (o2, n2, e2, i2) = a, b
    k1 = {k: v for k, v in i1.items() if k != 't'}; k2 = {k: v for k, v in i2.items() if k != 't'}
    ee = (e1 is None and e2 is None) or (e1 is not None and e2 is not None and e1.shape == e2.shape and bool((e1 == e2).all()))
    return n1 == n2 and k1 == k2 and o1.shape == o2.shape and bool((o1 == o2).all()) and ee

def _patch(g):
    """installed into gpu_la's globals just before its __main__ block: wraps combineV"""
    cpu = g["combineV"]; mode = os.environ.get("LA_CMB", "gpu")
    def combineV(n, x, v, ax, av, VB, nout=64, **kw):     # kw: count_all (LA_DEFER_ROWS only)
        if mode == "cpu": return cpu(n, x, v, ax, av, VB, nout, **kw)
        try:
            r = combine_fast(n, x, v, ax, av, VB, nout, **kw)
            print(f"  [la-fast] GPU combine: rank {r[3]['rank']}, {r[1]} kernel vectors in {r[3]['t']:.1f}s", flush=True)
        except Exception as e:
            print(f"  [la-fast] WARNING: GPU combine failed ({type(e).__name__}: {str(e)[:200]}) -- CPU combine", flush=True)
            try: g["cp"].get_default_memory_pool().free_all_blocks()
            except Exception: pass
            return cpu(n, x, v, ax, av, VB, nout, **kw)
        if mode == "check":
            rc = cpu(n, x, v, ax, av, VB, nout, **kw)
            print(f"  [la-fast] LA_CMB=check: GPU {r[3]['t']:.1f}s vs CPU {rc[3]['t']:.1f}s: {'MATCH' if _same(r, rc) else 'MISMATCH'} (CPU result used)", flush=True)
            return rc
        return r
    g["combineV"] = combineV

def _selftest(n, rank=128):
    import importlib.util
    spec = importlib.util.spec_from_file_location("gpu_la", BASE); gl = importlib.util.module_from_spec(spec)
    sys.argv = [BASE]; spec.loader.exec_module(gl)        # module import: gpu_la's __main__ block does not run
    VB = 256; rng = np.random.default_rng(20261020)
    x = rng.integers(0, 1 << 64, (n, 4), dtype=np.uint64); v = rng.integers(0, 1 << 64, (n, 4), dtype=np.uint64)
    # [ax|av] = n rows of 512 bits spanning a rank-`rank` space (as B x / B p0 at the end of block Lanczos)
    G = rng.integers(0, 2, (2 * VB, rank), dtype=np.uint8); Bb = rng.integers(0, 2, (n, rank), dtype=np.uint8)
    t = time.time(); ab = np.zeros((n, 2 * VB // 8), dtype=np.uint8); CH = 1 << 18
    for a in range(0, n, CH):
        bits = ((Bb[a:a + CH].astype(np.float32) @ G.T.astype(np.float32)).astype(np.int64) & 1).astype(np.uint8)
        ab[a:a + CH] = np.packbits(bits, axis=1, bitorder='little')
    ab = ab.view('<u8').reshape(n, 8); ax = np.ascontiguousarray(ab[:, :4]); av = np.ascontiguousarray(ab[:, 4:])
    print(f"selftest n={n} rank={rank}: inputs built in {time.time() - t:.1f}s", flush=True)
    t = time.time(); rg = combine_fast(n, x, v, ax, av, VB); tg = time.time() - t
    print(f"  GPU combine_fast: {tg:.2f}s  info {rg[3]}", flush=True)
    t = time.time(); rc = gl.combineV(n, x, v, ax, av, VB); tc = time.time() - t
    print(f"  CPU combineV:     {tc:.2f}s  info {rc[3]}", flush=True)
    ok = _same(rg, rc)
    print(f"SELFTEST {'IDENTICAL' if ok else 'MISMATCH'} (kernel words, count, spare planes, rank/cand/nonzero/indep/unsure): GPU {tg:.2f}s vs CPU {tc:.2f}s", flush=True)
    return 0 if ok else 1

# ---- --warm (2026-10-23): fill the CuPy/NVRTC kernel cache before the real matrix exists -------------------------
# A cold LA start spends ~17 s compiling (gpu_la's KS/KS256/ks256v2/ks256v3 kernels, the segspmv RawModule with its
# name expressions, cado_mat's parse kernels, the combine kernels, CuPy's own elementwise/reduction kernels). CuPy keeps
# every compiled binary in CUPY_CACHE_DIR keyed by source + options + device, and the kernel SOURCES depend only on the
# environment (LA_BLOCK, LA_SPMV, LA_SEG_G2*, LA_SEG_U*, LA_V256, LA_V3, LA_PARSE ...), never on the matrix. So the
# production path run end to end -- same interpreter, same environment, same script -- on a tiny synthetic CADO matrix
# compiles exactly the binaries the real LA will load. Nothing it does reaches the real run except cache files
# (written atomically by CuPy), so it cannot change any result. ds_cado.sh / factor_c169.sh run it in the background
# once the GPU is free (sieve stopped) while merge/replay keep the CPUs busy.
# The synthetic run forces >1 seg column part (LA_SEG_P / LA_SEG_BT_P = 2 unless set) so the multi-part build path the
# production matrix takes (5+5 parts at N 12.9M) is the one exercised; LA_MAXIT is dropped so the combine runs too.
# env: LA_WARM_N (ideals, default 60000), LA_WARM_DIR (scratch parent, default $TMPDIR)
def _warm():
    import subprocess, tempfile, shutil
    t0 = time.time()
    sys.path.insert(0, HERE); import cado_mat
    n = int(os.environ.get("LA_WARM_N", "60000"))
    td = tempfile.mkdtemp(prefix="la_warm.", dir=os.environ.get("LA_WARM_DIR") or None)
    try:
        path, nr = cado_mat.synth_cado(td, n, excess=64, kmin=10, kmax=40, seed=20261023)
        env = dict(os.environ)
        for k in ("LA_MAXIT", "LA_HASH", "LA_PROF", "LA_KER2"): env.pop(k, None)
        env["LA_CADO_NROWS"] = str(nr); env["LA_LOG_EVERY"] = "1000000"
        env.setdefault("LA_SEG_P", "2"); env.setdefault("LA_SEG_BT_P", "2")
        p = subprocess.Popen([sys.executable, os.path.realpath(__file__), path, os.path.join(td, "ker")], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        import signal
        def _term(*_):          # ds_cado.sh kills a warm-up that outlives LA_WARM_WAIT: take the child down too
            p.kill(); shutil.rmtree(td, ignore_errors=True); os._exit(143)
        signal.signal(signal.SIGTERM, _term)
        out, _ = p.communicate()
        r = subprocess.CompletedProcess(p.args, p.returncode, out)
        keep = [l for l in r.stdout.splitlines() if any(s in l for s in ("cado_mat", "seg", "loop starts", "LA done", "VERIFY", "la-fast", "WARNING", "FATAL", "Error"))]
        for l in keep[-12:]: print("  [la-warm] " + l.strip(), flush=True)
        print(f"[la-warm] done in {time.time() - t0:.1f}s (rc {r.returncode}, n {n}, CUPY_CACHE_DIR {os.environ.get('CUPY_CACHE_DIR', '/tmp/.cupy (gpu_la default)')})", flush=True)
        return 0 if r.returncode == 0 else 1
    finally:
        shutil.rmtree(td, ignore_errors=True)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--warm":
        sys.exit(_warm())
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(_selftest(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 128))
    src = open(BASE).read(); guard = 'if __name__=="__main__":'
    if src.count(guard) != 1:
        print(f"  [la-fast] WARNING: {BASE} has {src.count(guard)} main guards -- running it unpatched", flush=True)
    else:
        src = src.replace(guard, "_gla_fast_patch(globals())\n" + guard, 1)
    g = {"__name__": "__main__", "__file__": os.path.realpath(BASE), "__builtins__": __builtins__, "_gla_fast_patch": _patch}
    exec(compile(src, os.path.realpath(BASE), "exec"), g)
