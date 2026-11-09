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

# cado_mat.py -- load a CADO-NFS merged matrix (replay's X.sparse.bin) straight into gpu_la's
# in-memory layout, replacing the msieve path (replay --for_msieve + cado2msieve2 + msieve -nc2
# + parse_mat).  numpy only: no CuPy import, so it is testable on a CPU-only box.
#
# ---- CADO FORMAT (filter/replay.cpp flushSparse, bin=1, skip=0) --------------------------------
#   X.sparse.bin     for each ROW i (a relation-set; small_nrows of them):
#                        uint32 w, then w uint32 column indices          (little endian)
#   X.sparse.rw.bin  small_nrows uint32 = the row weights w (same values as above)
#   X.sparse.cw.bin  small_ncols uint32 = column weights
#   Columns are renumbered by DECREASING weight (replay.cpp renumber()), so column 0 is the
#   heaviest ideal. Rows are relation-sets, in the order of the index file written by the same
#   replay run (-index).  small_nrows >= small_ncols (replay asserts it); excess ~= purge -keep.
#
# ---- WHAT gpu_la NEEDS (parse_mat's return value) ----------------------------------------------
#   (nrows_B, ND, ncols_B, R_sp, C_sp, dcol) for B with B x = 0 solved over x in F2^ncols_B.
#   CADO factoring wants the LEFT kernel of M (bwc nullspace=left): x^T M = 0  <=>  M^T x = 0.
#   So B = M^T:  B rows = CADO columns (ideals), B columns = CADO rows (relation-sets), and
#   bit j of x[i] = "relation-set i is in kernel vector j" -- which is byte-for-byte the flat
#   kernel file CADO 'characters -ker' reads (small_nrows x 64 bits, one uint64 per row).
#   Dense rows: gpu_la packs up to 128 B-rows as per-column bitfields (dcol, 2 uint64 per column).
#   Because replay sorted columns by weight, the ND heaviest ideals are exactly CADO columns
#   0..ND-1, so they become dense rows 0..ND-1 with NO renumbering; all other entries keep their
#   CADO column index as the B row index (>= ND), which is the layout gpu_la's Bmul expects
#   (spmv writes every row, dense_fwd then XORs rows 0..ND-1 in).
#   C_sp is non-decreasing by construction (row-major walk) -- gpu_la relies on that to skip the
#   transpose sort (d_colBt = d_Ri).
import os, sys, time
import numpy as np

MAX_ND = 128   # gpu_la dcol = 2 uint64 per column


def _base(path):
    b = path
    if b.endswith(".bin"):
        b = b[:-4]
    return b


def cado_sidecars(path):
    b = _base(path)
    return b + ".rw.bin", b + ".cw.bin"


def _row_weights_by_walk(a):
    """Row weights by a sequential walk of X.sparse.bin (header word w, then w column indices).
    Used when X.sparse.rw.bin is missing or disagrees with the matrix. ~12.5M rows at c160 is a few
    seconds of Python -- only ever on this fallback path."""
    w = []
    pos = 0; L = len(a)
    while pos < L:
        k = int(a[pos]); w.append(k); pos += 1 + k
    if pos != L:
        raise ValueError("sparse.bin walk overran the file (pos=%d len=%d): not a CADO binary matrix" % (pos, L))
    return np.array(w, dtype=np.int64)


def load_cado_sparse(path, nd=None, verbose=True, with_map=False):
    """-> (nrows_B, ND, ncols_B, R_sp(int32), C_sp(int32), dcol(uint64, 2*ncols_B))
    with_map=True appends `rowmap`: None, or (n_cado_rows, int64 array of the CADO rows that ARE
    B's columns) when EMPTY merged rows were dropped from the LA (see below).

    EMPTY MERGED ROWS. A relation-set whose merged row is empty (merge summed its relations to a
    square) is a column of zeros in B. It is never given to the Lanczos solver: such columns are
    dropped here and the caller expands the kernel back to n_cado_rows with ZERO bits at those rows
    (still a kernel of the full B -- zero columns contribute nothing). Two layouts are handled:
      (1) CADO 9bb8fc079 replay writes weight 0 for the row in BOTH sparse.bin and rw.bin;
      (2) replay's sparsemat[i]==NULL branch writes the 0 header to sparse.bin but NOTHING to
          rw.bin, so rw.bin is SHORTER than the matrix: detected by the word-count identity and
          recovered by walking sparse.bin. Both are reported LOUDLY; a matrix the walk cannot
          explain (or whose row count disagrees with LA_CADO_NROWS, the index's nrows) raises."""
    t0 = time.time()
    if nd is None:
        nd = int(os.environ.get("LA_CADO_ND", "64"))
    nd = max(0, min(MAX_ND, int(nd)))
    want_rows = int(os.environ.get("LA_CADO_NROWS", "0") or 0)
    rwp, cwp = cado_sidecars(path)
    a = np.fromfile(path, dtype="<u4")
    w = None
    if os.path.exists(rwp):
        w = np.fromfile(rwp, dtype="<u4").astype(np.int64)
        if int(w.sum()) + len(w) != len(a) or (want_rows and len(w) != want_rows):
            wr = w
            w = _row_weights_by_walk(a)
            nz = w[w > 0]
            if len(wr) == len(nz) and np.array_equal(wr, nz):
                print("  [cado_mat] WARNING: %s has %d rows but the matrix has %d: replay omitted the %d EMPTY "
                      "merged row(s) from rw.bin -- recovered by walking the matrix" % (
                          os.path.basename(rwp), len(wr), len(w), len(w) - len(wr)), flush=True)
            else:
                raise ValueError("CADO matrix %s: %d words, rw.bin %d rows / %d entries, walk %d rows / %d "
                                 "entries: stale rw.bin or truncated sparse.bin" % (
                                     path, len(a), len(wr), int(wr.sum()), len(w), int(w.sum())))
            del wr, nz
    else:
        print("  [cado_mat] WARNING: %s missing -- walking the matrix to find row starts" % rwp, flush=True)
        w = _row_weights_by_walk(a)
    nrows_cado = len(w)
    if want_rows and nrows_cado != want_rows:
        raise ValueError("CADO matrix %s has %d rows but the replay index says %d" % (path, nrows_cado, want_rows))
    W = int(w.sum())
    if W + nrows_cado != len(a):
        # LOUD: a mismatch means rw.bin is from another run or the matrix is truncated.
        raise ValueError("CADO matrix %s: %d words but rw.bin says %d rows + %d entries = %d "
                         "(stale rw.bin or truncated sparse.bin)" % (path, len(a), nrows_cado, W, W + nrows_cado))
    starts = np.empty(nrows_cado, dtype=np.int64)
    starts[0] = 0
    if nrows_cado > 1:
        np.cumsum(w[:-1] + 1, out=starts[1:])
    hdr = a[starts].astype(np.int64)
    if not np.array_equal(hdr, w):
        bad = int(np.flatnonzero(hdr != w)[0])
        raise ValueError("CADO matrix %s: row %d header %d != rw.bin %d" % (path, bad, int(hdr[bad]), int(w[bad])))
    del hdr
    keep = np.ones(len(a), dtype=bool); keep[starts] = False
    vals = a[keep]; del keep, a, starts          # all column indices, row-major (uint32)
    if os.path.exists(cwp):
        ncols_cado = os.path.getsize(cwp) // 4
    else:
        ncols_cado = int(vals.max()) + 1 if len(vals) else 0
    if len(vals) and int(vals.max()) >= ncols_cado:
        raise ValueError("CADO matrix %s: column index %d >= ncols %d" % (path, int(vals.max()), ncols_cado))
    if ncols_cado > nrows_cado:
        raise ValueError("CADO matrix %s has more columns (%d) than rows (%d): no left kernel guaranteed"
                         % (path, ncols_cado, nrows_cado))
    if nrows_cado >= 2**31:
        raise ValueError("matrix too large for int32 indices")
    # ---- empty merged rows: drop them from B (see docstring) ----
    rowmap = None
    empty = np.flatnonzero(w == 0)
    if len(empty):
        live = np.flatnonzero(w > 0)
        print("  [cado_mat] WARNING: %d EMPTY merged row(s) (merge found a square: first at CADO row %d) -- "
              "dropped from the LA; their kernel bits are written as 0" % (len(empty), int(empty[0])), flush=True)
        rowmap = (nrows_cado, live)
        w = w[live]
    nb_cols = len(w)                              # B columns = non-empty CADO rows
    # The rows>=cols guard above ran BEFORE the empty rows were dropped. gpu_la solves the SQUARE
    # n x n problem with n = nb_cols (B rows >= n are neither counted nor verified), so B must not
    # have more rows (ideals) than columns (non-empty relation-sets): that would index past gpu_la's
    # n-sized row buffers on the GPU and let a non-kernel vector pass VERIFY. Fail LOUDLY instead.
    if ncols_cado > nb_cols:
        raise ValueError("CADO matrix %s: %d EMPTY merged rows exceed the excess -- %d ideals vs only %d "
                         "non-empty relation-sets; no left kernel guaranteed (rerun the filter with a larger "
                         "purge -keep)" % (path, nrows_cado - nb_cols, ncols_cado, nb_cols))
    nd = min(nd, ncols_cado)
    # B column index of every entry = (compacted) CADO row index (non-decreasing)
    cidx = np.repeat(np.arange(nb_cols, dtype=np.int32), w)
    del w
    dcol = np.zeros(2 * nb_cols, dtype=np.uint64)
    if nd > 0:
        dm = vals < nd
        rd = vals[dm]; cd = cidx[dm]
        # Within one CADO row the column indices are DISTINCT (a merged row is a GF(2) sum, repeated
        # ideals cancel), so per (row, word) the OR of the bits equals their SUM -> reduceat.
        # Two passes (word 0 = dense rows 0..63, word 1 = 64..127); cd stays non-decreasing inside
        # each pass, so run starts are just the change points.
        for hi in (0, 1):
            m = (rd >> 6) == hi
            if not m.any():
                continue
            c = cd[m]; bits = np.left_shift(np.uint64(1), (rd[m] & 63).astype(np.uint64))
            st = np.concatenate(([0], np.flatnonzero(c[1:] != c[:-1]) + 1))
            sums = np.add.reduceat(bits, st)
            dcol[2 * c[st].astype(np.int64) + hi] = sums
            del c, bits, st, sums, m
        # guard the distinctness assumption: popcount of dcol must equal number of dense entries
        pc = int(np.unpackbits(dcol.view(np.uint8)).sum())
        if pc != len(rd):
            raise ValueError("dense packing lost bits (%d set vs %d entries): duplicate column in a row?" % (pc, len(rd)))
        del rd, cd
        sm = ~dm; del dm
        R = vals[sm].view(np.int32); C = cidx[sm]
        del sm
    else:
        R = vals.view(np.int32); C = cidx
    del vals, cidx
    if verbose:
        print("  [cado_mat] %s: CADO %d rows (relation-sets) x %d cols (ideals), W=%d -> B = M^T %dx%d, "
              "ND=%d dense rows (%d bits), sparse nnz=%d, loaded in %.1fs"
              % (os.path.basename(path), nrows_cado, ncols_cado, W + 0, ncols_cado, nb_cols,
                 nd, W - len(R), len(R), time.time() - t0), flush=True)
    if with_map:
        return ncols_cado, nd, nb_cols, R, C, dcol, rowmap
    return ncols_cado, nd, nb_cols, R, C, dcol


# =================================================================================================
# LA_PARSE=gpu (2026-10-23): the same loader with the entry-level work on the GPU.
# The numpy loader above spends ~40 s at N 12.9M / W 1.85e9 (one 7.4 GB file): single-threaded boolean masks, a
# 7.4 GB compress, a 6.4 GB np.repeat, the dense split -- and gpu_la then copies R and C (12.8 GB) to the device.
# Here the raw words go to the device ONCE (pinned, double-buffered chunked read), the ROW-level bookkeeping (12.9M
# rows: weights, starts, empty rows, compacted index) stays in numpy exactly as above, and two warp-per-row kernels
# do the entry-level work:
#   pass 1: header check (a[start] == w), per-row sparse count, max column, dense bits OR-ed into dcol (atomicOr; a
#           bit already set = a duplicate dense column in a row, the condition the numpy popcount guard catches);
#   pass 2: stable compaction of the sparse entries (column >= ND) into R/C at offsets = exclusive cumsum of the
#           pass-1 counts: lane order within a row via __ballot_sync + __popc, rows in file order -- the order of
#           numpy's vals[~dm] / cidx[~dm], so R, C and dcol are EQUAL ELEMENT FOR ELEMENT (not merely the same matrix).
# R, C, dcol come back as CuPy arrays (gpu_la's run256/run128/64-bit paths cp.asarray() them: no copy, and the 12.8 GB
# host->device copy inside "seg SpMV plan built in" disappears too). Anything outside the plain layout (rw.bin
# missing/short/stale, cw.bin missing, odd file size) uses the numpy loader, which owns those diagnostics. A GPU-side
# failure (OOM, NVRTC) falls back to the numpy loader with a WARNING; matrix errors raise ValueError like it does.
# LA_PARSE = cpu | gpu (DEFAULT since s65 on key1g TD 130, 12.90M rows: IDENTICAL)
#          | check (both; compares R, C, dcol, shapes, rowmap; prints "LA_PARSE=check: ... MATCH|MISMATCH"; returns numpy's)
# No GPU here? `python3 cado_mat.py --selftest`: the GPU loader's host side + _NumpyDev (the two kernels transliterated
# lane by lane) against the numpy loader on synthetic matrices (dense 0/64/128, empty rows, duplicate and header errors).
# On a box: `python3 cado_mat.py --compare <X.sparse.bin> [reps]` times both loaders and compares them.
_PARSE_CU = r'''
#define FULLM 0xffffffffu
extern "C" __global__ void cm_pass1(const unsigned int* a, const long long* st, const int* w, const int* cr, long long nrows,
    unsigned int nd, unsigned long long* dcol, int* cnt, unsigned int* mx, int* err) {
  // one warp per CADO row r (warp-uniform loop): a[st[r]] must be w[r]; entries a[st[r]+1 .. st[r]+w[r]]
  const int lane = threadIdx.x & 31;
  const long long nw = ((long long)gridDim.x * blockDim.x) >> 5;
  for (long long r = (((long long)blockIdx.x * blockDim.x) + threadIdx.x) >> 5; r < nrows; r += nw) {
    const long long s = st[r]; const int k = w[r]; const int c = cr[r];
    if (lane == 0 && a[s] != (unsigned int)k) atomicMin(&err[0], (int)r);
    int ns = 0; unsigned int m = 0;
    for (int j = lane; j < k; j += 32) {
      const unsigned int v = a[s + 1 + j];
      m = v > m ? v : m;
      if (v < nd) {
        const unsigned long long bit = 1ULL << (v & 63);
        const unsigned long long old = atomicOr(&dcol[2LL * c + (v >> 6)], bit);
        if (old & bit) err[1] = 1;
      } else ns++;
    }
    for (int o = 16; o; o >>= 1) {
      ns += __shfl_down_sync(FULLM, ns, o);
      const unsigned int t = __shfl_down_sync(FULLM, m, o); m = t > m ? t : m;
    }
    if (lane == 0) { cnt[r] = ns; if (m) atomicMax(mx, m); }
  }
}
extern "C" __global__ void cm_pass2(const unsigned int* a, const long long* st, const int* w, const int* cr,
    const long long* os, long long nrows, unsigned int nd, int* R, int* C) {
  // stable compaction: entry j of row r with a value >= nd goes to os[r] + (number of such entries before j in the row)
  const int lane = threadIdx.x & 31; const unsigned int lt = (1u << lane) - 1u;
  const long long nw = ((long long)gridDim.x * blockDim.x) >> 5;
  for (long long r = (((long long)blockIdx.x * blockDim.x) + threadIdx.x) >> 5; r < nrows; r += nw) {
    const long long s = st[r] + 1; const int k = w[r]; const int c = cr[r]; long long o = os[r];
    for (int j0 = 0; j0 < k; j0 += 32) {
      const int j = j0 + lane; unsigned int v = 0; bool p = false;
      if (j < k) { v = a[s + j]; p = v >= nd; }
      const unsigned int b = __ballot_sync(FULLM, p);
      if (p) { const long long q = o + __popc(b & lt); R[q] = (int)v; C[q] = c; }
      o += __popc(b);
    }
  }
}
'''
_LAST_READ = [0.0]


class _Fallback(Exception):
    """the plain-layout preconditions do not hold: use the numpy loader (it owns those diagnostics)"""


class _CupyDev:
    """device side of load_cado_sparse_gpu on CuPy"""
    def __init__(self):
        import cupy as cp
        self.cp = cp
        m = cp.RawModule(code=_PARSE_CU)
        self.k1 = m.get_function("cm_pass1"); self.k2 = m.get_function("cm_pass2")
        self.t_read = 0.0

    def read(self, path, nwords):
        """the file's nwords uint32 on the device: pinned double-buffered chunks (chunk i+1 is read while i copies)"""
        cp = self.cp; t = time.time()
        d = cp.empty(nwords, dtype=cp.uint32); nbytes = nwords * 4
        if nbytes == 0:
            return d
        CHB = max(4096, int(float(os.environ.get("LA_PARSE_CHUNK_MB", "256")) * 2**20) // 4096 * 4096)
        try:
            bufs = [cp.cuda.alloc_pinned_memory(CHB) for _ in range(2)]
            hv = [np.frombuffer(b, dtype=np.uint8, count=CHB) for b in bufs]
            ev = [None, None]; st = cp.cuda.Stream(non_blocking=True); dv = d.view(cp.uint8)
            with open(path, "rb", buffering=0) as f:
                off = 0; i = 0
                while off < nbytes:
                    k = i & 1
                    if ev[k] is not None:
                        ev[k].synchronize()          # buffer k's previous copy has left the host buffer
                    want = min(CHB, nbytes - off); mv = memoryview(hv[k])[:want]; got = 0
                    while got < want:
                        r = f.readinto(mv[got:])
                        if not r:
                            raise ValueError("CADO matrix %s: short read at byte %d of %d" % (path, off + got, nbytes))
                        got += r
                    dv[off:off + want].set(hv[k][:want], stream=st)
                    ev[k] = st.record(); off += want; i += 1
            st.synchronize()
            del bufs, hv
        except ValueError:
            raise
        except Exception as e:      # no pinned memory etc.: plain read + pageable copy
            print("  [cado_mat] NOTE: pinned chunked read failed (%s: %s) -- np.fromfile + copy"
                  % (type(e).__name__, str(e)[:120]), flush=True)
            d.set(np.fromfile(path, dtype="<u4", count=nwords))
        self.t_read = time.time() - t
        return d

    def asarray(self, x):
        return self.cp.asarray(x)

    def word(self, a, i):
        return int(a[int(i)])

    @staticmethod
    def _grid(nrows):
        return (max(1, min(1 << 20, -(-int(nrows) // 8))),), (256,)      # 8 warps (rows) per block, grid-stride

    def pass1(self, a, st, w, cr, nrows, nd, ncb):
        cp = self.cp
        dcol = cp.zeros(2 * ncb, dtype=cp.uint64); cnt = cp.zeros(max(1, nrows), dtype=cp.int32)
        mx = cp.zeros(1, dtype=cp.uint32); err = cp.array([2**31 - 1, 0], dtype=cp.int32)
        if nrows:
            g, b = self._grid(nrows)
            self.k1(g, b, (a, st, w, cr, np.int64(nrows), np.uint32(nd), dcol, cnt, mx, err))
        e = cp.asnumpy(err)
        bad = -1 if int(e[0]) == 2**31 - 1 else int(e[0])
        return dcol, cnt[:nrows], int(cp.asnumpy(mx)[0]), bad, bool(e[1])

    def offsets(self, cnt):
        cp = self.cp; n = int(cnt.size)
        os_ = cp.zeros(max(1, n), dtype=cp.int64)
        if n > 1:
            os_[1:n] = cp.cumsum(cnt[:-1], dtype=cp.int64)
        tot = int(os_[n - 1]) + int(cnt[n - 1]) if n else 0
        return os_, tot

    def pass2(self, a, st, w, cr, os_, nrows, nd, tot):
        cp = self.cp
        R = cp.empty(tot, dtype=cp.int32); C = cp.empty(tot, dtype=cp.int32)
        if nrows and tot:
            g, b = self._grid(nrows)
            self.k2(g, b, (a, st, w, cr, os_, np.int64(nrows), np.uint32(nd), R, C))
        return R, C

    def done(self):
        cp = self.cp
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()


class _NumpyDev:
    """_CupyDev's interface in numpy with the two kernels transliterated warp by warp (32 lanes = one numpy chunk,
    __ballot_sync/__popc spelled out): tests load_cado_sparse_gpu's host side and the kernels' lane logic without a
    GPU. A python loop per row: small matrices only."""
    t_read = 0.0

    def read(self, path, nwords):
        return np.fromfile(path, dtype="<u4", count=nwords)

    def asarray(self, x):
        return np.asarray(x)

    def word(self, a, i):
        return int(a[int(i)])

    def pass1(self, a, st, w, cr, nrows, nd, ncb):
        dcol = np.zeros(2 * ncb, dtype=np.uint64); cnt = np.zeros(nrows, dtype=np.int32); mx = 0; bad = -1; dup = False
        lanes = np.arange(32)
        for r in range(nrows):
            s = int(st[r]); k = int(w[r]); c = int(cr[r])
            if int(a[s]) != k and bad < 0:
                bad = r                                        # atomicMin over rows: the first bad row
            ns = np.zeros(32, np.int64); m = np.zeros(32, np.uint64)
            for j0 in range(0, k, 32):                         # lane L handles entries j0 + L
                j = j0 + lanes; act = j < k; v = a[s + 1 + j[act]].astype(np.uint64)
                m[act] = np.maximum(m[act], v)
                dn = v < nd
                for vv in v[dn]:                               # atomicOr (any serialisation order)
                    bit = np.uint64(1) << (vv & np.uint64(63)); idx = 2 * c + int(vv >> np.uint64(6))
                    if dcol[idx] & bit:
                        dup = True
                    dcol[idx] |= bit
                ns[np.flatnonzero(act)[~dn]] += 1
            cnt[r] = int(ns.sum()); mm = int(m.max())
            if mm:
                mx = max(mx, mm)
        return dcol, cnt, mx, bad, dup

    def offsets(self, cnt):
        n = len(cnt); os_ = np.zeros(max(1, n), dtype=np.int64)
        if n > 1:
            os_[1:n] = np.cumsum(cnt[:-1], dtype=np.int64)
        return os_, (int(os_[n - 1]) + int(cnt[n - 1]) if n else 0)

    def pass2(self, a, st, w, cr, os_, nrows, nd, tot):
        R = np.full(tot, -7, dtype=np.int32); C = np.full(tot, -7, dtype=np.int32); lanes = np.arange(32)
        for r in range(nrows):
            s = int(st[r]) + 1; k = int(w[r]); c = int(cr[r]); o = int(os_[r])
            for j0 in range(0, k, 32):
                j = j0 + lanes; act = j < k
                v = np.zeros(32, np.uint64); v[act] = a[s + j[act]]
                p = act & (v >= nd)
                b = int(np.sum(p.astype(np.int64) << lanes))                      # __ballot_sync
                for ln in np.flatnonzero(p):
                    q = o + bin(b & ((1 << int(ln)) - 1)).count("1")              # __popc(b & lanemask_lt)
                    R[q] = int(v[ln]); C[q] = c
                o += bin(b).count("1")
        return R, C

    def done(self):
        pass


def load_cado_sparse_gpu(path, nd=None, verbose=True, with_map=False, dev=None):
    """load_cado_sparse with the entry-level work on the device `dev` (default CuPy). Same return value, element for
    element; R, C, dcol are device arrays. Raises _Fallback when the plain layout does not hold."""
    t0 = time.time()
    if nd is None:
        nd = int(os.environ.get("LA_CADO_ND", "64"))
    nd = max(0, min(MAX_ND, int(nd)))
    want_rows = int(os.environ.get("LA_CADO_NROWS", "0") or 0)
    rwp, cwp = cado_sidecars(path)
    if not (os.path.exists(rwp) and os.path.exists(cwp)):
        raise _Fallback("rw.bin or cw.bin missing")
    fsz = os.path.getsize(path)
    if fsz % 4:
        raise _Fallback("file size %d is not a multiple of 4" % fsz)
    L = fsz // 4
    w = np.fromfile(rwp, dtype="<u4").astype(np.int64)
    if int(w.sum()) + len(w) != L or (want_rows and len(w) != want_rows):
        raise _Fallback("rw.bin does not describe the matrix (empty rows omitted, stale, or LA_CADO_NROWS differs)")
    nrows_cado = len(w); W = int(w.sum())
    if nrows_cado >= 2**31:
        raise ValueError("matrix too large for int32 indices")
    ncols_cado = os.path.getsize(cwp) // 4
    if ncols_cado > nrows_cado:
        raise ValueError("CADO matrix %s has more columns (%d) than rows (%d): no left kernel guaranteed"
                         % (path, ncols_cado, nrows_cado))
    rowmap = None
    live = np.flatnonzero(w > 0); nb_cols = len(live)
    if nb_cols != nrows_cado:
        print("  [cado_mat] WARNING: %d EMPTY merged row(s) (merge found a square: first at CADO row %d) -- "
              "dropped from the LA; their kernel bits are written as 0"
              % (nrows_cado - nb_cols, int(np.flatnonzero(w == 0)[0])), flush=True)
        rowmap = (nrows_cado, live)
    if ncols_cado > nb_cols:
        raise ValueError("CADO matrix %s: %d EMPTY merged rows exceed the excess -- %d ideals vs only %d "
                         "non-empty relation-sets; no left kernel guaranteed (rerun the filter with a larger "
                         "purge -keep)" % (path, nrows_cado - nb_cols, ncols_cado, nb_cols))
    nd = min(nd, ncols_cado)
    # row-level bookkeeping on the host (12.9M rows: ~0.1 s), identical to load_cado_sparse's
    st = np.zeros(nrows_cado, dtype=np.int64)
    if nrows_cado > 1:
        np.cumsum(w[:-1] + 1, out=st[1:])
    cr = np.full(nrows_cado, -1, dtype=np.int32); cr[live] = np.arange(nb_cols, dtype=np.int32)
    dev = dev or _CupyDev()
    a = dev.read(path, L); _LAST_READ[0] = dev.t_read
    d_st = dev.asarray(st); d_w = dev.asarray(w.astype(np.int32)); d_cr = dev.asarray(cr)
    dcol, cnt, mx, bad, dup = dev.pass1(a, d_st, d_w, d_cr, nrows_cado, nd, nb_cols)
    if bad >= 0:
        raise ValueError("CADO matrix %s: row %d header %d != rw.bin %d" % (path, bad, dev.word(a, st[bad]), int(w[bad])))
    if W and mx >= ncols_cado:
        raise ValueError("CADO matrix %s: column index %d >= ncols %d" % (path, mx, ncols_cado))
    if dup:
        raise ValueError("dense packing lost bits: duplicate column in a row?")
    os_, tot = dev.offsets(cnt)
    R, C = dev.pass2(a, d_st, d_w, d_cr, os_, nrows_cado, nd, tot)
    del a, d_st, d_w, d_cr, cnt, os_
    dev.done()
    if verbose:
        print("  [cado_mat] %s: CADO %d rows (relation-sets) x %d cols (ideals), W=%d -> B = M^T %dx%d, "
              "ND=%d dense rows (%d bits), sparse nnz=%d, loaded in %.1fs (LA_PARSE gpu, file read %.1fs)"
              % (os.path.basename(path), nrows_cado, ncols_cado, W, ncols_cado, nb_cols,
                 nd, W - tot, tot, time.time() - t0, _LAST_READ[0]), flush=True)
    if with_map:
        return ncols_cado, nd, nb_cols, R, C, dcol, rowmap
    return ncols_cado, nd, nb_cols, R, C, dcol


def _host(x):
    if isinstance(x, np.ndarray):
        return x
    import cupy as cp
    return cp.asnumpy(x)


def _free_pool():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def same_load(g, c):
    """two with_map=True load results equal element for element? -> (bool, first difference)"""
    names = ("ncols_B", "ND", "nb_cols", "R", "C", "dcol", "rowmap")
    for nm, x, y in zip(names, g, c):
        if nm in ("R", "C", "dcol"):
            x = _host(x); y = _host(y)
            if x.dtype != y.dtype or x.shape != y.shape:
                return False, "%s: %s%s vs %s%s" % (nm, x.dtype, x.shape, y.dtype, y.shape)
            if not np.array_equal(x, y):
                i = int(np.flatnonzero(x != y)[0])
                return False, "%s[%d]: %d vs %d" % (nm, i, int(x[i]), int(y[i]))
        elif nm == "rowmap":
            if (x is None) != (y is None) or (x is not None and (x[0] != y[0] or not np.array_equal(x[1], y[1]))):
                return False, "rowmap differs"
        elif x != y:
            return False, "%s: %r vs %r" % (nm, x, y)
    return True, ""


def load_cado(path, nd=None, verbose=True, with_map=False, mode=None):
    """gpu_la's entry point: LA_PARSE = cpu (default) | gpu | check (see above)"""
    mode = mode or os.environ.get("LA_PARSE", "gpu") or "gpu"   # default gpu since s65 (sb52): IDENTICAL arrays + LA_HASH, 36-40 s -> 2-2.5 s
    if mode not in ("cpu", "gpu", "check"):
        raise ValueError("LA_PARSE=%s (must be cpu, gpu or check)" % mode)
    if mode == "cpu":
        return load_cado_sparse(path, nd, verbose, with_map)
    t = time.time()
    try:
        g = load_cado_sparse_gpu(path, nd, verbose, True)
    except _Fallback as e:
        print("  [cado_mat] NOTE: LA_PARSE=%s: %s -- numpy loader" % (mode, e), flush=True)
        return load_cado_sparse(path, nd, verbose, with_map)
    except ValueError:
        raise
    except Exception as e:
        print("  [cado_mat] WARNING: GPU parse failed (%s: %s) -- numpy loader" % (type(e).__name__, str(e)[:200]), flush=True)
        _free_pool()
        return load_cado_sparse(path, nd, verbose, with_map)
    tg = time.time() - t
    if mode == "check":
        t = time.time(); c = load_cado_sparse(path, nd, verbose, True); tc = time.time() - t
        ok, why = same_load(g, c)
        print("  [cado_mat] LA_PARSE=check: GPU %.1fs vs numpy %.1fs: %s%s (numpy result used)"
              % (tg, tc, "MATCH" if ok else "MISMATCH", "" if ok else " -- " + why), flush=True)
        del g; _free_pool()
        return c if with_map else c[:6]
    return g if with_map else g[:6]


def synth_cado(dirpath, n_ideals, excess=64, kmin=8, kmax=40, seed=1, empty=(), dup_dense=False):
    """write a small CADO-layout matrix <dirpath>/c.sparse.bin (+ .rw.bin/.cw.bin): n_ideals columns numbered by
    decreasing weight (as replay does), n_ideals+excess rows of distinct columns in arbitrary order. `empty` = rows
    written with weight 0 (layout (1) above); dup_dense repeats column 0 in row 0. -> (path, nrows)"""
    rng = np.random.default_rng(seed); nr = n_ideals + excess
    k = rng.integers(kmin, kmax + 1, nr)
    raw = (n_ideals * rng.random((nr, kmax)) ** 3).astype(np.int64)      # skewed: low columns heavy
    raw.sort(axis=1)
    keep = np.ones_like(raw, dtype=bool); keep[:, 1:] = raw[:, 1:] != raw[:, :-1]
    keep &= np.arange(kmax)[None, :] < k[:, None]
    for r in empty:
        keep[r] = False
    rows = [rng.permutation(raw[i][keep[i]]) for i in range(nr)]
    cw = np.bincount(np.concatenate(rows), minlength=n_ideals).astype(np.int64)
    perm = np.argsort(-cw, kind="stable"); ren = np.empty(n_ideals, np.int64); ren[perm] = np.arange(n_ideals)
    rows = [ren[x] for x in rows]
    if dup_dense:
        rows[0] = np.concatenate([rows[0], [0, 0]])
    os.makedirs(dirpath, exist_ok=True); path = os.path.join(dirpath, "c.sparse.bin")
    np.concatenate([np.concatenate([[len(x)], x]) for x in rows]).astype("<u4").tofile(path)
    rwp, cwp = cado_sidecars(path)
    np.array([len(x) for x in rows], dtype="<u4").tofile(rwp)
    cw[perm].astype("<u4").tofile(cwp)
    return path, nr


def _selftest():
    """numpy loader vs load_cado_sparse_gpu on _NumpyDev (kernel transliteration): layouts and error paths"""
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for i, (n, nd, empty, kmax) in enumerate([(3000, 64, (), 40), (2000, 128, (), 90), (2000, 0, (), 40),
                                                  (1500, 64, (5, 77), 40), (300, 64, (), 70), (100, 64, (), 40)]):
            path, nr = synth_cado(os.path.join(td, str(i)), n, seed=i + 1, empty=empty, kmax=kmax)
            c = load_cado_sparse(path, nd, False, True)
            g = load_cado_sparse_gpu(path, nd, False, True, dev=_NumpyDev())
            same, why = same_load(g, c)
            print("selftest %d: n %d nd %d empty %s nnz %d dense bits %d: %s %s"
                  % (i, n, nd, list(empty), len(c[3]), int(np.unpackbits(c[5].view(np.uint8)).sum()),
                     "IDENTICAL" if same else "MISMATCH", why))
            ok &= same
        # error paths raise in both: duplicate dense column; a row header swapped with the previous row's last entry
        path, nr = synth_cado(os.path.join(td, "dup"), 800, seed=9, dup_dense=True)
        for nm, fn in (("cpu", lambda: load_cado_sparse(path, 64, False)),
                       ("dev", lambda: load_cado_sparse_gpu(path, 64, False, dev=_NumpyDev()))):
            try:
                fn(); print("selftest dup %s: NO ERROR" % nm); ok = False
            except ValueError as e:
                print("selftest dup %s: ValueError (%s)" % (nm, str(e)[:60]))
        path, nr = synth_cado(os.path.join(td, "hdr"), 800, seed=10)
        a = np.fromfile(path, dtype="<u4"); w = np.fromfile(cado_sidecars(path)[0], dtype="<u4")
        p = int(w[0]) + 1 + int(w[1]) + 1                                  # header of row 2
        a[p - 1], a[p] = a[p], a[p - 1]; a.tofile(path)
        errs = []
        for fn in (lambda: load_cado_sparse(path, 64, False), lambda: load_cado_sparse_gpu(path, 64, False, dev=_NumpyDev())):
            try:
                fn(); errs.append("none")
            except ValueError as e:
                errs.append(str(e))
        print("selftest header: cpu %r / dev %r" % (errs[0][-40:], errs[1][-40:]))
        ok &= errs[0] == errs[1] and errs[0] != "none"
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


def _compare(path, reps=1):
    """box test (gpu/s65_lastart.sh): numpy loader vs GPU loader on a real matrix, timed, element for element"""
    rc = 0
    for r in range(reps):
        t = time.time(); c = load_cado_sparse(path, None, True, True); tc = time.time() - t
        t = time.time(); g = load_cado_sparse_gpu(path, None, True, True); tg = time.time() - t
        ok, why = same_load(g, c)
        print("PARSE rep %d: cpu %.2f s gpu %.2f s (file read %.2f s): %s %s"
              % (r + 1, tc, tg, _LAST_READ[0], "IDENTICAL" if ok else "MISMATCH", why), flush=True)
        del g, c; _free_pool()
        rc |= 0 if ok else 1
    return rc


def expand_kernel(x, rowmap):
    """Kernel over the compacted B columns -> flat kernel over ALL CADO rows (zeros at empty rows)."""
    if rowmap is None:
        return x
    n, live = rowmap
    if len(x) != len(live):
        raise ValueError("kernel has %d entries but %d non-empty rows were solved" % (len(x), len(live)))
    full = np.zeros(n, dtype=np.uint64)
    full[live] = np.asarray(x, dtype=np.uint64)
    return full


def index_nrows(indexpath):
    """First line of the replay index file (plain or .gz) = small_nrows."""
    import gzip
    op = gzip.open if indexpath.endswith(".gz") else open
    with op(indexpath, "rt") as f:
        return int(f.readline().split()[0])


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    if len(sys.argv) >= 3 and sys.argv[1] == "--compare":
        sys.exit(_compare(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1))
    nr, ND, nc, R, C, dcol = load_cado(sys.argv[1])
    print(nr, ND, nc, len(R), int(C[-1]) if len(C) else -1)
