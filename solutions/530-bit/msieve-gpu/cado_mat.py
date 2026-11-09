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
    nr, ND, nc, R, C, dcol = load_cado_sparse(sys.argv[1])
    print(nr, ND, nc, len(R), int(C[-1]) if len(C) else -1)
