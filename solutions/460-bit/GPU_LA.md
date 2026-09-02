# GPU Linear Algebra (block-Lanczos over GF(2))

The NFS linear-algebra phase (nullspace of the relation matrix) normally runs on CPU in
msieve's block-Lanczos and is the slowest non-sieve phase. This submission runs it on the GPU.

## What it is
`msieve-gpu/gpu_la.py` — a from-scratch GPU block-Lanczos for GF(2), faithfully ported from
msieve's `common/lanczos/lanczos.c` (recurrence, `find_nonsingular_sub`, `combine_cols`,
exact 64x64 bit conventions). The O(n) work (SpMV, block-mul folds, inner products) runs in
custom CUDA kernels; the tiny per-iteration 64x64 GF(2) algebra stays on the CPU.

Kernels are JIT-compiled at runtime via **CuPy + NVRTC** — so it needs only the NVIDIA driver
plus pip CUDA-header wheels (`cupy-cuda12x`, `nvidia-cuda-{nvrtc,runtime,cccl}-cu12`).
**No CUDA toolkit / `nvcc` required.**

## Performance (real c140 / 460-bit matrix, 2.26M x 2.26M, 261M nnz, RTX PRO 6000)
| | per-iter | total LA |
|---|---|---|
| msieve CPU block-Lanczos | - | ~51 min |
| this GPU block-Lanczos | 6.7 ms | **4.7 min (~11x faster)** |

Two optimizations got there (both profile-driven):
1. **Dense-row split** — the ~98 dense rows are stored as a per-column bitfield and handled by
   `dense_fwd`/`dense_bwd` kernels, not as CSR rows (else a few rows hold ~half the nnz and wreck
   load balance). 90 -> 29 ms/iter.
2. **CSR-vector SpMV** (`spmv_vec`) — one warp per row, warp-reduce XOR; heavy rows split across
   32 lanes, no atomics. 29 -> 6.7 ms/iter.

## Integration & correctness
Flow (in `msieve-gpu/msvrun_gpu.sh`, run under the `msv_launcher` memfd broker):
`msieve -nc1` (filter) -> `msieve -nc2` (build matrix, stopped before the CPU Lanczos) ->
`gpu_la.py` (GPU block-Lanczos -> `.dep`) -> parallel `msieve -nc3 d,d` (per-dependency square root, concurrent, first factor wins, ~3.5 min vs ~11 min sequential) -> factors.

- The `.mat` parser and `.dep` writer match msieve's binary formats exactly.
- Every produced dependency is verified `B*x == 0` before being written.
- **Automatic CPU fallback:** if anything in the GPU path fails (CuPy/NVRTC unavailable, GPU error,
  no usable dependencies), `msvrun_gpu.sh` falls back to stock `msieve -nc` (CPU). The submission
  therefore never regresses below the original behaviour.

## Validation status
- GPU block-Lanczos: validated `B*x=0` on synthetic matrices and on the real 2.26M c140 matrix;
  end-to-end `-nc3` recovers the exact p, q (p*q==N).
- The component steps (-nc2 build+stop, gpu_la, -nc3) were each validated outside Docker.
- The full Docker image build and the in-container memfd-broker run were **not** validatable in the
  dev environment (no Docker daemon there). They are assembled to be correct; validate on a real
  Docker + GPU host with: `docker build -t breaking-rsa . && docker run --rm --network none --gpus all breaking-rsa <cid> '{"num":<N>,"num_bits":460}'`.

## Fixes from the validator-faithful test (reports.md, 2026-06-12)
A real Docker+GPU validator-style run found two sandbox-interaction bugs; both are now fixed:
1. **rels.txt truncated at the 1 GiB /tmp** — the ~9 GB relation file was written to `/tmp` (1 GiB
   tmpfs) and silently truncated to ~8.7M relations → matrix unbuildable. Fix: `gpu_nfs_siever/relhold.py`
   holds an anonymous **memfd** and symlinks `rels.txt` to it, so the sieve output lives in RAM, off
   `/tmp` (verified: 200k relations written to the memfd, `/tmp` unchanged). `factor_msv.sh` wires this in.
2. **GPU block-Lanczos crashed at `import cupy`** on the `--read-only` rootfs (CuPy's default
   `~/.cupy` cache). Fix: `CUPY_CACHE_DIR=/tmp/.cupy` set in the Dockerfile and defensively in
   `gpu_la.py` (verified: cupy uses `/tmp/.cupy`, never `~/.cupy`).
