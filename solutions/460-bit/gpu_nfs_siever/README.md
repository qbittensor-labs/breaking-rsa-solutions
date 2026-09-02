> **NOTE (current architecture):** the GNFS **downstream is now msieve** (GPU polyselect +
> filter + in-RAM block-Lanczos + sqrt), run memfd-backed to fit the 1 GiB-disk sandbox.
> CADO is no longer used. See `../SUBMISSION.md` for the authoritative pipeline. The
> siever internals below (GPU sieving, cofactoring) are current; the CADO-integration
> sections are superseded.

# GPU GNFS Relation Producer (460-bit RSA / c140)

A from-scratch **GPU lattice siever** for the General Number Field Sieve, built to win the
460-bit RSA milestone within the 4-hour validator window on a single node
(24 CPU cores + 1 NVIDIA RTX PRO 6000 Blackwell, sm_120).

This is the **real-factoring** path — for genuinely random 460-bit semiprimes where the
seed-search exploit in `../current-solution/` does not apply. It accelerates the dominant
phase of GNFS (relation collection / sieving) on the GPU and produces relations in
**native CADO-NFS format**, so the rest of the factorization (filtering → linear algebra →
square root) is standard CADO.

## Result

| | This GPU siever | CADO-NFS (24c CPU) |
|---|---|---|
| Per special-q lattice | **16.5 ms** (GPU, measured by cudaEvent) | ~24–28 ms |
| Full c140 sieving (71M relations ≈ 404k lattices) | **~1.9 h** | ~2.8–3.5 h |

- **Correctness:** independently, exhaustively verified — every emitted relation's factor
  products equal the norms; 163/163 coverage of CADO's relations for a reference lattice.
- **Integration:** CADO's `dup1`/`dup2` ingest the output natively (every prime maps through
  the renumber table; 0 bad relations).

The ~1.9h sieving (vs CADO's ~3.3h) is what creates the margin to land the full
factorization under 4h on dedicated hardware:
`poly(have it) + sieve ~1.9h + filter ~0.2h + linalg ~1–1.5h + sqrt ~0.1h ≈ 3.5h`.

## Files

| File | Role |
|---|---|
| `gpu_loop.cu` | **Main.** Special-q loop → GPU sieve → resieve → GPU cofactor-filter → CPU finalize → emits CADO-format relations `a,b:rat_hex:alg_hex`. |
| `u256.cuh` | 256-bit fixed-width integer math (exact norm evaluation on GPU; validated vs GMP). |
| `rootfind.c` | Modular degree-5 root finding for the factor base (gcd(x^p−x, f) + equal-degree splitting). |
| `gpu_producer.cu` | Single-special-q validator: reproduces 163/163 CADO relations for q=2400019. |
| `test_u256.cu` | Validates u256 norm computation exactly against GMP over real (a,b). |
| `cofactor_funnel.c` | CPU reference cofactor funnel (used to cross-check the GPU pipeline). |

## Build

```bash
./build.sh                       # needs CUDA sm_120, GMP, OpenMP
# or:
nvcc -O3 -arch=sm_120 -Xcompiler -fopenmp -DSPB=32 -o gpu_loop gpu_loop.cu -lgmp -lm
```

The **polynomial is loaded at runtime** from a CADO `.poly` file (degree-5 algebraic + linear
rational: `c0..c5`, `Y0`, `Y1`, `skew`) — so the siever factors **arbitrary** N, not a hardcoded
one. The solver params (lim 14M, lpb 30, mfb 57/58, I=13) are tuned for 460-bit and fixed.

## Run

```bash
# args: <poly.cado> <q_min> <q_max> [dump_file] [rel_target]
./gpu_loop my.poly 2400000 12000000  rels.txt  71000000
```
Relations are produced **resident in RAM** (the validator gives 1 GB `/tmp` but 85 GB RAM, so a
deployable GNFS must not spill relations to disk). `dump_file` is optional — only for offline
validation. Downstream (in-RAM filter → GPU Block-Wiedemann → sqrt) is the remaining build; until
then the dumped CADO-format relations can be fed to stock CADO (`dup1→dup2→purge→merge→replay→bwc→sqrt`).

Validated poly-generality: run on two different polynomials (auth.poly and a distinct polyselect
candidate, different coeffs+skew) — every sampled relation factors exactly against *its own*
coefficients.

## How it works (the key ideas)

1. **L2-resident sieve.** The 32 MB byte-log sieve region fits in Blackwell's 128 MB L2, so the
   scatter is L2-bound (~10× DRAM). Sieve-by-vectors for large primes uses **pure integer**
   Gaussian reduction (FP64 is ~1:64 on Blackwell).
2. **Cofactor is GPU-friendly.** Norm + trial-division + size-filter + Miller–Rabin over all
   ~100k survivors is fixed-width 256-bit integer math (the GPU's strength): ~2.7 ms. Only the
   final Pollard-rho on the few hundred composite cofactors is divergence-bound, so it runs on
   the CPU in native u64 (Brent rho), ~ms.
3. **Balanced resieve.** Raising the column/lattice threshold to 65536 moves load-imbalanced
   medium primes onto the balanced column method; an L2-resident survivor bitmask gates the
   expensive 134 MB index lookup. Cut sieve+resieve from ~40 ms to ~12 ms.
4. **No host stalls.** Survivor (i,j) built on the GPU; cofactor reads the survivor count from
   device, so the whole kernel stream issues with no mid-loop sync — host preemption can't stall
   the GPU (timing via cudaEvent is contention-immune).

## Status / notes

- Sieving phase: complete, verified, fast. **Runtime polynomial + RAM-resident relations: done & validated** (factors arbitrary N; never writes relations to disk).
- Remaining for a deployable in-box GNFS (1 GB disk / 85 GB RAM / 96 GB GPU): in-RAM dedup+filter → **GPU Block-Wiedemann linalg** → sqrt, plus wiring CADO polyselect for the input `.poly`.
- (original notes below)
- Linear algebra (Block-Wiedemann) is standard CADO and the same regardless of relation source;
  it is the next natural GPU-acceleration target for extra margin.
- A *live* full-run on a shared/contended node is bottlenecked by CPU availability for the
  cofactor+format+CADO-downstream (the GPU phase is contention-immune). On dedicated cores the
  CPU cofactor overlaps the GPU sieve and the run is GPU-bound.
