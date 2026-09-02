# Breaking-RSA submission

A **tiered** RSA-semiprime factoring solution for the Enigma `breaking_rsa` challenge.
Builds to a single Docker image; entrypoint `breaking_rsa.py` emits the validator output
contract (base64 zip with `result.json`).

## Strategy (cheapest method first, within the 4 h wall-time budget)

| Tier | Method | Wins | Cost |
|---|---|---|---|
| 0 | trial division · Fermat near-square · Pollard-rho (pure Python) | malformed / small-factor / close-prime N | ~seconds |
| 1 | **GPU-GNFS** (full pipeline below) | genuinely random 460-bit semiprimes | ~3.5 h |

Tiers run in order and stop at the first success, time-boxed against the validator's wall clock.

## Tier 1 — the GPU-GNFS pipeline

```
N  →  msieve GPU polyselect  →  gpu_loop GPU lattice sieve  →  memfd-backed msieve
                                                               (filter + in-RAM block-Lanczos + sqrt)  →  p, q
```

Per-key and self-contained — given only N it selects a polynomial, sieves, and factors.
Driver: `gpu_nfs_siever/factor_msv.sh`.

1. **Polynomial selection** — `msieve -np` on the GPU (Blackwell sm_120). ~18 min.
2. **Lattice sieving** — `gpu_loop`, a from-scratch GPU special-q lattice siever (~17 ms/lattice,
   GPU-bound; cofactoring via 256-bit fixed-width int math + Montgomery Brent-rho). Emits
   relations in CADO/msieve-compatible format. ~2–3 h for ~75M relations.
3. **Downstream** — `msieve -nc` does duplicate removal, filtering, **block-Lanczos linear
   algebra (matrix held in RAM)**, and square root. ~1–1.5 h.

### Fitting the 1 GiB writable-disk sandbox

The validator gives only **1 GiB writable `/tmp`** but 85 GiB RAM. GNFS intermediates are
multi-GB (relations ~9 GB, large-primes ~2 GB, matrix ~1 GB). The broker **`msv_launcher`**
holds these in anonymous RAM (`memfd_create`) and exposes them to msieve as symlinks
`m.dat[.lp|.mat] → /proc/<broker>/fd/N`. msieve reads/writes them via ordinary `fopen`,
so nothing large ever touches disk; only small intermediates (`.hc/.cyc/.dep`) remain.
block-Lanczos keeps the matrix **in RAM**, so this is **robust to any matrix size**.

**Measured peak `/tmp` (real files):**
| case | matrix | peak `/tmp` |
|---|---|---|
| good poly (c140) | 1.92M × 1.92M | **114 MB** |
| weak poly (bigger matrix) | 2.65M × 2.65M | **204 MB** |

Both far under 1 GiB. (For comparison, CADO's Block-Wiedemann writes an 855 MB–1.2 GB
on-disk matrix cache — 957 MB for c140, **1389 MB (over budget) for the weak key**.)

## Validated

- Factored **c140** (460-bit) end-to-end from `gpu_loop` relations via the memfd-msieve
  downstream → exact p, q.
- Factored two **fresh random** 140-digit N end-to-end (per-key generality) → exact p, q,
  `p·q == N`.
- Entrypoint I/O contract + Tier 0 verified (`result.json` with correct p, q).

## Portability (matches the live validators)

Both validators: 1× RTX PRO 6000 Blackwell (sm_120), 26 cores, 96 GB; CPUs AMD EPYC 9555
(Zen5) and Intel Xeon 6980P (Granite Rapids). Binaries are built for this:
- `gpu_loop`, `sort_engine.so`: no AVX-512 / generic — run anywhere.
- `msieve`: `-march=x86-64-v4` (standard AVX-512 baseline supported by **both** vendors —
  no AMD-specific tuning that would fault on Intel).
- GPU kernel `stage1_core_sm86.ptx` is JIT-compiled by the driver to sm_120 at runtime.

## Build / run

```bash
docker build -t breaking-rsa .
docker run --rm --network none --gpus all breaking-rsa <challenge_id> '{"num": <N>, "num_bits": 460}'
```
Requires NVIDIA Blackwell GPU (sm_120), CUDA 12.8 base, recent driver.

## Files

| Path | Role |
|---|---|
| `breaking_rsa.py` | entrypoint — tiered orchestrator + output contract |
| `gpu_nfs_siever/gpu_loop` | from-scratch GPU lattice siever (+ source: `gpu_loop.cu`, `u256.cuh`, `rootfind.c`) |
| `gpu_nfs_siever/factor_msv.sh` | per-key GNFS driver (polyselect → sieve → downstream → verify) |
| `msieve-gpu/msieve` | GPU polyselect + filter + block-Lanczos + sqrt |
| `msieve-gpu/msv_launcher` | memfd broker (holds big files in RAM for the 1 GiB sandbox) |
| `msieve-gpu/msvrun.sh` | runs the memfd-backed msieve downstream |
| `msieve-gpu/cub/sort_engine.so`, `stage1_core_sm86.ptx` | GPU sort engine + polyselect kernel |
| `Dockerfile` | builds the image |
| `enigma_challenges/` | validator output-contract helpers |
