# pipeline_c169 -- the RSA-560 (c169) pipeline, forked from the RSA-530 solution

Started 2026-10-15 (docs/rsa560_plan.md W4). Every file here is a copy of the corresponding file in
`solutions/badf0faf-8c59-491a-a92a-3fe1cce623f7.zip` (reuse approved), except where this README says it
was changed. The 530 pipeline is dissected in `docs/rsa530_pipeline_anatomy.md`.

## Layout (the 530 convention: scripts next to the binaries they call)

| Here | 530 origin | Role |
|---|---|---|
| `factor_c169.sh` | `gpu_nfs_siever/factor_c160.sh` | orchestrator: polyselect, chunked sieve, incremental dedup, filter probes, downstream |
| `polysel160.py`, `polytest160.sh`, `las_ctl.py`, `memhold.py` | `gpu_nfs_siever/` | unchanged |
| `cado-filter/cado_filter160.sh`, `cado_incdup.sh`, `cado2msieve2.c` | `cado-filter/` | unchanged (binaries: CADO `freerel dup1 dup2 purge merge replay` + `cado2msieve2` expected in `CF_DIR`) |
| `msieve-gpu/ds_cado.sh`, `gpu_la.py`, `segspmv.py`, `cado_mat.py`, `la128host.c` | `msieve-gpu/` | downstream; `ds_cado.sh` gained `LA=bwc` |
| `vendor/` | `vendor/` | CADO patches, cofac_cpu source, cuda-sieve fast-emit patch, provenance |
| `breaking_rsa_530.py` | `breaking_rsa.py` | reference entrypoint (the c169 dispatch is still to do) |

Directories the scripts resolve: `CF_DIR` (filter scripts + CADO binaries), `MSV_DIR` (downstream scripts +
`characters`, `sqrt`, `replay`), `CS_DIR` (cuda-sieve `bench`, `fbgen_gpu`, `cofac_cpu`), `GLAS` (glas-gpu
binary), `BWC_DIR` (CADO `bwc.pl` + bwc programs). Locally: `third_party/cado-bin`, `third_party/msv-bin`
(symlink dirs, gitignored); in `docker/Dockerfile.c169-cpu`: `/app/cado-filter`, `/app/msieve-gpu`, `/app/bwc`.

## Changes vs the 530 script (ledger: docs/rsa560_ledger.md, W4 rows)

1. **Per-side large-prime bounds.** `LPB` became `LPB0`/`LPB1` everywhere (job and fb writers, freerel,
   filter env, las co-sieve, cofac_cpu, relation gates, downstream). Defaults are the plan's production set
   measured production set (2026-10-17, ledger X2 stack part 2): lpb 32/33, mfb 63/65, lambda 2.1, ncurves 17/17,
   lim 100M/150M, `GLAS_ADJUST=3` (the glas adjust-strategy port; `c173cog` 30/31 3LP overflows glas).
2. **Siever abstraction** `SIEVER=glas|bench|las` with one `siever_launch q0 q1 out log` function.
   `bench` is the 530 cuda-sieve path unchanged (fbgen_gpu, --candidates + cofac_cpu at 2LP, GPU --cofactor
   otherwise, bkthresh/allowance flags, checkpoint resume). `las`/`glas` take the las command line
   (`-fb1` from `makefb`, `-A SIEVE_A`, `-t SIEVE_T`, glas adds `-gpu`), write CADO-format relations, have
   their `#` statistics lines stripped before landing, are relaunched once on a failed chunk, and are killed
   at MATRIX (the running chunk is discarded). The chunk-1 relation gate uses `las_ctl.py gate` (exact
   norm check) for them. The CPU co-sieve defaults to off for las/glas (those cores go to facul).
3. **Preflight by mode**: cuda-sieve binaries only for `bench`; `glas-gpu` for `glas`; `las`+`makefb` for
   `las`; msieve-path binaries only for `DS_MODE=msieve`; the GPU check is skipped for `SIEVER=las LA=bwc`.
4. **`FREE_RAW=1`**: truncate a raw chunk's memfd when incdup commits it (needs `DS_MODE=cado`, `INCDUP=1`);
   an incdup failure is then fatal (no one-shot fallback). Default 0 until measured.
5. **`LA=bwc`** in `ds_cado.sh`: CADO `bwc.pl :complete` on the replay matrix, kernel `W` -> characters.
   The CPU-only validation path; production stays `LA=gpu` (gpu_la.py).
6. `COFAC_CPU` defaults to 1 only for `bench` at 2LP (`cofac_cpu` is a 2LP, <= 64-bit splitter).

7. **Predictive probing** (`PREDICT_PROBE=1`): after two SHORT probes the next probe is placed at the
   linear extrapolation of purge's singleton excess to `PRED_TARGET_EXCESS`, times `PRED_FACTOR` (1.02),
   clamped to [PROBE_STEP/2, 4 PROBE_STEP] past the last probe. `FIRST_PROBE` 135M (~87% of the expected
   155M), `PROBE_STEP` 5M, `CHUNK_RELS` 4M, late chunks 1.5M, `MAXCHUNKS` 100.
8. `MAXDEP` 10 in `ds_cado.sh` (p(miss) = 2^-10; saves ~1 GB of /tmp).
9. `polytest_las.sh`: the las/glas test-sieve ranking of the ropt top-K (same pick model as polytest160.sh);
   wired into `factor_c169.sh` for `SIEVER=las|glas` (`POLY_TS`, `TS_K`, `TS_MAX_SECS`).
10. `breaking_rsa.py` (from `breaking_rsa_530.py`): digits >= 166 dispatch to `pipeline_c169/factor_c169.sh`;
    `FULL_RUN_SECS` 13,000.

11. `CHUNK_W0` (default 200000, the c160/c169 first-chunk q-width). The 530 script hard-coded it; on the
    c115 E0.7 take 1 (2 las threads) that was a 6.4M-relation, 5.5 h first chunk (aborted after 89 min).
12. **Late-run fallback** `LATE_SECS` (9000): a run still without a matrix at that wall drops `PRED_FACTOR`
    to 1.0 and halves `PROBE_STEP`. The plan's "`required_excess` / `KEEP` fallback" was dropped as a
    non-lever: purge builds a matrix exactly when the singleton excess is positive, `-keep 160` is noise
    against ~155M relations, and CADO's `required_excess` default is already 0.

13. `cado_filter160.sh` purge input-completeness check sums the "# Done: Read N" lines per "Pass k" block:
    this tree's purge reads input files concurrently and prints one line per FILE (the 530 purge printed
    one per pass; E0.7 take 3 failed every probe on it).

Decided not to do: moving purged.gz/history/dependencies into memfds. At the production set (~155M
relations) `/tmp` holds renumber ~1.5 GB + purged.gz ~2 GB + history ~1 GB + 10 deps x ~150 MB + a 0.6 GB
staging chunk, about 7 GB of the 10 GiB. Revisit only if the relation count grows past ~250M.

Status (2026-10-16): the Linux end-to-end run PASSED on the c100 key in the CPU image (ledger E0.7 take 5:
polyselect -> polytest_las -> 12 las chunks with incremental dedup -> predictive probing -> MATRIX -> replay ->
bwc -> characters -> sqrt, p and q verified). Still to do (plan W4): the kill -9 / resume drill and the
memory-headroom guard drill (`las_ctl.py mem` under a cgroup cap).
