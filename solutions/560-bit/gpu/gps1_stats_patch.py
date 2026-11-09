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

"""gps1_stats_patch.py <gpu-siever dir> -- review 6 Phase A0 (docs/rsa560_codex_review6.md 6.1 / 7.1, branch codex-review6):
a default-off stage-timing diagnostic for gps1, the GPU polynomial collision search (gpu-siever/polyselect/).

Patches polyselect/gps1.h, polyselect/gps1_kernels.cu and polyselect/gps1.cpp of the Cognition tree at tag `latest`
(b5ca9c4f). The three files are pinned by sha256 (fail-closed on any drift); no other production patch touches them.
Idempotent: a tree that already carries the GPS1_STAGE_STATS marker in all three files is left alone.

  GPS1_STAGE_STATS unset / 0   unchanged: no events, no extra syncs, no output (one getenv at start-up).
  GPS1_STAGE_STATS=1           CUDA events on the search stream around the four stages of every gps1_run
                                 inv  (zero the match counter + kernel_invqq)
                                 fill (kernel_fill)
                                 sort (cub::DeviceRadixSort::SortPairs)
                                 scan (kernel_scan)
                               read once after the stream sync gps1_run already does (no extra device sync). Host wall
                               (omp_get_wtime) for the batch set-up + H2D copies, gps1_run, the match D2H + hit build,
                               the wait for the next ad's prepare(), the per-ad root H2D copies and the main thread's
                               wait on the previous flush thread (start_flush join). The flush itself (host candidate
                               build + GPU sopt + CPU redo) runs on its own thread; its completed time is reported as
                               the delta of gps1's own t_match / t_sopt counters.
  GPS1_STAGE_STATS=2           as 1, plus a padding count (entries with key KEY_NONE) by one extra small kernel after
                               the scan event (outside the timed stages, but inside the whole-search wall).

Output (stderr, one line each, prefix GPS1_STATS):
  GPS1_STATS ad=<ad> ...           once per completed leading coefficient
  GPS1_STATS total ...             at exit, all ads
  GPS1_STATS total_excl_first ...  at exit, all ads but the first (first-use CUDA/CUB set-up, cold caches)
Fields: ads batches items entries pad matches sort_bits passes_est sort_bytes_est inv_ms fill_ms sort_ms scan_ms
gpu_ev_ms (event ev0->ev4) run_wall_ms (host wall of gps1_run) setup_ms hits_ms prep_wait_ms roots_h2d_ms flush_join_ms
flush_done_ms sopt_done_ms search_ms (main-loop wall of the ad(s)) f_sort (sort_ms/search_ms) f_sort_gpu
(sort_ms/gpu_ev_ms); the totals add tail_ms (final flush + join) and whole_ms (t0 -> end).
passes_est / sort_bytes_est are an ESTIMATE, not read from CUB: ceil(sort_bits / 8) onesweep digit passes (CUB's
64-bit-key policy uses 8-bit digits on sm_80+) moving 2 x 12 B per entry per pass, plus one 8 B key read for the
histogram pass."""
import hashlib, os, sys

D = sys.argv[1]
P = os.path.join(D, 'polyselect')
FILES = {'gps1.h': '05d0f7c66a41326add1f612f3fc5f9cd9ded3bfffefd76e2cba21d964624b207',
         'gps1_kernels.cu': 'b359f4fa2a43c549059e86b8bb246a0158eaa65f11b3f06fa7ab580c1748c8ba',
         'gps1.cpp': 'a1346a98fbc7d2bf21b207c1637bba7d01b395fce32f46666408b24b5706059f'}
src = {f: open(os.path.join(P, f)).read() for f in FILES}
marked = [f for f in FILES if 'GPS1_STAGE_STATS' in src[f]]
if len(marked) == len(FILES): print('already patched'); sys.exit(0)
assert not marked, 'partially patched tree (%s); start from a clean checkout' % ' '.join(marked)
for f, h in FILES.items():
    assert hashlib.sha256(src[f].encode()).hexdigest() == h, '%s differs from the reviewed upstream text (latest b5ca9c4f)' % f

def sub(f, a, b, cnt=1):
    assert src[f].count(a) == cnt, (f, src[f].count(a), a[:80])
    src[f] = src[f].replace(a, b)

# ------------------------------------------------------------------------------------------------------------ gps1.h
sub('gps1.h', '''/* temp-storage size the radix sort needs for nentries pairs */
extern "C" size_t gps1_sort_tmp_bytes(uint64_t nentries);
''', '''/* temp-storage size the radix sort needs for nentries pairs */
extern "C" size_t gps1_sort_tmp_bytes(uint64_t nentries);

/* GPS1_STAGE_STATS (gpu/gps1_stats_patch.py, review 6 Phase A0): per-stage
 * event times of gps1_run, accumulated since the last gps1_stats_take.
 * Level 0 (env unset) records nothing. */
struct gps1_stats_t {
    double inv_ms, fill_ms, sort_ms, scan_ms, ev_ms;
    uint64_t runs, entries, pad, matches, sort_bits, passes_est, bytes_est;
};
extern "C" int gps1_stats_level(void);
extern "C" void gps1_stats_take(gps1_stats_t *out);
''')

# --------------------------------------------------------------------------------------------------- gps1_kernels.cu
sub('gps1_kernels.cu', '''__device__ uint64_t d_match_count;
''', '''__device__ uint64_t d_match_count;

/* ---- GPS1_STAGE_STATS begin ---- */
__device__ unsigned long long d_pad_count;

static gps1_stats_t gps1_stats_acc;

extern "C" int gps1_stats_level(void)
{
    static int level = -1;
    if (level < 0) {
        const char *e = getenv("GPS1_STAGE_STATS");
        level = e ? atoi(e) : 0;
        if (level < 0)
            level = 0;
    }
    return level;
}

extern "C" void gps1_stats_take(gps1_stats_t *out)
{
    *out = gps1_stats_acc;
    gps1_stats_acc = gps1_stats_t();
}

/* GPS1_STAGE_STATS=2: entries still KEY_NONE after the sort (padding) */
__global__ void kernel_stats_pad(const uint64_t *keys, uint64_t n)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && keys[i] == UINT64_MAX)
        atomicAdd(&d_pad_count, 1ull);
}
/* ---- GPS1_STAGE_STATS end ---- */
''')
sub('gps1_kernels.cu', '''    uint64_t zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_match_count, &zero, 8, 0,
                                       cudaMemcpyHostToDevice, st));
''', '''    /* GPS1_STAGE_STATS: events only when on; read after the existing sync */
    const int stats = gps1_stats_level();
    static cudaEvent_t ev[5];
    if (stats && !ev[0])
        for (int k = 0; k < 5; k++)
            CUDA_CHECK(cudaEventCreate(&ev[k]));
    if (stats)
        CUDA_CHECK(cudaEventRecord(ev[0], st));

    uint64_t zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_match_count, &zero, 8, 0,
                                       cudaMemcpyHostToDevice, st));
''')
sub('gps1_kernels.cu', '''                                                         G->qs, ndq, G->invqq);
    uint64_t total = (uint64_t) nroots * batch;
''', '''                                                         G->qs, ndq, G->invqq);
    if (stats)
        CUDA_CHECK(cudaEventRecord(ev[1], st));
    uint64_t total = (uint64_t) nroots * batch;
''')
sub('gps1_kernels.cu', '''                                              slots, G->keys, G->vals);

    uint64_t nentries = slots * batch;
''', '''                                              slots, G->keys, G->vals);
    if (stats)
        CUDA_CHECK(cudaEventRecord(ev[2], st));

    uint64_t nentries = slots * batch;
''')
sub('gps1_kernels.cu', '''                                               sort_bits, st));
    kernel_scan<<<''', '''                                               sort_bits, st));
    if (stats)
        CUDA_CHECK(cudaEventRecord(ev[3], st));
    kernel_scan<<<''')
sub('gps1_kernels.cu', '''                                                 G->matches);
    uint64_t nmatches;
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&nmatches, d_match_count, 8, 0,
                                         cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    assert(nmatches < GPS1_MAX_MATCHES);
''', '''                                                 G->matches);
    unsigned long long npad = 0;
    if (stats) {
        CUDA_CHECK(cudaEventRecord(ev[4], st));
        if (stats >= 2) {
            unsigned long long z = 0;
            CUDA_CHECK(cudaMemcpyToSymbolAsync(d_pad_count, &z, 8, 0,
                                               cudaMemcpyHostToDevice, st));
            kernel_stats_pad<<<(nentries + 255) / 256, 256, 0, st>>>(
                dkeys.Current(), nentries);
            CUDA_CHECK(cudaMemcpyFromSymbolAsync(&npad, d_pad_count, 8, 0,
                                                 cudaMemcpyDeviceToHost, st));
        }
    }
    uint64_t nmatches;
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&nmatches, d_match_count, 8, 0,
                                         cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    assert(nmatches < GPS1_MAX_MATCHES);
    if (stats) {
        float ms[5];
        for (int k = 0; k < 4; k++)
            CUDA_CHECK(cudaEventElapsedTime(&ms[k], ev[k], ev[k + 1]));
        CUDA_CHECK(cudaEventElapsedTime(&ms[4], ev[0], ev[4]));
        gps1_stats_t &S = gps1_stats_acc;
        S.inv_ms += ms[0]; S.fill_ms += ms[1]; S.sort_ms += ms[2];
        S.scan_ms += ms[3]; S.ev_ms += ms[4];
        uint64_t passes = ((uint64_t) sort_bits + 7) / 8;
        S.runs++; S.entries += nentries; S.matches += nmatches;
        S.pad += stats >= 2 ? npad : 0;
        S.sort_bits = sort_bits > (int) S.sort_bits ? sort_bits : S.sort_bits;
        S.passes_est += passes;
        S.bytes_est += nentries * (passes * 24 + 8);
    }
''')

# ---------------------------------------------------------------------------------------------------------- gps1.cpp
sub('gps1.cpp', '''/* the prep and flush threads update these while the main thread reads them */
static std::atomic<double> t_roots(0), t_crt(0), t_match(0), t_sopt(0);
''', '''/* the prep and flush threads update these while the main thread reads them */
static std::atomic<double> t_roots(0), t_crt(0), t_match(0), t_sopt(0);

/* ---- GPS1_STAGE_STATS begin (main thread only) ---- */
struct gps1_hstats_t {
    gps1_stats_t k = gps1_stats_t();
    uint64_t ads = 0, batches = 0, items = 0;
    double run_wall = 0, setup = 0, hits = 0, prep_wait = 0, roots_h2d = 0,
           flush_join = 0, flush_done = 0, sopt_done = 0, search = 0;
};
static gps1_hstats_t gps1_hs;   /* the current ad (host side) */

static void gps1_hstats_add(gps1_hstats_t &a, const gps1_hstats_t &b)
{
    a.k.inv_ms += b.k.inv_ms; a.k.fill_ms += b.k.fill_ms;
    a.k.sort_ms += b.k.sort_ms; a.k.scan_ms += b.k.scan_ms;
    a.k.ev_ms += b.k.ev_ms; a.k.runs += b.k.runs;
    a.k.entries += b.k.entries; a.k.pad += b.k.pad;
    a.k.matches += b.k.matches; a.k.passes_est += b.k.passes_est;
    a.k.bytes_est += b.k.bytes_est;
    a.k.sort_bits = std::max(a.k.sort_bits, b.k.sort_bits);
    a.ads += b.ads; a.batches += b.batches; a.items += b.items;
    a.run_wall += b.run_wall; a.setup += b.setup; a.hits += b.hits;
    a.prep_wait += b.prep_wait; a.roots_h2d += b.roots_h2d;
    a.flush_join += b.flush_join; a.flush_done += b.flush_done;
    a.sopt_done += b.sopt_done; a.search += b.search;
}

/* one GPS1_STATS line on stderr; host times are seconds, printed as ms */
static void gps1_hstats_print(const char *head, const gps1_hstats_t &a,
                              double tail, double whole)
{
    const double ms = 1e3;
    fprintf(stderr, "GPS1_STATS %s ads=%lu batches=%lu items=%lu entries=%lu pad=%ld "
            "matches=%lu sort_bits=%lu passes_est=%lu sort_bytes_est=%lu "
            "inv_ms=%.3f fill_ms=%.3f sort_ms=%.3f scan_ms=%.3f gpu_ev_ms=%.3f "
            "run_wall_ms=%.3f setup_ms=%.3f hits_ms=%.3f prep_wait_ms=%.3f "
            "roots_h2d_ms=%.3f flush_join_ms=%.3f flush_done_ms=%.3f "
            "sopt_done_ms=%.3f search_ms=%.3f f_sort=%.4f f_sort_gpu=%.4f",
            head, (unsigned long) a.ads, (unsigned long) a.batches,
            (unsigned long) a.items, (unsigned long) a.k.entries,
            gps1_stats_level() >= 2 ? (long) a.k.pad : -1L,
            (unsigned long) a.k.matches, (unsigned long) a.k.sort_bits,
            (unsigned long) a.k.passes_est, (unsigned long) a.k.bytes_est,
            a.k.inv_ms, a.k.fill_ms, a.k.sort_ms, a.k.scan_ms, a.k.ev_ms,
            a.run_wall * ms, a.setup * ms, a.hits * ms, a.prep_wait * ms,
            a.roots_h2d * ms, a.flush_join * ms, a.flush_done * ms,
            a.sopt_done * ms, a.search * ms,
            a.search > 0 ? a.k.sort_ms / (a.search * ms) : 0.0,
            a.k.ev_ms > 0 ? a.k.sort_ms / a.k.ev_ms : 0.0);
    if (tail >= 0)
        fprintf(stderr, " tail_ms=%.3f whole_ms=%.3f", tail * ms, whole * ms);
    fputc('\\n', stderr);
    fflush(stderr);
}
/* ---- GPS1_STAGE_STATS end ---- */
''')
sub('gps1.cpp', '''    if (flush_thread.joinable())
        flush_thread.join();
    flush_thread = std::thread(''', '''    double tj = omp_get_wtime();
    if (flush_thread.joinable())
        flush_thread.join();
    if (gps1_stats_level())     /* GPS1_STAGE_STATS */
        gps1_hs.flush_join += omp_get_wtime() - tj;
    flush_thread = std::thread(''')
sub('gps1.cpp', '''    if (nroots == 0)
        return;
    std::vector<uint64_t> qs, rq_hi(batch), rq_lo(batch);
''', '''    if (nroots == 0)
        return;
    const int stats = gps1_stats_level();   /* GPS1_STAGE_STATS */
    double ts0 = stats ? omp_get_wtime() : 0;
    std::vector<uint64_t> qs, rq_hi(batch), rq_lo(batch);
''')
sub('gps1.cpp', '''    double tg = omp_get_wtime();
    uint64_t nmatches = gps1_run(G, nroots, qs.size(), batch, M, slots);
    t_gpu += omp_get_wtime() - tg;
''', '''    double tg = omp_get_wtime();
    uint64_t nmatches = gps1_run(G, nroots, qs.size(), batch, M, slots);
    t_gpu += omp_get_wtime() - tg;
    double tg1 = stats ? omp_get_wtime() : 0;
    if (stats) {
        gps1_hs.setup += tg - ts0;
        gps1_hs.run_wall += tg1 - tg;
        gps1_hs.batches++;
    }
''')
sub('gps1.cpp', '''        hits.push_back(std::move(ht));
    }
}
''', '''        hits.push_back(std::move(ht));
    }
    if (stats)
        gps1_hs.hits += omp_get_wtime() - tg1;
}
''')
sub('gps1.cpp', '''    std::vector<hit_t> hits;
    double t0 = omp_get_wtime();
''', '''    std::vector<hit_t> hits;
    const int stats = gps1_stats_level();   /* GPS1_STAGE_STATS */
    gps1_hstats_t st_all, st_rest;
    double st_match0 = 0, st_sopt0 = 0;
    double t0 = omp_get_wtime();
''')
sub('gps1.cpp', '''    for (; mpz_cmp(ad, pa.admax) < 0; mpz_add_ui(ad, ad, pa.incr)) {
        prep_t P = next.get();
''', '''    for (; mpz_cmp(ad, pa.admax) < 0; mpz_add_ui(ad, ad, pa.incr)) {
        double ta0 = stats ? omp_get_wtime() : 0;
        prep_t P = next.get();
        double ta1 = stats ? omp_get_wtime() : 0;
        if (stats)
            gps1_hs.prep_wait += ta1 - ta0;
''')
sub('gps1.cpp', '''        CUDA_CHECK(cudaMemcpy(G.entry_off, P.entry_off.data(), P.entry_off.size() * 4, cudaMemcpyHostToDevice));
        uint64_t slots = P.entry_off.back();
''', '''        CUDA_CHECK(cudaMemcpy(G.entry_off, P.entry_off.data(), P.entry_off.size() * 4, cudaMemcpyHostToDevice));
        if (stats)
            gps1_hs.roots_h2d += omp_get_wtime() - ta1;   /* incl. the async launch of the next prepare */
        uint64_t slots = P.entry_off.back();
''')
sub('gps1.cpp', '''                   t_roots.load(), t_gpu, t_crt.load(), t_match.load());
        fflush(stdout);
    }
''', '''                   t_roots.load(), t_gpu, t_crt.load(), t_match.load());
        fflush(stdout);
        if (stats) {    /* GPS1_STAGE_STATS: one line per completed ad */
            gps1_hs.search = omp_get_wtime() - ta0;
            gps1_hs.ads = 1;
            gps1_hs.items = P.items.q.size();
            gps1_hs.flush_done = t_match.load() - st_match0;
            gps1_hs.sopt_done = t_sopt.load() - st_sopt0;
            st_match0 += gps1_hs.flush_done;
            st_sopt0 += gps1_hs.sopt_done;
            gps1_stats_take(&gps1_hs.k);
            char head[96];
            gmp_snprintf(head, sizeof head, "ad=%Zd", (mpz_srcptr) ad);
            gps1_hstats_print(head, gps1_hs, -1, -1);
            gps1_hstats_add(st_all, gps1_hs);
            if (st_all.ads > 1)
                gps1_hstats_add(st_rest, gps1_hs);
            gps1_hs = gps1_hstats_t();
        }
    }
    double t_tail0 = omp_get_wtime();
''')
sub('gps1.cpp', '''    flush_thread.join();
    printf("# gps1: total %lu matches''', '''    flush_thread.join();
    if (stats) {        /* GPS1_STAGE_STATS: totals */
        double t_end = omp_get_wtime();
        st_all.flush_join += gps1_hs.flush_join;    /* final start_flush */
        st_all.flush_done = t_match.load();
        st_all.sopt_done = t_sopt.load();
        gps1_hstats_print("total", st_all, t_end - t_tail0, t_end - t0);
        gps1_hstats_print("total_excl_first", st_rest, -1, -1);
    }
    printf("# gps1: total %lu matches''')

for f in FILES:
    open(os.path.join(P, f), 'w').write(src[f])
print('gps1_stats_patch: patched', ' '.join(FILES))
