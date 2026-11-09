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

"""glas_session_patch.py <gpu-siever/src/glas.cpp> <gpu-siever/src/sieve_core_gpu.cu> -- proposal 1 of
docs/rsa560_codex_review.md (codex-review): keep GLAS alive between q bands; separate publication from process lifetime.

Applies ON TOP of the production patch stack (sandbox_bootstrap.sh: adjust on glas.cpp; coprime, prefetch, xfuse,
region on sieve_core_gpu.cu). Every anchor is asserted to occur exactly once. Without -session the binary behaves as
before: same argument handling, same output stream, same per-band code (the band body runs once and breaks out of the
new loop; emit writes the same text to the same FILE*).

  glas-gpu <usual las args, no -q0/-q1/-out> -session CTL -session-ack ACK
    Initialises ONCE (factor bases, CUDA context, device/pinned slot buffers, tiny tables, cofactor strategies), writes
    'READY <pid>' to ACK, then reads commands line by line from CTL (FIFO or regular file; blank and '#' lines skipped):
      BAND <id> <q0> <q1> <outpath>   run the unchanged per-band body (special-q list, sieve/worker threads, ordered
                                      emitter, exact pass-3 checks); join every thread (all CPU and device cofactor work
                                      is done when the joins return); write the band's relation lines -- and ONLY those,
                                      in the one-shot order -- to <outpath>.tmp; fflush + fclose (+ fsync when
                                      GLAS_SESSION_FSYNC=1); check lines written == reports; rename to <outpath>; then
                                      write 'ACK <id> <nrels> <q1>' to ACK with one write(2).
      QUIT                            exit 0.
    EOF on CTL without QUIT (the driver died): exit 5. Any band error: _exit(1) via usage_error, leaving only a .tmp
    (never an un-ACKed published file). Per-q '#' comment lines are dropped from the band file (they are ~1/3 of the
    one-shot bytes and the driver strips them anyway); per-band summary lines go to stdout, a
    '# GLAS_SESSION band ...' line to stderr. ACK is opened O_APPEND (regular file or FIFO).
  Relation lines of a band are byte-identical to `grep -v '^#'` of a one-shot run on [q0,q1): same sq_list, same
  idx-ordered emitter, same pass-3 text.
  gpu_cofactor's per-THREAD device/pinned scratch (static thread_local) would be re-allocated -- and leaked -- by every
  band's fresh worker threads; with GLAS_COFAC_POOL=1 (set by -session) the same scratch sets come from a mutex-guarded
  free list instead (scratch only: results unchanged). Unset: the thread_local set exactly as before."""
import sys
if len(sys.argv) != 3:
    sys.exit('usage: glas_session_patch.py <glas.cpp> <sieve_core_gpu.cu>')
pg, pc = sys.argv[1], sys.argv[2]
g = open(pg).read(); c = open(pc).read()
if 'GLAS_SESSION' in g or 'GLAS_COFAC_POOL' in c:
    assert 'GLAS_SESSION' in g and 'GLAS_COFAC_POOL' in c, 'half-patched tree'
    print('already patched'); sys.exit(0)
assert 'glas_adjust_strategy' in g, 'needs glas_adjust_patch.py first (production stack)'
assert 'fill_xfuse_kernel' in c, 'needs the production sieve_core_gpu.cu stack (coprime, prefetch, xfuse, region)'


def sub(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:100])
    return s.replace(a, b)


# ---------------------------------------------------------------- glas.cpp
# 1. params
g = sub(g, '''    bool allow_largesq = false;
    side_params side[2];
};''', '''    bool allow_largesq = false;
    side_params side[2];
    std::string session_path, session_ack;   /* GLAS_SESSION: -session CTL -session-ack ACK */
};''')

# 2. arguments: -q0/-q1 optional in session mode (bands carry them)
g = sub(g, '''    P.poly_path = need("poly");
    P.q0 = strtoull(need("q0").c_str(), nullptr, 10);
    P.q1 = strtoull(need("q1").c_str(), nullptr, 10);''', '''    P.poly_path = need("poly");
    if (auto v = take("session"))   /* GLAS_SESSION */
        P.session_path = *v;
    if (auto v = take("session-ack"))
        P.session_ack = *v;
    if (P.session_path.empty() != P.session_ack.empty())
        usage_error("-session and -session-ack go together");
    if (P.session_path.empty()) {
        P.q0 = strtoull(need("q0").c_str(), nullptr, 10);
        P.q1 = strtoull(need("q1").c_str(), nullptr, 10);
    } else {
        if (auto v = take("q0"))
            P.q0 = strtoull(v->c_str(), nullptr, 10);
        if (auto v = take("q1"))
            P.q1 = strtoull(v->c_str(), nullptr, 10);
    }''')

# 3. helpers before main: command reader, relation-only writer
g = sub(g, '''}  // namespace

int main(int argc, char const ** argv)
{''', '''/* GLAS_SESSION: next command from the control stream.  1 = BAND (fields
 * set), 0 = QUIT; EOF without QUIT exits 5 (the driver is gone). */
int glas_session_next(FILE * ctl, std::string & id, uint64_t & q0,
                      uint64_t & q1, std::string & path)
{
    char * line = nullptr;
    size_t cap = 0;
    for (;;) {
        ssize_t const n = getline(&line, &cap, ctl);
        if (n < 0) {
            free(line);
            fprintf(stderr, "# GLAS_SESSION: control stream EOF without QUIT\\n");
            fflush(nullptr);
            _exit(5);
        }
        std::string l(line, (size_t)n);
        while (!l.empty() && (l.back() == '\\n' || l.back() == '\\r' || l.back() == ' '))
            l.pop_back();
        if (l.empty() || l[0] == '#')
            continue;
        if (l == "QUIT") {
            free(line);
            return 0;
        }
        std::vector<char> a(l.size() + 1), p(l.size() + 1);
        unsigned long long x0, x1;
        int end = 0;
        if (sscanf(l.c_str(), "BAND %s %llu %llu %s%n", a.data(), &x0, &x1,
                   p.data(), &end) != 4 || (size_t)end != l.size() ||
            x1 <= x0)
            usage_error("GLAS_SESSION: bad command '" + l + "'");
        id = a.data();
        q0 = x0;
        q1 = x1;
        path = p.data();
        free(line);
        return 1;
    }
}

/* GLAS_SESSION: write only the relation lines of an emitted text block
 * (per-q '#' headers and discards dropped), counting them. */
void glas_put_rels(std::string const & s, FILE * f, unsigned long & nlines)
{
    size_t pos = 0;
    while (pos < s.size()) {
        size_t e = s.find('\\n', pos);
        e = e == std::string::npos ? s.size() : e + 1;
        if (s[pos] != '#') {
            fwrite(s.data() + pos, 1, e - pos, f);
            nlines++;
        }
        pos = e;
    }
}

}  // namespace

int main(int argc, char const ** argv)
{''')

# 4. band loop head (before the special-q list); session init
g = sub(g, '''    /* special-q list */
    std::vector<std::pair<uint64_t, uint64_t>> sq_list;''', '''    /* GLAS_SESSION: everything above is process-lifetime state; the body
     * below is one band.  Without -session it runs once and breaks. */
    bool const glas_ses = !P.session_path.empty();
    FILE * const glas_main_out = out;
    double const glas_t_main = t_start;
    FILE * glas_ctl = nullptr;
    int glas_ack = -1;
    unsigned long glas_nbands = 0;
    if (glas_ses) {
        if (!P.out_path.empty())
            usage_error("-session and -out are exclusive (BAND names the output)");
        setenv("GLAS_COFAC_POOL", "1", 1);   /* reuse gpu_cofactor scratch across bands' threads */
        glas_ctl = fopen(P.session_path.c_str(), "r");
        if (!glas_ctl)
            usage_error("cannot open session control " + P.session_path);
        glas_ack = open(P.session_ack.c_str(),
                        O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0666);
        if (glas_ack < 0)
            usage_error("cannot open session ack " + P.session_ack);
        char rb[64];
        int const rl = snprintf(rb, sizeof rb, "READY %d\\n", (int)getpid());
        if (write(glas_ack, rb, rl) != rl)
            usage_error("session ack write failed");
        fprintf(stderr, "# GLAS_SESSION on: control %s, ack %s, init %.2f s\\n",
                P.session_path.c_str(), P.session_ack.c_str(),
                wallclock() - t_start);
        fflush(nullptr);
    }
    for (;;) {
    std::string glas_id, glas_outpath, glas_tmp;
    FILE * glas_rel_out = glas_main_out;   /* relation sink */
    FILE * out = glas_main_out;            /* comment sink */
    unsigned long glas_lines = 0;
    double const t_start = glas_ses ? wallclock() : glas_t_main;
    if (glas_ses) {
        uint64_t bq0 = 0, bq1 = 0;
        if (!glas_session_next(glas_ctl, glas_id, bq0, bq1, glas_outpath))
            break;
        if (!P.allow_largesq && bq1 > (UINT64_C(1) << P.side[P.sqside].lpb))
            usage_error("special-q range [q0,q1) exceeds 2^lpb" +
                        std::to_string(P.sqside) + " (GLAS_SESSION band " + glas_id + ")");
        if (is_gz(glas_outpath))
            usage_error("GLAS_SESSION: gzip band outputs are not supported");
        S.P.q0 = bq0;
        S.P.q1 = bq1;
        glas_tmp = glas_outpath + ".tmp";
        int const fd = open(glas_tmp.c_str(),
                            O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
        if (fd < 0 || !(glas_rel_out = fdopen(fd, "w")))
            usage_error("cannot open band output " + glas_tmp);
        out = stdout;
        fprintf(out, "# glas with q0=%" PRIu64 " q1=%" PRIu64 " (GLAS_SESSION band %s)\\n",
                P.q0, P.q1, glas_id.c_str());
    }

    /* special-q list */
    std::vector<std::pair<uint64_t, uint64_t>> sq_list;''')

# 5. ordered emitter: same text to the same FILE* by default; relation lines only in a session
g = sub(g, '''        while (!pending.empty() && pending.begin()->first == next_print) {
            fputs(pending.begin()->second.c_str(), out);''', '''        while (!pending.empty() && pending.begin()->first == next_print) {
            if (glas_ses)
                glas_put_rels(pending.begin()->second, glas_rel_out, glas_lines);
            else
                fputs(pending.begin()->second.c_str(), glas_rel_out);''')

# 6. band end: publish atomically, ACK, next command
g = sub(g, '''                proc_n[PN_FACUL].load());
    if (gzip.f)
        return gzip.close() != 0;''', '''                proc_n[PN_FACUL].load());
    if (!glas_ses)
        break;
    /* every sieve and worker thread is joined: the band is complete */
    if (fflush(glas_rel_out) != 0 || ferror(glas_rel_out))
        usage_error("GLAS_SESSION: write failed on " + glas_tmp);
    if (env_flag("GLAS_SESSION_FSYNC", false) && fsync(fileno(glas_rel_out)) != 0)
        usage_error("GLAS_SESSION: fsync failed on " + glas_tmp);
    if (fclose(glas_rel_out) != 0)
        usage_error("GLAS_SESSION: close failed on " + glas_tmp);
    if (glas_lines != reports)
        usage_error("GLAS_SESSION: band " + glas_id + " wrote " +
                    std::to_string(glas_lines) + " lines for " +
                    std::to_string(reports) + " reports");
    if (rename(glas_tmp.c_str(), glas_outpath.c_str()) != 0)
        usage_error("GLAS_SESSION: rename to " + glas_outpath + " failed");
    {
        char ab[512];
        int const al = snprintf(ab, sizeof ab, "ACK %s %lu %" PRIu64 "\\n",
                                glas_id.c_str(), glas_lines, P.q1);
        if (al <= 0 || al >= (int)sizeof ab || write(glas_ack, ab, al) != al)
            usage_error("GLAS_SESSION: ack write failed");
    }
    glas_nbands++;
    fprintf(stderr, "# GLAS_SESSION band %s q[%" PRIu64 ",%" PRIu64 ") %lu rels, %zu special-q, %.2f s\\n",
            glas_id.c_str(), P.q0, P.q1, glas_lines, sq_list.size(), elapsed);
    fflush(nullptr);
    }   /* GLAS_SESSION band loop */
    if (glas_ses) {
        fprintf(stderr, "# GLAS_SESSION quit after %lu bands, %.2f s\\n",
                glas_nbands, wallclock() - glas_t_main);
        close(glas_ack);
        fclose(glas_ctl);
    }
    if (gzip.f)
        return gzip.close() != 0;''')

# ---------------------------------------------------------------- sieve_core_gpu.cu
c = sub(c, '''    static thread_local cudaStream_t st;
    static thread_local cofac_pair * d_in;
    static thread_local cofac_result * d_out;
    static thread_local ecm_side_result * d_res;
    static thread_local ecm_lists lists[2];   /* this round, next round */
    static thread_local u128 * d_g;
    static thread_local cofac_pair * h_in;
    static thread_local cofac_result * h_out;
    static thread_local ecm_side_result * h_res;
    static thread_local uint32_t * h_cnt;
    static thread_local cudaEvent_t ev[2];
''', '''    /* GLAS_COFAC_POOL=1 (set by glas -session): the scratch set comes
     * from a free list instead of thread_local storage, so worker threads
     * that end with each band neither re-allocate nor leak it. */
    struct cofac_bufs {
        cudaStream_t st;
        cofac_pair * d_in;
        cofac_result * d_out;
        ecm_side_result * d_res;
        ecm_lists lists[2];   /* this round, next round */
        u128 * d_g;
        cofac_pair * h_in;
        cofac_result * h_out;
        ecm_side_result * h_res;
        uint32_t * h_cnt;
        cudaEvent_t ev[2];
    };
    static thread_local cofac_bufs tl_bufs{};
    static bool const pooled = [] {
        char const * v = getenv("GLAS_COFAC_POOL");
        return v && atoi(v) != 0;
    }();
    static std::mutex pool_m;
    static std::vector<cofac_bufs *> pool_free;
    cofac_bufs * bp = &tl_bufs;
    if (pooled) {
        std::lock_guard<std::mutex> const lk(pool_m);
        if (pool_free.empty())
            bp = new cofac_bufs{};
        else {
            bp = pool_free.back();
            pool_free.pop_back();
        }
    }
    struct pool_return {
        cofac_bufs * b;
        bool on;
        ~pool_return()
        {
            if (on) {
                std::lock_guard<std::mutex> const lk(pool_m);
                pool_free.push_back(b);
            }
        }
    } const pool_guard{bp, pooled};
    cudaStream_t & st = bp->st;
    cofac_pair *& d_in = bp->d_in;
    cofac_result *& d_out = bp->d_out;
    ecm_side_result *& d_res = bp->d_res;
    ecm_lists (&lists)[2] = bp->lists;
    u128 *& d_g = bp->d_g;
    cofac_pair *& h_in = bp->h_in;
    cofac_result *& h_out = bp->h_out;
    ecm_side_result *& h_res = bp->h_res;
    uint32_t *& h_cnt = bp->h_cnt;
    cudaEvent_t (&ev)[2] = bp->ev;
''')

open(pg, 'w').write(g); open(pc, 'w').write(c); print('patched')
