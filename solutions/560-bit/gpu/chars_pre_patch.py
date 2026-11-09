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

"""chars_pre_patch.py <cado-nfs/linalg/characters.cpp> -- split `characters` into a kernel-independent PRECOMPUTE
(beside the LA) and a fast COMBINE (after the LA), with the stock code doing both halves (ds_cado.sh CHARS_PRE=1).

`characters` = (1) build the small character matrix scmat (nrows(index) x nchars bits: every relation-set's XOR of
its relations' quadratic characters; depends only on poly, purged, index, lpb, nchar -- NOT on the kernel), then
(2) read the kernel, column-reduce it, multiply by scmat (+ heavyblock), Gaussian elimination of the product,
write the dependencies. (1) is ~50-65 s of reading/parsing 7 GB of relations at c169; (2) is a few seconds.
This patch adds two environment switches around (1), leaving (2) byte-for-byte the stock code:
  CADO_CHARS_SCMAT_OUT=<f>  build scmat as usual, write it to <f> (<f>.tmp + rename), exit 0. -ker is not read
                            (pass any -out, e.g. /dev/null).
  CADO_CHARS_SCMAT_IN=<f>   do not read purged/index at all: load scmat from <f>, then (2) unchanged.
File: "RSACHR01", u64 nrows, u64 ncols, u64 fingerprint (FNV-1a over the character list (p, r) and nchars), u64
index-file size, then nrows x ceil(ncols/64) little-endian u64 words (scmat[i][cg]). The load checks the
fingerprint against the character list it recomputes from -poly/-lpb/-nchar and the -index file's size, and exits
1 on any mismatch (ds_cado.sh then falls back to the full `characters`). Since the same create_characters() /
big_character_matrix() / small_character_matrix() code builds scmat in both paths and the code after it is
untouched, the output kernel is identical by construction; gpu/s70_chars.sh checks it (cmp) and times both.
Applies to the stock Cognition tree and to the tree with vendor/cado-characters-parallel.patch. Idempotent."""
import sys
p = sys.argv[1]; s = open(p).read()
if 'CADO_CHARS_SCMAT_IN' in s: print('chars_pre_patch: already patched'); sys.exit(0)

A = '    cado::filter_io_details::interpret_parameters(pl);\n'
B = '    unsigned int const small_nrows = scmat.nrows();\n'
assert s.count(A) == 1 and s.count(B) == 1, 'chars_pre_patch: anchors not found'
i = s.index(A) + len(A); j = s.index(B)
body = s[i:j]
assert 'scmat' in body and 'small_character_matrix' in body, 'chars_pre_patch: unexpected body'

helpers = r'''
/* {{{ RSA-560 CHARS_PRE (gpu/chars_pre_patch.py): scmat save/load */
static uint64_t rsa_chars_fp(std::vector<alg_prime_t> const & chars, int nchars)
{
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](uint64_t v) { for (int k = 0; k < 8; k++) { h ^= (v >> (8*k)) & 0xff; h *= 1099511628211ULL; } };
    mix((uint64_t) nchars); mix((uint64_t) chars.size());
    for (auto const & c : chars) { mix((uint64_t) c.p); mix((uint64_t) c.r); }
    return h;
}
static uint64_t rsa_fsize(const char * fn)
{
    struct stat sb[1];
    if (stat(fn, sb) < 0) { perror(fn); exit(EXIT_FAILURE); }
    return (uint64_t) sb->st_size;
}
static void rsa_scmat_write(const char * fn, blockmatrix const & m, uint64_t fp, uint64_t isz)
{
    std::string const tmp = std::string(fn) + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) { perror(tmp.c_str()); exit(EXIT_FAILURE); }
    setvbuf(f, NULL, _IOFBF, 1 << 22);
    uint64_t const hdr[4] = { (uint64_t) m.nrows(), (uint64_t) m.ncols(), fp, isz };
    int ok = fwrite("RSACHR01", 1, 8, f) == 8 && fwrite(hdr, 8, 4, f) == 4;
    for (unsigned int i = 0; ok && i < m.nrows(); i++)
        for (unsigned int cg = 0; ok && cg < m.ncols(); cg += 64) {
            uint64_t const w = m[i][cg];
            ok = fwrite(&w, 8, 1, f) == 1;
        }
    if (fclose(f) != 0 || !ok) { fprintf(stderr, "Error: writing %s failed\n", tmp.c_str()); exit(EXIT_FAILURE); }
    if (rename(tmp.c_str(), fn) != 0) { perror(fn); exit(EXIT_FAILURE); }
}
static blockmatrix rsa_scmat_read(const char * fn, unsigned int exp_ncols, uint64_t fp, uint64_t isz)
{
    FILE * f = fopen(fn, "rb");
    if (!f) { perror(fn); exit(EXIT_FAILURE); }
    setvbuf(f, NULL, _IOFBF, 1 << 22);
    char mg[8]; uint64_t hdr[4];
    if (fread(mg, 1, 8, f) != 8 || memcmp(mg, "RSACHR01", 8) != 0 || fread(hdr, 8, 4, f) != 4) {
        fprintf(stderr, "Error: %s is not a CHARS_PRE scmat file\n", fn); exit(EXIT_FAILURE);
    }
    if (hdr[1] != exp_ncols || hdr[2] != fp || hdr[3] != isz) {
        fprintf(stderr, "Error: %s does not match this run (ncols %" PRIu64 "/%u, fingerprint %016" PRIx64 "/%016" PRIx64
                ", index size %" PRIu64 "/%" PRIu64 ")\n", fn, hdr[1], exp_ncols, hdr[2], fp, hdr[3], isz);
        exit(EXIT_FAILURE);
    }
    uint64_t const sz = rsa_fsize(fn);
    uint64_t const wpr = (hdr[1] + 63) / 64;
    if (sz != 40 + hdr[0] * wpr * 8) { fprintf(stderr, "Error: %s has %" PRIu64 " bytes, want %" PRIu64 "\n", fn, sz, 40 + hdr[0]*wpr*8); exit(EXIT_FAILURE); }
    blockmatrix m(hdr[0], hdr[1]);
    m.set_zero();
    std::vector<uint64_t> row(wpr);
    for (uint64_t i = 0; i < hdr[0]; i++) {
        if (fread(row.data(), 8, wpr, f) != wpr) { fprintf(stderr, "Error: short read on %s\n", fn); exit(EXIT_FAILURE); }
        for (uint64_t k = 0; k < wpr; k++) m[i][64 * k] = row[k];
    }
    fclose(f);
    return m;
}
/* }}} */

int main(int argc, char const * argv[])
'''
M = 'int main(int argc, char const * argv[])\n'
assert s.count(M) == 1, 'chars_pre_patch: main not found'

new = r'''
    /* RSA-560 CHARS_PRE (gpu/chars_pre_patch.py): the stock scmat construction, unchanged, in a lambda */
    auto rsa_build_scmat = [&]() -> blockmatrix {
''' + body + r'''        return std::move(const_cast<blockmatrix &>(scmat));
    };
    const char * rsa_scin = getenv("CADO_CHARS_SCMAT_IN");
    const char * rsa_scout = getenv("CADO_CHARS_SCMAT_OUT");
    uint64_t const rsa_fp = rsa_chars_fp(chars, nchars);
    uint64_t const rsa_isz = rsa_fsize(indexname);
    blockmatrix scmat(0, 0);
    if (rsa_scin && *rsa_scin) {
        scmat = rsa_scmat_read(rsa_scin, chars.size(), rsa_fp, rsa_isz);
        fprintf(stderr, "CHARS_PRE: loaded small character matrix %u x %u from %s at wct=%.1fs\n",
                scmat.nrows(), scmat.ncols(), rsa_scin, wct_seconds()-wct0);
    } else {
        scmat = rsa_build_scmat();
    }
    if (rsa_scout && *rsa_scout && !(rsa_scin && *rsa_scin)) {
        rsa_scmat_write(rsa_scout, scmat, rsa_fp, rsa_isz);
        fprintf(stderr, "CHARS_PRE: wrote small character matrix %u x %u to %s at wct=%.1fs\n",
                scmat.nrows(), scmat.ncols(), rsa_scout, wct_seconds()-wct0);
        print_timing_and_memory (stdout, cpu0, wct0);
        return 0;
    }

'''
s = s[:i] + new + s[j:]
s = s.replace(M, helpers.lstrip('\n'), 1)
for inc in ('#include <string>\n', '#include <vector>\n', '#include <cinttypes>\n', '#include <cstring>\n'):
    s = s.replace('#include <sys/stat.h>\n', '#include <sys/stat.h>\n' + inc, 1) if inc not in s else s
open(p, 'w').write(s)
print('chars_pre_patch: patched', p)
