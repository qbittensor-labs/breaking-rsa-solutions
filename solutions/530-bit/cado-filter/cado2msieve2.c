// Copyright (C) 2026 qBitTensor Labs.
// Original author: an anonymous competition participant (Enigma / Breaking RSA competition).
// IP in custom components assigned to qBitTensor Labs under the Enigma rules.
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
// details. You should have received a copy of the license with this program;
// if not, see <https://www.gnu.org/licenses/>.

// cado2msieve2 -- single-pass, pipe-safe replacement for cado2msieve (same output contract).
//
// Why: at c160 the raw relations are ~20 GB spread over many RAM-backed chunk files (memfds).
// cado2msieve indexed EVERY raw relation by file offset (2^28-slot, 8.6 GB table with no full-table
// guard) and then fseek()ed back into ONE seekable file -- impossible across chunk files, and a pipe
// silently breaks the seek. This version inverts the join:
//   pass A: read purged (hex a,b in purge order; may be a pipe) -> table of the ~20-40M WANTED keys
//   pass B: stream all raw relation files ONCE (pipes/memfds fine), keep only wanted lines
//   emit  : wanted lines in purge order to <out_prefix>, identity map to <out_prefix>.purged
// Output is byte-identical in content/order to cado2msieve for the same inputs.
// CADO free relations (b == 0) are absent from the sieve dump and are re-emitted as "a,0:".
//
// usage: cado2msieve2 <purged_plain|-> <out_prefix> <npurged> <rels file> [<rels file> ...]
// build: gcc -O2 -march=x86-64-v3 -o cado2msieve2 cado2msieve2.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct { int64_t a; uint64_t b; long idx; } key_t_;   // idx = purge position (+1; 0 = empty)
static key_t_ *tab; static size_t tmask;
static char **line_of;          // per purge position: the relation text (NULL until found)
static uint32_t *len_of;
static unsigned char *is_free;
static long *next_pos;          // chain of further purge positions sharing a key (-1 = none)

static inline size_t hsh(int64_t a, uint64_t b) {
    uint64_t h = (uint64_t)a * 0x9E3779B97F4A7C15ULL ^ (b + 0x165667B19E3779F9ULL);
    h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 32;
    return (size_t)h & tmask;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <purged|-> <out_prefix> <npurged> <rels>...\n", argv[0]); return 1; }
    long npur = strtol(argv[3], NULL, 10);
    if (npur <= 0) { fprintf(stderr, "FATAL: bad npurged %s\n", argv[3]); return 1; }
    size_t tsz = 1; while (tsz < (size_t)npur * 2 + 1024) tsz <<= 1;   // load <= 0.5
    tmask = tsz - 1;
    tab = calloc(tsz, sizeof *tab);
    line_of = calloc(npur, sizeof *line_of); next_pos = malloc(npur * sizeof *next_pos);
    if (next_pos) { for (long k = 0; k < npur; k++) next_pos[k] = -1; }
    len_of = calloc(npur, sizeof *len_of); is_free = calloc(npur, 1);
    if (!tab || !line_of || !len_of || !is_free || !next_pos) { fprintf(stderr, "FATAL: alloc failed (npur=%ld)\n", npur); return 1; }

    // ---- pass A: purge order -> wanted keys ----
    FILE *fp = strcmp(argv[1], "-") ? fopen(argv[1], "r") : stdin;
    if (!fp) { perror("purged"); return 1; }
    char *line = NULL; size_t cap = 0; ssize_t n; long idx = 0, nfree = 0;
    while ((n = getline(&line, &cap, fp)) > 0) {
        if (line[0] == '#') continue;
        char *c = strchr(line, ','); if (!c) continue;
        *c = 0;
        int64_t a = (int64_t)strtoll(line, NULL, 16);
        uint64_t b = strtoull(c + 1, NULL, 16);
        if (idx >= npur) { fprintf(stderr, "FATAL: purged has more than npurged=%ld entries\n", npur); return 1; }
        if (b == 0) {                  // free relation: re-emitted as "a,0:" (msieve implies the factorisation)
            is_free[idx] = 1; line_of[idx] = NULL; len_of[idx] = (uint32_t)a;  // stash a below
            char tmp[64]; int k = snprintf(tmp, sizeof tmp, "%lld,0:\n", (long long)a);
            line_of[idx] = malloc(k + 1); memcpy(line_of[idx], tmp, k + 1); len_of[idx] = k; nfree++;
        } else {
            size_t i = hsh(a, b);
            int dup = 0;
            while (tab[i].idx) {
                if (tab[i].a == a && tab[i].b == b) {      // repeated key (never in real CADO output):
                    long q = tab[i].idx - 1;                // chain it, as cado2msieve emitted it twice
                    while (next_pos[q] >= 0) q = next_pos[q];
                    next_pos[q] = idx; dup = 1; break;
                }
                i = (i + 1) & tmask;
            }
            if (!dup) { tab[i].a = a; tab[i].b = b; tab[i].idx = idx + 1; }
        }
        idx++;
    }
    if (fp != stdin) fclose(fp);
    if (idx != npur) { fprintf(stderr, "FATAL: purged has %ld entries, npurged says %ld\n", idx, npur); return 1; }
    fprintf(stderr, "purge entries: %ld (%ld free)\n", npur, nfree);

    // ---- pass B: stream raw relations once, keep wanted ones (first occurrence) ----
    long nraw = 0, found = 0;
    for (int f = 4; f < argc; f++) {
        FILE *fr = fopen(argv[f], "r");
        if (!fr) { perror(argv[f]); return 1; }
        while ((n = getline(&line, &cap, fr)) > 0) {
            long long a; unsigned long long b;
            if (line[0] == '#' || sscanf(line, "%lld,%llu:", &a, &b) != 2) continue;
            nraw++;
            if (b == 0) continue;
            size_t i = hsh((int64_t)a, (uint64_t)b);
            while (tab[i].idx) {
                if (tab[i].a == (int64_t)a && tab[i].b == (uint64_t)b) {
                    long p = tab[i].idx - 1;
                    if (!line_of[p]) {
                        char *t = malloc(n); memcpy(t, line, n);
                        for (long q = p; q >= 0; q = next_pos[q]) { line_of[q] = t; len_of[q] = (uint32_t)n; found++; }
                    }
                    break;
                }
                i = (i + 1) & tmask;
            }
        }
        fclose(fr);
    }
    fprintf(stderr, "indexed %ld relations, found %ld of %ld wanted non-free\n", nraw, found, npur - nfree);

    // ---- emit in purge order ----
    char outrel[4096], outpur[4096];
    snprintf(outrel, sizeof outrel, "%s", argv[2]);
    snprintf(outpur, sizeof outpur, "%s.purged", argv[2]);
    FILE *orl = fopen(outrel, "w"), *opu = fopen(outpur, "w");
    if (!orl || !opu) { perror("out"); return 1; }
    fprintf(opu, "%ld\n", npur);
    for (long p = 0; p < npur; p++) {
        if (!line_of[p]) {
            // report the key for diagnosis
            fprintf(stderr, "FATAL: non-free relation missing at purge position %ld\n", p);
            return 1;
        }
        fwrite(line_of[p], 1, len_of[p], orl);
        fprintf(opu, "%ld\n", p);
    }
    if (fclose(orl) || fclose(opu)) { perror("close out"); return 1; }
    fprintf(stderr, "emitted %ld relations (%ld CADO free relations re-emitted)\n", npur, nfree);
    return 0;
}
