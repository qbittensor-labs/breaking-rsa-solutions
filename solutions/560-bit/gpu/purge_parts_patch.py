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

"""purge_parts_patch.py <cado-nfs/filter/purge.cpp> -- pass 2 writes its output through ONE checked stream, in input-file
order, from per-file memory buffers; no temporary part files.

Why (S1 validation failure, 2026-10-25, key 30): the Cognition purge writes pass 2 as one <out>.part<i> file per input
file BESIDE the final output and concatenates them afterwards. The pipeline points the final output at a memfd
(factor_c169.sh MEMFD_PURGED=1, cado-symlink-writethrough), but the parts were ordinary files next to that symlink,
i.e. on the validator's 10 GiB /tmp tmpfs. At c169 the parts are 7-8 GB on top of the ~4.6 GB already there: the tmpfs
overflowed, the failed part writes were never checked (fmt::print to a stream in a failed state is silent, close()
ignores pubsync()'s result, concatenate() ignores read failures), and purge returned 0 with a header line promising
more rows than the file held. The filter wrapper's log-count check caught it only because the log, also on /tmp, was
truncated too -- and the run then sat until the deadline: EmptyArtifacts.

What this patch does: each input file's retained relations are collected in a std::string; when the file is done the
buffer is handed to an ordered writer that appends buffers to the single final stream in input-file order (the parts
were concatenated in that order, so the output is byte-identical). Every write is checked, the stream is flushed and
checked before close, and for a plain output the final file size must equal the bytes written. Memory stays bounded:
for_each_file hands out file indices in increasing order, so only about nthreads buffers ever wait for their turn
(~8 MB each at c169). purge prints "# ordered output <file>: <n> parts, <bytes> bytes" on success; the wrapper
(cado_filter160.sh) requires that line under PURGE_ORDERED_REQUIRED=1 and re-counts the rows of the plain file.
Applies to the Cognition tree b5ca9c4 (with or without the write-through patch). Idempotent (marker CADO_PURGE_ORDERED)."""
import sys
p = sys.argv[1]; s = open(p).read()
if 'CADO_PURGE_ORDERED' in s: print('purge_parts_patch: already patched'); sys.exit(0)

# ---- 1. includes ---------------------------------------------------------------------------------------------
A = '#include <unistd.h>\n'
assert s.count(A) == 1, 'purge_parts_patch: include anchor not found'
s = s.replace(A, A + '#include <sys/stat.h>\n#include <cerrno>\n#include <iterator>\n#include <map>\n', 1)

# ---- 2. the parted_output struct -> ordered_output -----------------------------------------------------------
B0 = '/* Output file that pass 2 writes as one part per input file, so that\n'
B1 = '/* }}} */\n\nstruct purge_output_specification { /* {{{ */\n'
assert s.count(B0) == 1 and s.count(B1) == 1, 'purge_parts_patch: parted_output anchors not found'
i = s.index(B0); j = s.index(B1)
old = s[i:j]
assert 'struct parted_output' in old and 'void concatenate() const' in old, 'purge_parts_patch: unexpected parted_output body'
NEW_STRUCT = r'''/* RSA-560 CADO_PURGE_ORDERED (gpu/purge_parts_patch.py). Pass 2 used to write one <out>.part<i> file per
 * input file BESIDE the final output and concatenate them afterwards. With a final output that is a symlink to a
 * memfd (the pipeline's MEMFD_PURGED under the validator's 10 GiB /tmp tmpfs) the parts were ordinary files on
 * /tmp: 7-8 GB at c169, the tmpfs overflowed, the failed part writes were never checked, and purge returned 0
 * with a header promising more rows than the file held. Now each input file's retained relations are collected
 * in a memory buffer and the buffers are written to the ONE final stream in input-file order: byte-identical
 * output, no temporary files, every write checked, the final size checked against the bytes written. Memory is
 * bounded: for_each_file hands out file indices in increasing order, so only about nthreads buffers wait. */
#define CADO_PURGE_ORDERED 1
struct ordered_output { /* {{{ */
    std::string name;
    size_t nparts;
    ofstream_maybe_compressed out;
    std::mutex m;
    size_t next = 0;
    size_t bytes = 0;
    std::map<size_t, std::string> pending;

    /* Creates the final file with only the header line; the stream stays open until finish(). */
    ordered_output(std::string const & name, size_t nparts,
                   std::string const & header)
        : name(name)
        , nparts(nparts)
        , out(name)
    {
        if (!out.good())
            throw cado::error("cannot open {} for writing", name);
        out << header;
        if (!out.good())
            throw cado::error("write error on {}", name);
        bytes = header.size();
    }

    /* Called from the worker threads, once per input file, in any order. */
    void complete(size_t i, std::string && buf)
    {
        std::scoped_lock const lk(m);
        if (i >= nparts || !pending.emplace(i, std::move(buf)).second)
            throw cado::error("{}: part {} completed twice", name, i);
        while (!pending.empty() && pending.begin()->first == next) {
            auto it = pending.begin();
            out.write(it->second.data(),
                      static_cast<std::streamsize>(it->second.size()));
            if (!out.good())
                throw cado::error("write error on {} (part {})", name, next);
            bytes += it->second.size();
            pending.erase(it);
            next++;
        }
    }

    /* All parts must have been written; flush, close, and verify the plain file's size. */
    void finish()
    {
        std::scoped_lock const lk(m);
        if (next != nparts || !pending.empty())
            throw cado::error("{}: only {} of {} parts written", name, next,
                              nparts);
        out.flush();
        if (!out.good())
            throw cado::error("flush error on {}", name);
        out.close();
        if (get_suffix(name).empty()) {
            struct stat sb;
            if (stat(name.c_str(), &sb) != 0)
                throw cado::error("stat({}): {}", name, strerror(errno));
            if (static_cast<size_t>(sb.st_size) != bytes)
                throw cado::error("{}: size {} != {} bytes written", name,
                                  static_cast<size_t>(sb.st_size), bytes);
        }
        fmt::print("# ordered output {}: {} parts, {} bytes\n", name, nparts,
                   bytes);
        fflush(stdout);
    }
};
'''
s = s[:i] + NEW_STRUCT + s[j:]

# ---- 3. pass 2: buffers instead of part files -----------------------------------------------------------------
C = '        std::optional<parted_output> out;\n        std::optional<parted_output> outdel;\n'
assert s.count(C) == 1, 'purge_parts_patch: pass2 optional anchor not found'
s = s.replace(C, '        std::optional<ordered_output> out;\n        std::optional<ordered_output> outdel;\n', 1)

D0 = '''        for_each_file(files.size(), [&](size_t i) {
            ofstream_maybe_compressed out_i;
            ofstream_maybe_compressed outdel_i;
'''
D1 = '''        if (out)
            out->concatenate();
        if (outdel)
            outdel->concatenate();
'''
assert s.count(D0) == 1 and s.count(D1) == 1, 'purge_parts_patch: pass2 loop anchors not found'
i = s.index(D0); j = s.index(D1) + len(D1)
old = s[i:j]
for must in ('out->part_name(i)', 'fmt::print(out_i, "{}\\n", rel.line);', 'fmt::print(outdel_i, "{}\\n", rel.line);',
             'W[i] += static_cast<double>(rel.weight);', 'size_t const base = row_base[i];'):
    assert must in old, 'purge_parts_patch: unexpected pass2 body (%s)' % must
NEW_LOOP = '''        for_each_file(files.size(), [&](size_t i) {
            std::string buf_i;
            std::string bufdel_i;
            size_t const base = row_base[i];
            filter_rels<relation_type>(files[i], nullptr, nullptr,
                [&](relation_type & rel) {
                    if (M.is_active(base + rel.num)) {
                        W[i] += static_cast<double>(rel.weight);
                        if (out)
                            fmt::format_to(std::back_inserter(buf_i), "{}\\n",
                                           rel.line);
                    } else if (outdel) {
                        fmt::format_to(std::back_inserter(bufdel_i), "{}\\n",
                                       rel.line);
                    }
                });
            if (out)
                out->complete(i, std::move(buf_i));
            if (outdel)
                outdel->complete(i, std::move(bufdel_i));
        });

        if (out)
            out->finish();
        if (outdel)
            outdel->finish();
'''
s = s[:i] + NEW_LOOP + s[j:]
assert 'parted_output' not in s and 'part_name(' not in s and 'concatenate()' not in s, 'purge_parts_patch: leftovers'
open(p, 'w').write(s)
print('purge_parts_patch: applied to %s' % p)
