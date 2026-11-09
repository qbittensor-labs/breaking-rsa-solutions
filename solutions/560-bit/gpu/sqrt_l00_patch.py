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

"""sqrt_l00_patch.py <cado-nfs/sqrt/sqrt.cpp> -- level 00 of the algebraic square root without CADO's per-product
OpenMP regions and allocations (audit s21). Copies gpu/sqrt_l00.hpp next to sqrt.cpp and, at run time:
  SQRT_L00 unset/0  CADO's range product (default, unchanged: mpz_polymodF_mul through the *parallel* mpz_poly
                    interface, ~26 nested OpenMP regions and ~20 malloc/free per product, 115K products per range)
  SQRT_L00=1        CADO's own product through the not-parallel interface (same code, no OpenMP region)
  SQRT_L00=2        gpu/sqrt_l00.hpp: thread-local binary-counter tree, __int128 first level, Kronecker products
                    below 2^SQRT_L00_KS_BITS (default 12) packed bits, CADO's not-parallel Toom-Cook above
All three return the same (p, v) for a range (same pseudo-reduction mod the non-monic F, v counts the steps), so
the square root and its "square root is" line are unchanged. Laptop (M1, one thread, key-1-like ranges of 115K
pairs): CADO 0.92 s, mode 2 0.76 s, identical (p, v); with nested OpenMP active the default path took 126x longer.
The sandbox level 00 takes ~50 s per range (12,700 CPU-s per dependency), 50x the laptop: the overhead, not the
arithmetic, is the target."""
import os, shutil, sys
p = sys.argv[1]; s = open(p).read()
hdr = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sqrt_l00.hpp')
shutil.copyfile(hdr, os.path.join(os.path.dirname(os.path.abspath(p)), 'sqrt_l00.hpp'))   # always refresh the copy
if 'SQRT_L00' in s: print('already patched (header refreshed)'); sys.exit(0)

def sub(a, b):
    global s
    assert s.count(a) == 1, a[:120]
    s = s.replace(a, b)

sub('#include "mpz_polymodF.h"\n', '#include "mpz_polymodF.h"\n#include "sqrt_l00.hpp"   /* gpu/sqrt_l00_patch.py */\n')

# the floor level: release the per-thread storage of l00 once a thread has no range left
sub('''            fflush (stderr);
        }
        fclose_maybe_compressed_lock (fi, depname.c_str());
    }
    nab = ab;''', '''            fflush (stderr);
        }
        l00::release();
        fclose_maybe_compressed_lock (fi, depname.c_str());
    }
    nab = ab;''')

sub('''    ATTRIBUTE_NODISCARD
    T range_product(std::vector<ab_pair> const & ab) const {
        return product_of_leaves(ab, *this);
    }
    static bool is1(T & res) {''', '''    ATTRIBUTE_NODISCARD
    T range_product(std::vector<ab_pair> const & ab) const {
        /* SQRT_L00: 0 CADO (parallel interface), 1 CADO not parallel, 2 sqrt_l00.hpp (gpu/sqrt_l00_patch.py) */
        static const int l00_mode = [] {
            const char * e = getenv("SQRT_L00");
            const int m = e ? atoi(e) : 0;
            fmt::print(stderr, "level 00 range product: SQRT_L00={}\\n", m);
            return m;
        }();
        if (l00_mode == 2 && F->deg >= 1 && F->deg <= 8) {
            mpz_srcptr Fc[9];
            for (int i = 0; i <= F->deg; i++)
                Fc[i] = mpz_poly_coeff_const(F, i);
            l00::node r;
            l00::range_product(r, ab, Fc, F->deg);
            T res;
            mpz_poly_realloc(res->p, r.deg + 1);
            for (int i = 0; i <= r.deg; i++)
                mpz_swap(res->p->_coeff[i], r.c[i]);
            mpz_poly_cleandeg(res->p, r.deg);
            res->v = r.v;
            return res;
        }
        if (l00_mode == 1) {
            struct np_functions {
                using T = cxx_mpz_polymodF;
                mpz_poly_srcptr F;
                static T from_ab(cxx_mpz const & a, cxx_mpz const & b) { return cxx_mpz_polymodF_functions::from_ab(a, b); }
                static bool is1(T & x) { return cxx_mpz_polymodF_functions::is1(x); }
                static void set(T & y, T const & x) { mpz_polymodF_set(y, x); }
                static void set1(T & x) { mpz_polymodF_set_ui(x, 1); }
                void operator()(T & res, T const & a, T const & b) const { mpz_polymodF_mul(res, a, b, F); }
            } const np { F };
            return product_of_leaves(ab, np);
        }
        return product_of_leaves(ab, *this);
    }
    static bool is1(T & res) {''')
open(p, 'w').write(s); print('patched')
