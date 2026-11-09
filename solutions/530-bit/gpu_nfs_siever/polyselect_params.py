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

"""Size-aware polynomial-selection parameters for polyselect_best.sh.

Every constant in the shipped polyselect stage was tuned against a c151 key and hard-coded, so
the stage does not merely get *worse* on a larger N -- it FAILS. Measured on a c154 (512-bit)
key, 2026-09-03/04:

  * ``MIN_EVALUE=3e-12`` is the bar msieve uses to BANK a candidate. The best candidate in a
    283,607-candidate c154 pool scores 2.679e-12, so every slice banks nothing, ``pick_topk.py``
    finds no pool, and polyselect_best.sh takes its own ``exit 3`` at ~10 min.
  * ``FINE_HI=40000`` excludes the c154 winners outright: c5 = 42,300 (msieve) and 50,040 (CADO).
  * ``POLY_BREAKEVEN=43.3`` is above the entire achievable c154 band (ceiling 31.3 rel/lat), so
    the warning fires on every key and stops carrying information.

The validator passes NO environment into the container, so a hard-coded default IS the production
value. This module makes the three of them functions of ``digits(N)`` instead.

Scaling laws, both anchored on two MEASURED points (c151 and c154) rather than assumed:

  msieve "expecting poly E"   c151 8.10e-12 -> c154 4.81e-12   => exp(-0.1737 per digit)
  best measured yield         c151 42.5     -> c154 28.0       => exp(-0.1392 per digit)

``MIN_EVALUE`` keeps the shipped c151 value's *ratio* to the prediction (3.0/8.10 = 0.37), which is
comfortably below the ~0.55 x prediction that is actually achievable at both sizes (c151 ~4.5e-12,
c154 2.679e-12 measured -- 0.557 x its 4.81e-12 prediction).

``POLY_BREAKEVEN`` is re-defined. The shipped 43.3 was "the yield below which the sieve cannot reach
the crossover inside the wall"; at c154 nothing reaches that, so as a per-key discriminator it is
useless. It is now "clearly below the achievable band for this size" (0.92 x the size's ceiling),
which is what the warning was actually useful for. Configuration.md 5.2 already records that this
metric does not predict wall time.

The CADO table is CADO-NFS's own ``parameters/factor/params.cNNN``, read from the vendored tarball
(148/150/153/155/158). Those files agree on ``incr=420 nq=156250 nrkeep=72`` across the whole range
and vary only P/admin/admax, so only those three are interpolated.

usage: polyselect_params.py <N>   -> shell-eval-able PS_KEY=VALUE lines (PS_ prefix so a real
                                     environment override of MIN_EVALUE/FINE_HI/... still wins)
"""
import math
import sys

# digits -> (P, admin, admax) from CADO-NFS parameters/factor/params.cNNN
CADO_TABLE = [
    (148, 260000, 2160, 100000),
    (150, 290000, 2160, 140000),
    (153, 360000, 2520, 145000),
    (155, 390000, 4620, 180000),
    (158, 480000, 3360, 240000),
]


# -P at a 300 s / 24-core budget, MEASURED on the c154 key 2026-09-04. Four concurrent searches on
# 24 dedicated physical cores each, then an identical 150 s rootsieve; best MurphyE:
#     P=700k  2.762e-12      P=1M  3.036e-12  <-- interior optimum
#     P=1.5M  2.827e-12      P=2.5M 2.700e-12
# CADO's own params.cNNN says P=390000 for c155, and that is RIGHT FOR ITS OWN BUDGET, not ours:
# those files assume a distributed run of many machine-hours, where the search actually walks the
# whole `ad` range. At 300 s on 24 cores it covers only ~218 `ad` values, so the budget is far better
# spent making each one count. Using the table value (375k at c154) end-to-end measured **29.09
# rel/lat**; P=1e6 is what produces the 31.3 band.
# The SCALING is CADO's (P grows exp(0.0614/digit across params.c148..c158)); only the ANCHOR moved,
# from the table to the measured c154 optimum.
CADO_P_ANCHOR_DIGITS = 154
CADO_P_ANCHOR = 1.0e6
CADO_P_PER_DIGIT = 0.0614


def cado_params(d: int) -> dict:
    """Nearest-neighbour on the CADO table, log-interpolating P and admax between entries."""
    lo = CADO_TABLE[0]
    hi = CADO_TABLE[-1]
    for a, b in zip(CADO_TABLE, CADO_TABLE[1:]):
        if a[0] <= d <= b[0]:
            lo, hi = a, b
            break
    else:
        # outside the table: clamp to the nearest end and extrapolate P/admax log-linearly
        lo, hi = (CADO_TABLE[0], CADO_TABLE[1]) if d < CADO_TABLE[0][0] else (CADO_TABLE[-2], CADO_TABLE[-1])
    span = hi[0] - lo[0]
    t = 0.0 if span == 0 else (d - lo[0]) / span
    P = math.exp(math.log(lo[1]) * (1 - t) + math.log(hi[1]) * t)
    admax = math.exp(math.log(lo[3]) * (1 - t) + math.log(hi[3]) * t)
    admin = lo[2] if t < 0.5 else hi[2]
    # P from the measured anchor, not the table -- see the block above. `admax` is kept from the
    # table but is NOT binding at this budget: at 300 s the search reaches ad~13,500 of 200,000, and
    # the c5 that ends up in the output is the SIZE-OPTIMISED one (CADO's sopt translates/rotates, so
    # reported c5 ranges over +-4e8 here) rather than the search `ad`. It only has to be big enough
    # never to terminate the search early.
    P = CADO_P_ANCHOR * math.exp(CADO_P_PER_DIGIT * (d - CADO_P_ANCHOR_DIGITS))
    return {
        "PS_CADO_P": int(round(P / 1000.0) * 1000),
        "PS_CADO_ADMIN": int(admin),
        "PS_CADO_ADMAX": int(round(admax / 1000.0) * 1000),
        "PS_CADO_INCR": 420,
        "PS_CADO_NQ": 156250,
        "PS_CADO_NRKEEP": 72,
        "PS_CADO_SOPTEFFORT": 4,
    }


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    n = sys.argv[1].strip()
    if not n.isdigit():
        sys.exit("N must be a decimal integer")
    d = len(n)
    # Defensive clamp. Every law below is a two-point fit anchored at c151/c154 and is meaningful
    # only near that range; an out-of-range digit count (a malformed N, or a caller that passed
    # something other than N) would otherwise extrapolate to absurd values -- e.g. FINE_HI=4.6e25.
    # Clamping degrades to the nearest sane parameter set instead of producing a broken search.
    if not 120 <= d <= 200:
        sys.stderr.write("polyselect_params: digits=%d outside the fitted 120..200 band; "
                         "clamping (parameters are extrapolated, verify them)\n" % d)
        d = min(200, max(120, d))

    min_evalue = 3.0e-12 * math.exp(-0.1737 * (d - 151))
    # productive c5 zone: c151 tops out around 40k, c154's winners sit at 42.3k / 50.0k. 1.20/digit
    # reproduces both and keeps 32 slices of a sane width.
    fine_hi = max(40000.0, 40000.0 * (1.20 ** (d - 151)))
    fine_w = max(500, int(round(fine_hi / 32.0 / 5.0) * 5))
    breakeven = 43.3 * math.exp(-0.1392 * (d - 151))

    out = {
        "PS_DIGITS": d,
        "PS_MIN_EVALUE": "%.3g" % min_evalue,
        "PS_FINE_HI": int(round(fine_hi / 100.0) * 100),
        "PS_FINE_W": fine_w,
        "PS_POLY_BREAKEVEN": "%.1f" % breakeven,
    }
    out.update(cado_params(d))
    for k, v in out.items():
        print("%s=%s" % (k, v))


if __name__ == "__main__":
    main()
