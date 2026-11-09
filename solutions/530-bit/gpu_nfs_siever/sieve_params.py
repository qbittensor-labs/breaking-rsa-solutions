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

"""Size-aware sieve parameters for factor_msv.sh.

512-bit N is 154 or 155 digits (76.8 % / 23.2 %). The shipped sieve constants were calibrated at
151 digits, and off-anchor they fail silently: the ladder probes at the wrong relation count and
the run over- or under-sieves with nothing in the log to say why. This derives them per digit.

Below 152 digits the 151-digit constants are emitted VERBATIM -- that path is live and passing,
so it is reproduced, never re-derived.

usage: sieve_params.py <N>  ->  shell-eval-able SV_KEY=VALUE lines (SV_ prefix so a real
                                environment override still wins)
"""
import math
import sys

# (digits, QMIN, QSPLIT). QMIN is log-interpolated; QSPLIT is NOT -- see qsplit_for.
# (digits, QMIN, QSPLIT). QSPLIT here is REPORTED ONLY -- qsplit_for() derives the real one.
# The 155 row is MEASURED (2026-09-14), not extrapolated: the duplicate oracle was swept at
# QMIN = 6.3 / 9.3 / 13 / 18 M on the polynomial the bake-off actually selects for key 20260916,
# and integrated to the empirical requirement (43.6 M unique). Lattices needed:
#     6.3M +3.3%   9.3M +1.0%   13M +0.0%   18M +0.1%
# i.e. a broad plateau over 9-18 M with the shipped 6.3 M sitting OUTSIDE it, costing 3.3 % of
# sieve. ⚠️ The same sweep on the PREVIOUS winner cost only 1.2 %: duplicate structure is a
# property of the polynomial, so this anchor is only as good as the polynomial selection above it.
# Re-measure if the selector changes. Raw: 512bit-research/logs/qmin-c155-20260914.txt
ANCHORS = [(151, 1.0e6, 3.8e6), (154, 4.0e6, 6.8e6), (155, 13.0e6, 16.5e6)]

# (digits, unique relations the matrix needs, unique fraction at that QMIN).
# 155 is MEASURED: crossover bracketed to (55.40, 56.17] M raw, and dup2 reported 40,840,453
# unique from 56,600,360 raw. 151 is measured; 154 is still modelled.
# ⚠️ `frac` is the unique fraction AT THAT ROW'S QMIN, so it MUST move when QMIN moves -- it is
# not an independent constant. Raising QMIN to 13 M is precisely a reduction in self-duplication,
# and leaving frac at the 6.3 M value would over-estimate the raw relations needed, push
# `first_break` above the true crossover, and spend the QMIN saving on over-sieving before the
# first probe. The oracle's ABSOLUTE frac is biased low (it charges the whole q-range to the GPU
# and ignores las: it reads 0.6267 at 6.3 M where the c155d run measured 0.7602), so only its
# RATIO is used here: frac(13M)/frac(6.3M) = 0.6929/0.6267 = 1.1056, applied to the 0.7216 that
# was fitted against real runs -> 0.798.
REQ_ANCHORS = [(151, 39.86e6, 0.613), (154, 40.7e6, 0.663), (155, 40.5e6, 0.798)]

EXPECTED_SIEVE_SECS = 12413.0                    # 154-digit anchor, measured end-to-end
SIEVE_SECS_ANCHORS = [(154, 12413.0), (155, 12600.0)]


def expected_sieve_secs(d):
    """Expected sieve wall-seconds at `d` digits; log-interpolated, flat outside the anchors."""
    return _interp_log(d, SIEVE_SECS_ANCHORS, 1)


# GPU cost of one lattice, wall ms. Flat in q (4.8-5.0 measured across the whole c155 sweep --
# see the per-window `wall` figures in any bake-off log), which is what makes LATTICES, not
# relations, the right currency for comparing q-ranges.
MS_PER_LAT = 5.0


def lattices_between(a, b, h=2000.0):
    """Lattice count over special-q [a,b). Density is dL/dq = 1/ln q (special-q are primes)."""
    s = 0.0
    q = float(a)
    while q < b:
        s += h / math.log(q)
        q += h
    return s


def probe_windows(qsplit, sieve_secs, width=15000, fracs=(0.25, 0.50, 0.75)):
    """Bake-off probe windows placed at LATTICE quantiles of the GPU's production q-range.

    ⚠️ THIS IS A BUG FIX, NOT A TUNING KNOB. The shipped constant was
    `4000000:4015000 11000000:11015000 19000000:19015000`, chosen at c151/c154 where QMIN was
    1.0 M / 4.0 M. QMIN climbs with digits (6.3 M at c155) and QSPLIT with it, so by c155 those
    windows had fallen off the bottom of the range without anything checking:

        q =  4 M  -- OUTSIDE the production range entirely. The GPU starts at QSPLIT = 9.6 M and
                     las at 6.3 M, so NOTHING sieves there; it was still a third of the score.
        q = 11 M  --  3rd percentile of the GPU's lattices
        q = 19 M  -- 20th percentile

    i.e. the bake-off ranked candidates on the bottom fifth of the range while the GPU's median
    lattice sits at q ~ 33 M. MEASURED 2026-09-14 on key 20260916, the two polynomials from the
    c155d/c155e A/B: on the old windows the msieve poly wins on raw yield (28.8 vs 27.6) and on
    unique yield (24.32 vs 24.01) -- and it LOST end-to-end by 211 s. On these windows it still
    wins on raw (18.7 vs 18.0) but LOSES on unique (11.20 vs 11.36), which is the end-to-end
    result. Its duplicate rate at high q is worse (40.0 % vs 37.0 %), and high q is where 75 % of
    the lattices are. Window placement and the unique-vs-raw metric BOTH have to be right; either
    one alone still picks the loser.
    """
    total = sieve_secs * 1000.0 / MS_PER_LAT
    lo, hi = qsplit, qsplit * 100.0
    for _ in range(80):                       # bisect for the q where the sieve runs out of time
        mid = (lo + hi) / 2.0
        if lattices_between(qsplit, mid) < total:
            lo = mid
        else:
            hi = mid
    qend = (lo + hi) / 2.0
    wins = []
    for f in fracs:
        a, b = qsplit, qend
        for _ in range(60):
            m = (a + b) / 2.0
            if lattices_between(qsplit, m) / total < f:
                a = m
            else:
                b = m
        c = int(round((a + b) / 2.0 / 100000.0) * 100000)
        wins.append("%d:%d" % (c, c + width))
    return " ".join(wins), qend


SQPS = [(1.0e6, 11.8), (3.0e6, 13.1), (5.0e6, 13.7), (8.0e6, 14.2)]   # special-q/s, measured

# Throughput calibration on the raw integral. Per digit because two end-to-end measurements
# disagree by 7 %: las covers its range faster at 155 digits than the 154 fit predicts.
CAL_T_ANCHORS = [(154, 0.9448), (155, 0.8812)]


def cal_t(d):
    """Throughput calibration at `d` digits; log-interpolated, flat outside the anchors."""
    return _interp_log(d, CAL_T_ANCHORS, 1)


CAL_T = 0.9448                  # module default, so the d=154 regression check reads unchanged


def _loglog(pts, q):
    """pts interpolated in log-log, extrapolating on the end segments' own slope."""
    xs = [p[0] for p in pts]
    if q <= xs[0]:
        i = 0
    elif q >= xs[-1]:
        i = len(pts) - 2
    else:
        i = max(j for j in range(len(pts) - 1) if xs[j] <= q)
    a, b = pts[i], pts[i + 1]
    s = math.log(b[1] / a[1]) / math.log(b[0] / a[0])
    return a[1] * (q / a[0]) ** s


def _las_secs(a, b, n=20000, cal=None):
    """Seconds for las to sieve special-q in [a, b), under the production 20/4 pin."""
    if b <= a:
        return 0.0
    cal = CAL_T if cal is None else cal
    h = (b - a) / n
    tot = 0.0
    for i in range(n):
        q = a + h * (i + 0.5)
        tot += 1.0 / (_loglog(SQPS, q) * math.log(q))
    return tot * h * cal


def qsplit_for(qmin, secs=None, cal=None):
    """Where las hands over to gpu_loop: the q it reaches in the time the sieve lasts.

    QSPLIT is a THROUGHPUT answer, not a size-dependent constant -- interpolating it between
    anchors collapses the span off-anchor and idles 20 cores for hours. Bisected because the
    integrand has no closed form. Targets the crossing with no safety margin: too high strands a
    band nobody sieves, too low idles las, and both cost about the same.
    """
    secs = EXPECTED_SIEVE_SECS if secs is None else secs
    # hi is relative to qmin so the bracket cannot collapse once qmin passes a fixed ceiling.
    lo, hi = qmin, max(60.0e6, qmin * 4.0)
    for _ in range(90):
        mid = (lo + hi) / 2.0
        if _las_secs(qmin, mid, cal=cal) < secs:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _interp_log(d, pts, idx):
    xs = [p[0] for p in pts]
    if d <= xs[0]:
        lo, hi = pts[0], pts[1]
    elif d >= xs[-1]:
        lo, hi = pts[-2], pts[-1]
    else:
        for a, b in zip(pts, pts[1:]):
            if a[0] <= d <= b[0]:
                lo, hi = a, b
                break
    span = hi[0] - lo[0]
    t = 0.0 if span == 0 else (d - lo[0]) / span
    return math.exp(math.log(lo[idx]) * (1 - t) + math.log(hi[idx]) * t)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    n = sys.argv[1].strip()
    if not n.isdigit():
        sys.exit("N must be a decimal integer")
    d = len(n)
    if not 120 <= d <= 200:
        sys.stderr.write("sieve_params: digits=%d outside the fitted 120..200 band; clamping "
                         "(parameters are extrapolated, verify them)\n" % d)
        d = min(200, max(120, d))

    qmin = _interp_log(d, ANCHORS, 1)
    # QMIN is a two-point extrapolation and compounds fast past the anchors.
    if d > 155:
        sys.stderr.write("sieve_params: digits=%d -- QMIN=%.1f M is extrapolated %d digits past "
                         "the last measured anchor (154). Measure it before trusting this run.\n"
                         % (d, qmin / 1e6, d - 154))
    qsplit_anchor = _interp_log(d, ANCHORS, 2)   # reported only, to expose model divergence
    sieve_secs = expected_sieve_secs(d)
    qsplit = qsplit_for(qmin, sieve_secs, cal=cal_t(d))
    req = _interp_log(d, REQ_ANCHORS, 1)
    frac = _interp_log(d, REQ_ANCHORS, 2)

    raw = req / frac

    # Two regimes. Below the boundary the live 151-digit constants are emitted verbatim: they carry
    # a hand-tuned margin this model does not reproduce, and re-deriving them would change a
    # passing solver to buy nothing. Above it the model applies with the same relative margin.
    SHIPPED_DIGITS = 152
    SHIPPED = {"qmin": 1.0e6, "qsplit": 3.8e6, "rel_target": 72.0e6,
               "first_break": 65_075_000.0, "ecore": 216000.0, "max_probes": 8.0}
    MARGIN = SHIPPED["rel_target"] / (39.86e6 / 0.613)             # 1.108, the shipped over-sieve
    PROBE_RATIO = SHIPPED["first_break"] / SHIPPED["rel_target"]   # 0.904
    # Ladder rung 1. Coupled to REQ_ANCHORS' frac: raw = req/frac now ESTIMATES the crossover, so
    # the multiplier sits just under 1. Move the two together or not at all.
    PROBE_BAND_LO = 0.98
    PROBE_STEP = 750000.0       # must match factor_msv.sh's CKPT_BREAK_STEP default

    if d < SHIPPED_DIGITS:
        qmin, qsplit = SHIPPED["qmin"], SHIPPED["qsplit"]
        rel_target = SHIPPED["rel_target"]
        first_break = SHIPPED["first_break"]
        ecore = SHIPPED["ecore"]
        max_probes = SHIPPED["max_probes"]
    else:
        rel_target = raw * MARGIN
        first_break = min(raw * PROBE_BAND_LO, rel_target * PROBE_RATIO)   # may only move DOWN
        ecore = SHIPPED["ecore"] * (rel_target / SHIPPED["rel_target"])
        # The budget must reach the ceiling: a ladder that exhausts mid-walk falls through to
        # REL_TARGET and over-sieves. An unused probe is free; an unreachable ceiling is not.
        max_probes = math.ceil((rel_target - first_break) / PROBE_STEP) + 2

    def r(x, unit):
        return int(round(x / unit) * unit)

    # The model must reproduce the one configuration verified end-to-end. If it drifts, every
    # other digit count it emits is suspect too -- fail here rather than 4 h into a run.
    _chk = qsplit_for(4.0e6, 12413.0)
    if abs(_chk - 6.8e6) > 0.05e6:
        sys.exit("sieve_params: las model REGRESSION -- d=154 split is %.3f M, expected 6.80 M. "
                 "Refusing to emit parameters." % (_chk / 1e6))

    # The old anchor interpolation emitted a negative span past 155 digits. qsplit_for bisects
    # upward from qmin so this cannot recur; the guard stops a future edit reintroducing it.
    if qsplit <= qmin:
        sys.exit("sieve_params: NON-POSITIVE LAS_SPAN (QMIN %.2f M >= QSPLIT %.2f M) at digits=%d. "
                 "Refusing to emit parameters." % (qmin / 1e6, qsplit / 1e6, d))

    if d >= SHIPPED_DIGITS and abs(qsplit - qsplit_anchor) > 0.25e6:
        sys.stderr.write("sieve_params: QSPLIT %.2f M (las-throughput) vs %.2f M (old anchor "
                         "interpolation); las would have idled ~%.0f s at the anchor value\n"
                         % (qsplit / 1e6, qsplit_anchor / 1e6,
                            sieve_secs - _las_secs(qmin, qsplit_anchor)))

    out = {
        "SV_DIGITS": d,
        "SV_QMIN": r(qmin, 100000),
        "SV_QSPLIT": r(qsplit, 100000),
        # From the ROUNDED endpoints: factor_msv.sh computes the split it actually uses as
        # QMIN + LAS_SPAN, so independent rounding makes the log disagree with the run.
        "SV_LAS_SPAN": r(qsplit, 100000) - r(qmin, 100000),
        "SV_REL_TARGET": r(rel_target, 100000),
        "SV_CKPT_FIRST_BREAK": r(first_break, 25000),
        "SV_ECORE_PER_MREL": r(ecore, 1000),
        "SV_UNIQUE_NEEDED": r(req, 100000),
        "SV_EXPECT_RAW": r(raw, 100000),
        "SV_SPEC_MAX_PROBES": int(max_probes),
    }
    # Below SHIPPED_DIGITS the live 500-bit path runs; it is passing and its windows (4/11/19 M
    # against QSPLIT = 3.8 M) are genuinely inside its range. Emit nothing there so that path is
    # byte-identical, and polyselect_best.sh keeps its own default.
    if d >= SHIPPED_DIGITS:
        wins, qend = probe_windows(qsplit, sieve_secs)
        out["SV_PROBE_WINS"] = wins
        out["SV_PROBE_QEND"] = r(qend, 100000)
    # The contract at the top of this file is "shell-eval-able SV_KEY=VALUE lines", and every
    # consumer does a bare `eval` on the whole block (factor_msv.sh:43, polyselect_best.sh:56).
    # A value containing whitespace therefore MUST be quoted or eval splits it into an assignment
    # plus stray commands. 2026-09-14: SV_PROBE_WINS shipped unquoted and factor_msv.sh died with
    # `30700000:30715000: command not found` 9 s into a 4 h run. Quote at the source so no future
    # multi-word value can reintroduce it, rather than fixing it once at each call site.
    for k, v in out.items():
        v = str(v)
        print('%s=%s' % (k, '"%s"' % v if (v == "" or any(c.isspace() for c in v)) else v))


if __name__ == "__main__":
    main()
