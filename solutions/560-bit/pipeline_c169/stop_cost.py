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

"""stop_cost.py -- cost-based sieve stop rule for factor_c169.sh (STOP_RULE=cost).

Model (all knobs env, defaults calibrated on key1g / key1e / key1f, see the comments):
  N(rows)   = STOP_NFAC * purged_rows        purge-only probe "final nrows" -> merged TD-130 N
              key1g s61: 15.85/64.03, 14.18/56.20, 12.90/50.85 = 0.248 / 0.252 / 0.254
              key1e s23: 19.67/83.09 .. 17.79/74.28 = 0.237 .. 0.240 (32/33)          -> 0.25
  LA(N)     = N/STOP_ITER_DIV iterations x (STOP_C_NNZ_PS * nnz_row * N + STOP_DENSE_NS * N) per iteration
              N/iters 255.24 (key1g), 255.52 (key1f); nnz/row = W/N = 131.0 (key1g TD 130), 131.3 (key1e)
              key1g: 44.25 ms/it at W 1.6906e9, N 12.904M -> 25 ps x W + 0.2 ns x N = 44.9 ms/it (+1.4%)
  post(rows)= STOP_POST_US x purged_rows      merge (-t 20) 88 s at 50.8M rows (key1g), 163 s key1f
  C(point)  = LA(N) + post(rows)               the downstream seconds that still depend on the stop
Marginal saving per raw relation m(A): the secant of C between the last two matrix probes, decaying
exponentially in raw: m(A) = s * exp(-lam (A - mid)). lam = STOP_LAMBDA per M raw (default 0.023: s61's
two secants, LA ~ N^2: 50.2 -> 34.6 (x0.69) per 16M raw) until three points exist, then fitted from the
last two secants (clamped to [0.25, 4] x the default).
Cost per raw relation: (1 + STOP_MARGIN) / rate (rel/s of the GPU sieve, measured) plus STOP_PROBE_TAX
seconds per step (a matrix-era probe slows the GPU sieve ~21-23 s per chunk: key1g chunk 55 234 s and
chunk 61 241 s vs 209-218 s neighbours).
Decision at the landed raw A_L (>= the last probed raw A_K) for the next step of S raw:
  save_next = integral of m over [A_L, A_L + S];  cost_next = S x cost + STOP_PROBE_TAX
  save_next > cost_next -> continue
  else stop:  gain_landed = integral of m over [A_K, A_L] (relations landed after the last probe)
              gain_landed > T_final (final incdup + purge at full width) -> stop_final, else stop_probe

SPEC_MERGE=1 (proposal 4, pipeline_c169/spec_merge.sh): at a stop the orchestrator may hold a speculatively merged +
replayed matrix (READY, measured N_r / W_r on rows_r purged rows) or one still being prepared (rows_p, rem_p s left at its
own threads). With f = N_r / rows_r and w = W_r / N_r measured on that snapshot (else STOP_NFAC / STOP_NNZ_ROW):
  C_ready  = LA(N_r, W_r)                                       merge + replay already paid, not charged again
  C_inprog = rem_p + LA(f rows_p, w f rows_p)
  C_new    = handoff + merge + replay + LA(f rows_new, w f rows_new)
             stop_final: handoff = t_final, rows_new = cost_of^-1(cost_of(rows_K) - gain_landed)
             stop_probe: handoff = 0,       rows_new = rows_K
             merge = STOP_POST_US x rows_new (as in cost_of), replay = SPEC_REPLAY_US x rows_new (70 s / 50.8M, key1g)
The work after the LA (characters, sqrt) does not depend on the choice and is left out of all three.
A prepared matrix is used only if it beats C_new by SPEC_MARGIN_S (default 30 s; the LA model's error is ~1-2%).

STOP_SHAPE=1 (default 0 = the secant rule above, unchanged; docs/research/agents_2026-10-26/stop_rule.md): the SAME
decision, but the downstream cost C(A) comes from a fitted curve instead of the secant between the last two probes:
  crossover X_c : the singleton-excess line e = sx (A - X_c) through the matrix probes (least squares over >= 2 points,
                  sx clamped to [STOP_SX_LO, STOP_SX_HI]; one point: the knee slope STOP_SX, 0.20-0.26 measured per M raw)
  N(A)          = N0 (1 + u/STOP_SHAPE_A)^-STOP_SHAPE_B, u = A/X_c - 1 (keys 25-29, 31, arm B: 33 purge-only points to
                  0.67% rms), N0 by log-space least squares of NFAC x rows over all matrix points
  C(A)          = STOP_KAPPA x (N/1e6)^2 + STOP_POST_US x rows   (LA = kappa N^2: 9.62-10.02 s per (M rows)^2 on 7 keys)
  m(A)          = -dC/dA (closed form); A_star solves m = 1/rate + STOP_PROBE_TAX/step
One matrix point is enough (STOP_MINPTS, default 1 here), so a first matrix already past the optimum stops at once
instead of waiting a chunk for the second point. Any failure of the fit falls back to the secant rule for that call.

usage: stop_cost.py decide <points.tsv> <rate rel/s> <landed raw> <step raw>
         points.tsv: raw <TAB> purged_rows <TAB> sing_excess <TAB> dir <TAB> probe_secs
         prints ONE line: "<continue|stop_probe|stop_final> key=value ..."
       stop_cost.py la <N> [W]   -> predicted LA seconds (W default nnz_row x N)
       stop_cost.py xover <lpb0> <lpb1> [frac]  -> frac (default 1.0) x the expected raw-relation crossover of the regime
         (FIRST_PROBE_AUTO in factor_c169.sh; table XOVER below: measured for 31/32, 31/33, 32/33, modelled for the rest)
       stop_cost.py spec <points.tsv> <landed raw> <stop_probe|stop_final> <rows_r N_r W_r | - - -> <rows_p rem_p | - ->
         prints ONE line: "<use_ready|use_inprog|new> C_ready=.. C_inprog=.. C_new=.. ..."
"""
import math, os, sys

def env(k, d):
    v = os.environ.get(k, '')
    return float(v) if v.strip() else float(d)

NFAC = env('STOP_NFAC', 0.25)
C_NNZ = env('STOP_C_NNZ_PS', 25.0) * 1e-12
DENSE = env('STOP_DENSE_NS', 0.2) * 1e-9
NNZ_ROW = env('STOP_NNZ_ROW', 131)
ITDIV = env('STOP_ITER_DIV', 255.5)
POST = env('STOP_POST_US', 1.8) * 1e-6
LAM0 = env('STOP_LAMBDA', 0.023) / 1e6
MARGIN = env('STOP_MARGIN', 0.0)
TAX = env('STOP_PROBE_TAX', 20.0)
F_INC = env('STOP_FINAL_INC_SECS', 60.0)      # final incdup of the last landed chunk, all groups at once (vs ~90 s at par 1)
F_PURGE = env('STOP_FINAL_PURGE_F', 0.9)      # final purge at NCPU vs the last probe's (NT 12-16 beside the sieve)
# STOP_LA_SCALE (default 1.0 = unchanged): multiplies the LA term only (this box's LA speed relative to the card the
# constants above were calibrated on). 1.0 reproduces every logged verdict; a slower-LA card (molab RTX PRO 6000:
# x1.28 vs the GCP g4 card at N 17M, data/rsa560/audit_ops/laparts_sb111.txt) wants > 1, so the rule oversieves more.
# Set by hand, or by factor_c169.sh from the startup LA micro-benchmark (STOP_LA_BENCH=1, gpu_la_fast.py --bench).
LA_SCALE = env('STOP_LA_SCALE', 1.0)
if not (0.5 <= LA_SCALE <= 2.0): sys.exit('STOP_LA_SCALE out of range [0.5, 2.0]: %r' % LA_SCALE)
# ---- STOP_SHAPE=1: shape-prior rule (default off; see the docstring) ------------------------------------------------
SHAPE = int(env('STOP_SHAPE', 0))
KAPPA = env('STOP_KAPPA', 9.9) * LA_SCALE     # LA s per (M rows)^2 on the merged N: arm B 9.94, keys 25-29/31 9.62-10.02;
                                              # x STOP_LA_SCALE (slow_card.md: the shape rule needs the same card factor;
                                              # set ONE of STOP_LA_SCALE / a scaled STOP_KAPPA, not both)
SX = env('STOP_SX', 0.22)                     # excess per raw at the knee (M per M): 31/32 0.25-0.26, 31/33 0.21-0.23, 32/33 0.20-0.21
SX_LO, SX_HI = env('STOP_SX_LO', 0.12), env('STOP_SX_HI', 0.40)
SH_A, SH_B = env('STOP_SHAPE_A', 0.102), env('STOP_SHAPE_B', 0.44)
MINPTS = int(env('STOP_MINPTS', 1 if SHAPE else 2))

def shape(u): return (1.0 + u / SH_A) ** -SH_B

def shape_fit(pts):
    """pts {raw: (rows, excess, t)} of the MATRIX probes -> (xc, sx, N0, src)"""
    A = sorted(pts); n = len(A)
    if n >= 2:
        ma = sum(A) / n; me = sum(pts[a][1] for a in A) / n
        den = sum((a - ma) ** 2 for a in A); sx = sum((a - ma) * (pts[a][1] - me) for a in A) / den if den > 0 else SX
        src = 'fit'
        if not (SX_LO <= sx <= SX_HI): sx, src = SX, 'default(clamped)'
        xc = ma - me / sx
    else:
        sx, src = SX, 'default'; xc = A[0] - pts[A[0]][1] / sx
    if not (0 < xc <= A[-1] * 1.05): raise ValueError('crossover %.0f outside (0, 1.05 A_K]' % xc)
    us = [a / xc - 1 for a in A]
    if min(us) <= -0.5 * SH_A: raise ValueError('u_min %.3f below -a/2' % min(us))
    N0 = math.exp(sum(math.log(NFAC * pts[a][0]) - math.log(shape(u)) for a, u in zip(A, us)) / n)
    return xc, sx, N0, src

def shape_N(A, xc, N0): return N0 * shape(A / xc - 1)

def shape_C(A, xc, N0):   # downstream seconds that still depend on the stop: LA(N) + post(rows), rows = N / NFAC
    N = shape_N(A, xc, N0); return KAPPA * (N / 1e6) ** 2 + POST * N / NFAC

def shape_m(A, xc, N0):   # -dC/dA, seconds per raw relation (decreasing in A)
    N = shape_N(A, xc, N0); dN = N * (SH_B / SH_A) / (xc * (1.0 + (A / xc - 1) / SH_A))
    return (2.0 * KAPPA * N / 1e12 + POST / NFAC) * dN

def decide_shape(pts, rate, al, step):
    A = sorted(pts); ak, (rk, ek, tk) = A[-1], pts[A[-1]]
    xc, sx, N0, src = shape_fit(pts); nk = NFAC * rk
    base = 'shape=1 npts=%d A_K=%d N_K=%.0f LA_K=%.0f xc=%d sx_per_M=%.3f(%s) N0=%.0f' % (len(A), ak, nk, KAPPA * (nk / 1e6) ** 2, xc, sx, src, N0)
    if len(A) < MINPTS: return 'continue reason=need_%d_points %s' % (MINPTS, base)
    c = (1.0 + MARGIN) / rate; al = max(al, ak); tgt = c + TAX / step
    lo, hi = xc, max(al, ak) * 2.0
    if shape_m(lo, xc, N0) <= tgt: astar = lo
    elif shape_m(hi, xc, N0) >= tgt: astar = hi
    else:
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if shape_m(mid, xc, N0) > tgt: lo = mid
            else: hi = mid
        astar = (lo + hi) / 2.0
    save = shape_C(al, xc, N0) - shape_C(al + step, xc, N0); costn = step * c + TAX
    gain = shape_C(ak, xc, N0) - shape_C(al, xc, N0); tfin = F_INC + F_PURGE * tk
    kv = ('%s m_L_per_M=%.1f cost_per_M=%.1f A_star=%d N_star=%.0f A_L=%d step=%d save_next=%.0f cost_next=%.0f gain_landed=%.0f '
          't_final=%.0f rate=%.0f') % (base, shape_m(al, xc, N0) * 1e6, c * 1e6, astar, shape_N(astar, xc, N0), al, step, save, costn,
                                      gain, tfin, rate)
    if save > costn: return 'continue ' + kv
    if al > ak and gain > tfin: return 'stop_final ' + kv
    return 'stop_probe ' + kv

def la_secs(n, w=None):
    if w is None: w = NNZ_ROW * n
    return LA_SCALE * (n / ITDIV * (C_NNZ * w + DENSE * n))

def cost_of(rows):
    return la_secs(NFAC * rows) + POST * rows

def integ(s, lam, mid, a, b):
    if b <= a: return 0.0
    if lam <= 0: return s * (b - a)
    return s / lam * (math.exp(-lam * (a - mid)) - math.exp(-lam * (b - mid)))

def decide(path, rate, al, step):
    pts = {}; ex = {}
    for L in open(path):
        f = L.rstrip('\n').split('\t')
        if len(f) < 2 or not f[0].strip().isdigit() or not f[1].strip().isdigit(): continue
        pts[int(f[0])] = (int(f[1]), float(f[4]) if len(f) > 4 and f[4] else 0.0)
        try: ex[int(f[0])] = float(f[2])
        except (ValueError, IndexError): ex[int(f[0])] = 0.0
    A = sorted(pts)
    if not A: return 'continue reason=no_points'
    if SHAPE:
        try: return decide_shape({a: (pts[a][0], ex[a], pts[a][1]) for a in A}, rate, al, step)
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            sys.stderr.write('stop_cost.py: STOP_SHAPE fit failed (%s) -- secant rule for this call\n' % e)
    ak, (rk, tk) = A[-1], pts[A[-1]]
    nk = NFAC * rk; ck = cost_of(rk)
    base = 'npts=%d A_K=%d N_K=%.0f LA_K=%.0f' % (len(A), ak, nk, la_secs(nk))
    if len(A) < 2: return 'continue reason=need_2_points ' + base
    C = [cost_of(pts[a][0]) for a in A]
    s = (C[-2] - C[-1]) / (A[-1] - A[-2]); mid = (A[-1] + A[-2]) / 2.0
    lam, lsrc = LAM0, 'default'
    if len(A) >= 3:
        sp = (C[-3] - C[-2]) / (A[-2] - A[-3]); midp = (A[-2] + A[-3]) / 2.0
        if sp > 0 and s > 0 and mid > midp:
            lam = min(4 * LAM0, max(0.25 * LAM0, math.log(sp / s) / (mid - midp))); lsrc = 'fit'
    c = (1.0 + MARGIN) / rate
    al = max(al, ak)
    astar = (mid + math.log(s / c) / lam) if s > 0 else float(ak)
    save = integ(s, lam, mid, al, al + step); costn = step * c + TAX
    gain = integ(s, lam, mid, ak, al); tfin = F_INC + F_PURGE * tk
    kv = ('%s s_per_M=%.1f cost_per_M=%.1f lam_per_M=%.4f(%s) A_star=%d A_L=%d step=%d save_next=%.0f cost_next=%.0f '
          'gain_landed=%.0f t_final=%.0f rate=%.0f') % (base, s * 1e6, c * 1e6, lam * 1e6, lsrc, astar, al, step, save, costn,
                                                       gain, tfin, rate)
    if s > 0 and save > costn: return 'continue ' + kv
    if al > ak and gain > tfin: return 'stop_final ' + kv
    return 'stop_probe ' + kv

# Expected raw-relation crossover (first purge-only MATRIX, singleton excess >= 0) per lpb regime at mfb 62/64, lim
# 200M/450M, A 29, qmin 2.2M. The crossover is a property of the regime, not of the key (keys 25/26 at 31/32: 294.5M /
# 293.3M; keys 27/28/29 at 31/33: 365M / 363M / 368.5M; key 31 at 32/33: 441M). lim/lambda variants reach the same count
# sooner (same relations needed); mfb1 66 adds ~4% low-worth raw relations (crossover +4%, inside the FIRST_PROBE margin).
# 'measured' = GCP e2e probe brackets; 'model' = docs/research/agents_2026-10-25/regime_model.py (relation-worth model).
XOVER = {(31, 32): (293e6, 'measured'), (31, 33): (366e6, 'measured'), (32, 33): (441e6, 'measured'),
         (31, 34): (434e6, 'model'), (32, 34): (514e6, 'model'), (33, 33): (506e6, 'model'), (33, 34): (588e6, 'model')}
XOVER_RUNG = 1.2   # fallback for an unlisted regime: x1.2 raw per extra large-prime bit above 31/32 (measured 1.25 / 1.21)

def xover(lpb0, lpb1, frac=1.0):
    """frac x expected raw crossover; unlisted regimes extrapolate by XOVER_RUNG per bit above 31/32."""
    lpb0, lpb1 = int(lpb0), int(lpb1)
    if (lpb0, lpb1) in XOVER: base = XOVER[(lpb0, lpb1)][0]
    else: base = XOVER[(31, 32)][0] * XOVER_RUNG ** ((lpb0 - 31) + (lpb1 - 32))
    if not (0 < float(frac) <= 1.5): raise ValueError('frac must be in (0, 1.5]')
    return int(round(float(frac) * base / 1e6)) * 1000000

REPLAY_US = env('SPEC_REPLAY_US', 1.4) * 1e-6     # replay -t 20: 70 s on 50.8M purged rows (key1g)
SPEC_MARGIN = env('SPEC_MARGIN_S', 30.0)

def load_pts(path):
    pts = {}
    for L in open(path):
        f = L.rstrip('\n').split('\t')
        if len(f) < 2 or not f[0].strip().isdigit() or not f[1].strip().isdigit(): continue
        pts[int(f[0])] = (int(f[1]), float(f[4]) if len(f) > 4 and f[4] else 0.0)
    return pts

def rows_for_cost(c):
    lo, hi = 1.0, 1e10           # cost_of is increasing in rows: bisect cost_of(rows) = c
    for _ in range(200):
        mid = (lo + hi) / 2
        if cost_of(mid) < c: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def spec(path, al, act, ready, inprog):
    pts = load_pts(path); A = sorted(pts)
    if not A: return 'new reason=no_points'
    ak, (rk, tk) = A[-1], pts[A[-1]]
    al = max(al, ak)
    f, w, cal = NFAC, NNZ_ROW, 'model'
    if ready and ready[0] > 0 and ready[1] > 0: f, w, cal = ready[1] / ready[0], ready[2] / ready[1], 'ready'
    if act == 'stop_final' and len(A) >= 2 and al > ak:
        # the stop rule's own m(A): C(A_L) = C(A_K) - integral of m over [A_K, A_L] (same secant / lambda as decide)
        C = [cost_of(pts[a][0]) for a in A]
        s = (C[-2] - C[-1]) / (A[-1] - A[-2]); mid = (A[-1] + A[-2]) / 2.0; lam = LAM0
        if len(A) >= 3:
            sp = (C[-3] - C[-2]) / (A[-2] - A[-3]); midp = (A[-2] + A[-3]) / 2.0
            if sp > 0 and s > 0 and mid > midp: lam = min(4 * LAM0, max(0.25 * LAM0, math.log(sp / s) / (mid - midp)))
        gain = integ(s, lam, mid, ak, al) if s > 0 else 0.0
        rows_new = rows_for_cost(cost_of(rk) - gain); hand = F_INC + F_PURGE * tk
    else:
        rows_new = float(rk); hand = 0.0
    la = lambda rows: la_secs(f * rows, w * f * rows)
    c_new = hand + POST * rows_new + REPLAY_US * rows_new + la(rows_new)
    c_ready = la_secs(ready[1], ready[2]) if ready else float('inf')
    c_inp = (inprog[1] + la(inprog[0])) if inprog else float('inf')
    best, cb = 'new', c_new
    if c_ready < cb - SPEC_MARGIN: best, cb = 'use_ready', c_ready
    if c_inp < cb - (SPEC_MARGIN if best == 'new' else 0): best, cb = 'use_inprog', c_inp
    fmt = lambda x: 'na' if x == float('inf') else '%.0f' % x
    return ('%s C_ready=%s C_inprog=%s C_new=%s saving=%.0f act=%s A_K=%d A_L=%d rows_K=%d rows_new=%.0f handoff=%.0f '
            'merge_new=%.0f replay_new=%.0f LA_new=%.0f N_new=%.0f f=%.4f w=%.1f(%s)') % (
        best, fmt(c_ready), fmt(c_inp), fmt(c_new), c_new - cb, act, ak, al, rk, rows_new, hand, POST * rows_new,
        REPLAY_US * rows_new, la(rows_new), f * rows_new, f, w, cal)

if __name__ == '__main__':
    a = sys.argv[1:]
    if a and a[0] == 'decide' and len(a) == 5:
        print(decide(a[1], float(a[2]), int(float(a[3])), int(float(a[4]))))
    elif a and a[0] == 'spec' and len(a) == 9:
        rd = None if '-' in a[4:7] else tuple(map(float, a[4:7]))
        ip = None if '-' in a[7:9] else tuple(map(float, a[7:9]))
        print(spec(a[1], int(float(a[2])), a[3], rd, ip))
    elif a and a[0] == 'la' and len(a) in (2, 3):
        print('%.0f' % la_secs(float(a[1]), float(a[2]) if len(a) == 3 else None))
    elif a and a[0] == 'xover' and len(a) in (3, 4):
        print(xover(a[1], a[2], a[3] if len(a) == 4 else 1.0))
    else:
        sys.exit(__doc__)
