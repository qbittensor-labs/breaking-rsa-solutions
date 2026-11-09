#!/usr/bin/env bash
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

# ps_overlap.sh MODE WORK -- the CPU half of polyselect (sopt re-optimization -> topk -> polyselect_ropt -> best),
# split so most of it runs while gps1 is still searching on the GPU (PS_OVERLAP=1 in factor_c169.sh).
#   seq : the sequential path, line for line what factor_c169.sh does with PS_OVERLAP=0 (s72's reference)
#   a   : on gps1 batch 1: snapshot WORK/raw.p -> WORK/ov/raw1.p, sopt its SOPT_K best, topk ROPT_K, ropt them.
#         Pure speculation: it only fills the memo (WORK/ov/sA/*.out, WORK/ov/roptA.out).
#   b   : after gps1 stops (mode a finished): sopt only the soptin-list entries of the FINAL raw.p that are not in
#         the memo, append the sopt results in the order the sequential path appends them, topk ROPT_K, ropt only
#         the top-ROPT_K polys not in the memo, merge, `best`. Same raw.p -> same poly.cado as seq, provided sopt
#         and polyselect_ropt are deterministic per input polynomial (s72 checks). Any failure -> exit 2 with
#         WORK/raw.p restored, and the caller runs seq.
#   r   : (PS_MEMO_ROLL=1, PS_ADAPT extend batches; third argument = the batch number k) rolling memo: snapshot
#         WORK/raw.p -> WORK/ov/R<k>/raw.p, sopt only its soptin entries missing from every earlier memo, topk ROPT_K,
#         ropt only the top-ROPT_K polys missing from every earlier ropt memo. Pure speculation like a: it only adds
#         memo entries (ov/R<k>/s/*.out once ov/R<k>/sopt.done exists, ov/R<k>/ropt.out once ov/R<k>/done exists),
#         which b then reuses. A memo entry is keyed by the exact input polynomial, so b's pick does not depend on
#         which passes ran (same raw.p -> same poly.cado; E ties at the maximum aside, where seq itself picks by
#         thread completion order).
# Env: N CF NCPU ALIM RLIM (required); SOPT_EFF SOPT_K SOPT_MAX SOPT_BIN ROPT_K ROPT_EFF ROPT_MAX (factor_c169 defaults).
# Writes WORK/raw.p (sopt blocks appended), top.poly, ropt.out, poly.cado; log lines on stdout.
set -u
MODE=$1; WORK=$2; RTAG=${3:-}; HERE=${HERE:-$(cd "$(dirname "$0")" && pwd)}
: "${N:?}" "${CF:?}" "${NCPU:?}" "${ALIM:?}" "${RLIM:?}"
SOPT_EFF=${SOPT_EFF:-9}; SOPT_K=${SOPT_K:-500}; SOPT_MAX=${SOPT_MAX:-120}; SOPT_BIN=${SOPT_BIN:-$CF/sopt}
ROPT_K=${ROPT_K:-20}; ROPT_EFF=${ROPT_EFF:-6}; ROPT_MAX=${ROPT_MAX:-420}
# the result-relevant options ropt_run passes (keep in sync with it): ovroptmiss/ovroptmerge only reuse memo
# outputs printed with exactly these options (memo v2, review5 s6.1)
export PS_ROPT_OPTS="-ropteffort $ROPT_EFF -area 2.4e16 -Bf $ALIM -Bg $RLIM"
P="$HERE/polysel160.py"; OV="$WORK/ov"; _t0=$(date +%s)
say(){ echo "[ps_overlap $MODE $(( $(date +%s)-_t0 ))s] $*"; }
sopt_on(){ [ "$SOPT_EFF" -gt 0 ] 2>/dev/null && [ -x "$SOPT_BIN" ]; }
sopt_parts(){   # run sopt on every $1/in*.poly in parallel (as factor_c169.sh does)
  local _f _sp=()
  for _f in "$1"/in*.poly; do [ -e "$_f" ] || continue
    timeout --foreground "$SOPT_MAX" "$SOPT_BIN" -inputpolys "$_f" -sopteffort "$SOPT_EFF" \
        > "${_f%.poly}.out" 2> "${_f%.poly}.err" & _sp+=($!)
  done
  [ "${#_sp[@]}" -gt 0 ] && wait "${_sp[@]}"; return 0; }
# memo sources: pass a + every completed rolling pass (sopt parts only once that pass's sopt step finished; ropt outputs
# only from a pass that exited cleanly -- an incomplete ropt file is never reused by memo v2 either way)
memo_srcs(){
  SDIRS=("$OV/sA"); ROUTS=("$OV/roptA.out"); local _d
  for _d in "$OV"/R*; do [ -d "$_d" ] || continue
    [ -f "$_d/sopt.done" ] && SDIRS+=("$_d/s"); [ -f "$_d/done" ] && ROUTS+=("$_d/ropt.out"); done; return 0; }
ropt_run(){     # ropt_run IN OUT
  ( cd "$WORK" && timeout --foreground "$ROPT_MAX" "$CF/polyselect_ropt" -inputpolys "$1" -ropteffort "$ROPT_EFF" \
        -t "$NCPU" -area 2.4e16 -Bf "$ALIM" -Bg "$RLIM" > "$2" 2> "$2.err" ); }

case "$MODE" in
seq)
  if sopt_on; then
    _st=$(date +%s); rm -rf "$WORK/sopt"; mkdir -p "$WORK/sopt"
    if python3 "$P" soptin "$WORK/raw.p" "$SOPT_K" "$WORK/sopt/in" "$NCPU" > "$WORK/sopt/in.log" 2>&1; then
      sopt_parts "$WORK/sopt"
      if python3 "$P" soptout "$WORK/raw.p" "$WORK/sopt"/in*.out > "$WORK/sopt/out.log" 2>&1; then
        say "sopt effort $SOPT_EFF: $(cat "$WORK/sopt/in.log"); $(cat "$WORK/sopt/out.log") in $(( $(date +%s)-_st ))s"
      else say "WARNING: sopt produced nothing usable"; fi
    else say "WARNING: sopt input failed"; fi
  fi
  python3 "$P" topk "$WORK/raw.p" "$ROPT_K" "$WORK/top.poly" || exit 3
  ropt_run "$WORK/top.poly" "$WORK/ropt.out"; say "ropt rc=$? ($(grep -c 'MurphyE' "$WORK/ropt.out") MurphyE lines)"
  python3 "$P" best "$WORK/ropt.out" "$N" "$WORK/poly.cado" || exit 3 ;;
a)
  rm -rf "$OV"; mkdir -p "$OV/sA" "$OV/sB"; cp "$WORK/raw.p" "$OV/raw1.p"
  say "batch-1 snapshot: $(grep -c '^# Size-optimized' "$OV/raw1.p") candidates"
  if sopt_on && python3 "$P" soptin "$OV/raw1.p" "$SOPT_K" "$OV/sA/in" "$NCPU" > "$OV/sA/in.log" 2>&1; then
    sopt_parts "$OV/sA"; python3 "$P" soptout "$OV/raw1.p" "$OV/sA"/in*.out > "$OV/sA/out.log" 2>&1
    say "sopt: $(cat "$OV/sA/in.log"); $(cat "$OV/sA/out.log")"
  fi
  python3 "$P" topk "$OV/raw1.p" "$ROPT_K" "$OV/topA.poly" > /dev/null || exit 3
  ropt_run "$OV/topA.poly" "$OV/roptA.out"; say "ropt top-$ROPT_K rc=$? ($(grep -c 'MurphyE' "$OV/roptA.out") MurphyE lines)"
  touch "$OV/A.done" ;;
b)
  [ -f "$OV/A.done" ] || { say "no completed batch-1 pass"; exit 2; }
  cp "$WORK/raw.p" "$OV/raw.orig"; fail(){ say "FALLBACK: $*"; cp "$OV/raw.orig" "$WORK/raw.p"; exit 2; }
  if sopt_on; then
    _st=$(date +%s); rm -rf "$OV/sB" "$OV/ord"; mkdir -p "$OV/sB" "$OV/ord"
    memo_srcs; say "memo: ${#SDIRS[@]} sopt source(s), ${#ROUTS[@]} ropt output(s)"
    if _m=$(python3 "$P" ovsoptmiss "$WORK/raw.p" "$SOPT_K" "$OV/sB/in" "$NCPU" "${SDIRS[@]}" "$OV/sB"); then
      sopt_parts "$OV/sB"
      # the sequential path's soptout reads "$WORK/sopt"/in*.out in THIS shell's glob order: reproduce it
      _np=$(python3 "$P" ovsoptnparts "$WORK/raw.p" "$SOPT_K" "$NCPU") || fail "ovsoptnparts"
      for _i in $(seq 0 $((_np-1))); do : > "$OV/ord/in$_i.out"; done
      _ord=$(for _f in "$OV/ord"/in*.out; do _b=${_f##*/in}; printf '%s,' "${_b%.out}"; done)
      if _o=$(python3 "$P" ovsoptapply "$WORK/raw.p" "$SOPT_K" "$NCPU" "$_ord" "${SDIRS[@]}" "$OV/sB"); then
        say "sopt $_m re-run; $_o in $(( $(date +%s)-_st ))s"
      else say "WARNING: sopt produced nothing usable ($_o)"; fi
    else say "WARNING: sopt input failed ($_m)"; fi
  fi
  python3 "$P" topk "$WORK/raw.p" "$ROPT_K" "$WORK/top.poly" > /dev/null || fail "topk"
  memo_srcs
  _r=$(python3 "$P" ovroptmiss "$WORK/top.poly" "$OV/topB.poly" "${ROUTS[@]}") || fail "ovroptmiss"
  rm -f "$OV/roptB.out"; _st=$(date +%s)
  case "$_r" in "miss 0 "*) ;; *) ropt_run "$OV/topB.poly" "$OV/roptB.out" ;; esac
  say "ropt $_r re-run in $(( $(date +%s)-_st ))s"
  python3 "$P" ovroptmerge "$WORK/top.poly" "$WORK/ropt.out" "${ROUTS[@]}" "$OV/roptB.out" > /dev/null || fail "ovroptmerge"
  python3 "$P" best "$WORK/ropt.out" "$N" "$WORK/poly.cado" || fail "best" ;;
r)
  [ -f "$OV/A.done" ] || { say "no completed pass a"; exit 2; }
  [ -n "$RTAG" ] || { say "no batch number"; exit 64; }
  R="$OV/R$RTAG"; rm -rf "${R:?}"; mkdir -p "$R/s"; cp "$WORK/raw.p" "$R/raw.p"; memo_srcs
  say "batch-$RTAG snapshot: $(grep -c '^# Size-optimized' "$R/raw.p") candidates; memo ${#SDIRS[@]} sopt / ${#ROUTS[@]} ropt source(s)"
  if sopt_on && _m=$(python3 "$P" ovsoptmiss "$R/raw.p" "$SOPT_K" "$R/s/in" "$NCPU" "${SDIRS[@]}"); then
    sopt_parts "$R/s"; touch "$R/sopt.done"
    # the snapshot's sopt-appended raw.p (memo + this pass): only to choose which polys to ropt, part order is immaterial
    _np=$(python3 "$P" ovsoptnparts "$R/raw.p" "$SOPT_K" "$NCPU") || exit 3
    _o=$(python3 "$P" ovsoptapply "$R/raw.p" "$SOPT_K" "$NCPU" "$(seq -s, 0 $((_np-1)))" "${SDIRS[@]}" "$R/s") || _o="nothing usable"
    say "sopt $_m re-run; $_o"
  fi
  python3 "$P" topk "$R/raw.p" "$ROPT_K" "$R/top.poly" > /dev/null || exit 3
  _r=$(python3 "$P" ovroptmiss "$R/top.poly" "$R/topmiss.poly" "${ROUTS[@]}") || exit 3
  case "$_r" in "miss 0 "*) say "ropt $_r: nothing to run"; touch "$R/done"; exit 0 ;; esac
  ropt_run "$R/topmiss.poly" "$R/ropt.out"; _rc=$?
  say "ropt $_r re-run rc=$_rc ($(grep -c 'MurphyE' "$R/ropt.out" 2>/dev/null) MurphyE lines)"
  [ "$_rc" = 0 ] && touch "$R/done" ;;
*) echo "usage: ps_overlap.sh seq|a|b|r WORK [k]" >&2; exit 64 ;;
esac
