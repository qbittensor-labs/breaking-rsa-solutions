// CADO-NFS facul (ECM/P-1/P+1 strategy, LGPL, the same vendored tree as our other CADO binaries) as the
// primary 2LP splitter for cofac_cpu. One strategy object per thread and side (built lazily).
#include "cado.h"
#include <cstdint>
#include <memory>
#include "facul.hpp"
#include "facul_strategies.hpp"
#include "cxx_mpz.hpp"
extern "C" int facul_split2(uint64_t n, unsigned long fbb, int lpb, int side, uint64_t out[2])
{
    thread_local std::unique_ptr<facul_strategy_oneside> S[2];
    thread_local unsigned long sfbb[2] = {0, 0};
    thread_local int slpb[2] = {0, 0};
    side &= 1;
    if (!S[side] || sfbb[side] != fbb || slpb[side] != lpb) {
        /* as `testbench -strat`: mfb = 3*lpb keeps the default curve count sane for a 2LP cofactor */
        S[side] = std::make_unique<facul_strategy_oneside>(fbb, lpb, 3 * lpb, -1, 0);
        sfbb[side] = fbb; slpb[side] = lpb;
    }
    cxx_mpz N;
    mpz_set_ui(N, n);   /* unsigned long is 64-bit on LP64 */
    auto const res = facul(N, *S[side]);
    if (res.status == FACUL_NOT_SMOOTH) return 0;
    if (res.status != FACUL_SMOOTH) return -1;          /* MAYBE: caller falls back to rho */
    if (res.primes.size() != 2) return -1;
    for (int i = 0; i < 2; i++) {
        if (mpz_sizeinbase(res.primes[i], 2) > 64) return -1;
        out[i] = mpz_get_ui(res.primes[i]);
    }
    return 1;
}
