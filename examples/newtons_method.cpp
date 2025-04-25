// examples/newton_8bit.cpp
// -----------------------------------------------------------
// 8‑bit LoFloat demo (no plotting):
//   Solve  cos(x) – x = 0  with Newton’s method
//   x_{n+1} = x_n − f(x_n)/f'(x_n)
//   Forward iterations:   max_iter = 20
//   Dumps CSV with columns: k, x_rne, x_sr, x_exact
// -----------------------------------------------------------
#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>
#include "lo_float.h"

using namespace lo_float;

// ---------- 8‑bit parameter packs ------------------------------------------
using rne8 = float8_ieee_p<4, Rounding_Mode::RoundToNearestEven>;
using sr8  = float8_ieee_p<5, Rounding_Mode::StochasticRounding, 4>;

// ---------- one Newton step -------------------------------------------------
template<typename F>
inline void newton_step(F& x)
{
    const double xd  = static_cast<double>(x);
    const double fx  = 1.0/std::sin(xd) - xd;           // f(x)
    const double dfx = -std::cos(xd)/(std::sin(xd)*std::sin(xd)) - 1.0;         // f'(x)
    x = static_cast<F>(xd - fx / dfx);              // x ← x − f/f'
}

int main()
{
    constexpr int    max_iter = 20;
    constexpr double root_exact = 0.7390851332151607;   // high‑precision reference

    std::vector<int>    k;
    std::vector<double> x_rne, x_sr, x_exact;

    rne8 x_r = rne8(1.0);   // initial guess
    sr8  x_s = sr8(1.0);

    for (int i = 0; i <= max_iter; ++i)
    {
        k.push_back(i);
        x_rne  .push_back(static_cast<double>(x_r));
        x_sr   .push_back(static_cast<double>(x_s));
        x_exact.push_back(root_exact);

        newton_step(x_r);
        newton_step(x_s);
    }

    // Write CSV with the iterate history
    std::ofstream out("newton_8bit.csv");
    out << "k,x_rne,x_sr,x_exact\n";
    for (std::size_t i = 0; i < k.size(); ++i)
        out << k[i] << ","
            << x_rne[i] << ","
            << x_sr[i] << ","
            << x_exact[i] << "\n";

    std::cout << "✅ Wrote newton_8bit.csv with iterate history for both rounding modes\n";
    return 0;
}
