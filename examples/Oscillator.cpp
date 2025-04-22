// examples/oscillator_8bit.cpp
// -----------------------------------------------------------
// 8‑bit LoFloat demo (no plotting):
//   u'(t) = v(t),  v'(t) = −u(t),  u(0)=1, v(0)=0
//   Forward Euler, h = 0.05, T = 20
//   Dumps CSV with columns: t, u_rne, u_sr, u_exact
// -----------------------------------------------------------
#include <vector>
#include <fstream>
#include <cmath>
#include "lo_float.h"

using namespace lo_float;

// ---------- 8‑bit parameter packs ------------------------------------------


using rne8 = float8_ieee_p<5, RoundToNearestEven>;
using sr8  = float8_ieee_p<5, StochasticRounding, 3>;

// ---------- one Euler step --------------------------------------------------
template<typename F>
inline void euler_step(F& u, F& v, double h)
{
    const F du = v;
    const F dv = -u;
    u = static_cast<F>((double)u + h * (double)du);
    v = static_cast<F>((double)v + h * (double)dv);
}

int main()
{
    constexpr double T = 20.0, h = 0.0001;
    int N = static_cast<int>(T / h);
    std::cout << "N = " << N << "\n";

    N = 1;

    std::vector<double> t, u_rne, v_rne, u_sr, v_sr, u_exact, v_exact;
    t.reserve(N + 1);

    rne8 u_r = rne8(1.0), v_r = rne8(0.0);
    sr8  u_s = sr8(1.0), v_s = sr8(-0.0);

    std::cout << "u_s = " << static_cast<double>(u_s) << "\n";
    std::cout << "v_s = " << static_cast<double>(v_s) << "\n";

    for (int k = 0; k <= N; ++k) {
        double tk = k * h;
        t.push_back(tk);
        u_rne.push_back(static_cast<double>(u_r));
        v_rne.push_back(static_cast<double>(v_r));
        u_sr .push_back(static_cast<double>(u_s));
        v_sr .push_back(static_cast<double>(v_s));
        u_exact.push_back(std::cos(tk));
        v_exact.push_back(-std::sin(tk));

        std::cout << "iter k = " << k << "\n";
        std::cout << "u_r = " << static_cast<double>(u_r) << "\n";
        std::cout << "v_r = " << static_cast<double>(v_r) << "\n";
        std::cout << "u_s = " << static_cast<double>(u_s) << "\n";
        std::cout << "v_s = " << static_cast<double>(v_s) << "\n";

        euler_step(u_r, v_r, h);
        euler_step(u_s, v_s, h);

    }

    // Write CSV with u and v for both modes
    std::ofstream out("oscillator_8bit.csv");
    out << "t,u_rne,v_rne,u_sr,v_sr,u_exact,v_exact\n";
    for (std::size_t i = 0; i < t.size(); ++i)
        out << t[i] << ","
            << u_rne[i] << "," << v_rne[i] << ","
            << u_sr[i]  << "," << v_sr[i]  << ","
            << u_exact[i] << "," << v_exact[i] << "\n";

    std::cout << "✅ Wrote oscillator_8bit.csv with u(t) and v(t) for both rounding modes\n";
    return 0;
}