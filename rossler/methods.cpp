#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <cstdio>    // popen / pclose

// =========================================================================
// Rössler system parameters
// =========================================================================
constexpr double A = 0.2;
constexpr double B = 0.2;
constexpr double C = 5.7;

using State = std::array<double, 3>;

// =========================================================================
// Rössler derivatives
// dx/dt = -y - z
// dy/dt =  x + a*y
// dz/dt =  b + z*(x - c)
// =========================================================================
State rossler(const State& s) {
    double x=s[0], y=s[1], z=s[2];
    return { -y - z,
              x + A*y,
              B + z*(x - C) };
}

// =========================================================================
// Analytically derived second derivative
// d²x/dt² = -dy/dt - dz/dt
// d²y/dt² =  dx/dt + a*dy/dt
// d²z/dt² =  dz/dt*(x - c) + z*dx/dt
// =========================================================================
State rossler_second_deriv(double x, double /*y*/, double z,
                            double fx, double fy, double fz) {
    return { -fy - fz,
              fx + A*fy,
              fz*(x - C) + z*fx };
}

// =========================================================================
// Vector helpers
// =========================================================================
State add_scaled(const State& s, double h, const State& d) {
    return { s[0]+h*d[0], s[1]+h*d[1], s[2]+h*d[2] };
}
State add_scaled2(const State& s,
                  double a, const State& d1,
                  double b, const State& d2) {
    return { s[0]+a*d1[0]+b*d2[0],
             s[1]+a*d1[1]+b*d2[1],
             s[2]+a*d1[2]+b*d2[2] };
}

// =========================================================================
// Numerical methods
// =========================================================================

// Euler — O(h)
std::vector<State> euler_method(State init, double h, int steps) {
    std::vector<State> t(steps); t[0]=init;
    for(int i=0;i<steps-1;++i) t[i+1]=add_scaled(t[i],h,rossler(t[i]));
    return t;
}

// Heun — O(h²)
std::vector<State> heun_method(State init, double h, int steps) {
    std::vector<State> t(steps); t[0]=init;
    for(int i=0;i<steps-1;++i){
        State k1=rossler(t[i]), k2=rossler(add_scaled(t[i],h,k1));
        t[i+1]=add_scaled2(t[i],h/2,k1,h/2,k2);
    }
    return t;
}

// Taylor — O(h²), analytic second derivative
std::vector<State> taylor_method(State init, double h, int steps) {
    std::vector<State> t(steps); t[0]=init;
    for(int i=0;i<steps-1;++i){
        double x=t[i][0],y=t[i][1],z=t[i][2];
        State f =rossler(t[i]);
        State f2=rossler_second_deriv(x,y,z,f[0],f[1],f[2]);
        t[i+1]=add_scaled2(t[i],h,f,h*h/2,f2);
    }
    return t;
}

// Picard — trapezoidal fixed-point iterations
State picard_step(const State& s, double h, int iters=3){
    State x0=s, xc=s;
    for(int k=0;k<iters;++k)
        xc=add_scaled2(x0,h/2,rossler(x0),h/2,rossler(xc));
    return xc;
}
std::vector<State> picard_method(State init, double h, int steps, int iters=3){
    std::vector<State> t(steps); t[0]=init;
    for(int i=0;i<steps-1;++i) t[i+1]=picard_step(t[i],h,iters);
    return t;
}

// RK4 — O(h⁴)
std::vector<State> rk4_method(State init, double h, int steps){
    std::vector<State> t(steps); t[0]=init;
    for(int i=0;i<steps-1;++i){
        State k1=rossler(t[i]);
        State k2=rossler(add_scaled(t[i],h/2,k1));
        State k3=rossler(add_scaled(t[i],h/2,k2));
        State k4=rossler(add_scaled(t[i],h,  k3));
        t[i+1]={ t[i][0]+(h/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0]),
                 t[i][1]+(h/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1]),
                 t[i][2]+(h/6)*(k1[2]+2*k2[2]+2*k3[2]+k4[2]) };
    }
    return t;
}

// =========================================================================
// CSV export — columns: step x y z  (space separated)
// =========================================================================
void write_csv(const std::string& fn, const std::vector<State>& traj){
    std::ofstream f(fn);
    for(int i=0;i<(int)traj.size();++i)
        f << i <<" "<< traj[i][0] <<" "<< traj[i][1] <<" "<< traj[i][2] <<"\n";
}

// =========================================================================
// Gnuplot helper
// =========================================================================
void gnuplot(const std::string& script){
    FILE* gp = popen("gnuplot", "w");
    if(!gp){ std::cerr << "gnuplot not found. Install: brew install gnuplot\n"; return; }
    fputs(script.c_str(), gp);
    pclose(gp);
}

// =========================================================================
// Plot 1 — x(t) time-series comparison
// =========================================================================
void plot_xt(){
    gnuplot(R"GP(
set terminal pngcairo size 1300,600 enhanced font 'Arial,11'
set output 'rossler_xt.png'
set title  'Rossler System - x(t) Comparison'
set xlabel 'Time Step'
set ylabel 'x(t)'
set key top right
set grid lc rgb '#cccccc'

plot 'euler.csv'  u 1:2 w l lc rgb '#E24B4A' lw 1.0        t 'Euler',           \
     'heun.csv'   u 1:2 w l lc rgb '#EF9F27' lw 1.0        t 'Heun',             \
     'taylor.csv' u 1:2 w l lc rgb '#7F77DD' lw 1.0 dt 2   t 'Taylor',           \
     'picard.csv' u 1:2 w l lc rgb '#E87DB0' lw 1.0 dt 3   t 'Picard (3 iters)', \
     'rk4.csv'    u 1:2 w l lc rgb '#1D9E75' lw 1.4        t 'RK4'
)GP");
    std::cout << "Saved rossler_xt.png\n";
}

// =========================================================================
// Plot 2 — x-y phase plane (Rössler attractor), all methods
// =========================================================================
void plot_xy(){
    gnuplot(R"GP(
set terminal pngcairo size 1300,700 enhanced font 'Arial,11'
set output 'rossler_xy.png'
set title  'Rossler Attractor - x-y Phase Plane'
set xlabel 'x'
set ylabel 'y'
set key top right
set grid lc rgb '#cccccc'

plot 'rk4.csv'    u 2:3 w l lc rgb '#1D9E75' lw 0.8        t 'RK4',             \
     'euler.csv'  u 2:3 w l lc rgb '#E24B4A' lw 0.6        t 'Euler',           \
     'heun.csv'   u 2:3 w l lc rgb '#EF9F27' lw 0.6        t 'Heun',            \
     'taylor.csv' u 2:3 w l lc rgb '#7F77DD' lw 0.6 dt 2   t 'Taylor',          \
     'picard.csv' u 2:3 w l lc rgb '#E87DB0' lw 0.6 dt 3   t 'Picard (3 iters)'
)GP");
    std::cout << "Saved rossler_xy.png\n";
}

// =========================================================================
// Plot 3 — individual 3-D attractors (2x3 multiplot)
// =========================================================================
void plot_3d_individual(){
    gnuplot(R"GP(
set terminal pngcairo size 1400,1000 enhanced font 'Arial,10'
set output 'rossler_3d_individual.png'
set multiplot layout 2,3 \
    title 'Rossler Attractor - All Methods (3D Phase Space)' font 'Arial,13'

set xlabel 'X' font 'Arial,8'
set ylabel 'Y' font 'Arial,8'
set zlabel 'Z' font 'Arial,8'
set ticslevel 0
unset key

set title 'Euler'
splot 'euler.csv'  u 2:3:4 w l lc rgb '#E24B4A' lw 0.4 notitle

set title 'Heun'
splot 'heun.csv'   u 2:3:4 w l lc rgb '#EF9F27' lw 0.4 notitle

set title 'Taylor'
splot 'taylor.csv' u 2:3:4 w l lc rgb '#7F77DD' lw 0.4 notitle

set title 'Picard (3 iters)'
splot 'picard.csv' u 2:3:4 w l lc rgb '#E87DB0' lw 0.4 notitle

set title 'RK4'
splot 'rk4.csv'    u 2:3:4 w l lc rgb '#1D9E75' lw 0.4 notitle

unset multiplot
)GP");
    std::cout << "Saved rossler_3d_individual.png\n";
}

// =========================================================================
// Plot 4 — all methods overlaid in one 3-D view
// =========================================================================
void plot_3d_overlaid(){
    gnuplot(R"GP(
set terminal pngcairo size 1000,800 enhanced font 'Arial,11'
set output 'rossler_3d_overlaid.png'
set title  'Rossler Attractor - All Methods Overlaid'
set xlabel 'X'
set ylabel 'Y'
set zlabel 'Z'
set ticslevel 0
set key top right font 'Arial,9'

splot 'euler.csv'  u 2:3:4 w l lc rgb '#E24B4A' lw 0.4 t 'Euler',            \
      'heun.csv'   u 2:3:4 w l lc rgb '#EF9F27' lw 0.4 t 'Heun',              \
      'taylor.csv' u 2:3:4 w l lc rgb '#7F77DD' lw 0.4 t 'Taylor',            \
      'picard.csv' u 2:3:4 w l lc rgb '#E87DB0' lw 0.4 t 'Picard (3 iters)',  \
      'rk4.csv'    u 2:3:4 w l lc rgb '#1D9E75' lw 0.6 t 'RK4'
)GP");
    std::cout << "Saved rossler_3d_overlaid.png\n";
}

// =========================================================================
// main
// =========================================================================
int main(){
    constexpr double h     = 0.01;
    constexpr int    steps = 10000;
    const State      initial = {1.0, 1.0, 1.0};

    // Run simulations
    std::cout << "Running simulations...\n";
    auto euler_traj  = euler_method (initial, h, steps);
    auto heun_traj   = heun_method  (initial, h, steps);
    auto taylor_traj = taylor_method(initial, h, steps);
    auto picard_traj = picard_method(initial, h, steps, 3);
    auto rk4_traj    = rk4_method   (initial, h, steps);

    // Write CSVs
    write_csv("euler.csv",  euler_traj);
    write_csv("heun.csv",   heun_traj);
    write_csv("taylor.csv", taylor_traj);
    write_csv("picard.csv", picard_traj);
    write_csv("rk4.csv",    rk4_traj);
    std::cout << "Data written.\n\n";

    // Console summary
    std::cout << "Final state at step " << steps-1 << ":\n";
    auto pf=[&](const char* name, const std::vector<State>& t){
        std::cout << "  " << name
                  << "  x=" << t.back()[0]
                  << "  y=" << t.back()[1]
                  << "  z=" << t.back()[2] << "\n";
    };
    pf("Euler  ", euler_traj);
    pf("Heun   ", heun_traj);
    pf("Taylor ", taylor_traj);
    pf("Picard ", picard_traj);
    pf("RK4    ", rk4_traj);

    // Generate plots
    std::cout << "\nGenerating plots via gnuplot...\n";
    plot_xt();
    plot_xy();
    plot_3d_individual();
    plot_3d_overlaid();

    std::cout << "\nDone! 4 PNG files saved:\n"
              << "  rossler_xt.png\n"
              << "  rossler_xy.png\n"
              << "  rossler_3d_individual.png\n"
              << "  rossler_3d_overlaid.png\n";
    return 0;
}
