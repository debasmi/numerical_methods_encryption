#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <cstdio>    // popen / pclose

// =========================================================================
// Lorenz system parameters
// =========================================================================
constexpr double SIGMA = 10.0;
constexpr double RHO   = 28.0;
constexpr double BETA  = 8.0 / 3.0;

using State = std::array<double, 3>;

// =========================================================================
// Lorenz derivatives
// =========================================================================
State lorenz(const State& s) {
    double x=s[0], y=s[1], z=s[2];
    return { SIGMA*(y-x), x*(RHO-z)-y, x*y-BETA*z };
}
State lorenz_second_deriv(double x, double y, double z,
                           double fx, double fy, double fz) {
    return { SIGMA*(fy-fx), fx*(RHO-z)-fz*x-fy, fy*x+fx*y-BETA*fz };
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
std::vector<State> euler_method(State init, double h, int steps) {
    std::vector<State> t(steps); t[0]=init;
    for(int i=0;i<steps-1;++i) t[i+1]=add_scaled(t[i],h,lorenz(t[i]));
    return t;
}
std::vector<State> heun_method(State init, double h, int steps) {
    std::vector<State> t(steps); t[0]=init;
    for(int i=0;i<steps-1;++i){
        State k1=lorenz(t[i]), k2=lorenz(add_scaled(t[i],h,k1));
        t[i+1]=add_scaled2(t[i],h/2,k1,h/2,k2);
    }
    return t;
}
std::vector<State> taylor_method(State init, double h, int steps) {
    std::vector<State> t(steps); t[0]=init;
    for(int i=0;i<steps-1;++i){
        double x=t[i][0],y=t[i][1],z=t[i][2];
        State f=lorenz(t[i]);
        State f2=lorenz_second_deriv(x,y,z,f[0],f[1],f[2]);
        t[i+1]=add_scaled2(t[i],h,f,h*h/2,f2);
    }
    return t;
}
State picard_step(const State& s, double h, int iters=3){
    State x0=s, xc=s;
    for(int k=0;k<iters;++k)
        xc=add_scaled2(x0,h/2,lorenz(x0),h/2,lorenz(xc));
    return xc;
}
std::vector<State> picard_method(State init, double h, int steps, int iters=3){
    std::vector<State> t(steps); t[0]=init;
    for(int i=0;i<steps-1;++i) t[i+1]=picard_step(t[i],h,iters);
    return t;
}
std::vector<State> rk4_method(State init, double h, int steps){
    std::vector<State> t(steps); t[0]=init;
    for(int i=0;i<steps-1;++i){
        State k1=lorenz(t[i]);
        State k2=lorenz(add_scaled(t[i],h/2,k1));
        State k3=lorenz(add_scaled(t[i],h/2,k2));
        State k4=lorenz(add_scaled(t[i],h,  k3));
        t[i+1]={ t[i][0]+(h/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0]),
                 t[i][1]+(h/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1]),
                 t[i][2]+(h/6)*(k1[2]+2*k2[2]+2*k3[2]+k4[2]) };
    }
    return t;
}

// =========================================================================
// CSV export  (gnuplot data source — space separated, 1-indexed cols)
// cols: step(1)  x(2)  y(3)  z(4)
// =========================================================================
void write_csv(const std::string& fn, const std::vector<State>& traj){
    std::ofstream f(fn);
    for(int i=0;i<(int)traj.size();++i)
        f << i <<" "<< traj[i][0] <<" "<< traj[i][1] <<" "<< traj[i][2] <<"\n";
}

// =========================================================================
// Gnuplot helper — pipes a script string directly to gnuplot
// =========================================================================
void gnuplot(const std::string& script){
    FILE* gp = popen("gnuplot", "w");
    if(!gp){
        std::cerr << "ERROR: gnuplot not found.\n"
                  << "Install it with:  brew install gnuplot\n";
        return;
    }
    fputs(script.c_str(), gp);
    pclose(gp);
}

// =========================================================================
// Plot 1 — x(t) time-series comparison
// =========================================================================
void plot_xt(){
    gnuplot(R"GP(
set terminal pngcairo size 1200,600 enhanced font 'Arial,11'
set output 'comparison_x.png'
set title  'Lorenz System — Numerical Method Comparison'
set xlabel 'Time Step'
set ylabel 'x(t)'
set key top right
set grid lc rgb '#cccccc'

plot 'euler.csv'  u 1:2 w l lc rgb '#E24B4A' lw 1.0        t 'Euler',          \
     'heun.csv'   u 1:2 w l lc rgb '#EF9F27' lw 1.0        t 'Heun',            \
     'taylor.csv' u 1:2 w l lc rgb '#7F77DD' lw 1.0 dt 2   t 'Taylor',          \
     'picard.csv' u 1:2 w l lc rgb '#E87DB0' lw 1.0 dt 3   t 'Picard (3 iters)',\
     'rk4.csv'    u 1:2 w l lc rgb '#1D9E75' lw 1.4        t 'RK4'
)GP");
    std::cout << "Saved comparison_x.png\n";
}

// =========================================================================
// Plot 2 — individual 3-D attractors (2x3 multiplot grid)
// =========================================================================
void plot_individual(){
    gnuplot(R"GP(
set terminal pngcairo size 1400,1000 enhanced font 'Arial,10'
set output 'attractors_individual.png'
set multiplot layout 2,3 \
    title 'Lorenz Attractor — All Methods (3D Phase Space)' font 'Arial,13'

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
    std::cout << "Saved attractors_individual.png\n";
}

// =========================================================================
// Plot 3 — all methods overlaid in one 3-D view
// =========================================================================
void plot_overlaid(){
    gnuplot(R"GP(
set terminal pngcairo size 1000,800 enhanced font 'Arial,11'
set output 'attractors_overlaid.png'
set title  'Lorenz Attractor — All Methods Overlaid'
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
    std::cout << "Saved attractors_overlaid.png\n";
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

    // Write data
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
    plot_individual();
    plot_overlaid();
    std::cout << "\nDone! 3 PNG files saved in the current directory:\n"
              << "  comparison_x.png\n"
              << "  attractors_individual.png\n"
              << "  attractors_overlaid.png\n";
    return 0;
}
