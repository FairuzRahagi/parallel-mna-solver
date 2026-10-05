#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

#include "circuit.hpp"
#include "mna.hpp"
#include "solver.hpp"

// Simple series-resistor ladder (same topology as generate_testcases.cpp).
static Circuit makeLadder(int n_stages) {
    Circuit c;
    c.addVoltageSource(1, 0, 5.0);
    for (int i = 1; i < n_stages; ++i)
        c.addResistor(i, i + 1, 1.0);
    c.addResistor(n_stages, 0, 1.0);
    return c;
}

// Run solver callable REPS times, return minimum time in ms.
// The callable is expected to supply fresh copies of A and b on each invocation.
template<typename F>
static double minTime(F&& fn, int reps, SolverResult* out = nullptr) {
    double best = std::numeric_limits<double>::max();
    for (int r = 0; r < reps; ++r) {
        SolverResult res = fn();
        if (res.time_ms < best) {
            best = res.time_ms;
            if (out) *out = res;
        }
    }
    return best;
}

// Safely compute speedup, returns 0 if denominator <= 0.
static double speedup(double t_num, double t_den) {
    return (t_den > 0) ? t_num / t_den : 0.0;
}

int main(int argc, char* argv[]) {
    constexpr int REPS = 5;

    // Parse optional --size N argument (single-size mode for quick checks).
    std::vector<int> SIZES = {50, 100, 200, 300, 500, 700, 1000};
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--size") {
            int n = std::atoi(argv[i + 1]);
            if (n > 0) SIZES = {n};
        }
    }

    const int max_threads = omp_get_max_threads();
    const std::vector<int> thread_counts = {1, 2, 4, max_threads};

    SequentialSolver seq;
    OpenMPSolver     omp_solver;
    OpenCLSolver     ocl;

    // --- OpenCL init ---
    bool ocl_ok = ocl.initialize();
    std::cout << "\n=== MNA LU Benchmark ===\n";
    std::cout << "omp_get_wtick()    : " << omp_get_wtick() << " s\n";
    std::cout << "OpenMP max threads : " << max_threads << "\n";
    if (ocl_ok)
        std::cout << "OpenCL device      : " << ocl.getDeviceInfo() << "\n";
    else
        std::cout << "OpenCL             : UNAVAILABLE — " << ocl.error_msg << "\n";
    std::cout << "Runs per config    : " << REPS << " (minimum time reported)\n\n";

    // --- CSV ---
    std::ofstream csv("results_openmp.csv");
    csv << "matrix_size,seq_ms,"
           "omp1_ms,omp2_ms,omp4_ms,ompmax_ms,"
           "speedup_1,speedup_2,speedup_4,speedup_max,"
           "opencl_compute_ms,opencl_total_ms,"
           "speedup_opencl_compute,speedup_opencl_total\n";

    // ----------------------------------------------------------------
    // Section 1: OpenMP table
    // ----------------------------------------------------------------
    {
        const int W = 10;
        std::cout << "--- OpenMP results ---\n";
        std::cout << std::left
                  << std::setw(W) << "Size"
                  << std::setw(W) << "Seq(ms)"
                  << std::setw(W) << "OMP-1"
                  << std::setw(W) << "OMP-2"
                  << std::setw(W) << "OMP-4"
                  << std::setw(W) << "OMP-max"
                  << std::setw(W) << "S(1)"
                  << std::setw(W) << "S(2)"
                  << std::setw(W) << "S(4)"
                  << std::setw(W) << "S(max)"
                  << "OK?\n";
        std::cout << std::string(W * 10 + 4, '-') << "\n";

        for (int N : SIZES) {
            Circuit c = makeLadder(N);
            MNA     mna; mna.build(c);
            const Matrix A_orig = mna.getA();
            const Vec    b_orig = mna.getB();
            int ms = mna.getSize();

            // Re-copy A and b before every solver invocation so that
            // in-place modifications in one run do not affect later runs.
            SolverResult seq_res;
            double t_seq = minTime(
                [&]{ Matrix Ac = A_orig; Vec bc = b_orig;
                     return seq.solve(Ac, bc); },
                REPS, &seq_res);

            struct Run { double ms; SolverResult res; bool ok; };
            std::vector<Run> omp_runs;
            for (int nt : thread_counts) {
                SolverResult r;
                double t = minTime(
                    [&]{ Matrix Ac = A_orig; Vec bc = b_orig;
                         return omp_solver.solve(Ac, bc, nt); },
                    REPS, &r);
                bool ok = r.verified;
                if (ok && !r.x.empty()) {
                    double diff = 0;
                    for (size_t i = 0; i < r.x.size(); ++i)
                        diff = std::max(diff, std::abs(r.x[i] - seq_res.x[i]));
                    ok = (diff < 1e-6);
                }
                omp_runs.push_back({t, r, ok});
            }
            bool all_ok = true;
            for (auto& r : omp_runs) all_ok = all_ok && r.ok;

            double s1   = speedup(t_seq, omp_runs[0].ms);
            double s2   = speedup(t_seq, omp_runs[1].ms);
            double s4   = speedup(t_seq, omp_runs[2].ms);
            double smax = speedup(t_seq, omp_runs[3].ms);

            std::cout << std::fixed << std::setprecision(3) << std::left;
            std::cout << std::setw(W) << ms
                      << std::setw(W) << t_seq
                      << std::setw(W) << omp_runs[0].ms
                      << std::setw(W) << omp_runs[1].ms
                      << std::setw(W) << omp_runs[2].ms
                      << std::setw(W) << omp_runs[3].ms
                      << std::setw(W) << s1
                      << std::setw(W) << s2
                      << std::setw(W) << s4
                      << std::setw(W) << smax
                      << (all_ok ? "PASS" : "FAIL") << "\n";

            // Partial CSV row (OpenCL columns filled in Section 2 pass)
            csv << std::fixed << std::setprecision(4)
                << ms           << ","
                << t_seq        << ","
                << omp_runs[0].ms << "," << omp_runs[1].ms << ","
                << omp_runs[2].ms << "," << omp_runs[3].ms << ","
                << s1  << "," << s2  << "," << s4  << "," << smax << ","
                << "0,0,0,0\n";
        }
    }

    // ----------------------------------------------------------------
    // Section 2: OpenCL table
    // ----------------------------------------------------------------
    // Reopen CSV — redo in a single combined pass.
    csv.close();
    csv.open("results_openmp.csv");
    csv << "matrix_size,seq_ms,"
           "omp1_ms,omp2_ms,omp4_ms,ompmax_ms,"
           "speedup_1,speedup_2,speedup_4,speedup_max,"
           "opencl_compute_ms,opencl_total_ms,"
           "speedup_opencl_compute,speedup_opencl_total\n";

    std::cout << "\n--- OpenCL results ---\n";
    {
        const int W = 12;
        std::cout << std::left
                  << std::setw(W) << "Size"
                  << std::setw(W) << "Seq(ms)"
                  << std::setw(W) << "GPU cmp(ms)"
                  << std::setw(W) << "GPU tot(ms)"
                  << std::setw(W) << "Xfer-in"
                  << std::setw(W) << "Xfer-out"
                  << std::setw(W) << "S(cmp)"
                  << std::setw(W) << "S(total)"
                  << "OK?\n";
        std::cout << std::string(W * 8 + 4, '-') << "\n";

        for (int N : SIZES) {
            Circuit c = makeLadder(N);
            MNA     mna; mna.build(c);
            const Matrix A_orig = mna.getA();
            const Vec    b_orig = mna.getB();
            int ms = mna.getSize();

            // Sequential (reference) — fresh copy each run
            SolverResult seq_res;
            double t_seq = minTime(
                [&]{ Matrix Ac = A_orig; Vec bc = b_orig;
                     return seq.solve(Ac, bc); },
                REPS, &seq_res);

            // OpenMP runs (for CSV) — fresh copy each run
            struct Run { double ms; SolverResult res; };
            std::vector<Run> omp_runs;
            for (int nt : thread_counts) {
                SolverResult r;
                double t = minTime(
                    [&]{ Matrix Ac = A_orig; Vec bc = b_orig;
                         return omp_solver.solve(Ac, bc, nt); },
                    REPS, &r);
                omp_runs.push_back({t, r});
            }
            double s1   = speedup(t_seq, omp_runs[0].ms);
            double s2   = speedup(t_seq, omp_runs[1].ms);
            double s4   = speedup(t_seq, omp_runs[2].ms);
            double smax = speedup(t_seq, omp_runs[3].ms);

            // OpenCL — solve() takes A/b by value, so each call already
            // works on a fresh copy; no extra copy needed here.
            double ocl_cmp_ms = 0, ocl_in_ms = 0, ocl_out_ms = 0, ocl_tot_ms = 0;
            bool   ocl_pass   = false;
            std::string ocl_status = ocl_ok ? "" : "N/A";

            if (ocl_ok) {
                SolverResult best_r;
                double best_cmp = std::numeric_limits<double>::max();
                for (int r = 0; r < REPS; ++r) {
                    SolverResult res = ocl.solve(A_orig, b_orig);
                    if (res.success && res.time_ms < best_cmp) {
                        best_cmp   = res.time_ms;
                        ocl_in_ms  = ocl.transfer_in_ms;
                        ocl_out_ms = ocl.transfer_out_ms;
                        best_r     = res;
                    }
                }
                ocl_cmp_ms = best_cmp;
                ocl_tot_ms = ocl_in_ms + ocl_cmp_ms + ocl_out_ms;

                // Verify vs sequential
                if (best_r.verified && !best_r.x.empty()) {
                    double diff = 0;
                    for (size_t i = 0; i < best_r.x.size(); ++i)
                        diff = std::max(diff, std::abs(best_r.x[i] - seq_res.x[i]));
                    ocl_pass = (diff < 1e-6);
                }
                ocl_status = ocl_pass ? "PASS" : "FAIL";
            }

            double sc = speedup(t_seq, ocl_cmp_ms);
            double st = speedup(t_seq, ocl_tot_ms);

            std::cout << std::fixed << std::setprecision(3) << std::left;
            std::cout << std::setw(W) << ms
                      << std::setw(W) << t_seq
                      << std::setw(W) << ocl_cmp_ms
                      << std::setw(W) << ocl_tot_ms
                      << std::setw(W) << ocl_in_ms
                      << std::setw(W) << ocl_out_ms
                      << std::setw(W) << sc
                      << std::setw(W) << st
                      << ocl_status << "\n";

            csv << std::fixed << std::setprecision(4)
                << ms  << ","
                << t_seq << ","
                << omp_runs[0].ms << "," << omp_runs[1].ms << ","
                << omp_runs[2].ms << "," << omp_runs[3].ms << ","
                << s1  << "," << s2  << "," << s4  << "," << smax << ","
                << ocl_cmp_ms << "," << ocl_tot_ms << ","
                << sc  << "," << st  << "\n";
        }
    }

    csv.close();
    std::cout << "\nCSV written to: results_openmp.csv\n";

    std::cout << "\n=== results_openmp.csv ===\n";
    std::ifstream fin("results_openmp.csv");
    std::cout << fin.rdbuf();

    return 0;
}
