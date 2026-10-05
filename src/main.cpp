// Unified entry point for the parallel MNA circuit solver.
// Build variants (set by Makefile via -D flags):
//   HAS_OMP  — enables OpenMPSolver
//   HAS_OCL  — enables OpenCLSolver
//
// Usage:
//   ./solver --test
//   ./solver --bench N
//   ./solver --compare file.txt

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "circuit.hpp"
#include "mna.hpp"
#include "solver.hpp"

#ifdef HAS_OMP
#  include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Circuit builders (self-contained — no dependency on generate_testcases.cpp)
// ---------------------------------------------------------------------------

// Three equal-resistor chain:
//   GND -[Vs=5V]-> n3 -[R=1Ω]-> n2 -[R=1Ω]-> n1 -[R=1Ω]-> GND
//   => V(n1)=5/3≈1.667 V  (x[0], label N001)
//      V(n2)=10/3≈3.333 V (x[1], label N002)
//      V(n3)=5 V           (x[2], label N003, forced by VS)
static Circuit makeVoltageDivider() {
    Circuit c;
    c.addVoltageSource(3, 0, 5.0);
    c.addResistor(3, 2, 1.0);
    c.addResistor(2, 1, 1.0);
    c.addResistor(1, 0, 1.0);
    return c;
}

// Simple series loop: V=10 V, R=5 Ω → I=2 A, V(n1)=10 V
static Circuit makeSimpleLoop() {
    Circuit c;
    c.addVoltageSource(1, 0, 10.0);
    c.addResistor(1, 0, 5.0);
    return c;
}

// n_stages equal 1-Ω resistors in series from VS(5V) to GND.
// V(nk) = 5 - (k-1)*(5/n_stages),  V(n1)=5V (VS),  V(n_stages)=5/n_stages
static Circuit makeLadder(int n) {
    Circuit c;
    c.addVoltageSource(1, 0, 5.0);
    for (int i = 1; i < n; ++i)
        c.addResistor(i, i + 1, 1.0);
    c.addResistor(n, 0, 1.0);
    return c;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Run fn() up to reps times; return minimum time_ms, put best result in *out.
template<typename Fn>
static double minRun(Fn fn, int reps, SolverResult* out = nullptr) {
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < reps; ++i) {
        SolverResult r = fn();
        if (r.time_ms < best) {
            best = r.time_ms;
            if (out) *out = r;
        }
    }
    return best;
}

// Compare two solution vectors; return max element-wise absolute difference.
static double maxDiff(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) return 1e18;
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

static const char* pf(bool p) { return p ? "PASS" : "FAIL"; }

// Map "N001" -> node index 0 (node 1 → solution index 0).
static int labelToIndex(const std::string& label) {
    if (label.size() < 2 || label[0] != 'N') return -1;
    try { return std::stoi(label.substr(1)) - 1; }
    catch (...) { return -1; }
}

static void printUsage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " --test\n"
        << "  " << prog << " --bench N\n"
        << "  " << prog << " --compare file.txt\n";
}

// ---------------------------------------------------------------------------
// --test
// ---------------------------------------------------------------------------
static int runTests() {
    constexpr double TOL = 1e-4;
    SequentialSolver seq;
    int failures = 0;

    std::cout << "==============================\n";
    std::cout << " MNA Solver — Validation Suite\n";
    std::cout << "==============================\n\n";

    // ---- Test 1: voltage divider (3-R chain) ----------------------------
    {
        std::cout << "[1] Voltage divider  (3×1Ω, Vs=5V)\n";
        Circuit c = makeVoltageDivider();
        MNA mna; mna.build(c);
        auto r = seq.solve(mna.getA(), mna.getB());
        if (!r.success) { std::cout << "    FAIL — solver error: " << r.error << "\n\n"; ++failures; }
        else {
            double v1 = r.x[0], v2 = r.x[1], v3 = r.x[2];
            bool ok1 = std::abs(v1 - 5.0/3.0) < TOL;
            bool ok2 = std::abs(v2 - 10.0/3.0) < TOL;
            bool ok3 = std::abs(v3 - 5.0) < TOL;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "    V(N001) = " << v1 << "  expected " << 5.0/3.0  << "  " << pf(ok1) << "\n";
            std::cout << "    V(N002) = " << v2 << "  expected " << 10.0/3.0 << "  " << pf(ok2) << "\n";
            std::cout << "    V(N003) = " << v3 << "  expected 5.000000  " << pf(ok3) << "\n";
            bool pass = ok1 && ok2 && ok3;
            std::cout << "    => " << pf(pass) << "\n\n";
            if (!pass) ++failures;
        }
    }

    // ---- Test 2: simple loop --------------------------------------------
    {
        std::cout << "[2] Simple loop  (V=10V, R=5Ω  →  I=2A)\n";
        Circuit c = makeSimpleLoop();
        MNA mna; mna.build(c);
        auto r = seq.solve(mna.getA(), mna.getB());
        if (!r.success) { std::cout << "    FAIL — solver error: " << r.error << "\n\n"; ++failures; }
        else {
            // x[0]=V(n1)=10V, x[1]=branch current (negative convention: -2A)
            double v1 = r.x[0];
            double i1 = r.x[1];                        // branch current from VS
            double I_resistor = v1 / 5.0;              // I through R
            bool ok_v = std::abs(v1 - 10.0) < TOL;
            bool ok_i = std::abs(I_resistor - 2.0) < TOL;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "    V(n1)    = " << v1 << "  expected 10.000000  " << pf(ok_v) << "\n";
            std::cout << "    I(R)     = " << I_resistor << " A  expected 2.000000  " << pf(ok_i) << "\n";
            std::cout << "    I_branch = " << i1 << " A  (VS branch current)\n";
            bool pass = ok_v && ok_i;
            std::cout << "    => " << pf(pass) << "\n\n";
            if (!pass) ++failures;
        }
    }

    // ---- Test 3: ladder N=10 -------------------------------------------
    // 10 equal 1-Ω resistors in series with Vs=5V.
    // V(nk) = 5 - (k-1)*0.5:  V(n1)=5V, V(n5)=3V, V(n10)=0.5V
    {
        std::cout << "[3] Ladder N=10  (10×1Ω, Vs=5V)\n";
        Circuit c = makeLadder(10);
        MNA mna; mna.build(c);
        auto r = seq.solve(mna.getA(), mna.getB());
        if (!r.success) { std::cout << "    FAIL — solver error: " << r.error << "\n\n"; ++failures; }
        else {
            // In the MNA solution: x[0..9] are V(n1)..V(n10), x[10] is branch current.
            // V(nk) = 5*(n-k+1)/n with n=10 resistors and voltage at n1 forced to 5V.
            // Since V(n1) is forced: V(n1)=5, drop per resistor = 5/10=0.5V.
            auto check = [&](int k, double expected) {
                double got = r.x[k - 1];
                bool ok = std::abs(got - expected) < TOL;
                std::cout << "    V(n" << std::setw(2) << k << ")  = "
                          << std::fixed << std::setprecision(4) << got
                          << "  expected " << expected << "  " << pf(ok) << "\n";
                return ok;
            };
            bool pass = check(1, 5.0)
                      & check(2, 4.5)
                      & check(5, 3.0)
                      & check(9, 1.0)
                      & check(10, 0.5);
            std::cout << "    => " << pf(pass) << "\n\n";
            if (!pass) ++failures;
        }
    }

    // ---- Summary --------------------------------------------------------
    std::cout << "==============================\n";
    if (failures == 0)
        std::cout << "All tests PASSED.\n";
    else
        std::cout << failures << " test(s) FAILED.\n";
    std::cout << "==============================\n";
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// --bench N
// ---------------------------------------------------------------------------
static int runBench(int N) {
    constexpr int REPS = 5;
    constexpr double TOL = 1e-6;

    std::cout << "==============================\n";
    std::cout << " MNA Solver — Benchmark N=" << N << "\n";
    std::cout << "==============================\n";

    Circuit c = makeLadder(N);
    MNA mna; mna.build(c);
    const Matrix& A = mna.getA();
    const Vec&    b = mna.getB();
    int mat = mna.getSize();

    std::cout << "Ladder circuit: " << N << " stages, MNA matrix " << mat << "×" << mat << "\n";

#ifdef HAS_OMP
    int nt = omp_get_max_threads();
    std::cout << "OpenMP threads : " << nt << "\n";
#endif
#ifdef HAS_OCL
    OpenCLSolver ocl;
    bool ocl_ok = ocl.initialize();
    if (ocl_ok)
        std::cout << "OpenCL device  : " << ocl.getDeviceInfo() << "\n";
    else
        std::cout << "OpenCL         : unavailable (" << ocl.error_msg << ")\n";
#endif
    std::cout << "\n";

    // ---- Sequential ----
    SequentialSolver seq;
    SolverResult seq_res;
    double t_seq = minRun([&]{ return seq.solve(A, b); }, REPS, &seq_res);

    // Column widths
    const int W0=24, W1=12, W2=10, W3=22;
    auto hdr = [&]{
        std::cout << std::left
                  << std::setw(W0) << "Solver"
                  << std::setw(W1) << "Time (ms)"
                  << std::setw(W2) << "Speedup"
                  << "Result vs Sequential\n"
                  << std::string(W0+W1+W2+W3, '-') << "\n";
    };

    hdr();
    {
        auto fmtSp = [](double sp) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(2) << sp << "x";
            return os.str();
        };
        auto rowFmt = [&](const std::string& name, double t_ms, const SolverResult& res,
                          const std::string& note = "") {
            double sp = (t_ms > 0) ? t_seq / t_ms : 0.0;
            std::string verdict;
            if (note == "baseline") {
                verdict = "baseline";
            } else {
                bool ok = res.success && maxDiff(res.x, seq_res.x) < TOL;
                verdict = std::string(pf(ok)) + (note.empty() ? "" : "  " + note);
            }
            std::cout << std::fixed << std::setprecision(3)
                      << std::left << std::setw(W0) << name
                      << std::setw(W1) << t_ms
                      << std::setw(W2) << fmtSp(sp)
                      << verdict << "\n";
        };

        rowFmt("Sequential",   t_seq, seq_res, "baseline");

#ifdef HAS_OMP
        OpenMPSolver omp_solver;
        SolverResult omp_res;
        double t_omp = minRun([&]{ return omp_solver.solve(A, b, nt); }, REPS, &omp_res);
        rowFmt("OpenMP (" + std::to_string(nt) + " threads)", t_omp, omp_res);
#else
        std::cout << std::left << std::setw(W0) << "OpenMP"
                  << std::setw(W1) << "N/A"
                  << std::setw(W2) << "N/A"
                  << "not compiled in\n";
#endif

#ifdef HAS_OCL
        if (ocl_ok) {
            SolverResult ocl_res;
            double t_ocl_cmp = minRun([&]{ return ocl.solve(A, b); }, REPS, &ocl_res);
            double t_ocl_tot = ocl.transfer_in_ms + t_ocl_cmp + ocl.transfer_out_ms;
            rowFmt("OpenCL (GPU)",       t_ocl_cmp, ocl_res, "(compute only)");
            rowFmt("OpenCL+xfer",        t_ocl_tot, ocl_res, "(compute+transfer)");
        } else {
            std::cout << std::left << std::setw(W0) << "OpenCL"
                      << std::setw(W1) << "N/A"
                      << std::setw(W2) << "N/A"
                      << "not available\n";
        }
#else
        std::cout << std::left << std::setw(W0) << "OpenCL"
                  << std::setw(W1) << "N/A"
                  << std::setw(W2) << "N/A"
                  << "not compiled in\n";
#endif
    }
    std::cout << std::string(W0+W1+W2+W3, '-') << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// --compare file.txt
// ---------------------------------------------------------------------------
static int runCompare(const std::string& filename) {
    std::cout << "==============================\n";
    std::cout << " MNA Solver — Compare vs File\n";
    std::cout << "==============================\n";
    std::cout << "Circuit : voltage divider (3×1Ω, Vs=5V)\n";
    std::cout << "File    : " << filename << "\n\n";

    // Load reference values from file
    std::map<std::string, double> ref;
    {
        std::ifstream f(filename);
        if (!f) {
            std::cerr << "Error: cannot open '" << filename << "'\n";
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string name; double val;
            if (ss >> name >> val)
                ref[name] = val;
        }
    }
    if (ref.empty()) {
        std::cerr << "Error: no data in '" << filename << "'\n";
        return 1;
    }

    // Run sequential solver
    SequentialSolver seq;
    Circuit c = makeVoltageDivider();
    MNA mna; mna.build(c);
    auto r = seq.solve(mna.getA(), mna.getB());
    if (!r.success) {
        std::cerr << "Solver failed: " << r.error << "\n";
        return 1;
    }

    // Print comparison table
    constexpr double TOL = 1e-4;
    const int W = 12;
    std::cout << std::left
              << std::setw(8)  << "Node"
              << std::setw(W)  << "Reference"
              << std::setw(W)  << "Computed"
              << std::setw(W)  << "Diff"
              << "Status\n"
              << std::string(8 + W*3 + 6, '-') << "\n";

    int failures = 0;
    for (const auto& [name, expected] : ref) {
        int idx = labelToIndex(name);
        if (idx < 0 || idx >= static_cast<int>(r.x.size())) {
            std::cout << std::left << std::setw(8) << name
                      << std::setw(W) << expected
                      << std::setw(W) << "—"
                      << std::setw(W) << "—"
                      << "UNKNOWN NODE\n";
            ++failures;
            continue;
        }
        double got  = r.x[idx];
        double diff = std::abs(got - expected);
        bool   ok   = diff < TOL;
        if (!ok) ++failures;
        std::cout << std::fixed << std::setprecision(6)
                  << std::left  << std::setw(8)  << name
                  << std::setw(W)  << expected
                  << std::setw(W)  << got
                  << std::setw(W)  << diff
                  << (ok ? "OK" : "MISMATCH") << "\n";
    }

    std::cout << std::string(8 + W*3 + 6, '-') << "\n";
    std::cout << "Overall: " << pf(failures == 0) << "\n";
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(argv[0]); return 1; }

    std::string mode = argv[1];

    if (mode == "--test") {
        return runTests();
    }
    if (mode == "--bench") {
        if (argc < 3) { std::cerr << "--bench requires N\n"; return 1; }
        int N = std::atoi(argv[2]);
        if (N < 2) { std::cerr << "N must be >= 2\n"; return 1; }
        return runBench(N);
    }
    if (mode == "--compare") {
        if (argc < 3) { std::cerr << "--compare requires a filename\n"; return 1; }
        return runCompare(argv[2]);
    }
    printUsage(argv[0]);
    return 1;
}
