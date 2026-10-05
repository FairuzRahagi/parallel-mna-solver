#include "solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>

SolverResult OpenMPSolver::solve(Matrix A, Vec b, int num_threads) {
    using Clock = std::chrono::high_resolution_clock;
    const int nt = (num_threads <= 0) ? omp_get_max_threads() : num_threads;
    const int n  = static_cast<int>(b.size());

    // Keep originals for residual check (A and b will be destroyed by LU).
    Matrix A_orig = A;
    Vec    b_orig = b;

    SolverResult res;
    res.success  = false;
    res.verified = false;

    auto t0 = Clock::now();

    // ------------------------------------------------------------------
    // LU decomposition with partial pivoting
    // The inner elimination loop (independent rows) is parallelised with
    // OpenMP; pivot search and row swap remain sequential.
    // ------------------------------------------------------------------
    for (int k = 0; k < n; ++k) {

        // --- Partial pivot: find row with max |A[i][k]|, i >= k ---
        int    pivot_row = k;
        double max_val   = std::abs(A[k][k]);
        for (int i = k + 1; i < n; ++i) {
            double v = std::abs(A[i][k]);
            if (v > max_val) { max_val = v; pivot_row = i; }
        }

        if (max_val < 1e-14) {
            res.error = "Singular matrix at column " + std::to_string(k);
            res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            return res;
        }

        if (pivot_row != k) {
            std::swap(A[k], A[pivot_row]);
            std::swap(b[k], b[pivot_row]);
        }

        // --- Parallel elimination of rows below pivot ---
        #pragma omp parallel for schedule(dynamic) num_threads(nt)
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            A[i][k] = factor;                       // store multiplier (L)
            for (int j = k + 1; j < n; ++j)
                A[i][j] -= factor * A[k][j];
            b[i] -= factor * b[k];
        }
    }

    // ------------------------------------------------------------------
    // Backward substitution: Ux = b  (forward sub already done above)
    // ------------------------------------------------------------------
    Vec x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j)
            s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }

    res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    res.x       = x;
    res.success = true;

    // ------------------------------------------------------------------
    // Residual check: ||A_orig * x - b_orig||∞ < 1e-6
    // ------------------------------------------------------------------
    double max_res = 0.0;
    for (int i = 0; i < n; ++i) {
        double r = -b_orig[i];
        for (int j = 0; j < n; ++j)
            r += A_orig[i][j] * x[j];
        max_res = std::max(max_res, std::abs(r));
    }
    res.verified = (max_res < 1e-6);

    return res;
}
