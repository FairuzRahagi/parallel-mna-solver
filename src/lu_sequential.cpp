#include "solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>

SolverResult SequentialSolver::solve(Matrix A, Vec b) {
    using Clock = std::chrono::high_resolution_clock;
    const int n = static_cast<int>(b.size());
    SolverResult res;
    res.success = false;

    auto t0 = Clock::now();

    // --- LU decomposition with partial pivoting ---
    for (int k = 0; k < n; ++k) {
        int    pivot_row = k;
        double max_val   = std::abs(A[k][k]);
        for (int i = k + 1; i < n; ++i) {
            double v = std::abs(A[i][k]);
            if (v > max_val) { max_val = v; pivot_row = i; }
        }

        if (max_val < 1e-14) {
            res.error   = "Singular matrix (pivot < 1e-14) at column " + std::to_string(k);
            res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            return res;
        }

        if (pivot_row != k) {
            std::swap(A[k], A[pivot_row]);
            std::swap(b[k], b[pivot_row]);
        }

        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            A[i][k] = factor;
            for (int j = k + 1; j < n; ++j)
                A[i][j] -= factor * A[k][j];
            b[i] -= factor * b[k];
        }
    }

    // --- Backward substitution: Ux = b ---
    Vec x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j)
            sum -= A[i][j] * x[j];
        x[i] = sum / A[i][i];
    }

    res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    res.x       = std::move(x);
    res.success = true;
    return res;
}
