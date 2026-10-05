#include "mna.hpp"
#include <iomanip>
#include <iostream>

void MNA::build(const Circuit& c) {
    int n = c.getSize();
    A.assign(n, Vec(n, 0.0));
    b.assign(n, 0.0);

    // --- Resistors (G-stamps) ---
    for (const auto& r : c.resistors) {
        double G = 1.0 / r.resistance;
        int i = r.node_pos;
        int j = r.node_neg;
        if (i > 0) A[i-1][i-1] += G;
        if (j > 0) A[j-1][j-1] += G;
        if (i > 0 && j > 0) {
            A[i-1][j-1] -= G;
            A[j-1][i-1] -= G;
        }
    }

    // --- Voltage sources ---
    int node_offset = c.num_nodes - 1;
    for (const auto& vs : c.voltage_sources) {
        int k  = vs.branch_idx;
        int row = node_offset + k;   // extra row/col for branch current
        int p  = vs.node_pos;
        int n_ = vs.node_neg;

        // KCL rows: +1 at node_pos row, -1 at node_neg row
        if (p > 0) { A[p-1][row] += 1.0; A[row][p-1] += 1.0; }
        if (n_ > 0) { A[n_-1][row] -= 1.0; A[row][n_-1] -= 1.0; }

        b[row] += vs.voltage;
    }

    // --- Current sources ---
    for (const auto& cs : c.current_sources) {
        if (cs.node_pos > 0) b[cs.node_pos - 1] += cs.current;
        if (cs.node_neg > 0) b[cs.node_neg - 1] -= cs.current;
    }
}

void MNA::print() const {
    int n = getSize();
    std::cout << std::fixed << std::setprecision(4);

    // Header
    std::cout << "  [ A matrix ]";
    for (int j = 1; j < n; ++j) std::cout << "           ";
    std::cout << "  | b\n";

    for (int i = 0; i < n; ++i) {
        std::cout << "  [";
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(10) << A[i][j];
            if (j < n - 1) std::cout << " ";
        }
        std::cout << " ]  | " << std::setw(10) << b[i] << "\n";
    }
}
