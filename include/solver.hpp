#ifndef MNA_PROJECT_SOLVER_HPP
#define MNA_PROJECT_SOLVER_HPP
#include <string>
#include <vector>
#include "mna.hpp"

struct SolverResult {
    std::vector<double> x;            // node voltages + branch currents
    double              time_ms = 0;  // solve time in milliseconds
    bool                success  = false;
    bool                verified = false; // residual ||Ax-b||∞ < 1e-6
    std::string         error;
};

class SequentialSolver {
public:
    // A and b are passed by value — LU decomposition modifies them in place.
    SolverResult solve(Matrix A, Vec b);
};

class OpenMPSolver {
public:
    // num_threads == 0  →  use omp_get_max_threads()
    SolverResult solve(Matrix A, Vec b, int num_threads = 0);
};

// OpenCL solver — call initialize() once, then solve() any number of times.
// Requires kernels/lu_factor.cl and kernels/lu_eliminate.cl relative to CWD.
class OpenCLSolver {
public:
    bool        initialize();       // false if no OpenCL or build fails
    std::string getDeviceInfo();
    SolverResult solve(Matrix A, Vec b);
    // time_ms in SolverResult = GPU kernel compute time only.
    double      transfer_in_ms  = 0;    // host→device transfer for each solve()
    double      transfer_out_ms = 0;    // device→host transfer for each solve()
    std::string error_msg;

private:
    bool   ready_    = false;
    // Internal handles stored as void* so <CL/cl.h> is not needed here.
    void*  platform_ = nullptr;
    void*  device_   = nullptr;
    void*  context_  = nullptr;
    void*  queue_    = nullptr;
    void*  prog_factor_ = nullptr;  // compiled compute_factors program
    void*  prog_elim_   = nullptr;  // compiled lu_eliminate program
    void*  kern_factor_ = nullptr;
    void*  kern_elim_   = nullptr;
    std::string device_info_;
};

#endif // MNA_PROJECT_SOLVER_HPP
