#include "solver.hpp"

bool         OpenCLSolver::initialize()          { error_msg = "Built without OpenCL SDK"; return false; }
std::string  OpenCLSolver::getDeviceInfo()       { return ""; }
SolverResult OpenCLSolver::solve(Matrix, Vec)    { SolverResult r; r.error = "OpenCL unavailable"; return r; }
