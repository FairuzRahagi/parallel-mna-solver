#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Compute multipliers for column k and write them back into A's lower triangle
// so that forward-substitution on the CPU can use A as a combined L/U store.
__kernel void compute_factors(__global double* A,        // NOT const — we store L
                              __global double* factors,
                              const int n, const int k) {
    int i   = get_global_id(0);     // row offset: i=0 → row k+1
    int row = i + k + 1;
    if (row >= n) return;

    double f    = A[row * n + k] / A[k * n + k];
    factors[i]  = f;
    A[row * n + k] = f;             // store in L part for CPU forward-sub
}
