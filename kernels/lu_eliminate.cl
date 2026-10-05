#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Apply one step of Gaussian elimination: update the trailing (n-k-1)x(n-k-1)
// sub-matrix using the pre-computed multipliers for column k.
// Each work-item owns one (row, col) cell of the trailing sub-matrix.
__kernel void lu_eliminate(__global double* A,
                           __global const double* factors,
                           const int n, const int k) {
    int i   = get_global_id(0);     // row offset from k+1
    int j   = get_global_id(1);     // col offset from k+1
    int row = i + k + 1;
    int col = j + k + 1;
    if (row >= n || col >= n) return;
    A[row * n + col] -= factors[i] * A[k * n + col];
}
