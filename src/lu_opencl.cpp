#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include "solver.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string slurp(const std::string& path) {
    std::ifstream f(path);
    if (!f) return "";
    return {std::istreambuf_iterator<char>(f), {}};
}

static double eventMs(cl_event evt) {
    cl_ulong s = 0, e = 0;
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
    clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,   sizeof(e), &e, nullptr);
    return static_cast<double>(e - s) * 1e-6;  // nanoseconds → ms
}

// Round x up to the nearest multiple of align (for work-group sizing).
static size_t roundUp(size_t x, size_t align) {
    return ((x + align - 1) / align) * align;
}

static std::string clErrStr(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                    return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:           return "CL_DEVICE_NOT_FOUND";
        case CL_BUILD_PROGRAM_FAILURE:      return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_KERNEL_NAME:        return "CL_INVALID_KERNEL_NAME";
        case CL_OUT_OF_HOST_MEMORY:         return "CL_OUT_OF_HOST_MEMORY";
        default: {
            std::ostringstream os; os << "CL error " << err; return os.str();
        }
    }
}

// Build one program from source, printing the build log on failure.
static cl_program buildProg(cl_context ctx, cl_device_id dev,
                             const std::string& src, const std::string& label,
                             std::string& err_out) {
    const char* c = src.c_str();
    size_t      l = src.size();
    cl_int      err;
    cl_program  prog = clCreateProgramWithSource(ctx, 1, &c, &l, &err);
    if (err != CL_SUCCESS) { err_out = label + ": clCreateProgramWithSource " + clErrStr(err); return nullptr; }

    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL1.2", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logLen = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen);
        std::string log(logLen, '\0');
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logLen, &log[0], nullptr);
        err_out = label + " build failed:\n" + log;
        clReleaseProgram(prog);
        return nullptr;
    }
    return prog;
}

// ---------------------------------------------------------------------------
// OpenCLSolver::initialize()
// ---------------------------------------------------------------------------
bool OpenCLSolver::initialize() {
    // --- Platform ---
    cl_uint np = 0;
    clGetPlatformIDs(0, nullptr, &np);
    if (np == 0) { error_msg = "No OpenCL platforms found"; return false; }
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    auto plat = plats[0];

    // --- Device: prefer GPU, fall back to CPU ---
    cl_device_id dev = nullptr;
    cl_int err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 1, &dev, nullptr);
        if (err != CL_SUCCESS) { error_msg = "No OpenCL device"; return false; }
    }

    char dname[256] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
    char dver[128] = {};
    clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(dver), dver, nullptr);
    cl_uint cu = 0;
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
    std::ostringstream di;
    di << dname << "  [" << dver << "]  CUs=" << cu;
    device_info_ = di.str();

    // --- Context ---
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) { error_msg = "clCreateContext: " + clErrStr(err); return false; }

    // --- Command queue with profiling ---
    cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, props, &err);
    if (err != CL_SUCCESS) {
        error_msg = "clCreateCommandQueue: " + clErrStr(err);
        clReleaseContext(ctx);
        return false;
    }

    // --- Load and build kernels ---
    // Try two path prefixes: relative to CWD or one level up.
    auto tryLoad = [](const std::string& name) {
        for (const char* pfx : {"kernels/", "../kernels/"}) {
            auto s = slurp(std::string(pfx) + name);
            if (!s.empty()) return s;
        }
        return std::string{};
    };

    auto src_factor = tryLoad("lu_factor.cl");
    auto src_elim   = tryLoad("lu_eliminate.cl");
    if (src_factor.empty() || src_elim.empty()) {
        error_msg = "Cannot load kernel source files (kernels/lu_factor.cl / lu_eliminate.cl)";
        clReleaseCommandQueue(queue); clReleaseContext(ctx);
        return false;
    }

    cl_program prog_factor = buildProg(ctx, dev, src_factor, "lu_factor.cl",   error_msg);
    cl_program prog_elim   = buildProg(ctx, dev, src_elim,   "lu_eliminate.cl", error_msg);
    if (!prog_factor || !prog_elim) {
        if (prog_factor) clReleaseProgram(prog_factor);
        if (prog_elim)   clReleaseProgram(prog_elim);
        clReleaseCommandQueue(queue); clReleaseContext(ctx);
        return false;
    }

    cl_kernel kern_factor = clCreateKernel(prog_factor, "compute_factors", &err);
    if (err != CL_SUCCESS) { error_msg = "clCreateKernel compute_factors: " + clErrStr(err); return false; }
    cl_kernel kern_elim   = clCreateKernel(prog_elim,   "lu_eliminate",    &err);
    if (err != CL_SUCCESS) { error_msg = "clCreateKernel lu_eliminate: " + clErrStr(err); return false; }

    // Stash all handles as void* (avoids <CL/cl.h> in the header).
    platform_    = reinterpret_cast<void*>(plat);
    device_      = reinterpret_cast<void*>(dev);
    context_     = reinterpret_cast<void*>(ctx);
    queue_       = reinterpret_cast<void*>(queue);
    prog_factor_ = reinterpret_cast<void*>(prog_factor);
    prog_elim_   = reinterpret_cast<void*>(prog_elim);
    kern_factor_ = reinterpret_cast<void*>(kern_factor);
    kern_elim_   = reinterpret_cast<void*>(kern_elim);

    ready_ = true;
    return true;
}

std::string OpenCLSolver::getDeviceInfo() { return device_info_; }

// ---------------------------------------------------------------------------
// OpenCLSolver::solve()
// ---------------------------------------------------------------------------
SolverResult OpenCLSolver::solve(Matrix A_in, Vec b) {
    SolverResult res;
    res.success = false;

    if (!ready_) { res.error = "OpenCLSolver not initialised"; return res; }

    auto ctx         = reinterpret_cast<cl_context>(context_);
    auto queue       = reinterpret_cast<cl_command_queue>(queue_);
    auto kern_factor = reinterpret_cast<cl_kernel>(kern_factor_);
    auto kern_elim   = reinterpret_cast<cl_kernel>(kern_elim_);

    const int n = static_cast<int>(b.size());

    // --- Flatten A row-major ---
    std::vector<double> flat(static_cast<size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            flat[static_cast<size_t>(i) * n + j] = A_in[i][j];

    const size_t bytes_A       = flat.size() * sizeof(double);
    const size_t bytes_factors = static_cast<size_t>(n) * sizeof(double);

    cl_int err;

    // --- Allocate GPU buffers ---
    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes_A, nullptr, &err);
    if (err) { res.error = "clCreateBuffer A: " + clErrStr(err); return res; }
    cl_mem bufF = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes_factors, nullptr, &err);
    if (err) { clReleaseMemObject(bufA); res.error = "clCreateBuffer factors: " + clErrStr(err); return res; }

    // --- Transfer A host → device (profiled) ---
    cl_event write_evt;
    err = clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, bytes_A, flat.data(), 0, nullptr, &write_evt);
    if (err) { res.error = "clEnqueueWriteBuffer: " + clErrStr(err); goto cleanup; }
    clFinish(queue);
    transfer_in_ms = eventMs(write_evt);
    clReleaseEvent(write_evt);

    // --- GPU LU (no partial pivoting) ---
    // chrono::high_resolution_clock used for wall-clock; cl_event profiling
    // used for GPU-only compute time (eventMs) — nanosecond resolution.
    {
        using Clock = std::chrono::high_resolution_clock;
        auto t0 = Clock::now();

        for (int k = 0; k < n - 1; ++k) {
            int remaining = n - k - 1;

            // (a) compute_factors: 1D, global rounded up to multiple of local
            {
                size_t ls1 = static_cast<size_t>(std::min(64, remaining));
                size_t gs1 = roundUp(static_cast<size_t>(remaining), ls1);

                clSetKernelArg(kern_factor, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(kern_factor, 1, sizeof(cl_mem), &bufF);
                clSetKernelArg(kern_factor, 2, sizeof(cl_int), &n);
                clSetKernelArg(kern_factor, 3, sizeof(cl_int), &k);
                err = clEnqueueNDRangeKernel(queue, kern_factor, 1, nullptr, &gs1, &ls1, 0, nullptr, nullptr);
                if (err) { res.error = "enqueue compute_factors k=" + std::to_string(k) + ": " + clErrStr(err); goto cleanup; }
            }
            clFinish(queue);

            // (c) lu_eliminate: 2D, global rounded up to multiples of local
            {
                size_t lw       = static_cast<size_t>(std::min(16, remaining));
                size_t ls2[2]   = {lw, lw};
                size_t gs2[2]   = {roundUp(static_cast<size_t>(remaining), lw),
                                   roundUp(static_cast<size_t>(remaining), lw)};

                clSetKernelArg(kern_elim, 0, sizeof(cl_mem), &bufA);
                clSetKernelArg(kern_elim, 1, sizeof(cl_mem), &bufF);
                clSetKernelArg(kern_elim, 2, sizeof(cl_int), &n);
                clSetKernelArg(kern_elim, 3, sizeof(cl_int), &k);
                err = clEnqueueNDRangeKernel(queue, kern_elim, 2, nullptr, gs2, ls2, 0, nullptr, nullptr);
                if (err) { res.error = "enqueue lu_eliminate k=" + std::to_string(k) + ": " + clErrStr(err); goto cleanup; }
            }
            clFinish(queue);
        }

        res.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    // --- Transfer A device → host (profiled) ---
    {
        cl_event read_evt;
        err = clEnqueueReadBuffer(queue, bufA, CL_FALSE, 0, bytes_A, flat.data(), 0, nullptr, &read_evt);
        if (err) { res.error = "clEnqueueReadBuffer: " + clErrStr(err); goto cleanup; }
        clFinish(queue);
        transfer_out_ms = eventMs(read_evt);
        clReleaseEvent(read_evt);
    }

    // --- CPU forward substitution (Ly = b) ---
    // flat now stores L in strict lower triangle, U in upper triangle + diagonal.
    // L[i][i] = 1 (implicit), L[i][j] = flat[i*n+j] for j < i.
    {
        Vec y(n);
        for (int i = 0; i < n; ++i) {
            double s = b[i];
            for (int j = 0; j < i; ++j)
                s -= flat[static_cast<size_t>(i) * n + j] * y[j];
            y[i] = s;   // L[i][i] = 1
        }

        // --- CPU backward substitution (Ux = y) ---
        Vec x(n);
        for (int i = n - 1; i >= 0; --i) {
            double s = y[i];
            for (int j = i + 1; j < n; ++j)
                s -= flat[static_cast<size_t>(i) * n + j] * x[j];
            double diag = flat[static_cast<size_t>(i) * n + i];
            if (std::abs(diag) < 1e-14) {
                res.error = "Near-zero pivot at row " + std::to_string(i) + " (no pivoting in GPU path)";
                goto cleanup;
            }
            x[i] = s / diag;
        }

        // --- Residual check: ||A_orig * x - b||∞ ---
        double max_res = 0.0;
        for (int i = 0; i < n; ++i) {
            double r = -b[i];
            for (int j = 0; j < n; ++j)
                r += A_in[i][j] * x[j];
            max_res = std::max(max_res, std::abs(r));
        }

        res.x        = std::move(x);
        res.success  = true;
        res.verified = (max_res < 1e-6);
    }

cleanup:
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufF);
    return res;
}
