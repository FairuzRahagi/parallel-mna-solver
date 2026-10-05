# Parallel MNA Circuit Solver: Sequential vs OpenMP vs OpenCL

A circuit solver that builds the **Modified Nodal Analysis (MNA)** system `A x = b` for resistor/source networks and solves it with three interchangeable LU-decomposition backends: single-core sequential C++, multi-core **OpenMP**, and GPU **OpenCL**. Built for *EL4045 High Performance Computing* at Institut Teknologi Bandung, with a full IEEE-format paper.

Circuit simulation is a natural HPC case study: SPICE-class simulators spend most of their time solving a linear system, and dense LU is an `O(n^3)` compute-bound kernel (the same one LINPACK uses to rank supercomputers).

## Team

Fairuz Apuilla Rahagi, Adam Mohammad Febrian and Ahmad Ramadhan (group project, STEI ITB).

## Architecture

```
netlist / generator -> Circuit -> MNA stamping -> A, b
                                       |
                       +---------------+----------------+
                       |               |                |
                 Sequential LU     OpenMP LU        OpenCL LU
                 (partial pivot)   (parallel row    (GPU kernels per
                                    elimination)     pivot column)
                       |               |                |
                       +--------> residual check, timing, report
```

| Backend | Strategy |
|---|---|
| Sequential | LU with partial pivoting, the correct and numerically stable baseline |
| OpenMP | `#pragma omp parallel for schedule(dynamic)` over the independent row updates of each pivot column; pivot search stays sequential |
| OpenCL | Two kernels per pivot column (multiplier computation and 2D trailing-submatrix update); no pivoting on the GPU, so every result is verified by its residual `||Ax - b||` against a `1e-6` threshold |

All three use the same elimination algorithm, so timing differences reflect the hardware mapping and not the math. Forward/backward substitution runs on the CPU.

## Correctness

- `./solver --test` checks a voltage divider, a simple loop and a 10-stage ladder against analytically known answers and exits non-zero on failure.
- Netlists can be exported to LTspice (`./generate_testcases` writes `ladder6.net` and `mesh3x3.net`; copies are in `examples/`), and node voltages can be compared with `./solver --compare file.txt`.

## Benchmark results

Dense ladder circuits, matrix size `n` = number of nodes + 1. Each configuration is run 5 times and the minimum time is reported. Full data is in `results/results_openmp.csv`.

| n | Sequential (ms) | OpenMP, 4 threads (speedup) | OpenCL, kernel time (speedup) |
|---|---|---|---|
| 51 | 0.027 | 0.02x | 0.01x |
| 201 | 1.71 | 0.25x | 0.15x |
| 501 | 26.7 | 1.10x | 0.81x |
| 701 | 72.3 | 1.62x | 1.27x |
| 1001 | 248.0 | **2.64x** | **2.35x** |

What this shows:

- **Crossover:** for small systems, both parallel backends are slower than sequential because thread creation, data transfer and kernel launch cost more than the work. Parallelism starts to pay off at roughly n = 500 to 700.
- **More threads is not better:** at n = 1001, 4 OpenMP threads (2.64x) beat the maximum thread count (1.85x). Row elimination is memory-bound, so extra threads fight for memory bandwidth and may oversubscribe hyperthreads.
- **GPU overhead:** counting host-device transfers, the OpenCL total time speedup at n = 1001 is 2.25x versus 2.35x for kernel time alone.

The measurement machine's hardware details are not recorded in the repository, so absolute times will differ on other hardware.

## Build and run

Requirements: `g++` with C++17 and OpenMP. OpenCL is optional (NVIDIA CUDA Toolkit, Intel oneAPI, AMD ROCm or OCL_SDK_Light). Without it, a stub is linked and the GPU solver reports itself as unavailable.

```bash
bash run.sh                # detects OpenCL, builds everything, runs validation and benchmarks
make all                   # or build manually
./solver --test            # validation suite
./solver --bench 1000      # benchmark all backends on a 1000-stage ladder
./benchmark                # full OpenMP x OpenCL sweep, writes results_openmp.csv
```

`run.sh` handles WSL, MSYS2 and Git Bash. Detailed build notes, solver flags and circuit definitions are in [`docs/BUILD_AND_RUN.md`](docs/BUILD_AND_RUN.md).

## Repository layout

| Path | Content |
|---|---|
| `src/` | C++ sources: circuit model, MNA stamping, three LU solvers, CLI, benchmark, test-case generator |
| `include/` | Headers |
| `kernels/` | OpenCL kernels (`lu_factor.cl`, `lu_eliminate.cl`) |
| `examples/` | Example netlists and a reference voltage file |
| `results/` | Raw benchmark data |
| `report/` | IEEE-format paper (LaTeX source, PDF, figures, diagrams). Written in Indonesian. |
| `Makefile`, `run.sh` | Build system |

## Report

[`report/laporan.pdf`](report/laporan.pdf) (6 pages, in Indonesian) covers the problem definition, existing solutions (sparse solvers such as KLU, GPU libraries such as cuSOLVER and MAGMA), system design with data-flow diagrams, MNA stamping rules, the implementation, validation and analysis. The LaTeX source is `report/laporan.tex` and builds with `pdflatex` (IEEEtran class).
