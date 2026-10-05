# Parallel MNA Circuit Solver

Modified Nodal Analysis (MNA) circuit solver with three parallel backends:
sequential (LU decomposition), OpenMP (parallel elimination), and OpenCL (GPU).

## Requirements

- **g++** with C++17 support (`g++ --version` ≥ 7)
- **OpenMP** — usually bundled with g++ (`-fopenmp`)
- **OpenCL SDK** *(optional)* — one of:
  - NVIDIA CUDA Toolkit (provides OpenCL headers + `libOpenCL`)
  - Intel oneAPI / OpenCL Runtime
  - AMD ROCm
  - OCL_SDK_Light (lightweight SDK, no runtime — runtime comes from the GPU driver)
- Any OpenCL 1.2+ capable device (CPU or GPU)

On Windows, use the **MSYS2 MinGW x64** shell (not the plain MSYS2 shell — it
lacks `g++` on PATH).  OpenCL is auto-detected from the NVIDIA CUDA Toolkit or
OCL_SDK_Light; if absent, a no-op stub is linked and the GPU solver reports
unavailable at runtime.

## Quick start (recommended)

Open **MSYS2 MinGW x64**, navigate to the project, and run:

```bash
bash run.sh
```

This builds everything, runs the validation suite, and benchmarks at N=500 and
N=1000 in one step.

## Compile manually

```bash
make all          # builds everything: test_mna, generate_testcases, benchmark, solver
make sequential   # solver — sequential LU only  (no OpenMP/OpenCL required)
make openmp       # solver — sequential + OpenMP
make all_solvers  # solver — all three backends
make clean        # remove all build artefacts
```

All compile commands are printed during the build.

## Run

```bash
./solver --test              # validation suite (3 circuits, PASS/FAIL each)
./solver --bench 500         # benchmark three solvers on ladder N=500
./solver --bench 1000        # benchmark three solvers on ladder N=1000
./generate_testcases         # build performance table + export LTspice netlists
./benchmark                  # full OpenMP × OpenCL sweep + results_openmp.csv
```

## Solver modes

| Flag | Description |
|------|-------------|
| `--test` | Validates voltage divider, simple loop, and 10-stage ladder against analytically known answers. Exits with code 1 if any check fails. |
| `--bench N` | Builds a ladder circuit of N stages (MNA matrix size N+1), runs all available solvers 5 times each, reports minimum time and speedup versus sequential. |
| `--compare file.txt` | Reads reference node voltages from a plain-text file and compares them against the sequential solver on the voltage divider circuit. |

## Compare file format

One `node_name voltage` pair per line (spaces separated, `#` comments ignored):

```
# LTspice .op export — voltage divider (3×1Ω, Vs=5V)
N001 1.666667
N002 3.333333
N003 5.000000
```

Node label `N001` maps to internal node 1 (solution index 0), `N002` to node 2, etc.

## How to validate against LTspice

1. Generate netlists:
   ```bash
   ./generate_testcases
   ```
   Produces `ladder6.net` and `mesh3x3.net`.

2. Open `ladder6.net` in LTspice, run `.op` simulation.

3. Export node voltages:
   `File > Export Netlist > ASCII .raw`
   then manually format as `N001 <voltage>` per line.

4. Compare against this solver:
   ```bash
   ./solver --compare ladder6_ltspice.txt
   ```

## Circuit topology

### Ladder (generateLadder / makeLadder)

```
GND --[Vs=5V]--> n1 --[1Ω]--> n2 --[1Ω]--> ... --[1Ω]--> n_N --[1Ω]--> GND
```

MNA matrix size: N+1 (N node rows + 1 voltage-source branch row).
V(nk) = 5·(N−k+1)/N, so V(n1)=5V, V(nN)=5/N V.

### Voltage divider (3-resistor chain)

```
GND --[Vs=5V]--> n3 --[1Ω]--> n2 --[1Ω]--> n1 --[1Ω]--> GND
```

V(n1) = 5/3 ≈ 1.667 V (label N001)
V(n2) = 10/3 ≈ 3.333 V (label N002)
V(n3) = 5 V (label N003, VS-connected)

### Mesh (generateMesh)

G×G resistor grid, node (0,0) driven by Vs=5V, node (G−1,G−1) = GND.
All horizontal and vertical edges carry 1 Ω resistors.

## Parallel strategies

| Solver | Parallelism | Notes |
|--------|------------|-------|
| `SequentialSolver` | None | LU with partial pivoting (CPU) |
| `OpenMPSolver` | `#pragma omp parallel for` on row elimination loop | `schedule(dynamic)` per pivot column; partial pivoting sequential |
| `OpenCLSolver` | 2D GPU kernel per pivot column | No pivoting (GPU LU); residual verified post-solve |

## File structure

```
project/
├── include/
│   ├── circuit.hpp        — Circuit class, element structs
│   ├── mna.hpp            — MNA builder (G-stamp, VS-stamp)
│   └── solver.hpp         — SequentialSolver, OpenMPSolver, OpenCLSolver
├── kernels/
│   ├── lu_factor.cl       — compute multipliers for column k
│   └── lu_eliminate.cl    — update trailing submatrix (2D kernel)
├── src/
│   ├── circuit.cpp
│   ├── mna.cpp
│   ├── lu_sequential.cpp  — LU + partial pivoting (CPU)
│   ├── lu_openmp.cpp      — same + OpenMP inner loop
│   ├── lu_opencl.cpp      — OpenCL init, solve, timing
│   ├── main.cpp           — unified CLI (--test / --bench / --compare)
│   ├── benchmark.cpp      — sweep benchmark → results_openmp.csv
│   ├── generate_testcases.cpp — performance sweep + LTspice export
│   └── test_mna.cpp       — prints MNA matrices for two circuits
└── Makefile
```
