CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Iinclude
OMPFLAGS := -fopenmp

SRC_DIR  := src
OBJ_DIR  := obj

# ---------------------------------------------------------------------------
# OpenCL — detected by run.sh and passed in via environment variables.
# Set HAS_OCL=1, OCL_INC=<include path>, OCL_LIB=<.lib path> before make.
# If unset, the no-op stub is linked and the GPU solver reports unavailable.
# ---------------------------------------------------------------------------
HAS_OCL    ?= 0
OCL_INC    ?=
OCL_LDFLAG ?=

ifeq ($(HAS_OCL),1)
  OCLFLAGS := -I"$(OCL_INC)" -DCL_TARGET_OPENCL_VERSION=120
  OCL_OBJ  := $(OBJ_DIR)/lu_opencl.o
  $(info OpenCL enabled — $(OCL_INC))
else
  OCLFLAGS   :=
  OCL_LDFLAG :=
  OCL_OBJ    := $(OBJ_DIR)/lu_opencl_stub.o
  $(info OpenCL disabled — GPU solver will report unavailable)
endif

# ---------------------------------------------------------------------------
# Common objects (no special flags needed)
# ---------------------------------------------------------------------------
COMMON_OBJ := $(OBJ_DIR)/circuit.o \
               $(OBJ_DIR)/mna.o     \
               $(OBJ_DIR)/lu_sequential.o

.PHONY: all sequential openmp all_solvers clean

# ---------------------------------------------------------------------------
# Default: build every target
# ---------------------------------------------------------------------------
all: test_mna generate_testcases benchmark solver

# ---------------------------------------------------------------------------
# Object directory
# ---------------------------------------------------------------------------
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# ---------------------------------------------------------------------------
# Generic compile rule (no special flags)
# ---------------------------------------------------------------------------
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ---------------------------------------------------------------------------
# Per-file overrides for special flags
# ---------------------------------------------------------------------------
$(OBJ_DIR)/lu_openmp.o: $(SRC_DIR)/lu_openmp.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

$(OBJ_DIR)/lu_opencl.o: $(SRC_DIR)/lu_opencl.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OCLFLAGS) -c $< -o $@

$(OBJ_DIR)/benchmark.o: $(SRC_DIR)/benchmark.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(OCLFLAGS) -c $< -o $@

# main.cpp compiled three ways depending on which solvers are enabled
$(OBJ_DIR)/main_seq.o: $(SRC_DIR)/main.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/main_omp.o: $(SRC_DIR)/main.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DHAS_OMP -c $< -o $@

$(OBJ_DIR)/main_all.o: $(SRC_DIR)/main.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(OCLFLAGS) -DHAS_OMP -DHAS_OCL -c $< -o $@

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------

# MNA matrix printer (no solver dependency)
test_mna: $(COMMON_OBJ) $(OBJ_DIR)/test_mna.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# Circuit generator + LTspice exporter
generate_testcases: $(COMMON_OBJ) $(OBJ_DIR)/generate_testcases.o \
                    $(OBJ_DIR)/lu_sequential.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# Full benchmark (OpenMP + OpenCL)
benchmark: $(COMMON_OBJ) $(OBJ_DIR)/lu_openmp.o $(OCL_OBJ) $(OBJ_DIR)/benchmark.o
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ $(OCL_LDFLAG) -o $@

# ---- solver variants ----

# Sequential only (no OpenMP, no OpenCL required)
sequential: $(COMMON_OBJ) $(OBJ_DIR)/main_seq.o
	$(CXX) $(CXXFLAGS) $^ -o solver
	@echo "Built: solver  [sequential only]"

# Sequential + OpenMP
openmp: $(COMMON_OBJ) $(OBJ_DIR)/lu_openmp.o $(OBJ_DIR)/main_omp.o
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o solver
	@echo "Built: solver  [sequential + OpenMP]"

# All solvers (sequential + OpenMP + OpenCL)
all_solvers solver: $(COMMON_OBJ) $(OBJ_DIR)/lu_openmp.o $(OCL_OBJ) $(OBJ_DIR)/main_all.o
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ $(OCL_LDFLAG) -o solver
	@echo "Built: solver  [sequential + OpenMP + OpenCL]"

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -rf $(OBJ_DIR)
	rm -f  test_mna      test_mna.exe
	rm -f  generate_testcases generate_testcases.exe
	rm -f  benchmark     benchmark.exe
	rm -f  solver        solver.exe
	rm -f  results_openmp.csv ladder6.net mesh3x3.net
