#!/usr/bin/env bash
set -e

# ---------------------------------------------------------------------------
# OpenCL detection — handles three environments automatically:
#   WSL (Linux g++):    headers from /mnt/c/..., link via -lOpenCL (ICD loader)
#   MSYS2 bash:         headers and .lib from /c/...
#   Git Bash:           same as MSYS2
# ---------------------------------------------------------------------------
export HAS_OCL=0
export OCL_INC=""
export OCL_LDFLAG=""

_OS=$(uname -s)

if [[ "$_OS" == "Linux" ]]; then
    # WSL — g++ is Ubuntu/Linux, so use Linux paths and the ICD shared library
    for VER in v12.4 v13.1; do
        HDR="/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/$VER/include"
        if [ -f "$HDR/CL/cl.h" ]; then
            export HAS_OCL=1
            export OCL_INC="$HDR"
            # No libOpenCL.so symlink in WSL — link directly against the versioned .so
            _LIBOCL=""
            for _F in /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 \
                       /usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0; do
                [ -f "$_F" ] && { _LIBOCL="$_F"; break; }
            done
            export OCL_LDFLAG="${_LIBOCL:--lOpenCL}"
            echo "OpenCL detected: CUDA $VER (WSL — linking $OCL_LDFLAG)"
            break
        fi
    done
else
    # MSYS2 / Git Bash — g++ is MinGW, use /c/ paths and the Windows .lib
    for VER in v12.4 v13.1; do
        HDR="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/$VER/include"
        LIB="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/$VER/lib/x64/OpenCL.lib"
        if [ -f "$HDR/CL/cl.h" ]; then
            export HAS_OCL=1
            export OCL_INC="$HDR"
            export OCL_LDFLAG="\"$LIB\""
            echo "OpenCL detected: CUDA $VER (MSYS2 — linking $LIB)"
            break
        fi
    done
fi

if [ "$HAS_OCL" = "0" ]; then
    echo "OpenCL not found — building without GPU support"
fi

# ---------------------------------------------------------------------------
echo "=== Building ==="
rm -rf obj/
rm -f solver solver.exe test_mna test_mna.exe benchmark benchmark.exe generate_testcases generate_testcases.exe
make all

echo ""
echo "=== Testing ==="
./solver --test

echo ""
echo "=== Benchmarking (N=500) ==="
./solver --bench 500

echo ""
echo "=== Benchmarking (N=1000) ==="
./solver --bench 1000
