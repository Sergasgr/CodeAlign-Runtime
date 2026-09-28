#!/usr/bin/env bash
# Builds everything:
#   1. native C++/CUDA benchmark harness (CMake)  -> build/gemv_benchmark, build/gemm_benchmark
#   2. the three PyTorch CUDA extensions, in place -> codealign_runtime_{kernels,gemm,transformer}*.so (repo root)
#
# One CUDA toolkit is used for both: $CUDA_HOME if set, otherwise the first `nvcc` on PATH. It is passed to
# CMake explicitly because CMake otherwise prefers an nvcc next to the C++ compiler (e.g. Ubuntu's
# nvidia-cuda-toolkit in /usr/bin, CUDA 12.0), even when another toolkit comes first on PATH.
# Its major version must match the CUDA version torch was built for (13.x for the locked torch 2.13.0+cu130).
#
# Environment overrides:
#   CUDA_HOME=/usr/local/cuda-13.0                  toolkit to use
#   PYTHON="uv run python"                          interpreter used for the setup.py builds
#   CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120"     e.g. when no GPU is visible at configure time
#   TORCH_CUDA_ARCH_LIST="12.0"                      same for the PyTorch extensions
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-uv run python}"

# ---------------------------------------------------------------- pick and check the CUDA toolkit
if [[ -n "${CUDA_HOME:-}" ]]; then
    NVCC="${CUDA_HOME}/bin/nvcc"
else
    NVCC="$(command -v nvcc || true)"
fi
if [[ -z "${NVCC}" || ! -x "${NVCC}" ]]; then
    echo "error: nvcc not found. Install CUDA Toolkit 13.x or set CUDA_HOME=/path/to/cuda" >&2
    exit 1
fi
NVCC="$(readlink -f "${NVCC}")"
export CUDA_HOME="$(cd "$(dirname "${NVCC}")/.." && pwd)"
export CUDACXX="${NVCC}"
export PATH="${CUDA_HOME}/bin:${PATH}"

NVCC_VERSION="$("${NVCC}" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')"
# shellcheck disable=SC2086
TORCH_CUDA="$($PYTHON -c 'import torch; print(torch.version.cuda)')"
echo "==> CUDA toolkit: ${CUDA_HOME} (nvcc ${NVCC_VERSION}); torch built for CUDA ${TORCH_CUDA}"
if [[ "${NVCC_VERSION%%.*}" != "${TORCH_CUDA%%.*}" ]]; then
    echo "error: nvcc ${NVCC_VERSION} and torch (CUDA ${TORCH_CUDA}) have different major versions." >&2
    echo "       Point CUDA_HOME to a CUDA ${TORCH_CUDA%%.*}.x toolkit, e.g. CUDA_HOME=/usr/local/cuda ./build.sh" >&2
    exit 1
fi

# ---------------------------------------------------------------- 1. CMake harness
echo "==> CMake benchmarks"
# A cache from a configure with another nvcc cannot be reused
if [[ -f build/CMakeCache.txt ]] && ! grep -q "^CMAKE_CUDA_COMPILER:[A-Z]*=${NVCC}$" build/CMakeCache.txt; then
    echo "    (build/ was configured with a different nvcc: reconfiguring from scratch)"
    rm -rf build
fi
# shellcheck disable=SC2086
cmake -S . -B build -DCMAKE_CUDA_COMPILER="${NVCC}" ${CMAKE_ARGS:-}
cmake --build build -j

# ---------------------------------------------------------------- 2. PyTorch extensions
for ext in gemv gemm transformer; do
    echo "==> PyTorch extension: ${ext}_setup.py"
    $PYTHON "${ext}_setup.py" build_ext --inplace
done

echo "==> Done. Run the tests with: uv run pytest"
