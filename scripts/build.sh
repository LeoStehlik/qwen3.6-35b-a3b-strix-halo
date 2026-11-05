#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${1:-"$repo_root/work/prefill-stage2"}
test -f "$source_dir/CMakeLists.txt"
cmake -S "$source_dir" -B "$source_dir/build-vulkan" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG' \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG' \
  -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_WEBUI=OFF
targets=(llama-server llama-bench llama-perplexity test-backend-ops test-mtp-hidden)
if test -f "$source_dir/tests/test-fa-mr-numeric.cpp"; then
  targets+=(test-fa-mr-numeric)
fi
if test -f "$source_dir/tests/test-mtp-unused-head.cpp"; then
    targets+=(test-mtp-unused-head)
fi
for target in test-mmid-fp4 test-mmid-ownership; do
    if test -f "$source_dir/tests/$target.cpp"; then
        targets+=("$target")
    fi
done
cmake --build "$source_dir/build-vulkan" --target "${targets[@]}" -j "${BUILD_JOBS:-8}"
