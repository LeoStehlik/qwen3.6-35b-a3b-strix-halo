#!/usr/bin/env bash
set -euo pipefail
diagnostic_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(realpath -- "${1:?Usage: build-trace.sh MR_SOURCE OUTPUT_BINARY}")
output_binary=${2:?Usage: build-trace.sh MR_SOURCE OUTPUT_BINARY}
mkdir -p -- "$(dirname -- "$output_binary")"
g++ -std=c++17 -O2 \
  -I "$source_dir/include" -I "$source_dir/ggml/include" \
  -I "$source_dir/src" -I "$source_dir/vendor/nlohmann" \
  "$diagnostic_dir/trace-numerics.cpp" \
  -L "$source_dir/build-vulkan/bin" \
  -Wl,-rpath,"$source_dir/build-vulkan/bin" \
  -lllama -lggml-base -o "$output_binary"
