#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
g++ -O2 -std=c++17 main.cpp -o build/cpp_compute_O2
g++ -O3 -std=c++17 main.cpp -o build/cpp_compute_O3
echo "已构建：build/cpp_compute_O2 和 build/cpp_compute_O3"
