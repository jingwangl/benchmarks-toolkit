#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/out"
bash "$ROOT/envinfo.sh" > "$ROOT/out/env.txt" || true
bash "$ROOT/bench/cpp_compute/build.sh"
for th in 1 4 8; do
  LABEL="cpp_${th}" bash "$ROOT/bench/cpp_compute/run.sh" O2 $th
  LABEL="cpp_${th}" bash "$ROOT/bench/cpp_compute/run.sh" O3 $th
done
bash "$ROOT/bench/py_io/run.sh"
echo "基准测试运行完成。"
