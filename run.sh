#!/usr/bin/env bash
set -euo pipefail
cmd=${1:-all}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$cmd" in
  build)
    bash "$ROOT/bench/cpp_compute/build.sh"
    ;;
  run)
    bash "$ROOT/collect/run_benchmarks.sh"
    ;;
  analyze)
    mkdir -p "$ROOT/report/figs"
    python3 "$ROOT/analyze/parse.py" "$ROOT"/out/*_raw.csv -o "$ROOT/out/metrics.csv"
    python3 "$ROOT/analyze/plot.py" "$ROOT/out/metrics.csv" -o "$ROOT/report/figs"
    ;;
  report)
    ts=$(date)
    {
      echo "# Benchmark Report"
      echo
      echo "**Generated:** $ts"
      echo
      echo "## Environment"
      echo
      cat "$ROOT/out/env.txt" 2>/dev/null || true
      echo
      echo "## Summary Figures"
      echo
      for img in "$ROOT"/report/figs/*.png; do
        [ -f "$img" ] && echo "![](figs/$(basename "$img"))"
      done
    } > "$ROOT/report/REPORT.md"
    echo "Report written to report/REPORT.md"
    ;;
  all)
    $0 build
    $0 run
    $0 analyze
    $0 report
    ;;
  *)
    echo "Usage: $0 [build|run|analyze|report|all]"; exit 1;;
esac
