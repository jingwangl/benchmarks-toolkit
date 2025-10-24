#!/usr/bin/env bash
# LiDAR Processing Benchmark Runner
# Part of the Autonomous Vehicle Performance Analysis Toolkit

set -euo pipefail
cd "$(dirname "$0")"

# Configuration
out="$(realpath ../../out)/lidar_processing_raw.csv"
mkdir -p "$(dirname "$out")"

# Create header if file doesn't exist
[ -f "$out" ] || echo "bench,points,wall_ms,iterations" > "$out"

echo "Running LiDAR Processing Benchmark..."

# Test different point cloud sizes (simulating different LiDAR resolutions)
point_counts=(50000 100000 200000 500000)
iterations=5

for points in "${point_counts[@]}"; do
    echo "Testing with $points points..."
    
    for i in $(seq 1 $iterations); do
        # Run benchmark and capture output
        result=$(python3 lidar_bench.py --points $points --iterations 1 2>/dev/null | grep "lidar_processing," | tail -1)
        
        if [ -n "$result" ]; then
            echo "$result" >> "$out"
            echo "  Iteration $i: $(echo $result | cut -d',' -f3)ms"
        else
            echo "  Warning: No result captured for iteration $i"
        fi
    done
done

echo "LiDAR benchmark completed. Results written to $out"
