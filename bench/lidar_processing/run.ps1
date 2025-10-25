# LiDAR Processing Benchmark Runner - PowerShell Version
# Part of Autonomous Vehicle Performance Analysis Toolkit

$out = "..\..\out\lidar_processing_raw.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null

# Create header and clear existing data
"bench,points,wall_ms,iterations" | Out-File -FilePath $out -Encoding UTF8

Write-Host "Running LiDAR processing benchmark tests..."

# Test different point cloud sizes (simulating different LiDAR resolutions)
# Optimized for home computer compatibility
$pointCounts = @(5000, 10000, 20000, 50000)
$iterations = 3

foreach ($points in $pointCounts) {
    Write-Host "Testing point cloud with $points points..."
    
    for ($i = 1; $i -le $iterations; $i++) {
        # Run benchmark and capture output (including stdout and stderr)
        # Use absolute path to ensure correct execution
        $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
        $output = python "$scriptDir\lidar_bench.py" --points $points --iterations 1 --quiet --parallel 2>&1
        
        # Extract CSV line from output
        $csvLine = $output | Select-String "lidar_processing," | Select-Object -Last 1
        
        if ($csvLine) {
            $csvLine.Line | Add-Content -Path $out -Encoding UTF8
            $wallTime = ($csvLine.Line -split ',')[2]
            Write-Host "  Iteration $i : $wallTime ms"
        } else {
            Write-Host "  Warning: No result captured for iteration $i"
        }
    }
}

Write-Host "LiDAR benchmark tests completed. Results written to $out"