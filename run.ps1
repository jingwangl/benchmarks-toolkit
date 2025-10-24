# PowerShell version of the main run script
param([string]$cmd = "all")

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

switch ($cmd) {
    "build" {
        & "$ROOT\bench\cpp_compute\build.sh"
    }
    "run" {
        # Run traditional benchmarks
        & "$ROOT\collect\run_benchmarks.sh"
        
        # Run LiDAR processing benchmark
        Write-Host "Running LiDAR Processing Benchmark..."
        & "$ROOT\bench\lidar_processing\run.ps1"
    }
    "analyze" {
        New-Item -ItemType Directory -Force -Path "$ROOT\report\figs" | Out-Null
        python "$ROOT\analyze\parse.py" "$ROOT\out\*_raw.csv" -o "$ROOT\out\metrics.csv"
        python "$ROOT\analyze\plot.py" "$ROOT\out\metrics.csv" -o "$ROOT\report\figs"
    }
    "report" {
        # Generate environment info if not exists
        if (-not (Test-Path "$ROOT\out\env.txt")) {
            Write-Host "Generating environment information..."
            & "$ROOT\envinfo.sh" > "$ROOT\out\env.txt" 2>$null
        }
        
        # Generate comprehensive report using Python report generator
        Write-Host "Generating comprehensive performance report..."
        python "$ROOT\analyze\report_generator.py" -m "$ROOT\out\metrics.csv" -e "$ROOT\out\env.txt" -o "$ROOT\report\REPORT.md" -f "$ROOT\report\figs"
        Write-Host "Report written to report/REPORT.md"
    }
    "all" {
        & "$ROOT\run.ps1" build
        & "$ROOT\run.ps1" run
        & "$ROOT\run.ps1" analyze
        & "$ROOT\run.ps1" report
    }
    default {
        Write-Host "Usage: $0 [build|run|analyze|report|all]"; exit 1
    }
}
