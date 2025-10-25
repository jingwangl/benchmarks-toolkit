# 主运行脚本的PowerShell版本
param([string]$cmd = "all")


$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

switch ($cmd) {
    "build" {
        & "$ROOT\bench\cpp_compute\build.ps1"
    }
    "run" {
        # 运行C++计算基准测试
        Write-Host "Running C++ computational benchmark tests..."
        & "$ROOT\bench\cpp_compute\build.ps1"
        $threads = @(1, 4, 8)
        foreach ($th in $threads) {
            & "$ROOT\bench\cpp_compute\run.ps1" -opt O2 -threads $th
            & "$ROOT\bench\cpp_compute\run.ps1" -opt O3 -threads $th
        }
        
        # 运行Python I/O基准测试
        Write-Host "Running Python I/O benchmark tests..."
        Push-Location "$ROOT\bench\py_io"
        & ".\run.ps1"
        Pop-Location
        
        # 运行LiDAR处理基准测试
        Write-Host "Running LiDAR processing benchmark tests..."
        Push-Location "$ROOT\bench\lidar_processing"
        & ".\run.ps1"
        Pop-Location
    }
    "analyze" {
        New-Item -ItemType Directory -Force -Path "$ROOT\report\figs" | Out-Null
        python "$ROOT\analyze\parse.py" "$ROOT\out\*_raw.csv" -o "$ROOT\out\metrics.csv"
        python "$ROOT\analyze\plot.py" "$ROOT\out\metrics.csv" -o "$ROOT\report\figs"
    }
    "report" {
        # 生成环境信息（如果不存在）
        if (-not (Test-Path "$ROOT\out\env.txt")) {
            Write-Host "Generating environment information..."
            & "$ROOT\envinfo.sh" > "$ROOT\out\env.txt" 2>$null
        }
        
        # 使用Python报告生成器生成综合报告
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
