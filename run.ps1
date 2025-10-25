# 主运行脚本的PowerShell版本
param([string]$cmd = "all")

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

switch ($cmd) {
    "build" {
        & "$ROOT\bench\cpp_compute\build.sh"
    }
    "run" {
        # 运行传统基准测试
        & "$ROOT\collect\run_benchmarks.sh"
        
        # 运行LiDAR处理基准测试
        Write-Host "正在运行LiDAR处理基准测试..."
        & "$ROOT\bench\lidar_processing\run.ps1"
    }
    "analyze" {
        New-Item -ItemType Directory -Force -Path "$ROOT\report\figs" | Out-Null
        python "$ROOT\analyze\parse.py" "$ROOT\out\*_raw.csv" -o "$ROOT\out\metrics.csv"
        python "$ROOT\analyze\plot.py" "$ROOT\out\metrics.csv" -o "$ROOT\report\figs"
    }
    "report" {
        # 生成环境信息（如果不存在）
        if (-not (Test-Path "$ROOT\out\env.txt")) {
            Write-Host "正在生成环境信息..."
            & "$ROOT\envinfo.sh" > "$ROOT\out\env.txt" 2>$null
        }
        
        # 使用Python报告生成器生成综合报告
        Write-Host "正在生成综合性能报告..."
        python "$ROOT\analyze\report_generator.py" -m "$ROOT\out\metrics.csv" -e "$ROOT\out\env.txt" -o "$ROOT\report\REPORT.md" -f "$ROOT\report\figs"
        Write-Host "报告已写入 report/REPORT.md"
    }
    "all" {
        & "$ROOT\run.ps1" build
        & "$ROOT\run.ps1" run
        & "$ROOT\run.ps1" analyze
        & "$ROOT\run.ps1" report
    }
    default {
        Write-Host "用法：$0 [build|run|analyze|report|all]"; exit 1
    }
}
