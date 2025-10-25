# LiDAR处理基准测试运行器的PowerShell版本
# 自动驾驶车辆性能分析工具包的一部分

$out = "..\..\out\lidar_processing_raw.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null

# 创建标题并清空现有数据
"bench,points,wall_ms,iterations" | Out-File -FilePath $out -Encoding UTF8

Write-Host "正在运行LiDAR处理基准测试..."

# 测试不同的点云大小（模拟不同的LiDAR分辨率）
# 针对家用电脑兼容性进行了优化
$pointCounts = @(5000, 10000, 20000, 50000)
$iterations = 3

foreach ($points in $pointCounts) {
    Write-Host "正在测试 $points 个点的点云..."
    
    for ($i = 1; $i -le $iterations; $i++) {
        # 运行基准测试并捕获输出（包括stdout和stderr）
        # 使用绝对路径确保正确执行
        $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
        $output = python "$scriptDir\lidar_bench.py" --points $points --iterations 1 --quiet --parallel 2>&1
        
        # 从输出中提取CSV行
        $csvLine = $output | Select-String "lidar_processing," | Select-Object -Last 1
        
        if ($csvLine) {
            $csvLine.Line | Add-Content -Path $out -Encoding UTF8
            $wallTime = ($csvLine.Line -split ',')[2]
            Write-Host "  迭代 $i：$wallTime ms"
        } else {
            Write-Host "  警告：第 $i 次迭代未捕获到结果"
        }
    }
}

Write-Host "LiDAR基准测试完成。结果已写入 $out"
