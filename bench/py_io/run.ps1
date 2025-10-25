# PowerShell version of py_io run script - Fixed Total Data Size Mode
$out = "..\..\out\py_io_raw.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
"bench,block_kb,wall_ms,total_mb,blocks" | Out-File -FilePath $out -Encoding UTF8

$blockSizes = @(4, 64, 512)
$totalMb = 1  # Fixed total data size: 1MB

foreach ($kb in $blockSizes) {
    for ($i = 1; $i -le 3; $i++) {
        $result = python io_bench.py --block_kb $kb --total_mb $totalMb --mode fixed_total 2>$null
        if ($result) {
            $parts = $result -split ','
            $wall = ($parts[0] -split '=')[1]
            $blocks = ($parts[3] -split '=')[1]
            "py_io,$kb,$wall,$totalMb,$blocks" | Add-Content -Path $out -Encoding UTF8
        }
    }
}
Write-Host "Wrote results to $out"