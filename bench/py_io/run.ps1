# PowerShell version of py_io run script
$out = "..\..\out\py_io_raw.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null
"bench,block_kb,wall_ms,count" | Out-File -FilePath $out -Encoding UTF8

$blockSizes = @(4, 64, 512)
foreach ($kb in $blockSizes) {
    for ($i = 1; $i -le 3; $i++) {
        $result = python io_bench.py --block_kb $kb --count 20 --optimized 2>$null
        if ($result) {
            $wall = ($result -split '[=,]')[1]
            "py_io,$kb,$wall,20" | Add-Content -Path $out -Encoding UTF8
        }
    }
}
Write-Host "Wrote results to $out"