# PowerShell version of C++ compute benchmark runner
param(
    [string]$opt = "O2",
    [int]$threads = 1,
    [int]$iters = 90000000
)

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$bin = "$ROOT\build\cpp_compute_$opt.exe"
$out = "$ROOT\..\..\out\cpp_compute_raw.csv"

# Check if binary exists
if (-not (Test-Path $bin)) {
    Write-Error "Binary $bin not found. Run build.ps1 first."
    exit 1
}

# Create output directory
$outDir = Split-Path -Parent $out
if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

# Add header if file doesn't exist
if (-not (Test-Path $out)) {
    "bench,config,threads,wall_ms,iterations" | Out-File -FilePath $out -Encoding UTF8
}

# Run 5 times and append results for better statistical significance
for ($i = 1; $i -le 30; $i++) {
    $result = & $bin $iters $threads
    # Parse the output format: wall_ms=xxx,iterations=xxx,threads=xxx
    if ($result -match "wall_ms=([\d.]+)") {
        $wall = $matches[1]
        "cpp_compute,$opt,$threads,$wall,$iters" | Add-Content -Path $out -Encoding UTF8
    } else {
        Write-Warning "Failed to parse wall_ms from output: $result"
    }
}

Write-Host "Wrote results to $out"
