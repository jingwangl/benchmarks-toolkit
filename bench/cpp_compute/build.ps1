# PowerShell version of C++ build script
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = "$ROOT\build"

# Create build directory
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
}

# Build with O2 optimization
Write-Host "Building C++ compute benchmark (O2)..."
g++ -O2 -std=c++17 -pthread "$ROOT\main.cpp" -o "$buildDir\cpp_compute_O2.exe"

# Build with O3 optimization  
Write-Host "Building C++ compute benchmark (O3)..."
g++ -O3 -std=c++17 -pthread "$ROOT\main.cpp" -o "$buildDir\cpp_compute_O3.exe"

Write-Host "Build completed successfully."
