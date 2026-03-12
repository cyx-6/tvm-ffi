# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Install LLVM from conda-forge using micromamba (Windows).
# Usage: powershell -ExecutionPolicy Bypass -File tools/install_llvm.ps1 [version]
#   version defaults to LLVM_VERSION env var, then 22.1.0

param(
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $Version) {
    $Version = if ($env:LLVM_VERSION) { $env:LLVM_VERSION } else { "22.1.0" }
}
$Prefix = if ($env:LLVM_PREFIX) { $env:LLVM_PREFIX } else { "C:\opt\llvm" }

Write-Host "Installing LLVM $Version to $Prefix"

# Install micromamba
$MicromambaUrl = "https://micro.mamba.pm/api/micromamba/win-64/latest"
$MicromambaDir = "$env:TEMP\micromamba"
$MicromambaExe = "$MicromambaDir\Library\bin\micromamba.exe"

if (-not (Test-Path $MicromambaExe)) {
    Write-Host "Downloading micromamba..."
    New-Item -ItemType Directory -Path $MicromambaDir -Force | Out-Null
    $tarball = "$env:TEMP\micromamba.tar.bz2"
    Invoke-WebRequest -Uri $MicromambaUrl -OutFile $tarball
    # Extract using tar (available on Windows 10+)
    tar -xvjf $tarball -C $MicromambaDir
    if (-not (Test-Path $MicromambaExe)) {
        throw "Failed to extract micromamba"
    }
}
Write-Host "Using micromamba: $MicromambaExe"

# Create environment with LLVM
Write-Host "Creating conda environment with LLVM $Version..."
& $MicromambaExe create -p $Prefix -c conda-forge `
    "llvmdev=$Version" "clangdev=$Version" "compiler-rt=$Version" `
    zlib `
    -y
if ($LASTEXITCODE -ne 0) { throw "micromamba create failed" }

# Build libzstd from source with CMake (MSVC static)
$ZstdVersion = "1.5.7"
$ZstdTarball = "$env:TEMP\zstd-$ZstdVersion.tar.gz"
$ZstdSrc = "$env:TEMP\zstd-$ZstdVersion"

Write-Host "Building zstd $ZstdVersion from source..."
if (-not (Test-Path $ZstdTarball)) {
    Invoke-WebRequest -Uri "https://github.com/facebook/zstd/releases/download/v$ZstdVersion/zstd-$ZstdVersion.tar.gz" -OutFile $ZstdTarball
}
if (Test-Path $ZstdSrc) { Remove-Item -Recurse -Force $ZstdSrc }
tar -xzf $ZstdTarball -C $env:TEMP

$ZstdBuild = "$env:TEMP\_zstd_build"
if (Test-Path $ZstdBuild) { Remove-Item -Recurse -Force $ZstdBuild }

cmake -S "$ZstdSrc\build\cmake" -B $ZstdBuild `
    -DCMAKE_INSTALL_PREFIX="$Prefix" `
    -DZSTD_BUILD_SHARED=OFF -DZSTD_BUILD_STATIC=ON `
    -DZSTD_BUILD_PROGRAMS=OFF
if ($LASTEXITCODE -ne 0) { throw "zstd cmake configure failed" }

cmake --build $ZstdBuild --config Release --target install -j $env:NUMBER_OF_PROCESSORS
if ($LASTEXITCODE -ne 0) { throw "zstd build failed" }

# Cleanup
Remove-Item -Recurse -Force $ZstdSrc -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $ZstdBuild -ErrorAction SilentlyContinue

Write-Host "LLVM $Version installed to $Prefix"
