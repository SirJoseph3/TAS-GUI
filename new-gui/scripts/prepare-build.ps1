# Prepare Build Script for The Anime Scripter GUI
# This script copies all required files to the gerekenler folder

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$newGuiPath = Split-Path -Parent $scriptPath
$rootPath = Split-Path -Parent $newGuiPath
$gerekenlerPath = Join-Path $newGuiPath "gerekenler"

# Use TAS-257 for smaller Lib folder
$tas257Path = "C:\Users\Yusuf\Downloads\TAS-257-Windows"

Write-Host "Preparing build..." -ForegroundColor Cyan
Write-Host "Root: $rootPath" -ForegroundColor Gray
Write-Host "TAS-257: $tas257Path" -ForegroundColor Gray
Write-Host "Gerekenler: $gerekenlerPath" -ForegroundColor Gray

# Create gerekenler folder if not exists
if (-not (Test-Path $gerekenlerPath)) {
    New-Item -ItemType Directory -Force -Path $gerekenlerPath | Out-Null
}

# Files and folders to copy
$filesToCopy = @(
    "main.py",
    "image_upscale_cli.py",
    "python.exe",
    "pythonw.exe",
    "python3.dll",
    "python313.dll",
    "python313.zip",
    "python313._pth",
    "libcrypto-3.dll",
    "libffi-8.dll",
    "libssl-3.dll",
    "sqlite3.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "_asyncio.pyd",
    "_bz2.pyd",
    "_ctypes.pyd",
    "_decimal.pyd",
    "_elementtree.pyd",
    "_hashlib.pyd",
    "_lzma.pyd",
    "_multiprocessing.pyd",
    "_overlapped.pyd",
    "_queue.pyd",
    "_socket.pyd",
    "_sqlite3.pyd",
    "_ssl.pyd",
    "_uuid.pyd",
    "_wmi.pyd",
    "_zoneinfo.pyd",
    "pyexpat.pyd",
    "select.pyd",
    "unicodedata.pyd",
    "winsound.pyd",
    # Requirements files for dependency installation
    "extra-requirements-windows.txt",
    "extra-requirements-windows-lite.txt",
    "extra-requirements-linux.txt",
    "extra-requirements-linux-lite.txt",
    "deprecated-requirements.txt"
)

$foldersToCopy = @(
    "src",
    "ffmpeg",
    "presets",
    "Scripts",
    "weights",
    "custom_models"
)

# Lib folder will be copied from TAS-257 instead
$tas257FoldersToCopy = @(
    "Lib"
)

Write-Host "`nCopying files..." -ForegroundColor Yellow

foreach ($file in $filesToCopy) {
    $sourcePath = Join-Path $rootPath $file
    $destPath = Join-Path $gerekenlerPath $file
    
    if (Test-Path $sourcePath) {
        Copy-Item -Path $sourcePath -Destination $destPath -Force
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $file (not found)" -ForegroundColor DarkGray
    }
}

Write-Host "`nCopying folders..." -ForegroundColor Yellow

foreach ($folder in $foldersToCopy) {
    $sourcePath = Join-Path $rootPath $folder
    $destPath = Join-Path $gerekenlerPath $folder
    
    if (Test-Path $sourcePath) {
        # Remove existing folder first
        if (Test-Path $destPath) {
            Remove-Item -Path $destPath -Recurse -Force
        }
        Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
        Write-Host "  [OK] $folder" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $folder (not found)" -ForegroundColor DarkGray
    }
}

Write-Host "`nCopying folders from TAS-257 (lighter version)..." -ForegroundColor Yellow

foreach ($folder in $tas257FoldersToCopy) {
    $sourcePath = Join-Path $tas257Path $folder
    $destPath = Join-Path $gerekenlerPath $folder
    
    if (Test-Path $sourcePath) {
        # Remove existing folder first
        if (Test-Path $destPath) {
            Remove-Item -Path $destPath -Recurse -Force
        }
        Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
        Write-Host "  [OK] $folder (from TAS-257)" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $folder (not found in TAS-257)" -ForegroundColor DarkGray
    }
}

Write-Host "`nBuild preparation complete!" -ForegroundColor Cyan

# Fix CeLux dependencies - Copy celux_cuda from TAS-252 and add FFmpeg DLLs
Write-Host "`nFixing CeLux dependencies..." -ForegroundColor Yellow

$tas252CeluxCuda = Join-Path $rootPath "Lib\site-packages\celux_cuda"
$tas252Celux = Join-Path $rootPath "Lib\site-packages\celux"
$destCeluxCuda = Join-Path $gerekenlerPath "Lib\site-packages\celux_cuda"
$destCelux = Join-Path $gerekenlerPath "Lib\site-packages\celux"

# Copy celux_cuda from TAS-252 (has CUDA support)
if (Test-Path $tas252CeluxCuda) {
    if (Test-Path $destCeluxCuda) {
        Remove-Item -Path $destCeluxCuda -Recurse -Force
    }
    Copy-Item -Path $tas252CeluxCuda -Destination $destCeluxCuda -Recurse -Force
    Write-Host "  [OK] celux_cuda (from TAS-252)" -ForegroundColor Green
} else {
    Write-Host "  [SKIP] celux_cuda (not found in TAS-252)" -ForegroundColor DarkGray
}

# Copy FFmpeg DLLs to celux folder (for CPU decode)
$ffmpegDlls = @(
    "avcodec-61.dll",
    "avdevice-61.dll",
    "avfilter-10.dll",
    "avformat-61.dll",
    "avutil-59.dll",
    "swresample-5.dll",
    "swscale-8.dll",
    "dav1d.dll",
    "libx264-164.dll",
    "libx265.dll",
    "jpeg62.dll",
    "turbojpeg.dll"
)

if (Test-Path $tas252Celux) {
    foreach ($dll in $ffmpegDlls) {
        $srcDll = Join-Path $tas252Celux $dll
        $dstDll = Join-Path $destCelux $dll
        if (Test-Path $srcDll) {
            Copy-Item -Path $srcDll -Destination $dstDll -Force
        }
    }
    Write-Host "  [OK] FFmpeg DLLs copied to celux" -ForegroundColor Green
}

Write-Host "`nAll CeLux dependencies fixed!" -ForegroundColor Green
Write-Host "Run 'npm run build' to create the executable." -ForegroundColor Gray
