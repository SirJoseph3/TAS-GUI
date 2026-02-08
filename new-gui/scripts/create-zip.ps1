# Create ZIP distribution for The Anime Scripter GUI
# This script creates a portable ZIP distribution from the built app

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$newGuiPath = Split-Path -Parent $scriptPath
$distPath = Join-Path $newGuiPath "dist"
$winUnpackedPath = Join-Path $distPath "win-unpacked"
$version = "1.0.0"
$zipName = "The-Anime-Scripter-$version-win-x64.zip"
$zipPath = Join-Path $distPath $zipName

Write-Host "Creating ZIP distribution..." -ForegroundColor Cyan
Write-Host "Source: $winUnpackedPath" -ForegroundColor Gray
Write-Host "Output: $zipPath" -ForegroundColor Gray

if (-not (Test-Path $winUnpackedPath)) {
    Write-Host "Error: win-unpacked folder not found. Run 'npm run build' first." -ForegroundColor Red
    exit 1
}

# Remove existing zip if present
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
    Write-Host "Removed existing ZIP file" -ForegroundColor Yellow
}

Write-Host "`nCompressing files (this may take a while for large packages)..." -ForegroundColor Yellow

try {
    Compress-Archive -Path "$winUnpackedPath\*" -DestinationPath $zipPath -CompressionLevel Optimal
    
    $zipSize = (Get-Item $zipPath).Length
    $zipSizeGB = [math]::Round($zipSize / 1GB, 2)
    
    Write-Host "`nZIP distribution created successfully!" -ForegroundColor Green
    Write-Host "File: $zipName" -ForegroundColor Cyan
    Write-Host "Size: $zipSizeGB GB" -ForegroundColor Cyan
} catch {
    Write-Host "Error creating ZIP: $_" -ForegroundColor Red
    exit 1
}
