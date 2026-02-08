$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# This project ships a bundled Python; use it (not system python).
$py = Join-Path $root 'python.exe'
if (-not (Test-Path $py)) {
  throw "python.exe not found in $root"
}

Write-Host "[1/3] Installing PyInstaller (build-time dependency)" -ForegroundColor Cyan
& $py -m pip install --upgrade pyinstaller

Write-Host "[2/3] Building TAS-GUI.exe launcher" -ForegroundColor Cyan
# Clean previous PyInstaller build artifacts
Remove-Item -Recurse -Force (Join-Path $root 'build') -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force (Join-Path $root 'dist') -ErrorAction SilentlyContinue
Remove-Item -Force (Join-Path $root 'TAS-GUI.spec') -ErrorAction SilentlyContinue

$icon = Join-Path $root 'src/assets/icon.ico'
$iconArg = @()
if (Test-Path $icon) {
  $iconArg = @('--icon', $icon)
}

& $py -m PyInstaller --noconfirm --clean --onefile --noconsole --name 'TAS-GUI' @iconArg (Join-Path $root 'launch_gui.py')

Write-Host "[3/3] Copying result to project root" -ForegroundColor Cyan
Copy-Item (Join-Path $root 'dist/TAS-GUI.exe') (Join-Path $root 'TAS-GUI.exe') -Force

Write-Host "DONE: $(Join-Path $root 'TAS-GUI.exe')" -ForegroundColor Green
Write-Host "" 
Write-Host "NOTE:" -ForegroundColor Yellow
Write-Host "- This launcher EXE does NOT reduce size; it starts gui_app.py using the bundled pythonw.exe." -ForegroundColor Yellow
Write-Host "- Distribute TAS-GUI.exe together with the existing TAS folder contents (pythonw.exe, Lib/, weights/, ffmpeg/, etc.)." -ForegroundColor Yellow
