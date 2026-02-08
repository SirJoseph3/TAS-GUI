@echo off
echo Preparing build...
powershell -ExecutionPolicy Bypass -File "%~dp0prepare-build.ps1"
pause
