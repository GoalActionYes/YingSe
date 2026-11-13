@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.tools\vision-advisor-env\Scripts\python.exe"
if not exist "%PY%" (
  echo 尚未准备局部调色研究环境，请阅读 docs\research\2026-11-13-spatial-workbench.md。
  pause
  exit /b 1
)
"%PY%" "%ROOT%\experimental\adaptive-reference-20261113\launch_spatial_workbench.py"
if errorlevel 1 pause
