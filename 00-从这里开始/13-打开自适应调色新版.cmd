@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.tools\vision-advisor-env\Scripts\python.exe"
if not exist "%PY%" (
  echo 尚未准备本机研究环境，请阅读 docs\research\2026-11-13-multiref-workbench.md。
  pause
  exit /b 1
)
"%PY%" "%ROOT%\experimental\adaptive-reference-20261113\launch_spatial_workbench.py" --multiref %*
if errorlevel 1 pause
