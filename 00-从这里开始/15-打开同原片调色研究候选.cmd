@echo off
chcp 65001 >nul
setlocal
pushd "%~dp0.."
set "PYTHONPATH=%CD%;%CD%\.tools\tf-convert-env\Lib\site-packages"
if not exist ".tools\adaptive-env\Scripts\python.exe" (
  echo 需要映色本机研究环境，详见 docs\research\2026-11-24-same-capture-workbench.md。
  popd
  pause
  exit /b 1
)
".tools\adaptive-env\Scripts\python.exe" "experimental\adaptive-reference-20261113\same_capture_workbench.py" --open %*
if errorlevel 1 pause
popd
