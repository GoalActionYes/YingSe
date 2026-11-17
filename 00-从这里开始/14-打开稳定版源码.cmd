@echo off
chcp 65001 >nul
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%\.tools\vision-advisor-env\Scripts\python.exe"
if not exist "%PY%" (
  echo 缺少运行环境，请先按 docs\GETTING_STARTED.md 准备源码环境。
  pause
  exit /b 1
)
pushd "%ROOT%"
"%PY%" -c "from reference_color_app.media import ffmpeg_path; print('FFmpeg ready:', ffmpeg_path())"
if errorlevel 1 (
  echo 依赖检查失败，请按 docs\GETTING_STARTED.md 准备 FFmpeg。
  popd
  pause
  exit /b 1
)
echo 稳定版源码模式：请在浏览器打开下方本次打印的本地地址。
echo 保持此窗口运行；结束时按 Ctrl+C。工程单独保存在 dev-data\course-stable-1.3。
"%PY%" -m reference_color_app.server --port 0 --data-dir "%ROOT%\dev-data\course-stable-1.3" %*
set "APP_EXIT=%ERRORLEVEL%"
popd
if not "%APP_EXIT%"=="0" pause
exit /b %APP_EXIT%
