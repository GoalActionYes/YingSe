@echo off
chcp 65001 >nul
setlocal
set "APP=%~dp0..\dist\reference-color-current\ReferenceColor.exe"
if exist "%APP%" (
  start "" "%APP%"
  exit /b
)
echo 未找到旧路径下的便携包，正在启动稳定版源码。
call "%~dp014-打开稳定版源码.cmd" %*
exit /b %ERRORLEVEL%
