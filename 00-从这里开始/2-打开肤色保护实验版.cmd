@echo off
chcp 65001 >nul
setlocal
set "APP=%~dp0..\讨论\09-肤色保护实验版\源码开发版\ReferenceColor.exe"
if not exist "%APP%" (
  echo Full portable package is missing. Read README.md in this folder.
  pause
  exit /b 1
)
start "" "%APP%"
