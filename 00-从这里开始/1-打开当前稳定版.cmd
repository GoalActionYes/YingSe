@echo off
chcp 65001 >nul
setlocal
set "APP=%~dp0..\dist\reference-color-20261016-222448\ReferenceColor.exe"
if not exist "%APP%" (
  echo Full portable package is missing. Read README.md in this folder.
  pause
  exit /b 1
)
start "" "%APP%"
