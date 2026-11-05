@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0StartDemo.ps1"
if errorlevel 1 pause
