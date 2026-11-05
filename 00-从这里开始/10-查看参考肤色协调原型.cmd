@echo off
chcp 65001 >nul
set "DEMO=%~dp0..\讨论\15-参考肤色协调原型"
if not exist "%DEMO%\完整图片对照.png" (
 echo 本机阶段演示素材未生成。请查看源码中的研究记录，Git不包含媒体文件。
 pause
 exit /b 1
)
start "" "%DEMO%"
