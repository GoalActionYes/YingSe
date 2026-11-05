@echo off
chcp 65001 >nul
set "REVIEW=%~dp0..\讨论\14-构图适配研究验收\人物适用条件验证.html"
if not exist "%REVIEW%" set "REVIEW=%~dp0..\讨论\14-构图适配研究验收\构图适配研究展示.html"
if not exist "%REVIEW%" (
 echo 本机研究展示页尚未生成。Git只保存生成脚本与验收记录，不包含图像和运行依赖。
 echo 请查看 docs\research\content-balance-20261105 下的研究记录。
 pause
 exit /b 1
)
start "" "%REVIEW%"
