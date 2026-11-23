# 两版映色都未复现摄影师的局部提亮

摄影师用同一张室内混光原片给脸颊局部提亮。套用映色既有的 80% 强度后，目标右颊应增加 10.18 个亮度码值，预选墙面和杯身仍逐像素保持。固定关闭图为目标，将关闭或开启图作为参考，稳定 1.3 和 13 号分别从自己的工作台实际导出 PNG。再将两张参考按同一矩形裁切，保留脸颊、墙和杯子的三个评分区，目标完全保持。输出判断沿用事前门槛。[编辑来源和开关证据](2026-11-24-playraw-flash-room.md) · [事前协议](../../experimental/adaptive-reference-20261113/playraw-flash-room-workbench-protocol.md) · [本机对照总图](../../outputs/adaptive-reference-20261113/playraw-flash-room-workbench-v1/comparison.png)

| 工作台和参考范围 | 脸颊开关亮度中位数 | 保留的脸颊编辑投影 | 开启后脸颊与答案的误差 | 开启后墙面偏差 | 开启后杯身偏差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 稳定 1.3，完整 | −12.00 | −87.57% | 21.67 | 4.64 | 7.51 |
| 稳定 1.3，裁切 | 0.00 | 0.08% | 18.82 | 11.22 | 10.86 |
| 13 号，完整 | −1.29 | 0.25% | 11.14 | 4.33 | 5.09 |
| 13 号，裁切 | 0.00 | 1.06% | 9.46 | 7.88 | 2.65 |

亮度按 Rec. 709 系数从 RGB8 计算，其他误差为 RGB8 MAE。编辑投影比较开减关的实际 RGB 变化与已知编辑，负值表示整体响应偏向相反方向。墙与杯没有预期改动，表中偏差以原色为准。完整关闭图做参考时，两版都逐像素返回原图。稳定版开启完整参考后新增截断比例 9.17%，超出 2% 限制。13 号使用裁切关闭图时，墙本身已偏 7.81 RGB8，开启后偏 7.88 RGB8。即使只看开关，墙的变化不大，也不能把它视为保住了原色。[全部区域、误差与产物哈希](2026-11-24-playraw-flash-room-workbench-evidence.json)

四组均未通过。此前程序加半档提亮时，13 号只保留很小的响应。这次真实作者编辑也发生了同类漏调，漏调在完整参考下就出现，改用裁切后仍存在。后续把这项作者开关作为候选的开发回归，共同检查真实编辑响应与墙、杯保护。原生网格在另两组半档提亮上能够表达正确结果；本轮只衡量现有工作台的自动响应，尚未检验这张图的网格上限或跨原片识别。候选训练还需另一个真实多灯档与同款编辑配对来源，本轮没有增加该类配对。[现有网格表达范围](2026-11-24-gros-context-capacity.md) · [训练准入](2026-11-21-phase-cost-and-gates.md)

## 复现及独立复算

两次完整命令分别耗时 29.11 秒和 23.85 秒，包括本机 Python 启动、模型装载、图片导入、分析、PNG 导出和计分。八组同身份导出分别逐字节一致。独立从落盘 PNG 复算了 80 项区域读数，两遍均与原结果一致。两份裁切也逐像素对回原参考。素材是一张原片，四种比较同属一个已使用的开发内容。未训练或调整工作台，原片与完整成片保留本机。

从仓库根目录按[已有工作台的研究环境](2026-11-24-gros-mixed-light-exposure-toggle.md)准备环境，然后执行以下命令，第二遍换一个新的输出目录。证据同时保存了执行时文件的哈希与 Git 将末行标准化为 LF 后的哈希。

~~~powershell
$env:PYTHONPATH = @((Get-Location).Path, (Resolve-Path -LiteralPath '.tools/tf-convert-env/Lib/site-packages').Path) -join [IO.Path]::PathSeparator
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_playraw_flash_room_workbench.py "$base/playraw-flash-room-rawtherapee-v1" "$base/playraw-flash-room-workbench-new"
~~~
