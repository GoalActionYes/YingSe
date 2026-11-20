# 恢复线性亮度可修回部分灰阶，残余色偏仍超线

上一轮证明，仅在稳定版 1.3 和研究版 13 号的已有编辑量之间选择比例，不能覆盖部分静物灰卡所需的变化。原对照的数字染色在生成时声明保持线性亮度，本轮据此测试一个新约束。先取两套算法在数字编辑参考和原参考之间的输出差，按原对照的 80% 强度重放到目标原图。再保留候选的线性 RGB 通道比例，把亮度恢复到目标原图。遇到超色域时，沿相同亮度的中性轴压缩色度。目标正确成片和区域仅用于计分。实验仍使用参考编辑前后对，已知换灯与原参考臂仍保留目标原图。[事前协议](../../experimental/adaptive-reference-20261113/target-linear-luma-edit-protocol.md)

沿用六个目标状态、四个构图和 33 项整图及区域检查。RGB8 平均绝对误差上限为 3，风格响应投影需为 0.75 至 1.25。同一套原样或锁亮度的结果独立与门槛比较。

| 目标状态 | 稳定版 | 稳定版锁亮度 | 13 号 | 13 号锁亮度 |
| --- | ---: | ---: | ---: | ---: |
| MIT 客厅 | 2／4 | 2／4 | 1／4 | 1／4 |
| Gros 人像 | 5／5 | 5／5 | 2／5 | 2／5 |
| WHITE_green 灯 1 | 0／6 | 2／6 | 0／6 | 0／6 |
| WHITE_green 灯 8 | 1／6 | 1／6 | 0／6 | 3／6 |
| BLUE_red 灯 1 | 0／6 | 1／6 | 1／6 | 1／6 |
| BLUE_red 灯 8 | 0／6 | 2／6 | 3／6 | 5／6 |
| 合计 | 8／33 | 13／33 | 7／33 | 12／33 |

两套锁亮度算法各新增 5 项通过，各自原先已通过的区域均保留，且其中无超过 1 RGB8 的退化。BLUE_red 灯 8 的 13 号整图误差由 1.924 降至 1.359，编号 19 色卡仍为 3.300。WHITE_green 灯 1 的稳定版编号 23 色块误差由 6.998 降到 1.334，但编号 19 仍有 9.399。13 号在人像两侧白幕的误差仍为 5.325、5.660，风格投影从 0.565、0.530 降至 0.522、0.494。稳定版锁亮度的完整构图通过数仍为 1／4，13 号仍为 0／4。[逐区结果与文件哈希](2026-11-21-target-linear-luma-edit.json)

## 可保留的收益与后续范围

锁亮度的局部收益可重复，说明已有染色输出里的亮度变化确实会干扰这类编辑。对多项白幕和色卡反例，恢复亮度后仍有明显误差或迁移不足，下一轮应针对保亮度后剩余的色度方向与强度验证新表示。80% 的既有预期成片已经混合和量化，不能把残差全部解释成色度误差。量化前最大线性亮度差不超过 2.23×10⁻¹⁶，色域压缩最多约占像素的 0.0014%。六个目标的同图保持，以及单独构造的黑白边界与超色域检查均通过。产品未接入，普通两图输入、独立自然度及视频仍待验收。

## 复现

沿用 [MIT 对照](2026-11-21-mid-matched-wall-cast.md)与 [Beyond RGB 对照](2026-11-20-beyond-rgb-matched-cast.md)的成片和本机研究环境。从仓库根目录运行，选择尚未占用的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_target_linear_luma_edit.py outputs/adaptive-reference-20261113/target-linear-luma-edit-repro
~~~

原跑与复跑各生成 24 张成片、6 张对照和 1 份数值报告，31 个文件逐一核对 SHA-256 一致。本机输出在 outputs/adaptive-reference-20261113/target-linear-luma-edit-v1 和同级 target-linear-luma-edit-recheck。
