# 见过同一案例的暖色头也漏掉有意加暖

上一轮局部位置和池化两臂没能在留出的拍摄家族恢复有意暖色。这一轮把训练时的两组原输入交给同一批末轮权重，按原尺寸出图，仍为每臂 0／2 个完整方向通过。两臂已经在见过的案例上降低底色误差和换灯差，遗漏的有意暖色在 128 × 128 训练前向中也存在。跨来源和原尺寸渲染各自都无法单独解释这次失败。[原留出试验](2026-11-24-positional-warmth-formal.md) · [核验方案](../../experimental/adaptive-reference-20261113/positional-warmth-fitting-check.md)

| 已用于拟合的目标 | 比较对象 | 原尺寸两档底色误差，RGB8 | 原尺寸两档暖色投影 | 训练尺寸两档暖色投影 |
| --- | --- | ---: | ---: | ---: |
| MIT 客厅浅墙 | 13 号 | 12.83／15.68 | 6.66%／4.48% | — |
| MIT 客厅浅墙 | 池化 | 0.88／1.64 | 4.63%／6.04% | 8.26%／7.35% |
| MIT 客厅浅墙 | 局部位置 | 0.82／0.67 | 19.63%／21.81% | 20.55%／22.95% |
| Beyond RGB 白绿背景 | 13 号 | 21.43／19.17 | 0.39%／0.79% | — |
| Beyond RGB 白绿背景 | 池化 | 1.60／1.47 | 0.10%／0.13% | 0.13%／0.22% |
| Beyond RGB 白绿背景 | 局部位置 | 1.76／1.61 | 0.17%／0.05% | 0.25%／0.14% |

原暖色投影门槛为 75%–125%，同时要求各项误差 ≤3 RGB8。原尺寸四个实验头的关态和开态跨灯差均小于 1.47 RGB8。两种尺寸均存在漏暖，且原尺寸四头的暖色开关误差仍高于 3 RGB8。训练尺寸两档主比较为未经镜像的同一目标与参考。另检查三种镜像组合，全部 32 项投影及响应联合判定未过线。训练日志记的是运行中首末步各自批次的复合损失，包含全图、局部、开关和跨灯误差。损失下降与底色和跨灯改善相符，不能单独证明学到了应保留的暖色。EGM 复核了两种尺寸的实际对照范围。是否由参考表示、目标函数或训练容量等因素共同造成，本轮未拆开验证。[逐项数值](2026-11-24-positional-warmth-fitting-evidence.json) · [MIT 本机图板](../../outputs/adaptive-reference-20261113/positional-warmth-fitting-first/mit/local/comparison.png) · [Beyond RGB 本机图板](../../outputs/adaptive-reference-20261113/positional-warmth-fitting-first/beyond_rgb/local/comparison.png)

## 复现

从仓库根目录沿用已有研究环境和正式运行的四份本机权重。输出目录要新建。默认原尺寸出图，加 `--training-size-only` 时只保存训练尺寸读数。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$formal = "$base/positional-warmth-formal-v1"
$probe = 'experimental/adaptive-reference-20261113/check_positional_warmth_fitting.py'
& '.tools/adaptive-env/Scripts/python.exe' $probe $base $formal "$base/positional-warmth-fitting-replay"
if ($LASTEXITCODE -ne 0) { throw 'Native fitting check failed' }
& '.tools/adaptive-env/Scripts/python.exe' $probe $base $formal "$base/positional-warmth-fitting-small-replay" --training-size-only
~~~

原尺寸两遍分别约 13.58 和 12.92 秒，每遍 30 张结果、6 张图板与报告逐字节一致。用落盘图独立复算 21 条测区结果，检查的 10 张 13 号对应结果与正式试验逐像素一致。训练尺寸两遍分别约 3.48、3.58 秒，报告逐字节一致。此次使用已经见过的开发素材，未新增拟合、产品候选、人评或视频证据。局部位置路线继续停止，保留完整结果与原始门槛。
