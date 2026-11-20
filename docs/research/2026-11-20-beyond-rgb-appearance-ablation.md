# 参考外观减少了串色，同时损伤了有意染色响应

13 号对实拍换灯和匹配数字染色的成片很接近。本轮用已有扩展第 4 轮权重，只开关参考外观分支。目标、参考、真值、四个条件、80% 强度以及空间渲染入口均固定。两臂各 12 张成片，13 号仅作原结果参照。真实换灯和数字染色在不同参考图上发生，推理只读一张参考与一张目标。色卡和已知编辑只用于评分。[事前协议](../../experimental/adaptive-reference-20261113/beyond-rgb-appearance-ablation-protocol.md) · [对照来源与 13 号基线](2026-11-20-beyond-rgb-matched-cast.md)

| 固定目标 | 换灯造成的整图变化，关 → 开 | 有意染色整图响应误差，关 → 开 | 两种参考的成片差，关 → 开 |
| --- | ---: | ---: | ---: |
| WHITE_green 灯 1 | 8.156 → 3.959 | 5.765 → 7.769 | 2.112 → 7.543 |
| WHITE_green 灯 8 | 13.405 → 10.865 | 4.197 → 4.973 | 2.224 → 5.197 |
| BLUE_red 灯 1 | 9.739 → 7.287 | 4.369 → 4.464 | 0.663 → 7.442 |
| BLUE_red 灯 8 | 10.854 → 9.804 | 2.108 → 3.441 | 0.651 → 5.126 |

全部差值使用 RGB8 MAE。开启外观使四个目标的换灯反应均缩小，降幅 9.7%–51.5%，同时拉大了两种参考输出的差距。四个目标的有意染色误差也均上升。BLUE_red 灯 8 是关闭时唯一通过独立风格门槛的目标，开启后超过 3 RGB8 阈值。两臂整图联合均为 0／4，五块灰阶的联合结果均为 0／20。每个目标的同图保持检查两臂均逐像素通过。[逐区误差、权重与成片哈希](2026-11-20-beyond-rgb-appearance-ablation-evidence.json)

## 决定

现有参考外观输入确实改变了两个外观相近的参考对目标的作用，但这个冻结权重未把变化用在正确的编辑上。停止直接使用该分支作为这项修复，也不再次调整该权重的外观开关或强度。若继续利用空间信息，需要提供能区分两种来源的监督，并在新的拍摄家族中检查「同方向、同幅度」的实际误判。继续增加位置特征或追单一串色分数缺少当前成片支持。此次两个静物场景继续作开发负例，尚无独立白墙、真人、视频或软件画质验收结果。

## 复现

复用[匹配的真实换灯与数字染色对照](2026-11-20-beyond-rgb-matched-cast.md#复现)。从仓库根目录运行，需已有验证过的扩展第 4 轮权重。使用未占用的输出目录复跑。两份报告和 24 张成片、4 张拼图逐字节一致。图像和权重留本机。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_beyond_rgb_appearance.py "$base/beyond-rgb-matched-cast-prepared-v1" "$base/beyond-rgb-matched-cast-first-v1" "$base/beyond-rgb-appearance-replay"
~~~
