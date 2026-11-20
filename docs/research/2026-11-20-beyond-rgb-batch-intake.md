# 两组同灯号实拍通过曝光和色卡检查

此前 WHITE_green 满足测光和 Oppo 曝光条件，WHITE_red 的照度比超限。本轮固定灯号 1／8、Oppo 和原有门槛，批量核对其余九个未出图场景。先读 CRI，再给通过测光门槛的场景读相机标签。全部读数在查看照片前锁定。[批量协议](../../experimental/adaptive-reference-20261113/beyond-rgb-batch-precheck-protocol.md) · [旧预检](2026-11-20-beyond-rgb-exposure-precheck.md) · [官方数据](https://zenodo.org/records/16848482)

九组现场测光都合格，只有 BLUE_red 的 Oppo 曝光差不超过 0.10 EV。其余八组曝光差 0.281–1.175 EV。最终选 WHITE_green 与 BLUE_red，共四张 Oppo 带色卡原片。新一轮元数据范围共 110,592 字节，原片、色卡坐标和标签共 58,636,687 字节，均在预算内。逐项核对字节范围、ZIP 成员名、原始长度及 CRC32。[全部场景及固定成员](2026-11-20-beyond-rgb-batch-intake-evidence.json)

## 原片检查

| 场景 | 照度比 | CIE xy 距离 | 设置曝光差 | 最大色卡中心位移（占对角线） | 六块灰卡相机 RGB 色度距离 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WHITE_green | 1.03485 | 0.17915 | 0.000 EV | 0.0331% | 0.299–0.327 |
| BLUE_red | 1.06856 | 0.18155 | 0.081 EV | 0.0410% | 0.282–0.311 |

两组均通过机位、设定曝光和灰阶门槛，四张图的采样点均未到发布 RGB 的归一化上界。三通道 HDF5 无法核实传感器原始裁切。预览只添加固定幂次映射，未做白平衡。按旧路线的 D65 矩阵把 HDF5 的 `[1::4,1::4]` 样本转到线性 sRGB，裁切前越界比例分别为 WHITE_green 的 14.85%／5.49%，BLUE_red 的 35.57%／41.92%。低于旧极端灯的 54.51% 至 92.24%，但下一轮出图必须保留对裁切的控制。[事前像素门槛](../../experimental/adaptive-reference-20261113/beyond-rgb-batch-pixel-protocol.md) · [旧显影与裁切边界](2026-11-20-beyond-rgb-reference-light.md)

下一阶段可以用这两组对照生成当前算法及候选的实际成片。控制目标图和强度，同时测「只变化现场光色」与「只施加明确的后期风格」。原片在本阶段已经查看并用于材料筛选，属于新开发回归。此处的白色是实验布景和灰卡，独立白墙、真人及视频验收仍需另行完成。

## 复现

从仓库根目录运行，输出文件名需未占用。两轮批量筛选、测量报告和原片拼图复跑一致。前一轮元数据、原片取件和像素检查的默认入口也已复验。图像及原始标签只留本机。

~~~powershell
$src = 'experimental/adaptive-reference-20261113'
$base = 'outputs/adaptive-reference-20261113'
$cache = "$base/beyond-rgb-batch-v1"
$photos = "$base/beyond-rgb-batch-captures-v1"
$python = '.tools/tf-convert-env/Scripts/python.exe'
New-Item -ItemType Directory -Force $cache | Out-Null
& $python "$src/audit_beyond_rgb_batch.py" "$base/beyond-rgb-index-v1/central.bin" 'docs/research/2026-11-20-beyond-rgb-exposure-precheck-evidence.json' "$cache/cache" "$cache/replay.json"
& $python "$src/fetch_beyond_rgb_captures.py" "$cache/replay.json" $photos --batch
& $python "$src/audit_beyond_rgb_captures.py" $photos "$photos/replay.json" "$photos/replay.png" --batch
~~~
