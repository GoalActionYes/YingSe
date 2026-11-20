# 整体色偏接近时，两版未能分开实拍换灯和有意染色

固定 WHITE_green 和 BLUE_red 两个通过曝光与色卡检查的场景。对每个场景保留灯 1 实拍，再构造两张参考，一张使用灯 8 的实拍，一张在灯 1 图上施加保持线性亮度的整体数字染色。数字目标是匹配实拍的整体色偏。两个场景交叉互作参考，每个目标各固定两次灯光，共 24 张成片。稳定版与 13 号空间版均为 80% 强度，无手动项。色卡和直接对目标施加同款编辑的图只用于评估。[事前协议](../../experimental/adaptive-reference-20261113/beyond-rgb-matched-cast-protocol.md) · [原片与曝光核对](2026-11-20-beyond-rgb-batch-intake.md)

四张拍摄共用一个线性范围映射至 [0.02,0.98]，避免直接输出时的裁切。两份实拍换灯的整体对数色度变化范数为 1.060 和 1.074，数字染色与实拍的中位色偏差分别为 0.00017 和 0.00167。共同有效覆盖超过 99.98%，数字参考和直接编辑目标在该处理下均未裁切。两份参考都通过配对资格。仍存在实拍局部受光、反射及对比度差异，因此把这两种参考作为匹配整体色偏的对照。

## 实际成片

| 固定目标 | 换灯对参考基准的整图变化（稳定／空间） | 空间版的数字编辑响应误差 | 空间版的换灯／数字染色成片差 |
| --- | ---: | ---: | ---: |
| WHITE_green 灯 1 | 12.694／10.801 | 4.934 | 1.381 |
| WHITE_green 灯 8 | 16.053／15.381 | 5.126 | 1.901 |
| BLUE_red 灯 1 | 12.957／10.743 | 3.837 | 1.002 |
| BLUE_red 灯 8 | 14.795／11.165 | 1.924 | 0.803 |

以上差值为 RGB8 MAE。空间版在 BLUE_red 灯 8 的有意染色响应投影为 1.021，误差 1.924，过数字编辑门槛，同一目标换到实拍参考后却变化 11.165。四个目标中，两版都没有同时把换灯影响压在 3 RGB8 内并保持有意风格。稳定版的数字编辑单项通过 0／4，空间版 1／4。五块白到中灰色卡的联合通过数，两版均为 0／20。[逐区域分数与哈希](2026-11-20-beyond-rgb-matched-cast-evidence.json)

## 空间线索的决定

前一轮 MIT 实拍的全图色偏不足。本轮具备两组匹配控制后，旧空间离散度判断依旧 0／2。WHITE_green 的换灯离散度高于数字染色 0.053，却比原参考低 0.062。BLUE_red 换灯的离散度比原图高 0.214，但比数字染色低 0.137。两种失败都保留。结束依赖近灰网格空间离散度的固定判断。这两个新静物组可以用于候选回归，不能当作真人、独立白墙或视频证据。完整画质目标仍需要能减少现场光色串入，又保持明确数字编辑的可运行候选。

## 复现

先将四张原片和审计报告按[取件流程](2026-11-20-beyond-rgb-batch-intake.md#复现)写入本机。使用不同的新目录复跑。两次报告和 38 张图逐字节一致，原批次的默认准备入口也复跑一致。媒体留本机。

~~~powershell
$src = 'experimental/adaptive-reference-20261113'
$base = 'outputs/adaptive-reference-20261113'
$photos = "$base/beyond-rgb-batch-captures-v1"
$prepared = "$base/beyond-rgb-matched-cast-replay-input"
& .tools/tf-convert-env/Scripts/python.exe "$src/probe_beyond_rgb_transfer.py" prepare $photos "$photos/first.json" $prepared --batch --headroom
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe "$src/probe_beyond_rgb_matched_cast.py" $prepared "$base/beyond-rgb-matched-cast-replay"
~~~
