# 参考图换灯会改变目标中性色

把同一张目标图交给软件两次，只切换另一场景的暖灯参考和青蓝灯参考。用四张 Beyond RGB 带色卡实拍交叉组合，覆盖两个迁移方向及两档目标现场灯。稳定版使用现有图片分析与 LUT 渲染入口，空间版使用冻结的 13 号权重。两条流程均为 80% 强度、无手动项，输出 1024 × 768。色卡坐标只用于事后测量。同场景两灯拍摄的曝光标签存在 0.127 EV 和 0.424 EV 差异，实验结论对应拍摄状态整体改变。[实拍资格](2026-11-20-beyond-rgb-captures.md) · [实验协议](../../experimental/adaptive-reference-20261113/beyond-rgb-transfer-protocol.md)

## 两种显影与结果

首轮统一用标签的 D65 相机矩阵和 sRGB 曲线，不按单张白平衡。四张线性 sRGB 输入各有 54.5% 至 92.2% 的像素至少一个通道越界。在看过初次结果后，另做一次固定公共范围的显影敏感性复核。将四张图的线性通道统一从实测范围 −0.173 至 3.046 线性压缩到 0.02 至 0.98，再用同一 sRGB 曲线。复核输入及成片中，色卡采样都未到 RGB8 边界。压缩会抬高黑位并改变饱和度，因此把两种显影的成绩分别保存。[DNG 相机矩阵定义](https://android.googlesource.com/platform/external/dng_sdk/+/refs/heads/master/source/dng_camera_profile.h)

下表对每个固定目标报告五块白至中灰的色卡中，有几块在换参考时沿参考色度差方向移动至少 20%。分母分别是可测的五块色卡，同一场景与同一色卡反复出现。

| 目标 ← 参考场景 | 目标灯号 | 初轮稳定版／空间版 | 保留余量稳定版／空间版 |
| --- | ---: | ---: | ---: |
| MIXED_organic ← WHITE_rich | 6 | 5／5，5／5 | 5／5，5／5 |
| MIXED_organic ← WHITE_rich | 13 | 4／5，5／5 | 5／5，5／5 |
| WHITE_rich ← MIXED_organic | 6 | 2／5，5／5 | 3／5，3／5 |
| WHITE_rich ← MIXED_organic | 13 | 0／5，5／5 | 3／5，5／5 |

保留余量后，稳定版 16／20、空间版 18／20 达到本轮携带门槛，所有组合都存在同向响应。部分较暗灰阶的投影比例大于 1，说明沿该方向的成片色度变化甚至超过参考变化。空间版在浅色场景目标的两档灯下整幅输出变化更大，杂色静物目标的青蓝灯组合则由稳定版变化更大。浅色场景暖灯目标的白卡，在两条流程中都未过 20% 门槛。完整数值保留逐块幅度、正交误差、边界比例与整图变化。[数值证据与成片哈希](2026-11-20-beyond-rgb-reference-light-evidence.json)

## 决策

两种流程都会响应实拍参考的现场光色，在这组极端灯光中，现象并未随输入裁切消失。换灯也伴随曝光差，显影压缩影响颜色和明暗，因此这里只判断会不会带入光色，无法将响应评价为实际画质收益，也无法以本轮排列判定哪版更自然。

下一轮应为相同素材补上已知的有意编辑，分别比较「同一编辑换灯」和「同一灯下切换编辑」。只有当一项改动减少前者的传递，同时保留后者的方向和幅度，才值得带入原有四套开发对照。真人肤色、独立白墙、视频及自然评价仍按完整门槛验证。

## 复现

先完成[原片取件](2026-11-20-beyond-rgb-captures.md#复现)。准备阶段使用已有 HDF5 环境，渲染阶段使用已有自适应环境及 13 号权重。以下命令从仓库根目录执行，输出目录需使用未占用的名字。为复现首轮另用一个目录，省略 `--headroom`。两种显影各渲染两遍，报告与所有输出图像的 SHA256 分别一致。完整成片留本机。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$probe = 'experimental/adaptive-reference-20261113/probe_beyond_rgb_transfer.py'
$evidence = 'docs/research/2026-11-20-beyond-rgb-captures-evidence.json'
& .tools/tf-convert-env/Scripts/python.exe $probe prepare "$base/beyond-rgb-captures-v1" $evidence "$base/beyond-rgb-transfer-replay-input" --headroom
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe $probe render "$base/beyond-rgb-transfer-replay-input" "$base/beyond-rgb-transfer-replay"
~~~
