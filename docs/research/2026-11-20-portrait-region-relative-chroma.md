# 同类区域内的相对明暗仍不足以决定改色幅度

上一轮直接对应绝对亮度，两个方向都超过 95% 的目标像素用到了供样端点。本轮只在脸颊和近中性色各自的范围内对应暗、中、亮，让两边都能用到分段。在相同区域内用固定偏移作为对照。两臂均读取供样的开关前后图和目标原图，目标期望图用来评分；脸颊框为离线标注，近中性色范围依据原图像素计算。[事前协议](../../experimental/adaptive-reference-20261113/portrait-region-relative-chroma-protocol.md) · [上轮绝对明暗](2026-11-20-portrait-donor-tonal-chroma.md)

| 目标及区域 | 固定误差 | 区内相对明暗误差 | 区内相对明暗投影 |
| --- | ---: | ---: | ---: |
| 《Cuckoo》脸颊 | 2.140 | 2.247 | 1.648 |
| 《Cuckoo》近中性色 | 1.483 | 1.483 | 0.455 |
| 混合光人像脸颊 | 2.427 | 2.330 | 0.546 |
| 混合光人像近中性色 | 1.413 | 1.424 | 1.725 |

误差单位为 RGB8。投影 1 对应作者预期幅度，通过范围为 0.75–1.25。四区均有足够样本和预期色度信号，均因幅度越界失败，误差也没有一项下降达到 10%。两臂区域外保持原像素，最大量化亮度漂移为 0.407 码值。《Cuckoo》为 431,858 个区域外像素，混合光人像为 570,164 个。

《Cuckoo》脸颊的全图亮度中位排名约为 44 百分位，混合光人像的成人脸颊约为 99 百分位。两块脸都能在区内分段，但目标期望偏移的去亮度蓝通道中位数分别为 +3.234 和 +6.180 码值，交换时把强偏移传给了需要较弱修正的脸。混合光供样的近中性色在三个区内暗亮带的色度偏移中位数完全相同，分段与固定成片在《Cuckoo》近中性色的评分一致。两张照片的方向均有正投影，改色幅度仍差得很大。

结束用亮度分段迁移这组同款编辑的尝试。下一轮换一个可区分原因的问题：用已有明确光源或同材质实拍对照，核对真实的拍摄条件能否解释分段遗漏的幅度差，再决定软件能否从普通输入取得这类信息。两张现有人像均属开发素材。本轮无产品代码、自动定位或正式验收变更。[逐区读数及哈希](2026-11-20-portrait-region-relative-chroma-evidence.json) · [《Cuckoo》对照](../../outputs/adaptive-reference-20261113/portrait-region-relative-chroma-v1/cuckoo-comparison.png) · [混合光对照](../../outputs/adaptive-reference-20261113/portrait-region-relative-chroma-v1/mixed-comparison.png)

## 复现

在仓库根目录运行以下命令。两遍运行的四张成片、两张对照图与结果 JSON 逐字节一致。脚本从本地归档核对工作 RGB、两个期望图、脸颊登记文件和输入报告。

```powershell
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_portrait_region_relative_chroma.py outputs/adaptive-reference-20261113 outputs/adaptive-reference-20261113/rawtherapee-portrait-cross-v1 outputs/adaptive-reference-20261113/portrait-region-relative-chroma-replay
```
