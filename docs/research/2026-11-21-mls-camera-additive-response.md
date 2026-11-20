# 固定线性相机 RGB 增量未能修复跨场景换灯重放

两张照片的色度比例会受原本亮度影响。本轮用其他场景在不同灯下的线性相机 RGB 通道中位数差，叠加至目标原片代表值，再换算成两维对数色度增益。输入场景、掩膜、80% 亮度保持渲染、评分及留出方式沿用前轮。目标的蓝灯照片仅用于评分，人工区和预先知道的材质身份用于离线重放。[事前协议](../../experimental/adaptive-reference-20261113/mls-camera-additive-response-protocol.md) · [原固定色度重放](2026-11-21-mls-unpaired-material-response.md)

| 场景 | 原同材质固定色度误差 | 新同材质线性增量误差 | 新变化投影 | 位移统计资格 |
| --- | ---: | ---: | ---: | --- |
| 02 | 0.05041 | 0.15518 | 0.748 | 合格 |
| 03 | 0.00618 | 0.09246 | 0.842 | 合格 |
| 04 | 0.10017 | 0.20303 | 0.274 | 合格 |
| 05 | 0.08235 | 0.36690 | 1.687 | 不合格 |
| 06 | 0.08875 | 0.20951 | 2.379 | 不合格 |
| 09 | 0.03938 | 0.12794 | 1.662 | 合格 |

误差采用材质内两维对数色度中位响应的欧氏范数。四个可评分场景均未通过，误差均大于原固定色度方案。04 对应的绿色塑料棒由原投影 0.648 降至 0.274，09 从 1.193 升至 1.662，出现欠量和过冲。两个位移统计不合格的场景仅保留作描述。原始相机蓝通道中位数差在合格的两个绿色塑料棒场景 04 和 09 分别为 229 和 136，在两个合格蓝纸场景 02 和 03 分别为 1692 和 1756。这些差值是材质区域的统计量，不能据此反推单个像素的实际照度。所有零臂及掩膜外像素保持原图。[全部数值及成片哈希](2026-11-21-mls-camera-additive-response-evidence.json)

结束将其他场景的固定相机 RGB 增量用于同材质换灯估计。此次显示仅用目标原片的通道代表值做比例换算，不能补足具体场景下新增光量。结合前轮结果，继续调整固定材质查表的力度依据不足。后续产品路线须有能解释实际场景差异的独立线索，并同时检查普通两图下应保留的风格。该开发对照不能替代肤色、真实中性色、视频和独立画质验收。

## 复现

从仓库根目录运行，沿用[原试验](2026-11-21-mls-unpaired-material-response.md)的输入和输出报告。新输出目录须不存在。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$base = 'outputs/adaptive-reference-20261113'
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_mls_camera_additive_response.py "$base/mls-linear-pilot-inputs" "$base/mls-multiscene-intake-v1" "$base/mls-material-identity" "$base/mls-unpaired-material-response-v1/results.json" "$base/mls-camera-additive-response-repro"
~~~

两轮各 24 张成片和一份报告，25 个文件的 SHA-256 逐一一致。本机成片在 `mls-camera-additive-response-v1` 和同级复跑目录。完整报告 SHA-256 为 `1a2a0e60f933851ee7b6c4acffe0f14095cd93f813db0a4a70186577fc8eb92a`。
