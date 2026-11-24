# LSMI 辅助灯替换未补足局部暖色对照

之前的 CHROMA-MILL 红、蓝灯是高饱和端点。为检查真实室内的灯光变化能否更接近有意暖色，本轮用 LSMI 直接拍摄的三档输入。两档 `12` 和 `13` 共同保留灯 1，分别使用灯 2 或灯 3。另用 `123` 比较加灯和减灯。只读取衍生数据公开的同一日光白平衡预设成片。已用的暖色编辑保持固定，比较方向和幅度。[原始采集，第 3.2 节与表 2](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Large_Scale_Multi-Illuminant_LSMI_Dataset_for_Developing_White_Balance_Algorithm_Under_Mixed_Illumination_ICCV_2021_paper.pdf) · [衍生显影，第 5 节](https://openaccess.thecvf.com/content/ICCV2025/papers/Serrano-Lozano_Revisiting_Image_Fusion_for_Multi-Illuminant_White-Balance_Correction_ICCV_2025_paper.pdf) · [白蓝和红白证据](2026-11-24-chromamill-warmth-match.md) · [事前协议](../../experimental/adaptive-reference-20261113/lsmi-daylight-switch-warmth-protocol.md)

## 取样决策与边界

完整核对 Nikon ZIP64 的 24,806 个条目。955 个场景具有未遮罩的日光输入，其中 916 个只发布 `12`，39 个具有 `12`、`13`、`123`。按编号均匀固定六个场景。目录和尾部共 3,409,086 字节，档案约 11.4 GB。成功取样的 18 个成员使用 19,182,394 字节区间传输。初次按发布页的 600 × 800 规格检查失败。所有成员图头显示实际为 920 × 614，按原尺寸重新核验并出片。首轮读取产生的额外流量未完整计量。每个成员通过原路径、区间、长度、CRC 和哈希核对。[原发布及下载](https://color.cvc.uab.es/revisitingmiwb/#download) · [前次目录局部核验](2026-11-18-srgb-lsmi-intake.md)

每个场景只看 `12`，选浅色柜门或现成墙面，登记远处保护区。六个场景均找到可测材料，后续未改框。部分场景有相似的室内布置，整个小样按同一采集来源处理。18 张 PNG 均无 ICC 和 EXIF，按衍生论文的 sRGB 说明解释，仅显示已发布预设下的 RGB 变化。EGM 复核确认原论文记录了固定相机和三脚架，日光预设标签不能逐档证明显影参数或自然光稳定。原数据许可是 CC BY-NC 4.0，本次使用限本机课程研究诊断。源图、成片及配置均留本机。[上游许可和衍生研究关联](https://github.com/DY112/LSMI-dataset#-dataset-license) · [本机锚图和固定测区](../../outputs/adaptive-reference-20261113/srgb-lsmi-daylight-layout-v1/layout.png)

## 两遍出图结果

18 张开启和关闭的暖色对照中，15 张通过原素材门槛。失败的 3 张在测区达到有效区间的像素比例不足 80%，其中场景 938 的 `13` 为 0%。所有比较的整图最优整数平移均为零。EGM 要求候选再核实测区附近两处固定地标、材质对应和遮挡。因为数值没有联合命中，本轮止于初筛。

| 真实拍摄变化 | 有序方向 | 素材、信号和共同支持过线 | 方向过线 | 幅度过线 | 联合命中 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 替换辅助灯 `12 ↔ 13` | 12 | 8 | 2 | 0 | 0 |
| 从两灯到三灯或反向 | 24 | 18 | 0 | 0 | 0 |

场景 931 的左侧参考是 `12` 档。换到 `13` 后登记墙面的中位变化是 `[39, 0, −21]`，同图有意暖色的变化是 `[13, −3, −14]`，余弦 0.935，达到 0.90。编辑幅度为实际换灯的 43.7%，低于 75%。远处保护区同一次换灯也发生了 `[13, −28, −46]` 的变化。另一组方向过线的是场景 946，幅度比只有 0.154。这两个结果说明真实室内的辅助灯替换可以在方向上接近局部暖色，本轮预先选定的灯档仍缺小幅配对。[场景 931 本机成片](../../outputs/adaptive-reference-20261113/lsmi-daylight-switch-warmth-v1/Place931/comparison.png) · [场景 946 本机成片](../../outputs/adaptive-reference-20261113/lsmi-daylight-switch-warmth-v1/Place946/comparison.png) · [全部方向、哈希与失败读数](2026-11-24-lsmi-daylight-switch-evidence.json)

结束这六组取样，保留 15 张暖色开关和输入反例。下次有可控采集时，先在浅色表面锁定机位、曝光与白平衡，通过暖白灯的小档切换控制亮度和色度幅度，再制作相同的编辑开关。这个做法直接针对本次通过方向、幅度过大的问题。当前工作台仍只取得一个合格的暖墙匹配来源，训练、产品候选和独立验收尚待相关条件。[现有工作台比较](2026-11-24-matched-wall-warmth-workbench.md) · [补拍和使用条件](2026-11-21-capture-edit-intake.md)

## 复现

在仓库根目录使用既有研究 Python。从同一归档重建的日光输入索引与首次记录逐字节相同。复现导出时，取件目录沿用以下名称，每轮输出目录应未被占用。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$py = '.tools/adaptive-env/Scripts/python.exe'
$exp = 'experimental/adaptive-reference-20261113'
$base = 'outputs/adaptive-reference-20261113'
& $py "$exp/fetch_lsmi_daylight_switch.py" catalog "$base/lsmi-daylight-catalog-new.json"
& $py "$exp/fetch_lsmi_daylight_switch.py" fetch "$base/lsmi-daylight-catalog-new.json" "$base/srgb-lsmi-nikon-daylight-switch-input-v2"
~~~

出图时临时使用未占用的 V 盘映射到仓库根目录，使 RawTherapee 读取 ASCII 路径。

~~~powershell
if (Test-Path 'V:\') { throw 'V: already in use' }
subst.exe V: (Get-Location).Path
try {
    $env:PYTHONPATH = 'V:/;V:/.tools/tf-convert-env/Lib/site-packages'
    $env:OPENBLAS_NUM_THREADS = '2'
    $base = 'V:/outputs/adaptive-reference-20261113'
    $rt = "$base/rawtherapee-sky-pair-v1/runtime/RawTherapee_5.13_win64_x86_64_release/rawtherapee-cli.exe"
    & V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/probe_lsmi_daylight_switch_warmth.py $base $rt "$base/lsmi-daylight-switch-warmth-new"
    if ($LASTEXITCODE -ne 0) { throw "LSMI warmth probe returned $LASTEXITCODE" }
} finally {
    subst.exe V: /D
}
~~~

两遍出图耗时约 73.1 秒、54.7 秒。每遍的 60 张 PNG、12 份 PP3 和 1 份结果均逐文件一致。独立从落盘 PNG 重算两遍共 72 个方向的共同支持、通道变化及角度幅度，并核对准确的 18 个来源 CRC、保护区变化和输出 ICC。
