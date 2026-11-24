# Ribeira Hotel 实拍浅墙的日光与有意暖色对照

本轮查的是短时间的真实日光变化，可否为局部暖色减偏的研究补齐第二个独立拍摄来源。前一轮的直接／漫射白灯产生了比有意调色强得多的亮度差，所以选用作者已配准的同一建筑 15:03 和 15:05 实拍，只测这一组。先固定点位、显影与停止条件，再测正反两个方向。此组归开发诊断。[事前协议](../../experimental/adaptive-reference-20261113/ribeira-daylight-protocol.md) · [上一轮对照](2026-11-24-cl3an-neutral-warmth.md) · [作者论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7822868/)

常规 Figshare 下载入口返回 403。存档端点的总长度和 ETag 与固定版本相符，经已有代理续传后，688,272,614 字节的两份完整存档均核对了发布方 MD5。随包说明确认两份 `hsi` 为 2003 年 6 月 10 日同一建筑的实测辐亮度，采样范围为 400–720 nm、间距 10 nm，且未给出两张 BMP 共用曝光的证据。使用两份原始 `hsi` 与同一 CIE 1931 2° 观察者表积分，沿用同一个 XYZ 到 sRGB 的矩阵，曝光系数固定为首帧定义的 0.2312809391。原始数据、成片和发布附带文档保留在本机。Figshare 两条固定记录列 CC BY 4.0，随包说明与作者页面另有个人使用文字，本轮使用限于本机课程研究。[15:03 固定记录](https://api.figshare.com/v2/articles/13435715/versions/1) · [15:05 固定记录](https://api.figshare.com/v2/articles/13435736/versions/1) · [作者说明](https://personalpages.manchester.ac.uk/staff/d.h.foster/Short-Interval_Hyperspectral_Radiance_Image_Pairs/Short-Interval_Hyperspectral_Radiance_Image_Pairs.html)

首帧固定窗间浅墙和保护区，第二帧复核位置与材质，测区未见遮挡或阴影边缘移入。整图与两处窗框、墙面附近的配准块最优位移均为零。整图零位梯度相关为 0.9795，三个局部区域为 0.9912–0.9942。测区 Y 的逐像素比值中位数为 1.0779，xy 色度变化中位数为 `[0.00027, 0.00106]`。相邻浅墙与远侧立面的 Y 也分别上升约 7.6% 和 8.3%。固定曝光下，窗间墙裁切前出界像素从 0% 增至 20.89%，两块保护区均为 0%。这是本组真实受光变化及显示范围的测量，不能进一步分解为某种灯具或单一色温变化。[测区及共同显影](../../experimental/adaptive-reference-20261113/ribeira-daylight-protocol.md) · [逐项数值](2026-11-24-ribeira-daylight-evidence.json)

| 方向 | 三态有效覆盖 | 有效像素内实拍变化 RGB8 | 有效像素内局部暖色 RGB8 | 判定 |
| --- | ---: | --- | --- | --- |
| 15:03 → 15:05 | 1.42% | `[8, 8, 7]` | `[12, -3, -14]` | 覆盖、方向和幅度失败，余弦 −0.105，幅度比 1.404 |
| 15:05 → 15:03 | 0% | 无有效读数 | 无有效读数 | 覆盖失败 |

两档开关中 15:03 通过，编辑区红减蓝中位变化为 25 RGB8，亮度变化为 −1.10 RGB8，保护区保持。15:05 的关态有效覆盖只有 33.81%，低于 80% 要求。正反两个方向均未匹配。15:03 开关保留为辅助暖色正例，完整双帧保留为明暗变化和高光边界反例。该组的任一方向均不计入暖墙跨来源训练准入。保留首帧定义的曝光与原测区，不用事后改曝光或换墙增加命中。稳定版 1.3 和研究版 13 号的工作台仍是此前版本，跨场景画质与产品验证继续待达标。[原门槛](../../experimental/adaptive-reference-20261113/shared-local-warmth-protocol.md) · [跨来源研究条件](2026-11-24-spatial-condition-design.md)

## EGM 复核

第 5 阶段独立审阅指出，共同显影排除了逐帧调整，但裁切和 sRGB 编码仍会影响比较。实拍与有意编辑应共用起始关态及三态有效区域。局部配准需同时核查，编辑保护区只约束同一时刻的编辑开关。按这些意见追加了基于原始 XYZ 的亮度和色度证据，并用落盘图独立复算两个方向。复核未读取成片或结果。即使有方向过线，也需核对与 MIT copyroom 的材质可比性和两侧按来源留出，才能考虑监督试验。[审阅落实的协议](../../experimental/adaptive-reference-20261113/ribeira-daylight-protocol.md)

## 本机复现

从上述两条固定记录取得 ZIP，并按证据核对字节数和 MD5，取 `Ribeira_Hotel_1503.zip` 和 `Ribeira_Hotel_1505.zip`。常规入口不可达时，可从对应的 `pfigshare-u-files` 存档路径取件，路径分别以 `26172140/Ribeira_Hotel_1503.zip` 和 `26172143/Ribeira_Hotel_1505.zip` 结尾。`render_ribeira_daylight.py` 使用现有研究 Python、NumPy、Pillow 和 SciPy 1.18.1，后者可从此前本机 `scipy-capacity-20261122/wheel` 加载。以下命令从项目根目录在新目录生成一套结果。`capture.md` 可按固定来源、两次实拍时间、同一建筑、亚像素配准和共同显影方法重新编写。

~~~powershell
if (Test-Path 'V:\') { throw 'V: already in use' }
subst.exe V: (Get-Location).Path
if ($LASTEXITCODE -ne 0) { throw 'Drive mapping failed' }
try {
    $research = 'V:/outputs/adaptive-reference-20261113'
    $inputs = "$research/ribeira-daylight-source-v1"
    $env:PYTHONPATH = "$research/scipy-capacity-20261122/wheel"
    $env:OPENBLAS_NUM_THREADS = '2'
    $py = 'V:/.tools/adaptive-env/Scripts/python.exe'
    $experiment = 'V:/experimental/adaptive-reference-20261113'
    $cmf = "$research/skin-spectra-input/CIE_xyz_1931_2deg.csv"
    $a = "$research/ribeira-daylight-a-new"
    $b = "$research/ribeira-daylight-b-new"
    & $py "$experiment/render_ribeira_daylight.py" "$inputs/Ribeira_Hotel_1503.zip" $cmf 1503 $a
    if ($LASTEXITCODE -ne 0) { throw '15:03 render failed' }
    & $py "$experiment/render_ribeira_daylight.py" "$inputs/Ribeira_Hotel_1505.zip" $cmf 1505 $b --anchor $a
    if ($LASTEXITCODE -ne 0) { throw '15:05 render failed' }
    $env:PYTHONPATH = 'V:/;V:/.tools/tf-convert-env/Lib/site-packages'
    & $py "$experiment/warm_capture_preflight.py" "$a/srgb.png" "$b/srgb.png" "$inputs/capture.md" `
        "$research/ribeira-daylight-preflight-new" --edit 494 410 544 482 --guard 720 269 750 330 --guard 1045 268 1085 310
    if ($LASTEXITCODE -ne 0) { throw 'Warm-light preflight failed' }
} finally {
    subst.exe V: /D
}
~~~

两遍各自重新读取原件。每帧的 sRGB PNG、浮点 XYZ、裁切前出界掩码和显影记录逐字节一致。每遍预检独立完成，耗时分别 9.45、9.17 秒，11 张 PNG、2 份 PP3 和 1 份 JSON 逐文件一致。预检的五项既有单元测试通过。独立解码复查两遍共四个有序方向，以三态共用的 `[4, 251]` 通道范围核对覆盖，再取各自通道中位数并相减。用未裁切 XYZ 逐像素计算 Y 比值及 xy 差值，再取墙区中位数。[冻结身份与复算数值](2026-11-24-ribeira-daylight-evidence.json) · [本机四图板](../../outputs/adaptive-reference-20261113/ribeira-daylight-preflight-v1/render/comparison.png)
