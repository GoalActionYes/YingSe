# 第二个暖色匹配开发来源

Beyond RGB 的实验箱白背景新增一组匹配。在已登记区域内，实际把灯 7 换成灯 1 所带来的颜色变化，与在灯 7 照片上有意局部加暖的方向和幅度都过线。MIT 复印室以外现在有第二个可用来源，下一步可以用同一套工作台检查：换了参考拍摄来源之后，目标上的预期暖色是否仍被漏掉，肤色和白幕是否仍被参考灯光带偏。本轮完成的是素材核验。[旧工作台对照](2026-11-24-matched-wall-warmth-workbench.md) · [本机四图](../../outputs/adaptive-reference-20261113/beyond-rgb-near-white-v2/warm-verified-first/render/comparison.png)

## 从测光到出片

先补齐前次缺失的三份测光记录，完成原定四场的 12 个配对。十个未同时通过测光线。两个进入相机筛查，`WHITE_red` 1／9 设定曝光差 0.40512 EV，超过 0.10 EV。仅 `WHITE_blue` 1／7 同时通过测光、相机和原始成员齐全检查。色度距离为 0.01243，照度比为 1.14779。两档均为 OPPO Find X5 Pro、1/100 秒、f/1.7，ISO 为 225 与 228，设定曝光差为 0.01911 EV。读取两档的 HDF5、色卡和相机标签，共 38,259,750 字节。六个成员均校验了名称、解压长度和 CRC32。[前次测光与重试依据](2026-11-24-beyond-rgb-near-white.md) · [作者发布](https://zenodo.org/records/16848482)

两张以同一个 D65 相机矩阵、原曝光及 sRGB 显影。24 个色卡中心的最大位移为全幅对角线的 0.01767%。只读基准 1 档时登记右下白背景和保护区。拿到 7 档图后，确认材料、托盘、布边和遮挡对应。对整图及两处地标依次实际测到零位，梯度相关分别为 0.830、0.662、0.788。两张整图显影裁切约 20.49% 与 20.63%，登记背景区分别为 0 与 0.0464%。拍摄 ISO 有少量差异，照度是探头位置的读数，发布的三通道图像无法确认传感器裁切。结论限于该登记区域与固定显影。[取区、门槛与顺序](../../experimental/adaptive-reference-20261113/beyond-rgb-near-white-pixel-protocol.md) · [全部数值与文件哈希](2026-11-24-beyond-rgb-near-white-match-evidence.json)

## 两个方向的实际结果

两档各有一张调色关态和一张局部暖色开态。关态与输入逐像素一致，保护区及编辑范围之外的差值 P95 均为 0。以下 RGB8 为三态共同有效像素的逐通道中位变化。既定门槛要求共同覆盖至少 80%，两种变化各至少 3 RGB8，方向余弦至少 0.90，编辑与实拍幅度比 0.75–1.25。

| 方向 | 实拍换灯 | 有意加暖 | 共同覆盖 | 方向余弦 | 幅度比 | 判定 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 7 → 1 | [14, 0, −15] | [8, −3, −15] | 99.72% | 0.951 | 0.841 | 匹配 |
| 1 → 7 | [−14, 0, 15] | [7, −3, −15] | 98.56% | −0.936 | 0.820 | 颜色方向未通过 |

EGM Stage 5 复核核对了筛选回执和先取区后查看照片的顺序。复核指出，预检入口默认只测整图，两个局部地标必须各自调用其配准函数。上文记录了三个实际调用的结果。协议还补明旧预检双向规则，两向均保存，任一方向达原门槛可登记一组匹配。取区、候选、软件和数值门槛没有调整。Beyond RGB 的同实验室素材和衍生图按同一开发来源归组。白背景实验只证明静物上的匹配，真人肤色、场景白墙、质量盲评和视频仍需各自证据。[既有预检定义](2026-11-24-warm-capture-preflight.md)

## 复现

从仓库根目录运行。保留原固定 ZIP 中央目录，使用空输出目录和现有研究环境。上次缓存同时包含 `.range` 和对应的 HTTP 头文件。筛查命令复用现有脚本，缺测光时可从官方发布补取，完整回执已钉住 SHA-256。

~~~powershell
$src = 'experimental/adaptive-reference-20261113'
$base = 'outputs/adaptive-reference-20261113'
$run = "$base/beyond-rgb-near-white-replay"
$url = 'https://zenodo.org/records/16848482/files/beyondRGB.zip.002?download=1'
New-Item -ItemType Directory -Force "$run/WHITE_blue-light1","$run/WHITE_blue-light7" | Out-Null
& '.tools/adaptive-env/Scripts/python.exe' "$src/audit_beyond_rgb_near_white.py" "$base/beyond-rgb-index-v1/central.bin" "$run/cache" "$run/selection.json"
if ($LASTEXITCODE -ne 0) { throw 'Selection audit failed' }
$p1 = "$run/WHITE_blue-light1"
$p7 = "$run/WHITE_blue-light7"
curl.exe --proxy http://127.0.0.1:7897 --fail --silent --show-error --max-time 100 --range 38882445606-38900969943 --max-filesize 18524338 --dump-header "$p1/headers.txt" --output "$p1/archive.range" $url
if ($LASTEXITCODE -ne 0) { throw 'Light 1 capture unavailable' }
curl.exe --proxy http://127.0.0.1:7897 --fail --silent --show-error --max-time 100 --range 37939568809-37959304220 --max-filesize 19735412 --dump-header "$p7/headers.txt" --output "$p7/archive.range" $url
if ($LASTEXITCODE -ne 0) { throw 'Light 7 capture unavailable' }
& '.tools/tf-convert-env/Scripts/python.exe' "$src/prepare_beyond_rgb_near_white.py" "$run/selection.json" $run "$run/common-display"
if ($LASTEXITCODE -ne 0) { throw 'Capture audit failed' }
~~~

共显影后按[暖白灯预检命令](2026-11-24-warm-capture-preflight.md)映射 ASCII 盘符。将 `common-display/light1.png`、`common-display/light7.png` 送入预检。拍摄记录参数使用 `beyond-rgb-near-white-pixel-protocol.md`，区域传 `--edit 930 728 1007 756 --guard 109 203 158 273`。在现有 Python 环境对两张图的整图及协议的两个地标窗口分别调用 `warm_capture_preflight.registration()`。三态有效像素为每张各通道都在 4–251 内的同一组像素。各自取中位数，用实拍和编辑两向量的点积和长度独立复算两个方向。

测光缓存复跑回执相同。两遍显影的两张 PNG、两份裁切掩膜及来源报告逐字节一致。两遍暖色导出的 11 份 PNG、2 份 PP3 和 1 份 JSON 逐字节一致，两个方向均从落盘成片独立复算，已有预检的五项测试通过。本机保留发布原片、标签、范围和完整成片。下一次先把新参考接到现有两版工作台，预算 30–60 分钟，无训练。把新组与 MIT 组在目标、强度和成片条件一致时并列。之后据实际失败模式决定原定跨来源训练是否值得启动。[训练预算与原开发门槛](2026-11-24-spatial-condition-design.md)
