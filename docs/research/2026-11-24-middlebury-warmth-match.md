# Middlebury 同曝光换灯与有意暖色仍不匹配

沿用已经消费的 Middlebury Art 和 Dolls 场景，补齐 View 1 的三种灯光与三档快门。发布档案中的默认照片与此前同机位照片逐像素一致。两组均核对了 250、1000 和 4000 毫秒三个快门值。仅将不同灯位、同快门照片列入真实换灯比较。Art 用石膏侧脸作浅色测区，Dolls 用人偶布面。暖色参数和测量门槛与上一轮相同，沿用发布 RGB 码值并以 sRGB 作显影工作假设。[发布说明](https://vision.middlebury.edu/stereo/data/scenes2005/) · [事前协议](../../experimental/adaptive-reference-20261113/middlebury-warmth-match-protocol.md)

| 场景 | 暖色开关图通过 | 梯度相关达 0.70 | 两档均过素材线的等曝光方向 | 完整匹配 |
| --- | ---: | ---: | ---: | ---: |
| Art | 6／9 | 5／9 | 4／18 | 0／18 |
| Dolls | 7／9 | 9／9 | 12／18 | 0／18 |

所有关态逐像素保持。通过素材线的 13 张开关图可作为局部暖色开发正例，其中 11 张也过了换灯比较所需的梯度线。Dolls 在 250 毫秒下从灯位 1 换到 2，布面 RGB 的中位变化为 30、17、11。只开启有意编辑时，变化为 7、−2、−10，两向量余弦约 0.15。14 个方向同时有合格两档、共同像素和足够的实拍信号，仍全部未过 0.90 方向线。Art 第 3 档照明在石膏面的水平梯度与默认照片不一致，该组三档曝光均保留作光照结构反例。单项梯度差无法确定具体成因。[全部等曝光方向和读数](2026-11-24-middlebury-warmth-match-evidence.json) · [石膏面本机对照](../../outputs/adaptive-reference-20261113/middlebury-warmth-match-v1/Art/comparison.png) · [人偶布面本机对照](../../outputs/adaptive-reference-20261113/middlebury-warmth-match-v1/Dolls/comparison.png)

结束这两个场景的固定设置，跨来源暖色训练仍缺第二个匹配来源。下一次实拍核查选可独立设置 LED 色谱与功率的 Visillect MLS，理由是本轮换灯会同时提高三个通道，而下一来源有机会在固定快门下实测方向、幅度较接近的增红减蓝。核对固定参数及输入许可后，再沿用原暖色对照范围预先固定灯位和测区。现有 28 张暖色开发正例和 18 个室内外提亮回归继续保留。[后续光源的发布设置](https://arxiv.org/html/1908.06126#S4) · [前一轮暖色正例](2026-11-24-shared-local-warmth.md) · [室内外工作台回归](2026-11-24-cross-source-wall-lightening-workbench.md)

## 复现与核验

从发布页获取第三尺寸 `Art-2views-3illum-3exp.zip` 和 `Dolls-2views-3illum-3exp.zip`。两份档案合计 9,798,807 字节，校验值与下载前官方响应的 SHA-256 标签一致。媒体留在本机。需要此前冻结的默认图档案、RawTherapee 5.13 和研究 Python。在空闲 ASCII 映射盘根向脚本提供新输出路径。[官方档案目录](https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/zip-2views/) · [执行脚本](../../experimental/adaptive-reference-20261113/probe_middlebury_warmth_match.py)

两次完整渲染约 12.4 秒和 11.4 秒，结果逐字节一致，完整报告 SHA-256 为 `0013ef2a862c84b3c65ccc4198252e815f6bb9ba20b919c732fe98b47a177b3c`。独立核对 18 张发布捕获、两遍 108 张本机 PNG、8 份处理设置、4 张图板和 72 个等曝光方向。此次只显影和核查开发素材，产品画质和独立验收仍待候选验证。

~~~powershell
$root = 'V:/outputs/adaptive-reference-20261113'
$env:PYTHONPATH = 'V:/;V:/.tools/tf-convert-env/Lib/site-packages'
$cli = "$root/rawtherapee-sky-pair-v1/runtime/RawTherapee_5.13_win64_x86_64_release/rawtherapee-cli.exe"
& V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/probe_middlebury_warmth_match.py $root "$root/middlebury-warmth-input-v1" $cli "$root/middlebury-warmth-match-new"
~~~
