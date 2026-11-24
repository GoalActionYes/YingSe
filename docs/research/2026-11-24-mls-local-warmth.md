# 固定曝光的多档换光仍找不到与局部暖色同时同向同幅的组合

Visillect MLS 使用手动曝光和固定白平衡，四个 LED 颜色组合各有四个功率档。本次固定作者标注有白色背景的 02、03、05、22，遍历全部 18 档。所有 72 张图的 EXIF 都确认 1 秒、f/16、ISO 320 及相同白平衡模式。照片实际为 600×493，内嵌同一份线性 Rec.2020 ICC。使用相对色度转换到 sRGB，保持照片尺寸，仅用最近邻缩放作者 578×475 掩膜。修改在像素评分前冻结。编辑位置仅由作者白背景掩膜决定。色度参数和判定门槛与上一轮一致。[事前协议](../../experimental/adaptive-reference-20261113/mls-local-warmth-protocol.md) · [采集说明](https://arxiv.org/html/1908.06126#S4) · [数据许可](https://github.com/Visillect/mls-dataset#license--citation)

| 场景 | 编辑素材线 | 编辑、梯度均过 | 满足前置条件的有序方向 | 方向余弦 ≥0.90 | 完整匹配 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 02 | 9／18 | 8／18 | 42 | 2 | 0／306 |
| 03 | 7／18 | 6／18 | 25 | 1 | 0／306 |
| 05 | 7／18 | 7／18 | 36 | 1 | 0／306 |
| 22 | 10／18 | 7／18 | 30 | 6 | 0／306 |

前置条件包含两档素材与梯度、共同可测覆盖及足够的物理和编辑信号。133 个方向达到这些条件，10 个方向达到余弦门槛，3 个方向达到幅度比门槛，两者没有交集。例如 03 号将蓝绿光由 50% 切到 25%，白背景 RGB 中位变化为 11、−13、−39。有意暖色在原灯档的变化为 10、−4、−14。颜色方向余弦 0.9419，编辑幅度仅为实拍换光的 41.51%，低于预设 75%。全部 72 张关态逐像素保持，保护区 RGB8 差 P95 为零。达到素材线的 33 张图可作保留有意暖色的开发正例。[逐组数据和哈希](2026-11-24-mls-local-warmth-evidence.json) · [03 号本机成片](../../outputs/adaptive-reference-20261113/mls-local-warmth-v1/03/comparison.png)

已完成四个场景的全部灯档，不再添加相似灯档寻找这套暖色参数的匹配。现有室内外同款局部提亮配对和保护暖色正例用于下一候选的不同检查。第二个实拍暖色匹配来源依然缺失。下一阶段回到 EGM 已审过的局部条件与整图条件设计，先验证现有跨来源亮度数据能否支持可辨别的成片收益，同时保留已有肤色和暖色反例。跨来源暖色识别和完整产品验收照原要求验证。[EGM 设计](2026-11-24-spatial-condition-design.md) · [实际工作台亮度对照](2026-11-24-cross-source-wall-lightening-workbench.md) · [原暖色成片](2026-11-24-shared-local-warmth.md)

## 复现与核验

使用本机发布索引 `mls-warmth-index-audit-v1/preview-central-directory.bin`、材质归档和既有 RawTherapee 5.13。获取的范围上限为 18,837,473 字节。本次从既有缓存复用 6 张，66 个新文件的范围请求共 17,327,094 字节。文件名按 ZIP 实际目录核对，红及红绿档省略了蓝灯条目中的 `LED-` 前缀。两次完整渲染分别为 62.8 秒和 59.6 秒，结果逐字节相同，SHA-256 为 `a6d83b03017d87e0105ff9b408f2fba3a410b438556d25035f9c5d3f410fce0a`。独立核对 72 张 JPEG 的 CRC、哈希、ICC 和 EXIF，重新转换色彩并核对两遍 432 张 PNG、16 份处理设置和 8 张图板，重算全部 2,448 个方向。原图和出图留本机。[分段读取脚本](../../experimental/adaptive-reference-20261113/fetch_mls_warmth_sequences.py) · [执行脚本](../../experimental/adaptive-reference-20261113/probe_mls_local_warmth.py)

~~~powershell
$root = 'V:/outputs/adaptive-reference-20261113'
$env:PYTHONPATH = 'V:/;V:/.tools/tf-convert-env/Lib/site-packages'
$python = 'V:/.tools/adaptive-env/Scripts/python.exe'
$code = 'V:/experimental/adaptive-reference-20261113'
$cli = "$root/rawtherapee-sky-pair-v1/runtime/RawTherapee_5.13_win64_x86_64_release/rawtherapee-cli.exe"
& $python "$code/fetch_mls_warmth_sequences.py" "$root/mls-warmth-index-audit-v1/preview-central-directory.bin" "$root/mls-light-pilot-inputs" "$root/mls-local-warmth-input-new"
& $python "$code/probe_mls_local_warmth.py" $root "$root/mls-local-warmth-input-new" $cli "$root/mls-local-warmth-new"
~~~
