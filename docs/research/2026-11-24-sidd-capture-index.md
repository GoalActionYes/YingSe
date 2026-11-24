# SIDD 原片的曝光标签与发布索引不一致

上一轮未能用 FAU 档案取得第二个暖色匹配来源。SIDD 提供同一静物场景的物理换灯记录和原始 DNG。本轮先核验原片，判断它能否进入有意暖色与实际换灯的开发预审。2 个实例的索引均写 iPhone 7、ISO 100 和 1/160 秒，色温分别标为 5500 K 与 3200 K。查看原片后，4 张均记录 ISO 1600、1/2000 秒，SIDD 作者单独发布的 MAT 元数据给出同样的数值。现阶段无法确认索引与原片冲突的原因，停止这组的暖色匹配准入。[事前协议](../../experimental/adaptive-reference-20261113/sidd-fixed-cct-intake-protocol.md) · [原始索引](https://abdokamel.github.io/sidd/files/SIDD_URLs.txt) · [作者的第二镜像索引](https://abdokamel.github.io/sidd/files/SIDD_URLs_Mirror_2.txt) · [固定版本发布清单](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC-DNG/resolve/6d0169e354f11ce3d98c1ffa214f9acf9d373155/metadata/train_metadata.csv)

## 为什么取这组

先检查其他公开来源。FAU 的 Bleier 2011 年档案覆盖四个固定布景和同灯位色温滤片。对 6,976,383,112 字节的 ZIP 只读取 58,832 字节目录，核对 542 个成员后，再读取带 CRC 校验的说明和 8 个 PNG 的文件头。论文写明为避免过曝或欠曝，会在不同灯档间改变快门。文件名里的 `0` 表示曝光补偿，8 个 RGB16 PNG 在图像数据前均只有标准图头和时间块，没找到足以确认等快门的证据。未下载该档案的整张照片。[FAU 源档](https://www5.informatik.uni-erlangen.de/research/data/multi-illuminant-dataset/index.html) · [Bleier 等，原论文第 4 节](https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2011/Bleier11-CCA.pdf)

按场景、机型、ISO、快门和亮度类别归组，SIDD 官方可读索引的 160 个实例中有 4 组名义上的跨色温配对，全部来自静物场景 001。低 ISO 的 iPhone 两档均在独立的 DNG 重分发中可读，并且可以只取固定文件。每档按文件名排序选择第 1 和第 76 张，以同档复拍检查照片间自然差异。DNG 取样共传输 51,738,445 字节，另取两份原始 MAT 压缩包合计 1,816,957 字节。两份近 2 GB 的 DNG ZIP 均保持远程。原作者网页列数据集和代码为 MIT，重分发页要求保留 SIDD 的条款和归属。原片、元数据包和成片均留本机。[SIDD 下载说明](https://abdokamel.github.io/sidd/) · [保留原始 DNG 的重分发说明](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC-DNG)

## 实测结果与 EGM 复核

| 核验项 | 两组均有的结果 |
| --- | --- |
| 原 DNG 与作者 MAT | 4 个文件的原始文件名和字节数均对应。快门 1/2000 秒、ISO 1600、f/1.8、焦距 3.99 mm |
| 白平衡和传感器 | 4 张的 AsShotNeutral、相机矩阵、原始像素尺寸、CFA 排布及黑白位读数相同。DNG 的 BaselineExposure 分别是 2.26、2.25 EV |
| 全图构图初筛 | 6 个组合最佳平移均为 0，零位梯度相关 0.9646–0.9682 |
| 同档两张 | 共同显影后的整图平均绝对差约 3.37 RGB8，红绿蓝中位变化均为 0 |
| 跨档四种组合 | 平均绝对差约 8.98–8.99 RGB8。从 0028 到 0033，中位变化均为 `[-7, 5, 14]` |
| 显影前的传感器 | 未去马赛克的四组像素中位数在各自档内相同。0028 为 `[573, 616, 616, 562]`，0033 为 `[569, 642, 642, 592]` |

EGM 从独立上下文核对了原片哈希和曝光。Pillow 读取、直接读取 TIFF 的 EXIF 字段、原作者的 MAT 摘录三种检查一致。MAT 中的文件名与大小进一步支持重分发取到了对应文件。色差在显影前也可观测到，不能据此确定发布的色温数值或冲突原因。[原始拍摄和处理，论文第 3、4 节](https://openaccess.thecvf.com/content_cvpr_2018/papers/Abdelhamed_A_High-Quality_Denoising_CVPR_2018_paper.pdf)

复用旧 RawPy 函数时固定白平衡并关闭自动提亮。EGM 核对到它仍使用 LibRaw 的默认白点调整阈值，显影差异只用于描述这次预览。SIDD 论文在 iPhone 7 同档连拍中也观察到光学防抖造成的局部位移，零位全图初筛无法证明各个评分框都对齐。以后收到合格原片时，需要固定显影的有效白点和比例，并复用既有局部关键点检查覆盖编辑区和保护区。这组文件以来源标签冲突结束，未进入 RawTherapee 暖色开关、工作台候选或训练。[局部配准检查](../../experimental/adaptive-reference-20261113/audit_synclight_registration.py#L55) · [LibRaw 输出参数](https://www.libraw.org/docs/API-datastruct.html)

## 复现和用途

以下在仓库根目录执行。取样使用既有 `requests` 环境，核验只加载本机 DNG、Pillow、RawPy 0.28.0 和原项目研究依赖。可沿用本地取样，直接在新目录重复核验。需代理时，为取样脚本传入当前环境的代理地址。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$py = '.tools/adaptive-env/Scripts/python.exe'
& $py experimental/adaptive-reference-20261113/fetch_sidd_fixed_cct.py "$base/sidd-fixed-cct-intake-reproduce"
$env:PYTHONPATH = ".;$base/synclight-real-pixel-input/decoder"
$env:OPENBLAS_NUM_THREADS = '2'
& $py experimental/adaptive-reference-20261113/audit_sidd_fixed_cct.py "$base/sidd-fixed-cct-intake-reproduce" "$base/sidd-fixed-cct-audit-reproduce"
~~~

取样核对固定仓库版本、HTTP 区间、ZIP 成员 CRC 与 DNG 哈希。两遍本机解码与比较各约 6.3 秒，四张 PNG 和结果 JSON 逐文件一致。另从落盘 PNG 复算 6 组颜色差。从作者第二个 URL 索引按原始索引同一行取得 `0028_METADATA_RAW.zip` 与 `0033_METADATA_RAW.zip`，各取排序后的第 1、76 份 `.MAT`。以 SciPy 读取 `metadata.DigitalCamera` 的 `ExposureTime` 和 `ISOSpeedRatings`，并核对 `Filename` 末段和 `FileSize`。[数值、区间、原片与 MAT 哈希](2026-11-24-sidd-capture-index-evidence.json)

只有能够核对光源记录与相机设置的材料，才进入下一次局部暖色准入。现有的样本和输出可保留为取材核验的反例，用于防止把目录的拍摄标签当成逐张原始证据。
