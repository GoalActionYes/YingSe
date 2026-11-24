# VisibleFixture-60 的拍摄与显影依据

这轮核查了 TokenLight 的实拍开关灯基准，目标是找到能补齐跨来源暖墙对照的普通实拍。它与只给演示页面的来源相比，提供了成对文件清单及线性 EXR。先核对拍摄条件及公开文件，再决定是否值得下载原尺寸图片。按事前范围只取索引与文件头。[核验范围](../../experimental/adaptive-reference-20261113/visiblefixture60-exr-intake-protocol.md) · [作者基准仓库](https://github.com/Vrroom/VisibleFixture-60/tree/59657838536bd9cbedd70df67c43c584ce41c13e)

作者附录 A.2 记载，办公室实拍使用脚架固定 iPhone 机位，打开 ProRAW 自动曝光，逐次物理开关可见灯具，再从 DNG 生成 EXR 并用 Reinhard 处理显示成片。自动曝光可能响应灯具开关。附录未给出逐对曝光比例、统一白平衡或每张从 DNG 到 EXR 的共同转换记录。EXR 是线性容器，仍需证明两张的线性尺度可比较。[作者附录 A.2](https://arxiv.org/html/2604.15310v2#A1.SS2) · [OpenEXR 属性](https://openexr.com/en/latest/StandardAttributes.html)

固定发布的 `data.csv` 有 128 条有向记录，其中 67 条开灯、61 条关灯。对两个 EXR 路径排序归组后有 67 个文件对，其中 61 对含正反两条记录。统计说明发布索引的结构，同一个场景和图像可重复出现。索引有 121 个独立 EXR 路径、115 个 DNG 路径。公开存储的 `Exrs/` 和索引逐一对应，`Originals/` 列表为零，其中明确登记的 `Originals/IMG_CS1_5018.DNG` 直接读取返回 404。核查范围限作者在仓库写明的公开存储路径。[固定数据索引](https://github.com/Vrroom/VisibleFixture-60/blob/59657838536bd9cbedd70df67c43c584ce41c13e/data.csv) · [作者公开 EXR 目录清单](https://tokenlight-demo-test.s3.amazonaws.com/?list-type=2&prefix=visiblefixture60%2FExrs%2F)

121 份 EXR 的首 4096 字节均按 HTTP 206、字节范围和 ETag 核验并解析到完整属性表，共读取 495,616 字节。线性 EXR 存储总量约 1.91 GB。文件头均只有八个基础属性，分别记录通道、压缩、图幅、行序、像素宽高比和屏幕窗口。101 张显示图幅为 1365 × 1024，20 张为 1024 × 1365。无快门、光圈、ISO、相机、白平衡、色彩原色或逐帧曝光校正属性。没有下载全图、打开像素或运行暖色出片，方向和幅度未评估。若作者补出原 DNG 或逐帧曝光和共同处理记录，可重新核对这一来源的资格。

## EGM 复核的推进边界

独立复核确认，第二个匹配来源是当前跨来源暖色训练和留出的必要条件。两墙和三位人像的既有工作台对照共用一个匹配参考，增加目标数无法补齐参考来源。此前试跑的单张参考判断未过线，直接接成开关或继续用同批样例换特征缺乏新依据。课程交付可继续使用已有版本。跨场景升级仍需在原图门槛下完成实际候选、独立图片、真人评价和视频验收。[工作台的匹配参考](2026-11-24-matched-wall-warmth-workbench.md) · [单参考读出结果](2026-11-24-reference-signal-readout.md) · [训练与交付要求](2026-11-24-spatial-condition-design.md)

当前能重启训练准备的最小新证据是独立出处的浅色表面两档实拍、可核验的拍摄及共同显影记录，以及事前固定测区的两对编辑开关。先用既有工具出片并检查两档实际匹配，首组到件后估计 30–60 分钟。达到素材线后才安排训练前小测，正式独立验收仍另用封存来源。[首组材料如何核查](2026-11-24-warm-capture-preflight.md) · [上一组 CML 的具体缺项](2026-11-24-warm-white-source-intake.md)

## 复查文件身份

固定 `data.csv` 的 Git blob 为 `03d3cfc6c4d64df53451e988452d45fa7036ab30`，字节数 35,413，SHA-256 为 `036f6dae67cbf608f2eab4627d4e798155768844a12bca0057f4991faec44099`。按 `exr_1` 和 `exr_2` 去重，使用上述 S3 分目录列表读取对象大小和 ETag。对所列 EXR 请求 `Range: bytes=0-4095` 与对象 `If-Match`，校验响应区间和身份。按 OpenEXR 2 的属性名、属性类型、长度和值解析，直至空属性名。下面的哈希覆盖请求返回的 4096 字节，用于复查第一对索引与另一对文件。[OpenEXR 文件布局](https://openexr.com/en/latest/OpenEXRFileLayout.html)

| 公开 EXR 路径 | 对象字节数 | 首段 SHA-256 |
| --- | ---: | --- |
| `Exrs/IMG_CS1_4934.exr` | 15,720,796 | `3d13c1f4694dedb30dd9354e9c276968086f77905d31d1f6f6915e9695fdc99e` |
| `Exrs/IMG_CS1_4935.exr` | 15,742,351 | `9f2fd0655e0fa509c64bc0da482c073fae6f980b1acb6275a3a769efc7ad185b` |
| `Exrs/IMG_CS1_5017.exr` | 15,829,696 | `c6081c0ac46939ea7a696f3b4d6ea329cf552b02506b5a839a956d46a9f395d9` |
| `Exrs/IMG_CS1_5018.exr` | 15,878,560 | `4354e8de1753bd03c492f30f1839ac9243ff1e5ee080c56ae5c6322be7a9d979` |

逐文件记录保存在本机 `outputs/adaptive-reference-20261113/visiblefixture60-source-v1/exr-header-audit.json`。同时核对过 DIR、Pixel Cube 和 LightLab 的公开入口。DIR 发布的是改变屏幕发光位置的实拍与可合成的多灯设置，未找到小幅暖白灯的登记原片。Pixel Cube 作者页仍写数据待开放。LightLab 作者页公开了示例及处理说明，未找到成套 RAW 的下载入口。[DIR 论文与数据](https://michaelcsj.github.io/DIR/) · [Pixel Cube 作者发布页](https://yufanzhang82.github.io/PixelCube/) · [LightLab 作者发布页](https://nadmag.github.io/LightLab/)
