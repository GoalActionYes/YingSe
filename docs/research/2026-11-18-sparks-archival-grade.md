# 《Sparks》专业调色前后配对及跨镜头基线

已核验同一短片中两组不同镜头的正式调色前后图。近景参考远景时，13 号把青色明显带入窗外和安全帽高光，焊工脸部比作者调色更粉。反向时，13 号远景的天空变灰暖，作者调色的天空更青。两项观察来自本机诊断图。可打开[两向对照](../../outputs/adaptive-reference-20261113/sparks-archival-grade-baseline/comparison.jpg)，每行从左到右为目标原图、另一镜头参考、13 号默认强度调色及目标镜头的作者调色。

此前室内换灯配对有实际灯光变化。这一来源保留真实摄影，也提供同一内容进入正式调色前后的两个状态。它补足了已知原图与成片对，但同一影片的不同镜头可以应用不同修正，未核实具体调色操作或蒙版。Netflix 公开了线性 ACES 未调色档案母版 NAM 和调色档案母版 GAM，发行页面标明 CC BY 4.0。见[制作流程](https://netflixtechblog.com/engineers-making-movies-aka-open-source-test-content-f21363ea3781)、[发行页](https://opencontent.netflix.com/home)和[发行许可](https://s3.amazonaws.com/download.opencontent.netflix.com/sparks/creative-commons-attribution-4-intl-public-license.txt)。

固定取第 4592、9184 帧的两个母版版本，四份 EXR 共 212,476,400 字节。首次直连取第一份超时，随后经现有网络代理按相同对象及 ETag 完整取得。四份文件均与作者 MD5 一致。同组 EXR 时间码相同，缩略图内容一致，亮度梯度幅值相关为 0.9932、0.9880。用相同转换生成 512×270 sRGB8 诊断图。近景 GAM 上界裁切为 0.89% 通道样本，远景为 13.28%，远景另有 2.98% 通道样本低于下界。共同诊断转换会影响颜色误差，本轮结果只覆盖这种显示。见[事前协议](../../experimental/adaptive-reference-20261113/sparks-archival-grade-protocol.md)及[ACES2065-1 编码](https://docs.acescentral.com/encodings/aces2065-1/)。

两方向用冻结的 13 号、80% 强度和双 RGB 输入。每项误差都相对目标镜头同帧 GAM，以 RGB8 码值计。

| 目标 ← 参考 | 整图误差，原图 → 13 号 | 近中性色代理误差，原图 → 13 号 | 近中性色代理像素 |
| --- | ---: | ---: | ---: |
| 焊工近景 ← 双人远景 | 14.37 → 27.07 | 17.27 → 18.61 | 23,671 |
| 双人远景 ← 焊工近景 | 24.89 → 27.55 | 15.15 → 8.57 | 11,893 |

两方向的整图误差均增大，第二方向有近中性色代理收益。肤色和白墙未作专门区域测量。照片均来自同一影片，仅作开发诊断。这轮只验证配对可用和 13 号实际表现，软件尚无新候选。若据此进一步判断局部调色能否迁移，须先用统一标准显示转换复查远景的裁切影响，并检验不同镜头的真实编辑是否有共享响应。

两遍准备和出图的六张 PNG 及两份报告分别字节一致。另直接从 EXR 的行偏移读取中心扫描线，与解码值逐元素一致，复算两组梯度及全部整图和近中性色误差。见[原片身份与评分证据](2026-11-18-sparks-archival-grade-evidence.json)。

复现时按证据文件给出的对象 URL、ETag、尺寸和哈希取得四份 EXR，用短名 `nam-4592.exr` 等保存。按证据文件的四份 `frames` 条目提取 `name`、`url`、`etag`、`size`、`sha256` 写成 `intake.json` 数组。在仓库根目录使用有 OpenEXR 支持的 OpenCV 4.11.0 环境运行 `probe_sparks_archival_grade.py prepare <输入目录> <新准备目录>`。确认两组内容对应后，在现有自适应研究环境运行 `probe_sparks_archival_grade.py render <准备目录> <新输出目录>`。媒体和比较图留本机。
