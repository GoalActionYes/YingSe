# 映色的开源工具取舍

映色自行负责从参考图与目标图推断可迁移的风格、保护肤色与浅色区域、训练和部署空间调色模型，以及判断何时回退。媒体编解码、色彩标准转换、元数据读取和自动化测试采用成熟工具。创业材料按实际用途注明第三方工具及许可证。若申报有自研比例或指定组件限制，按该项目章程核对。

| 工具 | 映色可以复用什么 | 采用决定 |
| --- | --- | --- |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg) | 视频解码、编码、音轨和已知切点的辅助回归。现有便携包已带 FFmpeg，实际验证了 `scdet`。 | 用现有构建做本机测试。正式分发前，以将要打包的具体构建核对许可、对应源码与构建信息。[官方分发说明](https://ffmpeg.org/legal.html) |
| [Playwright](https://github.com/microsoft/playwright-python) | 浏览器自动化。统一检查导入、同帧预览、手动微调、撤销重做、点导出与下载。 | 复用现有浏览器回归脚本，优先补实际出现的操作回归。项目许可为 Apache 2.0。[许可](https://github.com/microsoft/playwright-python/blob/main/LICENSE) |
| [OpenColorIO](https://github.com/AcademySoftwareFoundation/OpenColorIO) | 标准色彩空间转换和 LUT 应用，科研对照已有相关用法。 | 保留作工具和参考结果。BSD 3-Clause。项目仍自行定义何时调哪里、如何估计风格。 |
| [Colour Science](https://github.com/colour-science/colour) | 复用标准 CIEDE2000 色差。可补充比较脸颊与白墙的误差。 | 已用 0.4.7 复核冻结真人成片，BSD 3-Clause。原门槛照常，色差数值不代替人类自然度评价。[实际对照](2026-11-22-gros-endpoint-colour.md)[许可](https://github.com/colour-science/colour/blob/develop/LICENSE) |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | 独立的自然视频切点对照，候选算法包括自适应检测。 | BSD 3-Clause。已有 FFmpeg 可以先覆盖合成硬切。只在有真实快速运动、闪光和人工标好的切点时扩大验证。 |
| [ExifTool](https://github.com/exiftool/exiftool) | 只读提取 EXIF、XMP、ICC 等信息。用于查拍摄和导出条件。 | 保留为需要 RAW／XMP 核验时的本机工具。许可随 Perl，可选 Artistic 或 GPL 条款，若打包需核对具体发行包。 |

PySceneDetect 和 ExifTool 未安装。Colour Science 的固定发行包仅由本机离线复测读取。代码、训练数据、权重和文档是不同发布物，若借外部模型作为离线比较，分别核对其来源和使用范围，再议是否参加产品操作。仓库公开本身不能证明可重新分发其模型或数据。

## 已用本机便携构建验证的切镜检查

仓库根目录定位现有稳定 1.3 便携包的 FFmpeg，复跑以下示例。黑、白、黑各两秒，24 fps，人工已知边界为第 2 秒与第 4 秒。直接使用过滤器的检测时间作提示。

~~~powershell
$ff = (Get-ChildItem -LiteralPath 'outputs/course-portable-20261118-v1/packages/reference-color-20261118-015906' -Recurse -File -Filter '*ffmpeg*.exe' | Select-Object -First 1).FullName
$graph = 'color=black:s=160x90:r=24:d=2[a];color=white:s=160x90:r=24:d=2[b];color=black:s=160x90:r=24:d=2[c];[a][b][c]concat=n=3:v=1:a=0'
& $ff -nostdin -hide_banner -nostats -f lavfi -i $graph -vf 'scdet=threshold=10' -an -f null -
~~~

第一次与复跑均报 2 和 4 秒，纯黑六秒对照无触发。纯黑中插入第 48 帧即时白色闪光时，第 2 秒有一次误报。该构建可以承担已知切点的快速回归，实际视频仍需拿人工标注的切点对照。此测试没有启动映色推理，也没有测完整视频导出性能或颜色稳定性。[scdet 官方说明](https://ffmpeg.org/ffmpeg-filters.html#scdet)

实测二进制为 Gyan FFmpeg 7.1，SHA-256 为 `2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3`。`-buildconf` 含 `--enable-gpl --enable-version3 --enable-static` 与 `--enable-libx264`。`-L` 声明该二进制是 GPL v3 或更高版本。这项事实只识别具体便携构建的许可和来源。产品正式分发方案可比较单独获取媒体工具与按对应条款分发经过核对的构建，比较的同时验证原导出能力。研究记录不包含二进制或样本视频。
