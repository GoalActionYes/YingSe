# 映色的开源工具取舍

映色自行负责从参考图与目标图推断可迁移的风格、保护肤色与浅色区域、训练和部署空间调色模型，以及判断何时回退。媒体编解码、色彩标准转换、元数据读取和自动化测试采用成熟工具。创业材料按实际用途注明第三方工具及许可证。若申报有自研比例或指定组件限制，按该项目章程核对。

| 工具 | 映色可以复用什么 | 采用决定 |
| --- | --- | --- |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg) | 视频解码、编码、音轨和已知切点的辅助回归。现有便携包已带 FFmpeg，实际验证了 `scdet`。 | 用现有构建做本机测试。正式分发前，以将要打包的具体构建核对许可、对应源码与构建信息。[官方分发说明](https://ffmpeg.org/legal.html) |
| [Playwright](https://github.com/microsoft/playwright) | 浏览器自动化。可检查导入、同帧预览、手动微调、撤销重做、导出下载和离线评审页面。 | 继续用现有 Playwright Core 脚本，优先覆盖已出现的操作问题。项目许可为 Apache 2.0。[本次复测版本的许可](https://github.com/microsoft/playwright/blob/v1.58.2/LICENSE) |
| [OpenColorIO](https://github.com/AcademySoftwareFoundation/OpenColorIO) | 标准色彩空间转换和 LUT 应用，科研对照已有相关用法。 | 保留作工具和参考结果。BSD 3-Clause。项目仍自行定义何时调哪里、如何估计风格。 |
| [Colour Science](https://github.com/colour-science/colour) | 复用标准 CIEDE2000 色差。可补充比较脸颊与白墙的误差。 | 已用 0.4.7 复核冻结真人成片，BSD 3-Clause。原门槛照常，色差数值不代替人类自然度评价。[实际对照](2026-11-22-gros-endpoint-colour.md)[许可](https://github.com/colour-science/colour/blob/develop/LICENSE) |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | 独立的自然视频切点对照，候选算法包括自适应检测。 | BSD 3-Clause。已有 FFmpeg 可以先覆盖合成硬切。只在有真实快速运动、闪光和人工标好的切点时扩大验证。 |
| [ExifTool](https://github.com/exiftool/exiftool) | 只读提取 EXIF、XMP、ICC 等信息。用于查拍摄和导出条件。 | 保留为需要 RAW／XMP 核验时的本机工具。许可随 Perl，可选 Artistic 或 GPL 条款，若打包需核对具体发行包。 |
| [jsPsych](https://github.com/jspsych/jsPsych) | 可在浏览器中展示图像、安排试题顺序并导出评分，采用 MIT 许可证。 | 暂不接入。现有离线页已支持当前的对照评分与保存。需要跨多场次分配不同题目或采用更复杂的呈现顺序时，再用少量试题验证接入收益。[数据接口](https://www.jspsych.org/latest/reference/jspsych-data/) |

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

## 2026-11-24 离线评审工具复测

直接复用已有两份离线页面及其 64 组历史开发对照。在新建本机目录中复制两份 HTML 和各自的清单，四份文件均与原件 SHA-256 一致。沿用 `verify_coverage_human_review.cjs`，由 Playwright Core 1.58.2 驱动 Edge 158.0.4370.33，Node 版本为 24.22.3。浏览器检查实际耗时 26.6 秒，使用现有本机工具，零新增费用，无需训练。[原材料与方法身份](2026-11-17-human-diagnostic.md) · [复测脚本](../../experimental/adaptive-reference-20261113/verify_coverage_human_review.cjs)

两份页面均正常显示对照图，初始都是 0 票，缺少评审编号时提示填写。两份都成功导出 1 组的部分结果和 32 组的完整结果，保留「无法判断」，所导出结果的材料哈希与清单一致，刷新后恢复作答。新浏览器上下文仍为 0 票，共记录 0 个页面错误。自动测试固定使用 `AUTOMATED TEST` 标记。本次范围为页面验证，真实评审人数与票数均为 0。

复测本机目录为 `outputs/adaptive-reference-20261113/open-source-review-check-20261124-v1`。`browser-checks.json` 的 SHA-256 为 `72ab924d817bfc23885f74453f893a7a49323efa8ec14058af9526bef724f2f3`。用新的本机输出目录复现，分别放入 `part-1`、`part-2` 的原 HTML 与清单。运行上述脚本时依次传入新目录、Playwright Core 的 `index.js` 路径和 Edge 可执行文件路径。完整报告、截图和图片保留本机。

后续正式对照使用同一目标原图、同一参考图下稳定版与合格候选的实际成片。页面隐去版本名称，评审分别判断贴近参考、画面自然度及人物与环境的协调性。素材确认后，第一份小批量材料生成和核验估计用 1–2 小时，评审组织和结果回收另计。以后有标好切点、闪光和快速运动的素材时，预计用 1–2 小时试用 PySceneDetect，对照已知标注与现有 FFmpeg 结果，计入离线视频测试，无需训练。
