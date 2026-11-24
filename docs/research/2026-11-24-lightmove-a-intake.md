# LightMove-A 实拍挪灯配对准入

13 号在已有暖墙开发案例中既受实拍灯色影响，也漏掉有意局部暖色。此前 LSMI 的实际换灯变化在方向接近时仍明显大于同款编辑。这里核对一个不同的采集方式：保持灯具和房间相同，挪动灯具改变它对局部浅墙的贡献。待验证的是这种公开配对能否保留足够拍摄证据，并产生与既有暖色编辑幅度相当的真实变化。[工作台失败](2026-11-24-matched-wall-warmth-workbench.md) · [LSMI 结果](2026-11-24-lsmi-daylight-switch.md) · [作者论文第 3.2.2 和第 4 节](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_LightMover_Generative_Light_Movement_with_Color_and_Intensity_Controls_CVPR_2026_paper.pdf)

只读取作者固定发布 `e1e412a12fc5ace912f736cbc725dfa986e157f4` 的 `Lightmove-A`。研究使用以数据卡标注的 Apache-2.0 为依据，图像及作者逐图原始登记留本机。先对照文件夹和标注条目，从同时发布源片、目标片的住宅与工作空间中，按编号为每个原片名字前缀取第一组。前缀只用于让取样覆盖不同记录分组，来源仍只算一个数据集。目录预查得到六组。限取这六组的十二张 JPG，共 2,434,172 字节，核对长度、发布 SHA-256 和 JPEG 元数据，不取生成结果、掩膜或整包。[数据卡](https://huggingface.co/datasets/ZGZzz/LightMover-Benchmark) · [固定发布](https://huggingface.co/datasets/ZGZzz/LightMover-Benchmark/commit/e1e412a12fc5ace912f736cbc725dfa986e157f4)

进入出图前，必须能核对同一相机、机位和焦距，确认曝光参数、白平衡和显影固定，排除自动 HDR 和自动局部色调。用逐档原件或作者的具体采集与显影说明佐证。相同的 JPEG 色彩空间、相机型号或 EXIF 手动白平衡标签均不足以单独证明所有档位使用相同显影。缺任一必要证据就记录该项未确认，结束这批出图准入。开图前先检查标签和尺寸，只有过线组才在源片固定浅色材质区、远处保护区和靠近测区的两处可辨物理地标，再核实目标片中结构、材质和遮挡。灯具本体、移动后露出的遮挡区域和明显新反光不能成为测区。

若通过拍摄和测区核验，沿用 RawTherapee 既有暖色开关、原有效覆盖和通道端点限制。同一浅色测区内，两种变化的平均绝对通道幅度各至少 3 RGB8，方向余弦至少 0.90，编辑与实拍幅度比在 0.75–1.25。保留远处与编辑区外变化，复算两遍看结果能否重现。六组都没有联合命中就结束取样，有命中才用既有工作台交叉比较。目录及取件约 10–15 分钟，达到出图条件另留 25–45 分钟，使用现有 Python 和编辑器。此轮只判断开发素材资格。[原暖色协议和计分](../../experimental/adaptive-reference-20261113/shared-local-warmth-protocol.md)

## 实际取件和复核

固定发布的 `Lightmove-A` 有 200 个完整目录，共 400 张 JPG、600 张 PNG 和 1 份标注 JSON。JSON 有 254 条记录，其中 54 条无发布目录。200 条可用索引均登记同一场景下的两次不同原片，共 200 个场景编号和 400 个 DNG 文件名。六组取样编号为 010、012、015、021、027、051。固定发布的子集未含对应 DNG，作者工具表所列为成片、标注和掩膜。[目录登记与哈希](2026-11-24-lightmove-a-intake-evidence.json) · [作者评测工具说明](https://github.com/GengzeZhou/LightMover_eval#lightmover-benchmark--zgzzzlightmover-benchmark)

12 张 JPG 共 2,434,172 字节，各自匹配发布的 LFS 长度和 SHA-256。每张都是 1024 × 1024、三通道、8 位、JFIF。Pillow 检查和独立文件头扫描均只看到 APP0，找不到 EXIF、ICC 或其他 APP 段。标注文件有 140,217 字节，核对了发布的 Git blob 和 SHA-256。首次 URL 读取遇到连接超时，改用本机既有网络代理按文件续取。此处只统计最终核验的 JPG 大小，重试流量未计。

EGM 复核指出，若要把差异归到灯具摆位，还需该配对的灯具输出设置和环境光稳定记录。MIT 复印室已有灯位变化的配对，LightMove-A 的增量应是另一个可核查的采集来源。作者论文确认了实拍挪灯的任务，但这批发布没有为选定配对补充机身、焦距、快门、ISO、白平衡和共同显影的证据。六组均止于素材准入，墙面方向与幅度未测。继续扩大同格式 JPG 下载无法补足现已确认的记录缺口。[本机 EGM 审查](../../outputs/adaptive-reference-20261113/lightmove-a-source-intake-v1/zoom_out_report.md)

这一步关闭 LightMove-A 的公开成片作为第二个暖墙来源的取样。公开替代来源只有在同时有原件和固定显影依据，或作者给出能对应到具体图对的记录时才值得取件。可控补拍直接解决这个问题。按现有三组预算执行，并在第一组开拍时先检查灯档和导出的变化幅度，再扩到另外两组。暖墙真实对照的新增合格来源仍为 0，待开发候选和后续独立验收。[原补拍及时间预算](2026-11-21-capture-edit-intake.md)

## 复现

先从[固定版本的文件记录 API](https://huggingface.co/api/datasets/ZGZzz/LightMover-Benchmark/revision/e1e412a12fc5ace912f736cbc725dfa986e157f4)读取 `siblings`，筛选 `Lightmove-A/` 路径。把[发布的标注 JSON](https://huggingface.co/datasets/ZGZzz/LightMover-Benchmark/resolve/e1e412a12fc5ace912f736cbc725dfa986e157f4/Lightmove-A/lightmove_eval.json)与目录取交集，按上文规则重建六组。各组的发布元数据从 API 路径 `/api/datasets/ZGZzz/LightMover-Benchmark/tree/<固定版本>/Lightmove-A/<组号>` 取得，下载路径使用同版本 `/resolve/<固定版本>/Lightmove-A/<组号>/<文件>`。用文件长度和 LFS 的 `oid` 核对取得的 JPG，再用 Pillow 的 `size`、`applist`、`getexif()` 和 `info.get("icc_profile")` 复查。独立核对可逐一读取从 SOI 到 SOS 前的 JPEG 段，检查 APP 标识和 SOF 的精度、尺寸及通道数。每项预期结果见数值摘要。[逐图证据](2026-11-24-lightmove-a-intake-evidence.json)
