# 公开来源能否提供换灯与同一编辑的四张对照

当前两版都会漏掉真实的脸颊提亮。下一版若要同时保护未编辑的肤色和白墙，并保留作者确实做过的局部调色，训练前仍需要检查同一拍摄内容下的两种光照，以及一项在两种光照下均能独立开关的编辑。此轮先核对指向同一场景的公开帖子、作者给出的拍摄说明和侧车。资料排查的预算为 30–45 分钟，先查完整可见目录。出现两份原片和两份作者设置时，优先取不超过 7 MB 的相机 JPEG 和不超过 128 KiB 的编辑设置核对，无需先下载较大的 RAW。准入条件仍沿用原四图方案。[既有失败成片](2026-11-24-playraw-flash-room-workbench.md) · [素材准入与拍摄方案](2026-11-21-capture-edit-intake.md)

## 最接近条件的一组有两张原片，但拍摄与编辑都不同

《Camera evaluation》的两次实拍各有公开 CR2，回帖第 44 楼由同一编辑者给出两份 darktable 3.8.0 侧车。直接取相机 JPEG 后，左图拍的是餐桌和柜子，右图拍的是台灯与挂画，不能按同机位同内容归组。两张尺寸相同、焦距皆为 50 mm，但拍摄参数不同。两份侧车里主曝光与 RGB 色阶参数相同，RGB 曲线和白平衡参数不同，色彩区域模块在第一张禁用、第二张启用。共用部分参数无法提供同一个局部操作在两档都可开关的证据。本轮止于 JPEG 和侧车，没有取两份合计显示约 52 MB 的 CR2。[原帖及许可](https://discuss.pixls.us/t/camera-evaluation/28283) · [第二次拍摄](https://discuss.pixls.us/t/camera-evaluation/28283/20) · [编辑者的两份设置](https://discuss.pixls.us/t/camera-evaluation/28283/44)

| 相机 JPEG | 快门 | 光圈 | ISO | 焦距 |
| --- | ---: | ---: | ---: | ---: |
| IMGA7451 | 1/200 秒 | f/6.3 | 16000 | 50 mm |
| IMGA7460 | 1/125 秒 | f/7.1 | 1600 | 50 mm |

## 其他可疑配对的结果

| 公开来源 | 核对结果 |
| --- | --- |
| [Foul blue LED lights](https://discuss.pixls.us/t/foul-blue-led-lights/36063) | 共 54 楼。公开一张 RAW，有局部修正侧车。另一张由作者说明是不同角度，公开的是已处理的图。无蓝灯的官方照片不具备原机位和处理可控的证据。 |
| [First offering in still life](https://discuss.pixls.us/t/first-offering-in-still-life/45708) | 共 14 楼。第一张有原片和设置，作者后来展示换滤片的第二次尝试。第二张在所查发布中只有展示图，缺可核对的原片和对应编辑开关。 |
| [Stage lighting at local concert](https://discuss.pixls.us/t/stage-lighting-at-local-concert/39828) | 共 21 楼。同场演出的四张额外成片仅见 JPG，所有可见 RAW 链接只指向最初的 `XH1F4774.RAF`。 |
| [Variable Lighting Workflow](https://discuss.pixls.us/t/variable-lighting-workflow/53701) | 共 44 楼。帖子有局部白平衡和建筑提亮的作者处理记录，所有可见 RAW 链接指向同一张 `P1020943.RW2`。 |
| [Serge Ramelli 的单灯教程](https://www.youtube.com/watch?v=YoIw-9Y9X8c) | 作者讲解中提到拍有闪光和无闪光，也实际展示复制修图设置。复制设置的那一对用于清除画面中的人物和器材，其中一张是清空后的背景底图。未核实另有固定主体、可分别取得、带该编辑开关的一对源文件。 [讲解字幕](https://filmot.com/sidebyside/YoIw-9Y9X8c/en/en/English/English/All%2Byou%2BNEED%2Bis%2Bone%2BLIGHT%2B%7C%2BTUTORIAL!) |
| [PixTalk 论文 §3](https://openaccess.thecvf.com/content/ICCV2025/papers/Conde_PixTalk_Controlling_Photorealistic_Image_Processing_and_Editing_with_Language_ICCV_2025_paper.pdf) | 多项 Lightroom 编辑可追溯到每张 RAW；论文里的色温和闪光白平衡变体是同张原片的数字显影设置，未报告所需的一组实景重复换灯。 |

这次新增合格四图组为 **0**。两侧各有已验证的素材，实际换灯仍用 Gros 等受控采集，真实的局部调色用已经重放并经过两版工作台的作者对照。将两者用在各自的回归项目。启动依赖交叉监督的训练，还要取得可核对的同场景多灯档与对应编辑记录。停止继续下载本页候选的 RAW，也不在这批素材上重复调局部强度。下一次只在来源明确提供可查的四项记录，或拿到按现有方案实际采集的一小组时，再重开此项准入。独立保留集仍封存。[已有实际换灯证据](2026-11-24-gros-mixed-light-transfer.md) · [设备、采集和训练预算](2026-11-21-phase-cost-and-gates.md)

## 核对范围和复现

从论坛公开主题 `.json` 读取可见帖子 ID，再用 `/t/<主题 ID>/posts.json?post_ids[]=<ID>` 读取首批之外的帖子。五个主题共 178 楼，实际取得的 ID 与各自发布流一致。只记录上传链接及原片文件名，统计范围是五个主题当时可见的发布内容。共下载 566,586 字节的公开索引、两张相机 JPEG 6,378,479 字节、两份侧车 30,934 字节。RAW 下载为零。相机 JPEG 的 SHA-1 与发布路径对应，再核对 EXIF 及侧车模块启用状态和参数哈希。响应、相机 JPEG、设置和本机并排图均留在忽略目录 `outputs/adaptive-reference-20261113/public-capture-edit-source-check-v1`。未修改产品或运行训练。[文件哈希与结构化证据](2026-11-24-public-capture-edit-check-evidence.json)
