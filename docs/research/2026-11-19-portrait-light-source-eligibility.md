# 实拍肤色光照对照：三套采集库的准入结果

本轮只核验作者资料，回答一个问题：能否直接加入一小批实拍人像，隔离换灯与有意后期染色，复查目前集中在静物上的结论。准入需要同人物、同相机视角、可分辨的采集灯色状态、小批可取得且使用范围明确；还需单独保存有意后期编辑和姿态检查。未下载图像、未安装、未申请账号或联系作者。

| 数据 | 有价值的证据 | 目前的决定 |
| --- | --- | --- |
| BecomingLit | 16 台相机和 40 盏高显色白光 LED。单灯帧和全亮帧交错采集，人物在做表情或说话；可按参与者、序列、相机选取子集。 | 适合研究方向光和运动。不能把相邻帧当姿态严格不变的换灯色正例，也不能把白光方向变化当有色灯对照。整库超过 1.5 TB，访问须审批，只向获批准机构提供非商业研究用数据。暂不准入。见[固定下载说明](https://github.com/jonathsch/becominglit-dataset/blob/4b3ac9b74b281e63a1646436ba439c66dbed3f96/README.md)和[论文 §3](https://papers.nips.cc/paper_files/paper/2025/file/811449856c80b13a1818ea3d777961eb-Paper-Conference.pdf)。 |
| HumanOLAT | 真人全身 HDR RGB，包括白光、环境光和颜色梯度，值得用于判断彩光对真正肤色的影响。作者建议的预处理存档为 10 GB。 | 需要机构邮箱、说明用途、抄送导师及审批；未核验可匿名取得且使用范围明确的同组原始小样。先保留为候选，不下载整库。见[作者数据说明](https://github.com/TMT22/HumanOLAT#quick-start)。 |
| Shadows & Lumination | 每图带两个光源的测量和区域标注，覆盖室内外与多个相机。 | 作者的公开资料未给出本实验需要的同人物、同姿态换灯组及有意调色前后图。只适合作为混合光识别线索，不将区域标注当成需要迁移的风格。见[作者介绍](https://ilijad.github.io/shal.html)和[原论文](https://www.sciencedirect.com/science/article/pii/S095741742300547X)。 |

这次没有发现能直接补齐缺口的小批人像。已有 POLAR 检查结论仍有效：[公开预览核查](2026-11-19-polar-preview-intake.md)。下一次若取得许可明确的实拍小样，先核对脸部位姿和灯色，再把每个场景的重复拍摄、实际换灯、受控后期染色分开编号。采集条件和作者前后图只用于离线判定；软件仍只接收目标和参考两张图。静物开发对照可以继续用于筛错，不能代替肤色、白墙的独立验收。

近场 near_ps 两组真人可按文件取用，作者标定中最近的两个灯位仍相距 105 毫米，无法用这些实拍配对隔离换灯色。Meyer 等有同位置滤片采集，待核实原图入口与使用范围。[标定核查](2026-11-19-near-ps-portrait-intake.md)

补查了两个直接研究肤色和灯光的来源。**优先保留 PolyU Face Color Constancy 作为候选。** [原论文](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/33/1/15)报告八名真人、1,299 张图，分别测量前景和背景的两个光源，并调节人像前方 LED，尽量不影响背景。这样的采集设计与主体和背景分开受光的问题贴近。但[学校的论文记录](https://research.polyu.edu.hk/en/publications/facial-centric-color-constancy-dataset-to-improve-scenario-specif/)、[实验室出版目录](https://www.polyucolorlab.com/publication.html)和论文未提供可核对的原图下载、使用许可或按人物与场景对应的公开配对索引。论文插图不代替原图。即使以后拿到数据，1,299 张图也不能当作 1,299 个独立人像，首先按原始拍摄和参与者分组。

**排除《Color Constancy Using Faces》的配对作为换灯验证。** [作者论文 §II](https://htchen.github.io/awb/awb_single.pdf)的 472 对照片是一人一组，区别是取景里有没有灰卡，两张在同一灯光下拍摄。灰卡给的是灯色测量，可用于另一类问题；不能将有卡／无卡当白灯／彩灯对照。

至此结束从这些公开预览和论文寻找立即可用的肤色换灯素材。恢复此项取样需要带明确许可的小组原图和可核对的换灯顺序；具备这些条件后，再做成片对照。未下载图片、未运行新成片，以上是数据准入结果。

后续补查真人换灯视频：已从固定公开索引核对 MMPD 的 33 个完整静止组，仍缺可直接使用的小样许可；COHFACE 和 CHROMA-FIT 的变化条件不能单独解释灯色。[视频采集来源核查](2026-11-19-portrait-video-source-audit.md)

后续核实 Meyer 与 Oulu 两套真人换灯设计，原片和使用范围仍缺可直接查证的小样。也核对了 CI 与三套闪光对照的数据入口。[复核记录](2026-11-20-actual-portrait-light-pairs.md)

后续从 Gros 的开放数据中取到三组暖／冷灯原片。白幕、主体灯与背景灯有采集说明，自动白平衡和不同灯档光量须纳入解释。[已取得的六张原片与限制](2026-11-20-gros-portrait-light-intake.md)
