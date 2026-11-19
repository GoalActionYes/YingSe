# 真人换灯视频：采集分组已核实，原片未取得

只核查此前未登记的真人换灯视频。需要至少一小组有授权的同人物静止片段、可追溯的灯状态及重复或足够接近的白光基线，才能判定真实换灯与后期染色的区别。本轮只读取公开论文、使用条件和 MMPD 的 51,484 字节索引，不下载视频、申请账号或联系他人。

| 来源 | 已核实的条件 | 当前决定 |
| --- | --- | --- |
| MMPD | 固定索引有 660 条、无重复文件键。筛选静止且未运动后，33 人各有低强度 LED、高强度 LED、白炽灯、自然光四种状态，各一段，共 132 段。 | 能按人物分组；目前不能确认白平衡、灯色测量或固定曝光。作者要求机构邮箱提交签署的用途协议，供非商业研究；公布的精简包仍约 48 GB、80×60。未取得适用小样。[固定索引](https://github.com/thuhci/MMPD_rPPG_Dataset/blob/fdfb79703c3807565ee3f395e39fdadf3f165143/meta_label.csv) · [固定使用说明](https://github.com/thuhci/MMPD_rPPG_Dataset/tree/fdfb79703c3807565ee3f395e39fdadf3f165143#access-and-usage) |
| COHFACE | 每人两段室内灯、两段开窗自然光；可用同灯重复估计拍摄波动。40 人共 160 段。 | 切换灯与窗光同时改变光线方向和分布，未给出校准的有色灯状态；不适合作为仅换灯色的对照。原视频受限，需非商业研究许可及机构签署。[采集论文 §2.2](https://publications.idiap.ch/attachments/papers/2017/Heusch_ARXIV_2017.pdf) · [原始发布及访问条件](https://zenodo.org/records/4081054) |
| CHROMA-FIT | 同一批人分别拍摄室内和室外，同种五角度拍摄流程；有实测皮肤颜色。 | 出入室内外同时换背景和光线。论文与实验室公开目录未核对到可直接取得、许可明确的小样。只能作为后续评估不同环境的线索。[论文](https://openaccess.thecvf.com/content/WACV2024W/DVPBA/papers/Pangelinan_The_CHROMA-FIT_Dataset_Characterizing_Human_Ranges_of_Melanin_for_Increased_Tone-Awareness_WACVW_2024_paper.pdf) |

MMPD 索引 blob 为 `75580bca7b64db90024462308fc6bd4ce18d2a9e`，SHA-256 为 `61d69c50c1b7709c06d2501d7d77fa42f281e8336a9a0908144039ab587c12e8`。筛选 `motion=Stationary` 且 `exercise=False` 后，以参与者字段对灯状态计数，并按全表的三个条件独立交叉计数。原表保留本机，Git 只记录合计数值。

以上来源暂不进入算法拟合或出图。它们都未提供本项目所需的同一人物、固定采集条件、可测灯色再加有意后期染色的整组小样。若以后取得授权，只需先检查少量静止帧的姿态、曝光和采集身份，不需要整库下载。风格迁移开发和后续视频验收仍须分别验证；本轮无新画质结果。
