# 成对实拍监督减弱了蓝灯串色，也误改了普通参考

从固定 CHROMA-MILL 手机训练分区新增 Scene3_a、Scene9_b、Scene10_a、Scene23_a，共 16 张实拍、33,748,383 字节。每组四图尺寸一致，八处白／蓝比较的最优整数位移均为零，零位相关为 0.800–0.968。用同场景、同强度、同编辑状态的白灯图作为离线监督。两臂各拟合 88,576 个像素样本，来源同为四个场景。对照使用一个共享的二次 RGB 残差映射，候选额外允许参考的全图 RGB 均值参与残差计算。向 13 号提供校正后的普通 RGB 参考，强度固定 80%。此前的 Scene1_a、Scene2_a 整场景未进入拟合，仍属于项目开发对照。[事前协议](../../experimental/adaptive-reference-20261113/pair-supervised-reference-correction-protocol.md) · [原始基线](2026-11-19-chromamill-capture-cross.md)

| 目标近中性色 | 原换灯差范围 | 共享映射 | 全图条件映射 |
| --- | ---: | ---: | ---: |
| Scene1_a | 29.37–35.84 | 9.45–11.39 | 4.42–7.68 |
| Scene2_a | 27.99–36.25 | 10.57–13.02 | 2.39–5.00 |

范围包含两档灯强与开关风格四种对照，单位 RGB8。全图条件臂八次换灯差均至少减半，只有一次达到 ≤3 的绝对门槛。Scene2_a 的近中性色在两档蓝灯下，Film A 投影由 −0.117／−0.085 变为 0.687／0.559，方向转正，幅度仍不足。两目标的整图与近中性色共 16 项风格比较中只有一项通过。白灯组最差误差增量为 1.39 RGB8，对照臂为 7.24，候选通过该项限制。[反向修复后的成片](../../outputs/adaptive-reference-20261113/pair-supervised-reference-correction-v1/conditioned/Scene2_a-blue-213-on-13.png)

六张非 CHROMA-MILL 的普通参考及 Film B 参考全部超出 1.5 码值退化线。候选直接改变参考 RGB 的差异为 11.19–22.81，实体色卡 Film B 最严重；这项只衡量参考预处理变化，未把它计为目标画质成绩。两臂均未通过完整开发条件，结束直接应用该固定拟合。结果支持继续重视成对实拍提供的风格方向信息，同时证明候选处理普通参考存在明显退化。[被误改的 Film B 参考](../../outputs/adaptive-reference-20261113/pair-supervised-reference-correction-v1/conditioned/hdr_testimage-film_b.png)

两遍各 79 份文件逐件一致，含 32 张目标成片、32 张校正参考、12 张负例、2 份系数与 1 个报告。另从落盘图独立重算源与成片哈希、风格、换灯差和负例。共享解码代码调整后，上一轮的 33 份产物也逐件复现。已有环境设仓库根目录、OpenCV 和 OpenColorIO，运行 `probe_pair_supervised_reference_correction.py <研究输出根目录> <16 张训练图与收据目录> <新输出目录>`。训练素材可用上一轮取材模块的 `fetch(目录, ("Scene3_a", "Scene9_b", "Scene10_a", "Scene23_a"), 48 * 1024**2)` 获取，固定发布及 SHA-256 见证据。输入、成片和系数留本机，软件未接入，独立自然评价及视频验收仍待完成。[复现脚本](../../experimental/adaptive-reference-20261113/probe_pair_supervised_reference_correction.py) · [数值证据](2026-11-19-pair-supervised-reference-correction-evidence.json)
