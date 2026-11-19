# 实测监督模型能认出白蓝灯，直接校正仍丢失有意风格

上一轮六张普通参考的错误改动同时涉及明暗与色度，色度 MAE 为 7.96–15.80 RGB8。本轮转用作者手机权重 CTI-Retinexformer，使用其实测光色监督得到的白点估计和校正图。沿用同一批拍摄参考，比较两种传入 13 号的方式：完整校正，以及只使用校正色度并保留原参考 Y′。作者模型训练覆盖 CHROMA-MILL 的训练分区，这两场只作已消费开发测试。固定权重按官方提交身份核验，模型和作者代码仅在本机按论文说明运行，未核实再分发许可。[作者说明](https://github.com/CVC-Color/CHROMA-MILL/tree/bf607cec57739932ad2800501b1aa1601bfe4fe5) · [事前协议](../../experimental/adaptive-reference-20261113/cti-reference-screen-protocol.md)

按预测 CIE xy 到两种实测白点的距离分类，16 张参考均判对灯色，八组 Film A 开关前后分类均一致。真实灯色仅用于评分。每个分支各出 16 张目标图，已知风格与测量区域继续只用于离线评分。

| 目标近中性色 | 13 号原换灯 RGB8 差 | 作者完整校正后 | 保留参考明暗后 |
| --- | ---: | ---: | ---: |
| Scene1_a，风格关／开 | 29.37–35.84 | 0.50–5.28 | 18.13–26.87 |
| Scene2_a，风格关／开 | 27.99–36.25 | 0.13–5.41 | 16.23–32.47 |

完整校正只有四张关闭状态达到 ≤3 的换灯线。开风格后，蓝光参考的近中性色方向在两个场景、两档强度全部反向。Scene1_a 蓝灯 B_241 的方向投影由基线 0.175 变为 −0.074，保留明暗后仍为 −0.035。完整校正的白灯目标最坏误差增加 14.18 RGB8；保留明暗将这项压到 1.63，却没有任何近中性色换灯对照同时通过降幅与绝对门槛。两种方式作用于六张常规参考均超过 1.5 RGB8 退化线，完整校正变化 5.07–12.47，保留明暗为 3.90–8.79。[完整校正成片](../../outputs/adaptive-reference-20261113/cti-reference-screen-v1/author/Scene1_a-blue-241-on-13.png) · [保留参考明暗成片](../../outputs/adaptive-reference-20261113/cti-reference-screen-v1/chroma/Scene1_a-blue-241-on-13.png)

完成该固定作者模型的直接校正筛查，两臂均未达标。受控灯色识别是可复核的信号，只有两个同域场景，仍缺更多拍摄组、肤色和有意光色的验证。未形成软件更新。

两遍各 77 份产物逐件匹配。独立复核 44 张参考、32 张目标成片、CIE xy 分类、亮度保持及风格与换灯指标。复现时将本机已核对 SHA-256 的 `archs.py` 与手机权重 `cti-phone.pth` 放同一目录，将仓库根目录、OpenCV 与已有 einops wheel 加入 `PYTHONPATH`，运行 `probe_cti_reference_screen.py <研究输出根目录> <本机作者资产目录> <新输出目录>`。原模型文件、权重和媒体留在本机。[复现脚本](../../experimental/adaptive-reference-20261113/probe_cti_reference_screen.py) · [数值与哈希](2026-11-19-cti-reference-screen-evidence.json)
