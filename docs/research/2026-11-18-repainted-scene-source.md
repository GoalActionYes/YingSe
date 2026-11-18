# 找到可将物体颜色与灯光分开验证的实拍档案

之前按亮度和颜色匹配不同照片时，相近的像素仍可能来自不同材质。本轮核对实拍发布能否提供两类可区分的变化：同一表面换光，以及相同灯光下表面颜色改变。先限定为档案目录、说明与许可核查，未读取图片。见[核查范围](../../experimental/adaptive-reference-20261113/repainted-scene-source-protocol.md)。

- **FAU 2011 喷灰实验。** 同一场景先拍原色，再喷灰并在原灯位重拍。ZIP 目录实际有 chalk、figures、fruits 和 rabbit 四组，每组 17 种照明、3 档曝光均有原色与喷灰对应，共 204 对。每组还有 9 个与零档曝光对应的照明真值。发布方指出喷灰会改变表面间反射，反射影响仍须单独检查。档案约 6.98 GB，HTTPS 支持范围读取，取得目录与说明只用 59,928 字节。发布页及档案说明要求引用论文，所查内容未列明确的数据许可；本轮未将照片纳入后续样本。见[官方发布](https://www5.informatik.uni-erlangen.de/research/data/multi-illuminant-dataset/index.html)。
- **Visillect MLS。** 发布说明提供表面掩膜、表面反射光谱和二者对应表，许可为 CC BY-SA 4.0。已通过发布方列出的 FTP 地址读取预览 ZIP 的目录，确认 24 组场景各有相同 18 种灯设置，共 432 张。预览档案为 93,125,444 字节，仅取末尾 49,929 字节，重读哈希一致。预览掩膜、表面光谱和线性 RGB 档案的 FTP 长度分别为 63,177、910,419 和 2,000,043,792 字节。HTTPS 端返回 403，FTP 官方入口可读。像素、掩膜和材质对应表还未核对。见[官方数据格式及许可](https://github.com/Visillect/mls-dataset#license--citation)。
- **ICTPolarReal。** 作者公开样例含偏振照片与反射估计所需文件，默认样例约 400 MB、覆盖一个相机视角。代码仓库的 MIT 许可不能确认该照片档案的许可。先保留为线索。见[作者发布说明](https://github.com/jingyangcarl/ICTPolarReal)。

FAU 两遍目录报告逐字节一致。另逐场景复核曝光、灰色后缀配对和灯光真值文件名。MLS 用相同 FTP 范围重取目录并核对哈希、场景与灯设置全集。见[身份及数值](2026-11-18-repainted-scene-source-evidence.json)。本轮没有获取图片、训练或改变软件。

下一轮优先核查 MLS 的表面掩膜与光谱名称能否把**同一种材质**关联到不同场景。若能关联，就固定同材质换灯和同灯换材质两套实拍对照，先检验基线误差，再决定某个新线索能否进入正常两图推理。即使这项诊断通过，肤色自然度、编辑意图和视频仍须另行验证。

复现时在现有研究环境运行[核查脚本](../../experimental/adaptive-reference-20261113/audit_repainted_scene_source.py)并传入新 JSON 路径。默认检查 FAU，加 `--mls` 检查 MLS。完整索引及下载的目录字节保存在本机。
