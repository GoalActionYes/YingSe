# 混合光实拍的 sRGB 白平衡配对核查

白球服换背景对照检验人物是否被误带偏。还需要实拍正例检验参考中的空间色彩变化确实有效时能否迁移。这一轮核查 sRGB-LSMI 的两份公开存档，仅取网页、固定源码和 ZIP 目录片段。作者使用同一实景在不同光照下的拍摄结果，以色卡校准单光源成片，再将其亮度匹配到混合光成片。这样的目标可以检验空间白平衡，作者未提供局部艺术调色意图或人类自然度评分。见[预设核查范围](../../experimental/adaptive-reference-20261113/srgb-lsmi-intake-protocol.md)、[发布页](https://color.cvc.uab.es/revisitingmiwb/)和[论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Serrano-Lozano_Revisiting_Image_Fusion_for_Multi-Illuminant_White-Balance_Correction_ICCV_2025_paper.pdf)。

- **小批访问。** Nikon 存档 11,401,528,965 字节，Sony 存档 13,083,047,934 字节。两份都返回 `Accept-Ranges: bytes`。Nikon 六次有效区间读取返回 HTTP 206，逐项核对实际长度、归档总长和 ETag。ZIP64 目录本身约 3.41 MB，本轮只取目录片段及尾部记录。
- **配对键。** Nikon 样本目录片段有 145 个键同时出现在 `input` 与 `target`，另有 127 个键同时出现在带遮罩的两类目录。键由原场景标识与光照组合组成。作者的载入程序检查每个键的五种输入和一个目标。本轮只证明目录条目相交，未查完整成员集合、配准、真实人像覆盖或跨场景对应关系。同一场景的光照和白平衡变体应保留在同一分组。见[固定载入程序](https://github.com/davidserra9/revisitingMIWB/blob/fc6696a13f02829d5559464e9ffdc188dabe0c8c/data.py#L30-L65)。
- **使用条件。** 固定代码库的两份分组文件为 0 字节。Nikon 存档目录片段中的分组文件也为 0 字节。原 LSMI 标明 CC BY-NC 4.0。sRGB 衍生发布页提供下载并要求引用，未列独立的图像许可条款。代码库的 MIT 文件对应软件，图像使用范围及正式分组仍待确认。见[上游数据许可](https://github.com/DY112/LSMI-dataset#-dataset-license)、[衍生发布页](https://color.cvc.uab.es/revisitingmiwb/#download)和[固定代码许可](https://github.com/davidserra9/revisitingMIWB/blob/fc6696a13f02829d5559464e9ffdc188dabe0c8c/LICENSE)。

首个目录末段的起止字节偏早 1,000 字节，审计拒绝该响应。按 ZIP64 宣告的目录末端重取后，两遍报告一致。原响应保留。共保存 831,162 字节，含首次范围响应。另用 Python 标准库从五段目录重新解析 3,645 条完整记录，复算两项配对键交集，并逐项核对有效保存响应的哈希。见[验核脚本](../../experimental/adaptive-reference-20261113/audit_srgb_lsmi_index.py)和[小型证据摘要](2026-11-18-srgb-lsmi-intake-evidence.json)。复现时可将保存的响应目录 `outputs/adaptive-reference-20261113/srgb-lsmi-intake` 作为脚本唯一参数。

本阶段到元数据为止。确认图像使用范围后，可按原场景分组小批取样，先检查配准和局部色度变化，再决定能否作为背景换物体之外的正向开发对照。正式独立画质评估、自然度和人物色彩仍需相应证据。本轮未取图像或权重，也未得到新的调色输出。
