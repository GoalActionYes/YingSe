# ARRI Encounters 提供可核对的原镜头清单

ARRI 为 ALEXA 35 调色工作流发布了 Encounters 的镜头指南、H.264 代理入口与 AWG4／LogC4 TIFF 档案入口。指南逐条给出原镜头代码和题材标签，可补足已有开源视频对照缺少上游镜头身份的问题。工作流测试用途见 ARRI 官方描述。本轮只查这些公开文档与访问元数据。[运行前协议](../../experimental/adaptive-reference-20261113/arri-encounters-source-protocol.md) · [官方页面](https://www.arri.com/en/learn/camera-systems/camera-sample-footage-reference-image) · [2023-06-16 指南](https://www.arri.com/resource/blob/297210/17b095dc68842f8dca22f614810e0134/alexa-35-encounters-sample-logc4-clips-data.pdf)

- 指南 17 个编号对应 16 个原镜头代码。04 和 05 同源，不能跨拟合和验收分开。按日期代码可先保守分为 11 组，仍需再排查同一人物、场景跨日期拍摄的情况。
- 指南的 06 描述室内外明暗差与肤色，08 描述城市中灰色建筑。逐镜头网页显示的 TAR 文件名与原镜头代码一致，日期代码不同。说明可为肤色和近中性色准备有源身份的技术对照。灰色建筑标签不是像素级中性色标注。[06 肤色镜头入口](https://f.io/YoTDWoFh) · [08 灰色建筑镜头入口](https://f.io/cqYQ0cV1)
- 两个合集网页和两个逐镜头网页均返回 HTTP 200，公开页面显示下载功能启用。未核实 TAR 大小、逐帧访问、代理与原片的帧对应，或可匹配的作者创意调色成片。代理标题仅证明它是该工作流的预览入口，不能当作创意调色答案。[代理入口](https://f.io/Jft16WLA) · [TIFF 合集入口](https://f.io/D26S9jpk)

五份文档响应合计 1,941,300 字节。通过代理取得指南的第一次 TLS 握手失败，切到直接连接后取得完整 PDF。两份本机摘要逐字节一致，另用独立 PDF 提取器复核 17 条镜头、16 个不同代码和 11 个日期组。四个网页的响应与文件名已核对。未下载媒体或读取画面，PST50 保留集继续封存。[数值与源文件哈希](2026-11-19-arri-encounters-source-evidence.json)

后续可先验证 06 与 08 的小范围逐帧访问，再确认色彩空间与成片角色。这两条候选适合作为有明确原片身份的技术测试；独立自然度评估仍需要真实参考与明确的验收材料。用已有工作区 `pypdf` 运行 `audit_arri_encounters_source.py <本机指南与四份HTML目录> <新JSON>`。本机文档位于 `outputs/adaptive-reference-20261113/arri-encounters-source-v1`。
