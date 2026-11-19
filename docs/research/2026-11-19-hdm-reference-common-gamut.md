# 共用色域仍不足以拆开 Showgirl 参考的明暗和颜色

把《Showgirl 1》同帧原片和作者成片的亮度限制在两种色度都能表达的范围，组合成四张 SDR 参考。浮点计算均在色域内，没有用色度压缩换取亮度。参考图之间仍有可测的明暗和色度差，但只在 12.28% 的位置具备预定的暗档亮度，低于 25% 门槛。RGB8 量化后，相同亮度档的最大逐像素线性 Y 差仍有 0.00604，高于 0.005 门槛。[四张参考对照](../../outputs/adaptive-reference-20261113/hdm-reference-common-gamut-v1/comparison.png) · [事前协议](../../experimental/adaptive-reference-20261113/hdm-reference-common-gamut-protocol.md)

| 输入检查 | 实测 | 预设 |
| --- | ---: | ---: |
| 共同暗档 Y ≥0.01 的参考像素比例 | 12.28% | ≥25% |
| 相同亮度档，量化后最大线性 Y 差 | 0.00604 | ≤0.005 |
| 合格像素内两档亮度的平均 Y 差 | 0.53307 | ≥0.02 |
| 合格像素内两种颜色的中位 u′v′ 距离 | 0.04583 | ≥0.005 |
| 合格像素内同一颜色换亮度的最大中位 u′v′ 漂移 | 0.00120 | ≤0.002 |

按预设规则停在参考制备，未加载 13 号权重、未运行扑克和 Bistro。上轮直接提亮参考需要压缩色度，这轮避开压缩后仍缺足够亮度支持。结束用这两张 Showgirl SDR 图做固定亮度与色度归因。先前原片参考已经造成的 Bistro 偏移仍有效。后续若继续拆分这两个因素，优先检查原生 SDR、同帧且已知编辑的素材，先检验信号覆盖再运行模型。[上一轮的实拍输出](2026-11-19-hdm-reference-edit-toggle.md)

运行 `probe_hdm_reference_common_gamut.py <本机研究输出根目录> <新输出目录>`。两遍四张参考、对照图与 JSON 逐字节一致；复算暗档共同支持为 18,110／147,456 像素。在独立的合成正例上验证了通过分支及亮度主效应计分。没有新增软件改动或画质分数，原始参考和生成图留本机。[脚本](../../experimental/adaptive-reference-20261113/probe_hdm_reference_common_gamut.py) · [数值和哈希](2026-11-19-hdm-reference-common-gamut-evidence.json)
