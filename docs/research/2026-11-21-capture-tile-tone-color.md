# 平滑区的明暗色度关系未跨来源区分实拍和数字染色

新墙面对照表明参考明暗和色度都会干扰目标。试验从单张参考读取相对平滑区域的明暗和色比关系，再用其他来源拟合最近中心。留出按完整拍摄来源划分，同一布景和不同相机均不跨折。参考原图只提供监督标签，并未参与留出照片特征。与旧的局部边缘差相比，本轮使用亮度排序后平滑区域之间的有符号色度变化。[事前协议](../../experimental/adaptive-reference-20261113/capture-tile-tone-color-protocol.md) · [旧边缘对照](2026-11-20-capture-chroma-edges.md)

| 留出来源 | 实拍识别 | 原参考误触 | 数字编辑误触 | 缺测 |
| --- | ---: | ---: | ---: | ---: |
| CHROMA-MILL | 4／6 | 6／6 | 4／6 | 4 |
| Beyond RGB | 1／2 | 1／2 | 1／2 | 0 |
| Gros 真人 | 0／3 | 0／3 | 0／3 | 0 |
| MIT 复印室 | 0／1 | 0／1 | 0／1 | 0 |

实拍分母包含缺测。CHROMA-MILL 缺测包括两张实拍和两张数字染色，平滑区域有效像素不足。总体 5／12 个实拍识别，五张数字编辑误触。13 号墙面与肤色对照的三个参考均未触发保护，实际输出六张均逐像素匹配原成片，联合 0／2。墙面串色仍为 15.14／15.73 RGB8，双颊为 12.38／9.58。若直接把这套开关用于其他来源，误触的数字编辑会回退原图；这五项为判别反例，不是实际目标渲染成绩。[客厅实际输出](../../outputs/adaptive-reference-20261113/capture-tile-tone-color-v1/living-comparison.png) · [人物实际输出](../../outputs/adaptive-reference-20261113/capture-tile-tone-color-v1/portrait-comparison.png)

结束用固定明暗区色比触发保护。实拍光照的空间结构无法由这四项分档值可靠辨认。可保留这些配对提供拍摄变化与编辑变化的监督，但下一步应有额外的物理光场、材质证据或经过跨来源验证的表示，避免继续在同批反例上换分档或阈值。本轮未修改产品，独立验收状态不变。两次各六张输出、两张对照和报告一致，SHA-256 为 `e5ea6d809e495ab3cd9a7b0ea30e11b00a17292491961bb921f6bd8b6e0ee550`。[逐案判定和成片哈希](2026-11-21-capture-tile-tone-color-evidence.json)

沿用视觉 Python、原 OpenCV 路径及本机 DeepWB-audit/vendor 辅助依赖，运行 [脚本](../../experimental/adaptive-reference-20261113/probe_capture_tile_tone_color.py) 并传新输出目录。不下载或安装依赖。
