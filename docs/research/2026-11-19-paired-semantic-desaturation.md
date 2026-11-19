# 已知参考编辑区域后，局部特征仍未保住目标应保留的颜色

本轮用 PSBench 人像 145 和鸟照 248 的作者前后图标出参考中实际去色、保色的位置。用本机冻结的 DINOv2-small 在原片灰度图上匹配局部特征，再对目标对应区域执行固定 80% 的去色，保持目标原始亮度。目标作者成片只用于最后计分。两个方向互换目标和参考，另比较整图去色与原 13 号成片。[事前协议](../../experimental/adaptive-reference-20261113/paired-semantic-desaturation-protocol.md) · [原始成片和案例准入](2026-11-19-psbench-background-desaturation.md)

| 目标 ← 参考 | 参考去色／保色网格数 | 目标两区平均去色权重 | 去色区误差，13 号 → 局部对应 | 保色区变化 |
| --- | ---: | ---: | ---: | ---: |
| 人像 ← 鸟照 | 218／0 | 0／0 | 6.72 → 13.94 | 0.00 |
| 鸟照 ← 人像 | 421／9 | 1／1 | 7.44 → 1.74 | 6.39 |

误差和变化均以亮度分离后的色度码值计。鸟照参考中，保色测量像素没有占据任何一个网格的过半面积，低于事前要求的五个网格。人像因此保持原片，漏掉应有的去色。鸟照方向的去色区响应投影为 0.803，误差为 1.74，但去色和保色两区都收到全部去色权重。整图结果逐像素等于全图去色控制，保色区变化 6.39，超过 1.5 上限。两张无编辑对照均逐像素保持目标原图。两方向联合为 0/2。[人像对照](../../outputs/adaptive-reference-20261113/paired-semantic-desaturation-v1/145-comparison.png) · [鸟照对照](../../outputs/adaptive-reference-20261113/paired-semantic-desaturation-v1/248-comparison.png)

这条固定对应路线分别受限于参考端的小范围保色样本和跨图匹配的选择性。两张照片的作者意图相似，实际范围和操作参数未确认相同。停止该固定网格分类，保留两种失败。后续若考虑借助真实编辑监督，需先有同款操作、多种构图，以及能保留小区域编辑证据的表示，再考虑普通两图输入。13 号与课程版均未改动。

两次运行的九份文件逐件匹配。从落盘 PNG 独立复算了六张成片的亮度、全图色度误差、测量区响应与误差，并核对无编辑原样输出。复现使用已有 DINOv2-small 权重和本机视觉环境，将仓库根目录与实验脚本目录加入 `PYTHONPATH`，运行 `probe_paired_semantic_desaturation.py <本机研究输出根目录> <新输出目录>`。[脚本](../../experimental/adaptive-reference-20261113/probe_paired_semantic_desaturation.py) · [数值和哈希](2026-11-19-paired-semantic-desaturation-evidence.json)
