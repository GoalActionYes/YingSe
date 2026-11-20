# 外观支路只读取明暗，整图改善伴随近中性色过调

沿用两张实拍人像和同一项色彩平衡编辑。候选把参考图的灰阶明暗副本送入外观支路，参考的颜色统计仍读取原始彩色图，目标、权重与 80% 强度也固定。每次预测只需目标图与参考图。原完整模型的四张对照逐像素复现，两张同图输入逐像素保持。[事前协议](../../experimental/adaptive-reference-20261113/portrait-reference-luma-only-protocol.md) · [参考路径诊断](2026-11-20-portrait-reference-paths.md)

| 目标及区域 | 编辑响应投影，完整 → 灰阶外观 | 关态误差，完整 → 灰阶外观（RGB8） |
| --- | ---: | ---: |
| 《Cuckoo》脸颊 | 0.632 → 0.572 | 24.54 → 26.24 |
| 《Cuckoo》浅树皮 | 0.321 → 0.295 | 30.76 → 32.87 |
| 混合光人像整图 | 0.726 → 0.975 | 12.90 → 11.76 |
| 混合光人像共同近中性色 | 1.400 → 1.603 | 25.80 → 24.47 |

编辑投影 1 为预期幅度，原门槛为 0.75–1.25。混合光人像的整图响应恢复，误差也从 1.85 降至 1.09，近中性色却进一步过调。关态误差仍超过 3。《Cuckoo》脸颊、浅树皮和共同近中性色的幅度均更弱。两方向均未达到联合门槛。这个固定的灰阶输入提供了实际可出图的对照，结果支持停止该转换。去色后的输入超出已有训练分布，结论限于这份冻结权重与两个已用开发方向。[全量读数与哈希](2026-11-20-portrait-reference-luma-only-evidence.json)

用现有视觉研究环境，将仓库根目录加入 `PYTHONPATH`，执行 `experimental/adaptive-reference-20261113/probe_portrait_reference_luma_only.py <研究输出根目录> <13 号人像输出目录> <扩展版人像输出目录> <新输出目录>`。两遍四张候选成片与数值报告逐字节一致。脚本与数值进入 Git，实图和模型保留本机。[本机《Cuckoo》成片](../../outputs/adaptive-reference-20261113/portrait-reference-luma-only-v1/cuckoo-on.png) · [本机混合光人像成片](../../outputs/adaptive-reference-20261113/portrait-reference-luma-only-v1/mixed-on.png)
