# 按目标亮度调整局部调色力度，脸颊通过而胸口仍偏色

统一减少色差曾使《Cuckoo》两处肤色通过，却让混合光原片两个留出折退化。本轮检验原始肤色亮度能否预测所需力度。在混合光原片的两个训练折分别拟合亮度到力度的关系，再冻结到《Cuckoo》。只调整固定颜色方向的力度，目标编辑器开态在保存预测成片后才用于评分。两张照片都已用于开发，肤色位置由已有区域给定。[事前协议](../../experimental/adaptive-reference-20261113/playraw-luma-gain-protocol.md) · [统一倍数的反例](2026-11-20-playraw-cross-editor-gain.md)

| 区域 | 固定色差 MAE（RGB8） | 亮度预测 MAE（RGB8） | 亮度预测投影 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 《Cuckoo》脸颊 | 3.094 | 2.667 | 1.090 | 通过 |
| 《Cuckoo》胸口 | 4.502 | 4.074 | 1.150 | 超过 3.00 |
| 混合光原片高权重区，留出折 0 | 1.582 | 1.632 | 0.989 | 通过 |
| 混合光原片高权重区，留出折 1 | 1.623 | 1.653 | 0.995 | 通过 |

《Cuckoo》两份落盘图的区域结果一致。脸颊和胸口所有像素均在两个拟合的共同亮度范围内，倍数中位数分别为 0.975 和 0.969。亮度所解释的减弱使脸颊过线，胸口仍不达标，且目标渲染没有触发色域压缩。源照片过渡区误差分别由 0.554／0.548 降至 0.540／0.539，区外逐像素保持。

因此停止用这一亮度规则恢复力度。源照片的亮度范围已经覆盖目标两处，继续扩大亮度支持范围没有直接依据。若推进力度自适应，需要额外能区分显影响应的观测证据，并在更多未用于选择规则的照片上验证。父实验全图复现未过线的记录仍适用。本轮没有升级映色软件，也未启动独立画质或视频验收。

## 复现

在已有视觉环境设置 `PYTHONPATH` 为仓库根目录、`OPENBLAS_NUM_THREADS=2`，运行 `python experimental/adaptive-reference-20261113/probe_playraw_luma_gain.py <playraw-mixed-light-cross-editor-v2 目录> <新输出目录>`。输入的同级目录需有 `playraw-mixed-light-skin-v1` 和 `playraw-mixed-light-operator-v1`。两次运行各四张 PNG 与报告逐件一致。用整数亮度权重从落盘图独立复算表内指标及区外像素。报告 SHA256 为 `f91b3a5d0463ad87dc92be084871ecd467cc061d6e6f7e3736619a7ca8f00300`。[本机目标成片](../../outputs/adaptive-reference-20261113/playraw-luma-gain-v1/target-0.png) · [复现脚本](../../experimental/adaptive-reference-20261113/probe_playraw_luma_gain.py)
