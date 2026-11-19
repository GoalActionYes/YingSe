# 统一参考曝光减轻部分肤色波动，仍未保住暖调

在五组实拍人像的标称、轻欠曝、轻过曝参考上，分别开关同一个程序脸部暖调。用单张参考的全图中位亮度估计曝光增益，在线性 RGB 中将它拉向 0.18，再用冻结 13 号出图。模型只读目标和一张参考；标称帧和作者目标只用于比较。沿用 80% 强度。[事前协议](../../experimental/adaptive-reference-20261113/dast-reference-exposure-normalization-protocol.md) · [原开发基线](2026-11-19-dast-local-edit-cross.md)

| 检查 | 原 13 号 | 统一参考曝光 |
| --- | ---: | ---: |
| 换曝光的目标肤色对数色度差 | 0.132–0.349 | 0.045–0.259 |
| 肤色差小于 0.03 | 0/20 | 0/20 |
| 有意暖调方向投影 | 0.232–0.514 | 0.086–0.513 |
| 三档风格、保护与稳定均通过 | 0/5 | 0/5 |

参考归一没有新增通道裁切，但仍不能把曝光造成的肤色变化降至预设范围，有意暖调也继续偏弱。停止固定全图曝光归一；这批是真实曝光和程序暖调，不包含实际换灯或独立画质评估。未接入软件。

两遍各 30 张成片和报告的哈希全部一致，核对 30 张冻结基线并复算原报告分数。将仓库根目录加入已有研究环境的 `PYTHONPATH`，设 `OPENBLAS_NUM_THREADS=1`，运行 `python experimental/adaptive-reference-20261113/probe_dast_reference_exposure_normalization.py <研究输出根目录> <新输出目录> <dast-local-edit-cross-v1/results.json 的 SHA-256>`。[数值](2026-11-19-dast-reference-exposure-normalization-evidence.json)
