# 白球服两项调色贡献都偏弱，部分其他场景确有抵消

固定扩展第 4 轮及 80% 强度。在同一目标上，用参考编辑关／开分别生成网格，沿用原采样并除以各自 alpha 后，交叉组合矩阵和偏移。两项都关、都开的 28 份重建成片与现有归档逐像素一致。另存 28 张只切矩阵或只切偏移的成片。两张参考的前后关系供归因，目标已知编辑只供评分。[事前协议](../../experimental/adaptive-reference-20261113/frozen-affine-terms-protocol.md) · [四套冻结结果](2026-11-19-frozen-candidate-development-crosscheck.md)

| 场景和测量范围 | 只切矩阵 | 只切偏移 | 全开 |
| --- | ---: | ---: | ---: |
| 白球服 C1 整图明暗 | 0.167 | 0.094 | 0.276 |
| 白球服 F1 整图明暗 | 0.207 | 0.014 | 0.229 |
| 8021–8690 F1 整图明暗 | −1.273 | 1.655 | 0.362 |
| 焊工用扑克作参考，Film B 整图 | 0.588 | −0.240 | 0.370 |
| 焊工用扑克作参考，Film B 近中性色 | 0.735 | −0.407 | 0.358 |

白球服两背景中，平均两种组合顺序后，矩阵和偏移的贡献仍均为正。没有一项达到 0.75，因此这一例缺的是足够的同向响应。8021–8690 的 F1 确有强烈抵消，但只切偏移又超过 1.25。焊工的矩阵保留更多 Film B 变化，近中性色仍低于 0.75 且增量误差为 3.60 RGB8，高于 3。Film B 六个方向中，任一单项均未通过整图与近中性色联合门槛。结束固定关闭一项的尝试。结论限于这份冻结权重及两套已使用的开发素材。[白球服只切矩阵](../../outputs/adaptive-reference-20261113/frozen-affine-terms-v1/bg-872-1268-F1-matrix.png) · [焊工只切矩阵](../../outputs/adaptive-reference-20261113/frozen-affine-terms-v1/sparks_4592-from-poker_fullshot-matrix.png)

在既有研究环境中将仓库根目录加入 Python 模块路径，执行 `python probe_frozen_affine_terms.py <新输出目录>`。两遍的 28 张成片和报告逐件相同。从落盘图独立复算 20 个测量范围内的投影、增量误差及贡献之和。本轮未接入软件。[复现脚本](../../experimental/adaptive-reference-20261113/probe_frozen_affine_terms.py) · [数值及哈希](2026-11-19-frozen-affine-terms-evidence.json)
