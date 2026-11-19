# 保住 13 号的色度分量仍无法兼顾保色与换款

上轮证明，解除限幅无法补足固定候选的风格。本轮保持同样的目标和参考、冻结参数、近中性色范围、亮度及总长度上限。只改变长度的分配，把候选沿 13 号改色方向的分量限制为原幅度的 1–2 倍，再把剩余长度分给垂直方向。两图输入及其他渲染环节与前轮相同。[事前协议](../../experimental/adaptive-reference-20261113/tst-preserve-color-component-protocol.md) · [幅度排查](2026-11-20-tst-color-cap-ablation.md)

| 开发组 | 13 号／旧候选／方向约束的换款投影 | 13 号／旧候选／方向约束的保色 MAE（RGB8） |
| --- | ---: | ---: |
| 2 | 0.696／0.334／0.679 | 3.188／2.572／3.332 |
| 3 | 0.666／0.540／0.689 | 1.986／1.791／2.186 |
| 8 | 0.660／0.477／0.833 | 2.542／2.072／3.423 |
| 10 | 0.224／0.148／0.119 | 2.883／2.871／3.158 |
| 23 | 0.680／0.302／0.504 | 1.776／1.905／2.017 |

相对旧候选，有四组换款投影上升，但五组保色误差均增加，局部联合仍为 0/5。第 8 组换款幅度进入 0.75–1.25，但同款换参考差为 4.688，保色误差为 3.423，均未通过。第 23 组仍低于 13 号。整图换款 7/24，最差额外整图 MAE 为 0.245 RGB8。对每个参考保住原方向分量，不能保证两参考之间的差值同样被保住。结束使用原预测器方向并强制保留 13 号分量的方案。[第 8 组本机成片](../../outputs/adaptive-reference-20261113/tst-preserve-color-component-v1/8-2.png) · [第 23 组本机成片](../../outputs/adaptive-reference-20261113/tst-preserve-color-component-v1/23-2.png)

## 复现

在既有视觉环境将仓库根目录加入 `PYTHONPATH`，设置 `OPENBLAS_NUM_THREADS=2` 和 `OMP_NUM_THREADS=2`，运行 `python experimental/adaptive-reference-20261113/probe_tst_preserve_color_component.py <新的输出目录>`。72 张原候选图逐像素复现。两遍各 72 张诊断图及报告逐件一致，独立检查作用域外不变、同图保持、亮度舍入误差和五组局部评分。报告 SHA256 为 `24beb13ace47bb520f31031a65fcf19da7802741c6b3a061960d92dc8e823734`。产品和独立验收未改。[脚本](../../experimental/adaptive-reference-20261113/probe_tst_preserve_color_component.py)
