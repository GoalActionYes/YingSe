# 跳过颜色网格的空间平滑仍未恢复风格

冻结扩展第 4 轮的权重、参考与目标、颜色引导及原尺寸渲染。每对输入预测一次调色网格，控制按原方法做 XY 平滑，试验直接渲染同一份未平滑网格。两个流程都使用 80% 强度。控制与全部 72 张归档成片逐像素一致，同图输入保持原图。[事前协议](../../experimental/adaptive-reference-20261113/frozen-grid-smoothing-protocol.md) · [四套归档对照](2026-11-19-frozen-candidate-development-crosscheck.md)

| 检查 | 原平滑 | 跳过平滑 |
| --- | ---: | ---: |
| 白球服 C1／F1 明暗响应 | 0.276／0.229 | 0.276／0.226 |
| 白球服保护区最大换背景差 | 4.57 RGB8 | 4.58 RGB8 |
| 四组背景保护区最大差，案例 8021–8690 | 26.88 RGB8 | 29.46 RGB8 |
| Film B 最高整图响应，扑克用色卡作参考 | 0.628 | 0.642 |
| FACD 最坏近中性色额外误差，相对 13 号 | 14.61 RGB8 | 14.45 RGB8 |
| 实体色卡关闭风格的最大色度误差 | 6.74 RGB8 | 7.48 RGB8 |

背景 0/4、Film B 0/6、FACD 0/7，实体色卡 2/7，四套原联合门槛均未通过。焊工两种参考的整图与近中性色响应均下降；火焰 Hudson 的近中性色绝对误差虽由 23.67 降至 23.51 RGB8，仍明显高于 13 号的 9.06。结束用移除固定空间平滑恢复风格的尝试。这里检验的是现有权重下的单步改动，不能推断重新训练对平滑的依赖，也不能推断剩余误差由哪一种参考线索造成。[白球服本机成片](../../outputs/adaptive-reference-20261113/frozen-grid-smoothing-v1/background/872-1268/comparison.png) · [换背景退化](../../outputs/adaptive-reference-20261113/frozen-grid-smoothing-v1/background/8021-8690/comparison.png) · [Film B 本机成片](../../outputs/adaptive-reference-20261113/frozen-grid-smoothing-v1/film_b/poker_fullshot-from-hdr_testimage-comparison.png)

使用已有研究环境，将仓库根目录加入 Python 模块路径，再执行 `python probe_frozen_grid_smoothing.py <新输出目录>`。两次运行的 72 张原尺寸成片、18 张对照图和报告逐件一致。从落盘图复核全部 72 对像素差和数值。仍为已消费素材的开发诊断，未接入软件。[复现脚本](../../experimental/adaptive-reference-20261113/probe_frozen_grid_smoothing.py) · [数值和哈希](2026-11-19-frozen-grid-smoothing-evidence.json)
