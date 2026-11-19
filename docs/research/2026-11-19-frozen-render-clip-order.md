# 推迟渲染取整未恢复白球服明暗响应

在冻结扩展第 4 轮上保持输入、模型、分析网格和 80% 强度。把全强度成片先裁切、取整再混合的路径，与在浮点值上混合、最后裁切和取整的路径比较。28 份先裁切对照与归档逐像素一致。[事前协议](../../experimental/adaptive-reference-20261113/frozen-render-clip-order-protocol.md) · [归档成片的开发复查](2026-11-19-frozen-candidate-development-crosscheck.md)

| 检查 | 先取整 | 最后取整 |
| --- | ---: | ---: |
| 白球服 C1 明暗响应 | 0.2760 | 0.2773 |
| 白球服 F1 明暗响应 | 0.2291 | 0.2223 |
| 白球服人物及中性色最大换背景差 | 4.575 RGB8 | 4.555 RGB8 |
| Film B 六方向最高整图响应 | 0.6284 | 0.6299 |
| 背景检查联合通过数 | 0/4 | 0/4 |
| Film B 检查联合通过数 | 0/6 | 0/6 |

28 张成片中，模型返回的浮点 RGB 和 80% 混合结果均无超出 [0, 1] 的通道。两条路径的逐张整图平均绝对差为 0.170–0.228 RGB8，单通道差最多 1 RGB8。白球服两个背景与六个 Film B 方向仍低于 0.75 的明暗或整图响应门槛。结束通过最后一步裁切或取整改变风格幅度的尝试。模型内部在反变换前还有一次限幅，本轮结论只适用于模型返回之后的渲染顺序。[白球服本机对照图](../../outputs/adaptive-reference-20261113/frozen-render-clip-order-v1/background/872-1268/comparison.png)

研究环境执行 `python probe_frozen_render_clip_order.py <新输出目录>`，需要源仓库根目录进入 Python 模块路径。两遍生成的 28 张原尺寸图、10 张对照图和报告逐件一致。另从磁盘复核 28 张输出及其归档像素差，独立复算 14 项整图／明暗投影。当前仅有开发验证，未接入软件。[复现脚本](../../experimental/adaptive-reference-20261113/probe_frozen_render_clip_order.py) · [数值与文件哈希](2026-11-19-frozen-render-clip-order-evidence.json)
