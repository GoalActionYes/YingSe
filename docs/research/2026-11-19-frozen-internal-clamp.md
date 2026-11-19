# 内部曲线外推未恢复白球服和 Film B 的风格幅度

沿用扩展第 4 轮冻结权重、同一实拍输入和 80% 强度。只在模型内部仿射值越界时延展原单调曲线末段。保留原 RGB8 渲染和强度混合顺序。按原公式重建的浮点控制与模型逐带相同，28 张控制成片与原归档逐像素相同。[事前协议](../../experimental/adaptive-reference-20261113/frozen-internal-clamp-protocol.md) · [上轮外层取整对照](2026-11-19-frozen-render-clip-order.md)

| 检查 | 实际外推 | 仅延展越界通道时的响应上限 |
| --- | ---: | ---: |
| 白球服 C1 明暗响应 | 0.27604 | 0.27604 |
| 白球服 F1 明暗响应 | 0.22911 | 0.22911 |
| Film B 六方向最高整图响应 | 0.62841 | 0.62848 |

白球服四张图仅有上端越界，涉及 0.684%–1.369% 的通道。这些通道已达到原渲染允许的最亮值，延展后四张图逐像素相同。全部 28 张中 21 张不变，其余任一通道最多变 1 RGB8，单张最大的整图平均绝对变化为 0.02741 RGB8。背景和 Film B 的原联合门槛分别为 0/4 和 0/6。反例中，扑克用色卡参考时，两态各有约 12.4% 通道低于内部下限，仍没有足够的可显示变化补回风格。[白球服本机对照](../../outputs/adaptive-reference-20261113/frozen-internal-clamp-v1/background/872-1268/comparison.png) · [扑克本机对照](../../outputs/adaptive-reference-20261113/frozen-internal-clamp-v1/film_b/poker_fullshot-from-hdr_testimage-comparison.png)

上限允许每个越界通道在两个开关状态中独立选择，只保留单调方向和最终 RGB8 可达限制，并借助已知编辑选择最有利的方向。八项背景明暗和六项 Film B 整图上限均低于 0.75。结束在冻结输出的越界处调延展曲线。结论限于保持曲线区间内响应及现有渲染顺序的方案。要继续改善，需要让进入曲线前的映射或区间内响应本身传递更多有效风格。未验证训练阶段的限幅影响。

使用既有研究环境，将仓库根目录加入 Python 模块路径，再运行 `python probe_frozen_internal_clamp.py <新输出目录>`。两遍的 28 张成片、10 张对照图和报告逐件一致。已有内部仿射测试三项通过。另通过穷举小型区间核查投影上界公式，从落盘图复核全部成片变化。无软件接入或独立验收。[复现脚本](../../experimental/adaptive-reference-20261113/probe_frozen_internal_clamp.py) · [数值和文件哈希](2026-11-19-frozen-internal-clamp-evidence.json)
