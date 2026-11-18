# 成对监督未改善联合画质，局部统计的冷调收益仍不稳定

训练和开发按照片组隔离。两臂使用同样的两图描述和 80% 渲染，唯一差别是候选新增两项拟合目标：同一参考换背景时维持同样的编辑参数，同一背景开关编辑时预测正确的增量。另在原结果保存后运行同尺寸、只用全图分位数的控制，核查白球服的冷调收益来自哪部分。两次对照合计保存 128 张实际成片。[事前协议](../../experimental/adaptive-reference-20261113/crossed-edit-pair-supervision-protocol.md) · [白球服两臂对照](../../outputs/adaptive-reference-20261113/crossed-edit-pair-supervision-v1/872-1268/comparison.png)

成对监督把训练中的参数换背景误差从 0.082 降到 0.056，开关增量误差却由 0.087 升到 0.097。开发四组中，两个拟合臂联合通过均为 **0/4**。成对臂的最差整图退化为 0.33 个 RGB8 码值，未超过 2，但白球服冷调开时的人物换背景差从 0.00 增至 0.33，原背景冷调投影基本不变；白球服两背景的对比度投影仍只有 0.385 和 0.168。结束固定成对目标。

同尺寸的特征归因比较显示，加入按亮度排名汇总的局部统计后，八项冷调响应中七项比全图分位数更接近预期，幅度至少为 0.10。白球服原背景为 **0.085 → 0.940**。但 `8021` 换背景的投影由 **0.826 → 0.369**，退化超过上限。该例人物换背景色差同时从 5.11 降到 0.27 码值，说明稳定性改善可以与风格丢失同时出现。特征分项和整体画质仍未达标，保留有意冷调与失败反例，不把七项局部收益算作升级。[同尺寸分位数白球服](../../outputs/adaptive-reference-20261113/crossed-edit-quantiles-only-v1/872-1268/comparison.png) · [换背景退化](../../outputs/adaptive-reference-20261113/crossed-edit-pair-supervision-v1/8021-8690/comparison.png)

本轮只有固定闭集的程序编辑，色彩参数由普通两图估计，评分才使用程序目标和人工范围。结果不涵盖开放参考、独立画质或视频，不更换既有模型。两个实验各自复跑后，成片、对照板和报告逐文件一致，另从落盘图独立复算 128 项整图误差和有效编辑投影。复现设仓库根目录为 `PYTHONPATH`，用研究环境运行 `probe_crossed_edit_pair_supervision.py <新输出目录>`，同尺寸控制加 `--quantiles-only`。[脚本](../../experimental/adaptive-reference-20261113/probe_crossed_edit_pair_supervision.py) · [数值与哈希](2026-11-19-crossed-edit-pair-supervision-evidence.json)
