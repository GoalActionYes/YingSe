# 真实作者配对改用近邻重放，部分误差下降但通过数减少

此前在已知 Film A／B 的受控对照中，RGB 八近邻曾补回部分白球服风格。本轮检查它能否替换实际作者配对的标签生成方法。候选只替换映射，同一参考原片、同款参考成片、目标原片、棋盘训练和留块均与旧三次色表相同。两种方法都用 80% 强度，候选沿用先将近邻估计量化为 RGB8 的方式。全部 10 个原合格参考槽进入比较，另 62 个原不具备测量资格的槽照录。[事前协议](../../experimental/adaptive-reference-20261113/tst-author-neighbor-replay-protocol.md) · [原作者配对](2026-11-20-tst-paired-donor-edit.md)

| 开发槽 | 风格投影，三次色表 → RGB 近邻 | 近中性色 MAE，三次色表 → RGB 近邻 |
| --- | ---: | ---: |
| 3-2，换款 | 0.778 → 0.913 | 5.119 → 2.300 |
| 20-0，同款 | 0.920 → 0.706 | 2.627 → 3.301 |
| 20-1，同款 | 0.885 → 0.700 | 3.016 → 3.038 |
| 21-1，同款 | 0.931 → 0.625 | 6.186 → 3.486 |
| 23-1，同款 | 0.628 → 0.591 | 4.324 → 8.038 |

原三次色表通过 9/10，近邻通过 6/10。三处原通过组的风格投影低于预定 0.75。第 21 组近中性色误差虽然明显下降，风格变化幅度反而减弱。第 23 组原反例仍未恢复。近邻相对原三次色表的整图留块误差最差增加 1.340 RGB8，超过 0.25 的替换限制。也保留第 3 组换款和第 16 组局部误差下降的成片。结束统一替换标签生成方法，监督仍须按实际颜色变化验证，不把相近的 RGB 或同编号当成足够的编辑证据。[本机第 3 组改善](../../outputs/adaptive-reference-20261113/tst-author-neighbor-replay-v1/3-2-neighbors.png) · [本机第 23 组反例](../../outputs/adaptive-reference-20261113/tst-author-neighbor-replay-v1/23-1-neighbors.png)

这十槽来自六张已经用于开发的目标图。算法读取了参考原图，目标作者成片只用于评分。证据用于衡量潜在训练标签，并不代表普通两图的自动画质。

## 复现

在本机视觉研究环境将仓库根目录与已有 SciPy 1.16.3 加入 `PYTHONPATH`，设置 `OPENBLAS_NUM_THREADS=2`、`OMP_NUM_THREADS=2`，运行 `python experimental/adaptive-reference-20261113/probe_tst_author_neighbor_replay.py <新的输出目录>`。10 张旧色表成片逐像素复现。两遍各 10 张新成片与报告逐件同哈希，并从落盘成片复算 10 个槽的评分。报告 SHA256 为 `132e11e4343e627ee0bfacc66ffa229a385a1c2902a7d7a193c3aeb3b9fb1b6a`。[脚本](../../experimental/adaptive-reference-20261113/probe_tst_author_neighbor_replay.py)
