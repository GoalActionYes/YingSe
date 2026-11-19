# 补齐已知调色的颜色样本后，投影幕区域恢复

固定参考颜色并使用已知调色差值后，上一轮白衣和雪地恢复，投影幕中的脸仍然变化不足。本轮只增加 13,824 个均匀 RGB8 色样，和原参考色样合并去重，使用同一已知调色生成差值。两臂的八近邻估计、80% 强度和评分区域相同。新色样由固定网格生成，构建时无需读取目标原图、目标成片或区域。[事前协议](../../experimental/adaptive-reference-20261113/paired-neighbor-uniform-support-protocol.md) · [原位置校准结果](2026-11-20-paired-neighbor-labels.md)

| 留出区域 | 换款投影，原样本 → 补样 | 响应 MAE，原样本 → 补样（RGB8） |
| --- | ---: | ---: |
| 投影幕中的脸，F1 背景 | 0.623 → 0.999 | 0.860 → 0.316 |
| 投影幕中的脸，C1 背景 | 0.944 → 0.993 | 0.649 → 0.322 |
| 白球服，C1 背景 | 0.819 → 0.887 | 0.851 → 0.586 |
| 白球服，F1 背景 | 0.840 → 0.866 | 0.775 → 0.663 |
| 雪地，F1 背景 | 0.990 → 0.988 | 0.074 → 0.103 |

投影幕 F1 的最近／第八近邻距离中位数由 13.15／42.37 减至 5.10／13.08 RGB8。该区投影进入 0.75–1.25，MAE 低于 3 且下降 0.544，达到预设改善线。两种背景的白衣通过，原单参考保护检查和相对校准臂的保护检查均通过。雪地误差略增 0.029，仍在预定许可范围内。全部缺测保留。[本机投影幕图板](../../outputs/adaptive-reference-20261113/paired-neighbor-uniform-support-v1/8021-8690/comparison.png) · [本机白衣图板](../../outputs/adaptive-reference-20261113/paired-neighbor-uniform-support-v1/872-1268/comparison.png)

同一估计方法获得更近的已知色样后，足以恢复这组开发对照的投影幕区域。和上轮一起提供两个受控结果：校准参考差值可以修复白衣和雪地，增加已知色样可以修复仍然不足的投影幕。后续可用这些成片做已知全局编辑的离线正对照。普通用户输入只有调色后的参考图，能否从中估计这套变化仍待验证。真实自然度、独立图片和视频验收尚未进行。

## 复现

沿用原视觉环境，将仓库根目录、OpenColorIO 2.5.2 和 SciPy 1.16.3 加入 `PYTHONPATH`，设置 `OPENBLAS_NUM_THREADS=4`，运行 `python experimental/adaptive-reference-20261113/probe_paired_neighbor_uniform_support.py <新的输出目录>`。原校准臂的 16 张归档成片逐像素复现。两次新运行的 16 张成片、四张图板和报告逐件同哈希。另读成片复算 26 项评分，并在同一掩膜对比补样前后的欧氏近邻距离和退化限制。报告 SHA256 为 `4c9eec182a1b00ef5ddc77ad1b185d4e2bfea0d1ee4349c67a1be379b2c84c3f`。[复现脚本](../../experimental/adaptive-reference-20261113/probe_paired_neighbor_uniform_support.py)
