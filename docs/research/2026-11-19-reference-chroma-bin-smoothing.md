# 平滑参考色度的相邻档，未减轻白球服串色

固定 13 号、80% 强度、目标统计和渲染。只把参考 Cb、Cr 的每档计数按 1/4、1/2、1/4 分配给邻档与原档，保持通道总量和 Y 直方图。用四组实拍背景换色的 HSL／曲线开关，以及三种实拍内容间六个 Film B 方向，分别检验背景串色和创意调色响应。[事前协议](../../experimental/adaptive-reference-20261113/reference-chroma-bin-smoothing-protocol.md)

| 检查 | 原 13 号 → 平滑后 |
| --- | --- |
| 白球服，编辑开启时的近中性色换背景差 | 7.680 → 7.682 RGB8 |
| 白球服，两背景间各自的明暗响应投影 | 0.220／0.251 → 0.220／0.252 |
| 785–3553，编辑开启时的近中性色换背景差 | 8.24 → 9.04 RGB8 |
| 焊工，分别用扑克、色卡作参考的 Film B 近中性色投影 | −0.087／−0.447 → −0.106／−0.454 |

四组换背景联合为 0/4，六个 Film B 方向没有一组同时通过整图和近中性色响应门槛。14 个未编辑参考的候选成片与原 13 号相差最多 0.65 RGB8，通过本轮 1.5 的变动限制。该限制只衡量换统计方法引起的变化，未编辑参考下原 13 号已有的偏色仍在。保留小幅改善和退化，结束这条固定相邻档平滑。此处只检查一个固定核和已有权重，尚不能据此判断其它统计表示的效果。[白球服对照](../../outputs/adaptive-reference-20261113/reference-chroma-bin-smoothing-v1/872-1268/comparison.png) · [焊工用色卡参考的成片](../../outputs/adaptive-reference-20261113/reference-chroma-bin-smoothing-v1/sparks_4592-from-hdr_testimage-film_b-smooth.png)

28 张原 13 号控制均逐像素复现。两次各 28 张候选成片、四张图板和一份报告逐件匹配。另从落盘图独立复算候选身份、背景换色误差、HSL／曲线与 Film B 的响应。使用已有研究 Python，将仓库根目录和本机 OpenCV 加入 `PYTHONPATH`，执行 `probe_reference_chroma_bin_smoothing.py <新输出目录>`。未训练，未接入产品，开发素材继续按原分组使用。[复现脚本](../../experimental/adaptive-reference-20261113/probe_reference_chroma_bin_smoothing.py) · [数值与哈希](2026-11-19-reference-chroma-bin-smoothing-evidence.json)
