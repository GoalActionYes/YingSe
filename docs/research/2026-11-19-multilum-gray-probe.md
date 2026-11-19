# 同次实拍的灰球读数改善部分近中性色，整图仍可能退化

从已核验的 MIT Multi-Illumination 两份场景归档中，读取 50 张室内照片、50 张同灯位灰球照片和两张材质图。固定每个场景的 14 号原片，按灰球中心区域的对数色度变化重放其它 24 个真实灯位，保持目标明暗，再与对应实拍和无修正对照比较。材质图供离线评分，数字编号沿用作者文件。此次没有下载新媒体。[事前协议](../../experimental/adaptive-reference-20261113/multilum-gray-probe-protocol.md) · [数据发布](https://projects.csail.mit.edu/illumination/) · [实拍及光探针方法](https://arxiv.org/html/1910.08131)

| 场景 | 可测换灯方向 | 整图色度误差至少降低 20% | 至少一项材质误差增加超过 0.05 |
| --- | ---: | ---: | ---: |
| main_experiment120 | 15 | 4 | 1 |
| kingston_dining10 | 24 | 1 | 8 |

两个场景都未通过。Kingston 的 24 号灯位在近中性色上由 0.275 降至 0.158，同一成片的整图误差从 0.251 升至 0.300，材质标记 20 从 0.185 升至 0.315。该场景最大材质退化出现在 20 号灯位，误差增量为 0.160。指标是线性对数色度的 RMS，不能换算成 RGB8 码值或审美分。两个 14 号同灯测试均逐像素保持，全量成片进入限域的最大比例约为 0.058%。[Kingston 24 号重放成片](../../outputs/adaptive-reference-20261113/multilum-gray-probe-v1/kingston_dining10/dir-24.png) · [原 14／24 实拍](2026-11-18-multilum-pilot.md)

这两室内场景支持一个具体边界。即使有同次拍摄的灰球，单个位置的颜色也无法可靠代表全部区域。全图统一校正能保护一部分近中性色，同时会改变其它区域应保留的颜色。停止该固定灰球全图重放。比较基于 JPEG 的固定 sRGB 解码、有限场景和作者材质类别，照明方向、表面反射和处理管线都可影响结果。不能从本对照断定哪种影响占主导，也未验证普通两图输入、人物肤色或新产品。

两遍各 50 张成片及一份报告逐件一致。独立以对数通道差重算 50 份灰球读数、50 个成片哈希、489 项区域误差和两个同灯负例。已有 NumPy、Pillow 环境可运行 `probe_multilum_gray_probe.py <两份归档所在目录> docs/research/2026-11-18-multilum-pilot-evidence.json <新输出目录>`。完整数值与成片只留本机，Git 记录逐灯位摘要和哈希。[复现脚本](../../experimental/adaptive-reference-20261113/probe_multilum_gray_probe.py) · [数值与哈希](2026-11-19-multilum-gray-probe-evidence.json)
