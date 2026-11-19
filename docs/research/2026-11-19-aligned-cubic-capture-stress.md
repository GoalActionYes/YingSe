# 两次实拍换灯强度触发了同画面三次拟合

固定三次拟合和现有配准门槛。补测 47 个尺寸完全相同的不同拍摄方向，包括人物重复拍摄及曝光变化、静物换灯强或灯色、跨场景和跨时点。数据均为已用于其它开发检查的本机实拍。[事前协议](../../experimental/adaptive-reference-20261113/aligned-cubic-capture-stress-protocol.md) · [三次拟合首轮](2026-11-19-aligned-cubic-pilot.md)

| 误触 | 原判定拒绝依据 | 新成片比 13 号的整图差 | 近中性色相对原图变化，13 号 → 新成片 |
| --- | --- | ---: | ---: |
| CHROMA-MILL 场景 1，白灯高档目标、低档参考 | 局部误差超限 | 15.67 RGB8 | 4.19 → 10.49 RGB8 |
| CHROMA-MILL 场景 2，白灯低档目标、高档参考 | 留出 P95 超限 | 3.40 RGB8 | 15.15 → 11.84 RGB8 |

两组都保留同一画面构图和白灯颜色，仅改变灯强后重新拍摄。结构相关分别为 0.997 和 0.994。三次查表使留出误差分别降到 1.239 和 1.371 RGB8，达到了启用门槛，但它不能凭此确认参考是同一原片后期调色。近中性色数值测量的是相对目标原图的变化，供解释误触的实际影响。第二例变化更小，仍然是误触。另有三次沿用了原仿射入口的接受，42 次继续回退。[场景 1 本机成片](../../outputs/adaptive-reference-20261113/aligned-cubic-capture-stress-v1/CHROMA-Scene1_a-white-241-to-anchor-comparison.png) · [场景 2 本机成片](../../outputs/adaptive-reference-20261113/aligned-cubic-capture-stress-v1/CHROMA-Scene2_a-white-241-from-anchor-comparison.png)

14 个同一原始帧的正例中有 8 个 Film A 对触发三次拟合，整图响应全部通过，其中 6 个有足够共同近中性色且该区也通过，另两个保留缺测。两个作者调色同帧对照均回退。三次拟合在同原片的已知编辑上仍有可复现收益，但本轮触发了事前停止条件。保留为离线工具，结束依据几何相似和颜色拟合误差直接自动接入的尝试。下一条路线需有额外可核验的同原片证据或明确操作意图。跨场景普通两图质量仍待解决。

复现使用已有研究环境，将仓库根目录加入 Python 模块路径，设 `OPENBLAS_NUM_THREADS=4`，执行 `python probe_aligned_cubic_capture_stress.py <新输出目录>`。两遍的四张误触成片、两张对照图、八张正例成片与报告逐件一致。另独立复算两次误触的 RGB 差，核对八张正例的成片身份。未接入软件。[复现脚本](../../experimental/adaptive-reference-20261113/probe_aligned_cubic_capture_stress.py) · [数值与哈希](2026-11-19-aligned-cubic-capture-stress-evidence.json)
