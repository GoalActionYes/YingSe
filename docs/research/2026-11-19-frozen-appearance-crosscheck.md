# 关闭参考外观会减轻 Hudson 退化，也会损失部分有效风格

冻结扩展第 4 轮的权重和正常两图输入，只通过既有开关移除参考外观分支。逐通道、联合颜色统计、目标路径、原生输出尺寸与 80% 强度保持。对已登记的 72 个组合分别生成完整与关闭输出，沿用四套实拍的门槛。[事前协议](../../experimental/adaptive-reference-20261113/frozen-appearance-crosscheck-protocol.md) · [原四套检查](2026-11-19-frozen-candidate-development-crosscheck.md)

| 对照 | 13 号 | 完整候选 | 关闭外观 |
| --- | ---: | ---: | ---: |
| 白球服开启编辑时的近中性色换背景差，RGB8 | 7.68 | 3.87 | 4.82 |
| 火焰 Hudson 的近中性色误差，RGB8 | 9.06 | 23.67 | 18.97 |
| 焊工用扑克参考的 Film B 近中性色投影 | −0.087 | +0.358 | +0.194 |
| 焊工用色卡参考的 Film B 近中性色投影 | −0.447 | +0.216 | −0.130 |
| 实体色卡响应达标区域 | 2/7 | 2/7 | 1/7 |

关闭外观使 FACD 的 42 张成片中 19 张近中性色误差下降，23 张上升。其最坏额外误差由 14.61 降至 9.91 RGB8，仍超过 2。背景、FACD、Film B 的联合达标数分别为 0/4、0/7、0/6，关闭后未改变。色卡关闭风格的最大色度误差从 6.74 降至 6.12，仍超 3，且两个灰阶色块的风格方向反转。[白球服成片](../../outputs/adaptive-reference-20261113/frozen-appearance-crosscheck-off-v1/background/872-1268/comparison.png) · [Hudson 反例](../../outputs/adaptive-reference-20261113/frozen-appearance-crosscheck-off-v1/facd/379484/comparison.png) · [焊工反例](../../outputs/adaptive-reference-20261113/frozen-appearance-crosscheck-off-v1/film_b/sparks_4592-from-hdr_testimage-comparison.png)

同一检查点下，外观分支确实参与了部分串色改善和风格方向修复，也加重了部分退化。关闭时与 13 号仍有差异，说明其它训练后的权重和颜色统计路径也会影响成片。这里检验的是固定检查点的开关，无法推断从头训练去掉分支会怎样。结束直接移除分支作为普遍修复，不把 19 张改善作为上线依据。四套素材保留作开发对照，未打开独立验收。

完整臂的 91 份文件逐字节复现先前结果，关闭臂两次各 91 份文件匹配。另从落盘图验证 72 张成片身份，独立复算背景换色、FACD 近中性色、Film B 两范围和色卡主要指标。复现沿用原四套检查脚本，传入新输出目录，追加 `--disable-appearance`。[复现脚本](../../experimental/adaptive-reference-20261113/probe_frozen_candidate_development_crosscheck.py) · [数值摘要和哈希](2026-11-19-frozen-appearance-crosscheck-evidence.json)
