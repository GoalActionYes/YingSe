# 扩展第 4 轮在白球服和焊工上有改善，四套联合检查仍未通过

固定最强保留候选扩展第 4 轮，使用原生空间渲染、80% 强度和正常两张 RGB 输入。在同一流程完成 16 张实拍换背景、42 张 FACD、12 张 Film B 和 2 张实体色卡成片。已知效果图与评分区域只供离线评分。13 号使用冻结成片作为基线。本轮预先核对 132 份文件，并验证同图输入逐像素保持。[事前协议](../../experimental/adaptive-reference-20261113/frozen-candidate-development-crosscheck-protocol.md)

| 套件 | 候选结果 | 原门槛 |
| --- | --- | --- |
| 人物换背景与有意编辑 | 0/4 组，最大保护区差 26.88 RGB8 | 至少 3/4，最大差 ≤3 |
| FACD 同款换参考 | 0/7 组，近中性色最坏额外误差 14.61 RGB8 | 至少 6/7，额外误差 ≤2 |
| Film B 换风格 | 0/6 方向同时满足整图和近中性色要求 | 投影 0.75–1.25、响应误差 ≤3 |
| 实体色卡 | 2/7 个区域响应达标，关闭风格最大色度误差 6.74 | 至少 5/7，关闭态最大误差 ≤3 |

- 白球服有意编辑开启时，换背景导致的近中性色差由 7.68 降至 3.87 RGB8，人物内近中性色由 6.30 降至 4.57。明暗响应两种背景下仍只有约 0.28 和 0.23。另两组的保护区差反而增加。[白球服对照](../../outputs/adaptive-reference-20261113/frozen-candidate-development-crosscheck-v1/background/872-1268/comparison.png)
- 焊工用扑克和色卡作参考时，Film B 近中性色响应从 −0.087／−0.447 转为 +0.358／+0.216。两组方向转正，幅度仍不足，响应误差为 6.43／7.67 RGB8。[焊工用色卡参考的对照](../../outputs/adaptive-reference-20261113/frozen-candidate-development-crosscheck-v1/film_b/sparks_4592-from-hdr_testimage-comparison.png)
- FACD 的火焰场景 `379484` 用 `523992` 作为 Hudson 参考时，近中性色与已知成片的差由 9.06 增至 23.67 RGB8。色卡也仍只有 2/7 个区域达标，关闭风格的最高误差虽由 7.51 降至 6.74，仍超过上限。[FACD 反例](../../outputs/adaptive-reference-20261113/frozen-candidate-development-crosscheck-v1/facd/379484/comparison.png) · [色卡对照](../../outputs/adaptive-reference-20261113/frozen-candidate-development-crosscheck-v1/chart/comparison.png)

扩展第 4 轮继续作为参考和回归候选。它能在部分目标上减轻串色并修正风格方向，但同时还有响应不足和明显退化，现有软件暂不采用。四套全是已消费的开发材料。下一候选需要在同一批检查中同时保住白球服的改善、恢复风格幅度，并避开 Hudson 的退化，再考虑独立评估。

两遍各 72 张输出、18 张对照图和一份报告逐字节一致。另从落盘图核对 72 个成片身份，复算四套各自的主要数值。原色卡混合候选的默认入口仍逐字节生成原两张图及对照。复现以已有研究环境将仓库根目录和 OpenCV 加入 `PYTHONPATH`，执行 `probe_frozen_candidate_development_crosscheck.py <新输出目录>`。完整逐例结果和图片保留本机。[复现脚本](../../experimental/adaptive-reference-20261113/probe_frozen_candidate_development_crosscheck.py) · [数值摘要和哈希](2026-11-19-frozen-candidate-development-crosscheck-evidence.json)
