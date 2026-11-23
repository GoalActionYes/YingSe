# 同款局部提亮切换三档实拍灯，现有两版均未保住提亮幅度

在两人的低、中、高灯档原片上分别开启同一项固定的半档曝光。使用已登记的人脸羽化范围。六张开启态都通过素材门槛，两颊增加 7.92–22.79 个 Y′ 码值，两处白幕和脸外保持原色。两张冷灯目标按相同方法给出离线答案，并沿用 80% 强度入分。这个半档曝光是本项目施加的控制，照片作者没有对这六张照片逐一调色。[事前协议](../../experimental/adaptive-reference-20261113/gros-mixed-light-exposure-toggle-protocol.md) · [已有关闭态成片](2026-11-24-gros-mixed-light-transfer.md)

实际工作台固定跨人物目标，每次只读普通目标图和一张已提亮的参考。与相同灯档下两版原来下载的关闭态做差，再对照目标需要的改变量。区域与答案只用于评分。保留每个灯档的响应、成片偏离和跨档差异。[成片与预期结果](../../outputs/adaptive-reference-20261113/gros-mixed-light-exposure-toggle-v1/2-from-4-comparison.png) · [逐区数值与哈希](2026-11-24-gros-mixed-light-exposure-toggle-evidence.json)

| 检查 | 稳定 1.3 | 13 号 |
| --- | ---: | ---: |
| 跨人物、跨三档灯完整通过 | 0／2 个方向 | 0／2 个方向 |
| 两颊编辑投影范围，原门槛 0.75–1.25 | −0.564 至 0.205 | 0.010 至 0.158 |
| 仅切换提亮开关，白幕最坏变化 | 21.150 RGB8 | 2.233 RGB8 |

稳定版用第 4 包低灯档作参考、第 2 包作目标时，开启提亮反使目标双颊的平均 Y′ 降低 7.67 和 10.04。两份映射都未走已对齐参考支路，保留两份实际输出。13 号的双颊响应方向在这些控制中均为正，但只有 1.0%–15.8% 的期望响应。其白幕开关差小于 3 RGB8，关闭态已有随拍摄灯档串色，完整的白幕保持仍未通过。报告分别记录这些现象，避免把开关差小解读为实际保色或成功迁移。[反向提亮的低灯档关闭成片](../../outputs/adaptive-reference-20261113/gros-mixed-light-transfer-repeat-v1/2-from-4-stable-low.png) · [开关后成片](../../outputs/adaptive-reference-20261113/gros-mixed-light-exposure-toggle-v1/2-from-4-stable-low-on.png)

这批数据补齐了同一可重复操作在三档实拍下的开关控制，可以同时筛查误迁移现场光和漏掉局部明暗风格。保留这一套固定对照，暂停变动半档强度继续追分。独立、真实作者的编辑配对仍待补齐。没有改动工作台、模型、产品权重、视频或独立评估素材。

## 核验与复现

两遍共用 69.7 秒，每遍各保存 6 张编辑参考、2 张目标答案、12 张实际工作台成片、2 张对照图和报告，23 件产物逐字节一致。由冻结的原图、关态和本轮保存的成片复算 8 张生成对照、72 项开关响应和 72 项灯档间响应差。报告 SHA-256 为 `a12aaa38a01e0ca459b7cfbc0e2c555420c020159b7399e66fc81f9b551bf6b0`。沿用[前一轮的两个固定工作台及原研究环境](2026-11-24-gros-mixed-light-transfer.md#核验与复现)，将脚本路径换为 `experimental/adaptive-reference-20261113/probe_gros_mixed_light_exposure_toggle.py`，指定同一研究输出根目录和新运行目录。原图、成片和内部工作台数据留本机。[复现脚本](../../experimental/adaptive-reference-20261113/probe_gros_mixed_light_exposure_toggle.py)
