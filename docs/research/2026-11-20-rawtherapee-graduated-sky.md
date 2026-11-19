# 第二个实拍天空开关通过前景保护

上一轮的 P1005231 在关闭任一局部工具后仍会改动石墙，因此改用另一位摄影者的原片。thumper 提供了公开 RAW，Jacques Desmis 公开说明自己的渐变滤镜用于压暗天空，并提供完整 PP3。只下载一份 RAW 和一份 PP3，大小分别为 18,867,286 与 25,758 字节。原帖声明 CC BY-SA，原件、设置及成片留本机。[thumper 原片与声明](https://discuss.pixls.us/t/show-your-best-shadow-highlight-techniques/55731) · [Desmis 的说明和 PP3](https://discuss.pixls.us/t/do-you-ever-use-the-a-or-b-curves/55864/11)

开态使用原作者设置。关态只改变 `[Gradient]` 中的 `Enabled`，保留其他局部工具、色彩和构图。先导出关态并登记固定区域，再导出开态，避免根据差异挑选容易通过的位置。原有保护线继续适用。[协议及区域](../../experimental/adaptive-reference-20261113/rawtherapee-graduated-sky-protocol.md)

| 区域 | 中位亮度变化，RGB8 | 亮度绝对变化中位数，RGB8 |
| --- | ---: | ---: |
| 主天空 | −22.718 | 22.718 |
| 侧天空 | 0.000 | 0.072 |
| 前景草地 | +0.213 | 0.213 |
| 前景右侧树岸 | +0.213 | 0.213 |
| 天空及枝条交界 | +0.787 | 0.787 |

主天空下降超过 2，两处前景绝对变化均小于 1，判定通过。侧天空与枝条交界的结果也保留，以体现局部滤镜的作用范围。与已有 R6_06283 合计取得两份独立摄影内容的开发开关，可以用于双向跨场景出图比较。它们共享局部压暗天空的意图，编辑工具、参数和空间覆盖各不相同。[本机关闭态](../../outputs/adaptive-reference-20261113/rawtherapee-graduated-sky-v1/off/off.png) · [本机开启态](../../outputs/adaptive-reference-20261113/rawtherapee-graduated-sky-v1/on/on.png)

使用既有 RawTherapee 5.13 单线程导出 3662 × 5488 RGB8 图像，ICC 均为 RTv4_sRGB。原导出任务在最后一张重复图落盘前中断，核对既有设置后只补跑这一张。开、关各自两遍解码像素一致。渐变开关的字节差、首遍各区域的整数亮度复算和两份审计报告的 SHA256 均已核对，旧天空开关报告也逐字节复现。用 `render_rawtherapee_graduated_sky.py <素材目录> <CLI> <新输出目录> <off 或 on>` 导出，将四个状态目录交给 `audit_rawtherapee_graduated_sky.py <素材目录> <新报告>`。报告 SHA256 为 `aaafb4d68acac62b53dc779177861e1099e2e506f6847712a603803e2b9a1337`。映色评分和正式画质验收尚未开展。
