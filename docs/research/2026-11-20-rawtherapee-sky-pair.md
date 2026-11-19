# 取得两个实拍天空编辑开关，一组仍会影响前景

使用 RawTherapee 5.13 导出了两张此前缺少可测量开关对照的实拍照片。P1005231 使用 Tamas Papp 的原片与 Sunhillow 的处理记录，R6_06283 使用 Karl 的原片与 Martin Werner 的处理记录。两幅原片各来自独立拍摄，原片作者均在帖子声明 CC BY-SA。设置和媒体只留本机。[P1005231 原片及第 16 楼设置](https://discuss.pixls.us/t/play-raw-mask-and-darken-sky/43881/16) · [R6_06283 原片及第 2 楼设置](https://discuss.pixls.us/t/vignette-correction/48970/2)

这次从编辑者原始 PP3 导出开态。关态只将 `[Locallab]` 的 `Enabled=true` 替换为 `Enabled=false`，保留其余字节。P1005231 的局部点是全范围模式，文件中另有 Retinex 相关局部设置。R6_06283 的局部点是普通模式；两者关闭的是各自的局部操作组，不能合称相同调色参数。[测量协议与关态登记区域](../../experimental/adaptive-reference-20261113/rawtherapee-sky-pair-protocol.md)

| 实拍 | 导出尺寸 | 主天空中位亮度变化 | 前景绝对变化中位数，两处 |
| --- | --- | ---: | ---: |
| P1005231 | 2048 × 1536 | −38.13 | 3.86／0.72 |
| R6_06283 | 1804 × 1205 | −8.71 | 0／0 |

表中是单线程复查结果，单位为 RGB8 亮度码值。双线程时 P1005231 两态重复导出分别有 7,098 和 17,852 个通道值变化，最大差为 3／6 RGB8。改为单线程后，两张照片的四种输出均逐像素复现，从各登记区域用整数亮度系数再计算，门槛判定相同。

P1005231 的石墙被一并压暗，树边测量区中位亮度下降 7.00，保留为前景串色反例。R6_06283 通过了主天空至少压暗 2、两处前景变化中位数均小于 1 的门槛，侧向天空也压暗了 2.14。两张一起尚不能构成通过核查的跨内容局部监督对照。[R6 关态](../../outputs/adaptive-reference-20261113/rawtherapee-sky-pair-v1/single-thread/off/R6_06283-off.png) · [R6 作者开态](../../outputs/adaptive-reference-20261113/rawtherapee-sky-pair-v1/single-thread/on/R6_06283-on.png)

使用官方 Windows x64 ZIP 隔离解包，大小为 121,500,459 字节，SHA256 为 `ba29d23f3f12097c925bdab6d80e831a84ed3a88a81ce94606e4dcc796fa87ee`。CLI 声明版本为 5.13，配置和缓存位于本机研究输出目录。P1005231 原件重用既有归档，R6_06283 下载为 10,285,202 字节。两份 PP3 合计 39,641 字节。[官方发布及资产](https://github.com/RawTherapee/RawTherapee/releases/tag/5.13)

用 `render_rawtherapee_sky_pair.py <素材目录> <CLI 路径> <新输出目录> <off 或 on>` 导出。双线程原始检查使用 `--threads 2`。将四次输出放在 `off`、`on`、`off-repeat`、`on-repeat`，用 `audit_rawtherapee_sky_pair.py <包含四个目录及原 PP3 的目录> <新报告路径>` 核查。单线程核查重复报告 SHA256 为 `3c8256292d0c92f11df08cc81c6bd865866f541188ee3de6339fc4b202045bf5`。原件、日志及出图保存在本机 `outputs/adaptive-reference-20261113/rawtherapee-sky-pair-v1`。映色推理和独立验收未参与这轮素材验证。
