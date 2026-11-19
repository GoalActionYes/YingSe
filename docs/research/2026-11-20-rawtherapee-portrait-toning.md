# 实拍人像的全图色彩平衡会一并改变树皮和草地

Andrea 在《Cuckoo》中说明，绿色植被的反光影响了人像颜色。chaimav 提供 RawTherapee 的完整处理参数，也列出了黑位、通道混合和阴影／中间调／高光色彩平衡。本轮固定原片与这份参数，只切换 `[ColorToning]` 的 `Enabled` 字段，检查这项调整在脸部与周围区域之间的差别。配置的 `Splitco` 模式将中间调和高光的绿色参数分别设为 −10 与 −20，阴影绿色参数为 0。其余工具保留原配置。[原片与 CC BY-SA 4.0 声明](https://discuss.pixls.us/t/cuckoo-problems-with-color-balance/29959) · [编辑说明与配置](https://discuss.pixls.us/t/cuckoo-problems-with-color-balance/29959/10) · [事前协议](../../experimental/adaptive-reference-20261113/rawtherapee-portrait-toning-protocol.md)

先从关态的 900 × 599 成片登记脸颊、浅灰树皮和后方草地。浅灰树皮只作为画面内低饱和材料代理，原材质未进行中性标定。数值取每个像素的红、蓝去亮度分量开关差的绝对值均值，再取区域中位数。以下是带 `RTv4_sRGB` 配置的 RGB8 码值测量。[登记范围](../../experimental/adaptive-reference-20261113/rawtherapee-portrait-toning-regions.json)

| 测量区域 | 像素数 | 色度改变量 | RGB 各通道变动中位数 |
| --- | ---: | ---: | ---: |
| 脸颊 | 224 | 4.29 | +3、−2、+4 |
| 浅灰树皮 | 648 | 5.44 | +7、−2.5、+4 |
| 绿色草地 | 1,595 | 13.47 | +20、−5、+11 |

脸颊达到 2 码值的有效变化门槛，树皮和草地均超过 1.5 的保护上限。编辑改变脸部颜色的同时，对两处环境样本的影响更大。保留这组作为整图去绿的开发反例。它记录的是一个真实编辑操作的像素变化，不能推出哪一态肤色更自然。本轮没有跨场景推理，也没有把这组开发素材放入独立验收。[关闭态](../../outputs/adaptive-reference-20261113/rawtherapee-portrait-toning-v1/off/image.png) · [开启态](../../outputs/adaptive-reference-20261113/rawtherapee-portrait-toning-v1/on/image.png)

原档案为 93,109,546 字节，仅含一份 ARW。首次下载在 80,532,058 字节处超时，按服务端范围续传补齐，完整 ZIP 的 SHA1 与发布端文件标识吻合。作者 PP3 标注 5.8 开发版，本机用 5.13 重放。配置的暗场和平场路径 `/szeva` 在本机不可用，另做清空这些字段的对照以核查本机影响。原片、PP3、导出和日志均留在本机。

开、关各自的重复导出和路径清空控制逐像素一致，ICC 内容相同。PNG 文件哈希的差异来自 `eXIf` 块，图像数据块一致。独立核查了 ZIP CRC、作者 PP3 的唯一开关变化，从落盘图重算三处得分。两遍数值报告的 SHA256 均为 `c0175701b1ff06967c704ffea4460994f6a747538e5284b5a9e67d7f442fb740`。

复现时在现有视觉 Python 环境将仓库根目录加入 `PYTHONPATH`。参考[复现脚本](../../experimental/adaptive-reference-20261113/probe_rawtherapee_portrait_toning.py)的 `render <素材目录> <CLI> <新输出目录> <状态>`，状态为 `off`、`on`、`off-repeat`、`on-repeat`、`off-clear-paths`、`on-clear-paths`。脚本核对 ARW、PP3 和 CLI 哈希。把六个同名状态目录置于同一根目录，运行 `audit <根目录> <区域 JSON> <新报告路径>`。如需后续检验局部肤色修正，与此例对照的样本应有可溯源的局部开关，并在其它区域保有接近不变的像素证据。
