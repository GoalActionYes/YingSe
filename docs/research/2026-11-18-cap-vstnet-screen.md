# CAP-VSTNet 自动分区候选未通过跨场景偏色筛查

在六组已消费的背景换色对照上，运行 CAP-VSTNet 摄影模型与本机 ADE20K SegFormer B0。该方法在可逆网络的特征空间按自动分区匹配参考颜色。两张 RGB 是唯一推理输入，人工蒙版用于离线测量。沿用固定的 80% 混合和原开发门槛。SegFormer B0 代替作者示例中的 B5，本轮结论限于这套适配。见 [固定协议](../../experimental/adaptive-reference-20261113/cap-vstnet-screen-protocol.md)、[作者仓库](https://github.com/linfengWen98/CAP-VSTNet/tree/76a14b6db19a8145365631c8300c3a1abfc715bf)和 [作者论文](https://openaccess.thecvf.com/content/CVPR2023/html/Wen_CAP-VSTNet_Content_Affinity_Preserved_Versatile_Style_Transfer_CVPR_2023_paper.html)。

| 项目 | 结果 | 门槛 |
| --- | ---: | ---: |
| 人物串色减少且保留整图着色 | 1 / 8 | 至少 6 / 8 |
| 近中性色平均串色变化 | 增加 90.25% | 至少减少 15% |
| 近中性色最差方向 | 增加 29.20 码值 | 增加不超过 1 码值 |
| 对比度与去饱和 | 0 / 12 | 12 / 12 |

最严重的近中性色退化发生在滑雪案例 `785-3553` 的绿色背景干预。相对各自原参考的输出，13 号的串色差为 3.36 码值，候选为 32.56 码值。候选的雪地大面积染绿。分区参与了计算，但在这套配置中未能抵抗参考背景颜色传播。画面描述来自本机对照观察。

源码和语义映射表逐项核对 Git blob，模型从作者公开目录取得，大小与发布目录的 16,433,547 字节一致，记录 SHA256 并用 `weights_only=True` 严格加载。首次源码核对命中了 Git 自动换行过滤，切换原始字节后通过。首次推理的分割标签为 int64，被 Pillow 拒绝，改为与作者入口相同的 uint8 后在新目录完成。复核独立读取 42 对 PNG、重算背景差分和保留投影，门槛一致。推理循环约 6.26 秒，范围包括图像读取、自动分割、特征变换、写盘和统计，未计首次加载。见 [数值与身份](2026-11-18-cap-vstnet-screen-evidence.json)。

结束这一固定配置的画质筛查，未替换现有软件、下载独立验收图或启动追加训练。作者代码标为 MIT，独立发布权重仅用于本机研究，未核实验收或再分发资格。要复现，从仓库根目录设置 `$env:PYTHONPATH = (Get-Location).Path`，用 `.tools/vision-advisor-env/Scripts/python.exe` 运行 `experimental/adaptive-reference-20261113/probe_cap_vstnet.py <新输出目录>`。输入权重留在 `.tools/CAP-VSTNet-audit` 和 `.tools/scene-segformer`，结果在 `outputs/adaptive-reference-20261113/cap-vstnet-screen-v2`。
