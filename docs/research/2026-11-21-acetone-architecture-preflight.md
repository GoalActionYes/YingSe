# AceTone 发布版架构与两图入口通过本机预检

上一轮给定已知色表时，AceTone 编解码器在六张真人实拍上保住了既定风格。是否值得下载完整模型，先看作者模型和本机运行环境能否接上。此次按固定提交取得模型类、配置、权重索引及 tokenizer 资料，共约 11.6 MB。实验沿用本机 PyTorch 2.6.0+cu124、Transformers 4.57.1，未安装新依赖。作者类与模型卡见以下链接。[事前协议](../../experimental/adaptive-reference-20261113/acetone-architecture-preflight-protocol.md) · [作者实现](https://github.com/martian422/AceTone/tree/916393b3f26bdf89c3d939cc5f2a9a3c115ccbc5) · [固定发布版本](https://huggingface.co/Vivre/AceTone-3B-PST-Preview/tree/9416b259dec1bf7bb01ba2dab38cf6470cfa98f0)

完整尺寸模型在 meta 设备上构造。使用本机 Qwen2.5-VL 已有的名称转换，公开索引中的 825 个名称与模型状态逐项对应，未发现遗漏或冲突。文本输出矩阵形状为 151936×2048。这里只读了索引，真实权重的张量内容、逐张量形状与加载峰值尚未验证。

小尺寸模型沿用作者的两图和 token 接口，使用随机权重。真实 tokenizer 将两张合成图分别转成 1×4×4 网格，共得到 8 个视觉占位 token。两图交换后的最后位置 logits 最大绝对差约 0.0244，证明前向接收了两张图及其顺序。CPU 使用 PyTorch SDPA，连续生成 3 个 token。另独立核对 256 个合法色表 token 和开关边界，合法 64 码通过，四种畸形序列均被拒绝。随机生成的 token 只用于验证生成接口，真实模型何时生成合法色表尚待测试。[输入身份和运行数值](2026-11-21-acetone-architecture-preflight-evidence.json)

接下来优先验证精度不变的本机加载方案，例如把少量完整层移到 CPU 或磁盘，先取得小规模真实两图成片再考虑量化。本机整卡约 8 GiB，公开参数体积约 7.57 GiB，推理还需留出图像张量和生成缓存。接口预检没有证明完整 BF16 模型能直接全放在 GPU。没有动稳定版、13 号及封存素材。

## 复现

在本机 `outputs/adaptive-reference-20261113/acetone-architecture-assets` 中，将证据清单中的源码依照路径放入 `upstream/model`，将发布模型的八个 JSON 文件放入 `checkpoint`。从仓库根目录运行，使用未占用的报告名。

~~~powershell
$assets = 'outputs/adaptive-reference-20261113/acetone-architecture-assets'
& .tools/vision-advisor-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_acetone_architecture_preflight.py $assets "$assets/preflight-repro.json"
~~~

两次复算与首次报告 SHA-256 一致，结果为 `16ed33db626b3b8fdd03cb6cb65bcdc2387dbbc59cf74f3a12abe339b4adb002`。
