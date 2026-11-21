# VCG 权重与作者模型四个组件逐项匹配

此前预处理可以出图，后续扩散步骤还未验证。将作者规定的 Diffusers 0.21.4 单独解包到本机研究目录，复用已有 Torch、Transformers、Accelerate、Safetensors 和 Einops。现有 0.36.0 首次导入作者 ReferenceNet 时，缺少已移除的 `PositionNet`。使用独立路径后，作者 ReferenceNet 和完整推理类皆可导入。旧库引用的网络下载接口在本次探针中被显式禁用，防止导入时改取外部权重。原 Python 环境和作者源码未修改。[事前协议](../../experimental/adaptive-reference-20261113/vcg-model-loading-protocol.md) · [作者依赖](https://github.com/seunghyuns98/VideoColorGrading/blob/9d58052bd70eb68ed684003619e62795825f346d/fast_env.sh) · [PyPI 发布](https://pypi.org/project/diffusers/0.21.4/)

| 组件 | 匹配张量 | 处理 |
| --- | ---: | --- |
| CLIP 视觉模型 | 200 | 转包多一层视觉投影，使用支持它的结构做全量映射 |
| VAE | 248 | 16 个旧注意力键作一一名称映射 |
| ReferenceNet | 662 | 原作者结构直接对应 |
| L-Diffuser | 686 | 六通道输入、三通道输出的 U-Net 直接对应 |

每项索引与空壳模型核对名称和形状，最终无缺项、无额外项、无形状不一致。首次 CLIP 使用作者直接分类时，199 个键相符，唯一额外键是 `visual_projection.weight`。后续渲染仍应读取作者 `ImageEncoder.py` 使用的投影前池化特征。此次另外通过 HTTP Range 取得三个真实张量切片，共 4,352 字节，分别来自 CLIP、ReferenceNet、L-Diffuser。每项经本机内存映射进入对应模型的 CUDA 参数槽，保持原 dtype，逐值取回，并卸载回 meta。小切片仅验证传输方法，不能用于估计完整推理峰值或画质。[固定作者源码](https://github.com/seunghyuns98/VideoColorGrading/tree/9d58052bd70eb68ed684003619e62795825f346d) · [固定转包](https://huggingface.co/Kijai/VCG_comfy/blob/13229792541f7361072a6eefb21402e9e4cc4d68/checkpoints/vcg_combined_fp16.safetensors)

两遍各约 4 秒，报告逐字节一致。来源文件、原作者源码、切片和研究依赖仅留本机。需要以完整哈希确认后的 4.12 GB 权重生成图像，才能比较默认预处理和关闭预处理的真实最终效果。这里不声称有新画质效果，本轮也没有软件变更。[运行脚本](../../experimental/adaptive-reference-20261113/probe_vcg_model_loading.py) · [数值与文件身份](2026-11-21-vcg-model-loading-evidence.json)

从仓库根目录运行，使用本机来源目录 `outputs/adaptive-reference-20261113/vcg-source-intake-v1`、作者源码目录 `outputs/adaptive-reference-20261113/vcg-author-9d58052bd70eb68ed684003619e62795825f346d`，以及未占用的报告文件，依次为脚本的三个参数。使用 `.tools/vision-advisor-env/Scripts/python.exe`，将 `.tools/ModFlows-audit/vendor` 附加到进程模块路径。原输出文件为同级 `vcg-model-loading-v2.json`。
