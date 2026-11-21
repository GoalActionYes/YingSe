# VideoColorGrading 默认预处理已出图，完整扩散模型待测

为验证另一种普通目标与单参考图的推理路线，核查了 ICCV 2025 的 VideoColorGrading。它先以参考颜色统计变换目标，再根据输入与参考的特征生成 16³ 色表。此前的 AceTone 是预测量化色表 token，二者的推理机制不同。这里直接执行固定作者源码中的预处理三个函数，不加载扩散网络。作者提供 `ncc` 关闭预处理的参数。单帧目标使用既有 80% 输出评分，参考按作者 CLI 缩放到 512×512。[论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Shin_Video_Color_Grading_via_Look-Up_Table_Generation_ICCV_2025_paper.pdf) · [固定作者源码](https://github.com/seunghyuns98/VideoColorGrading/blob/9d58052bd70eb68ed684003619e62795825f346d/utils/util.py) · [事前协议](../../experimental/adaptive-reference-20261113/vcg-preprocess-protocol.md)

| 目标 ← 参考 | 默认，风格关闭仍偏离原片，冷／暖 | 默认，整图风格投影，冷／暖 | 默认，仅换参考灯的整图差，关／开 | 两臂通过区域 |
| --- | ---: | ---: | ---: | ---: |
| 01 ← 02 | 32.08／29.96 | −0.086／−0.040 | 6.23／5.83 | 各 0／5 |
| 02 ← 01 | 33.92／32.66 | 0.459／0.034 | 4.74／5.32 | 各 0／5 |

误差单位为 RGB8。关闭预处理的八张输出逐像素保持原片，没有风格响应。默认预处理的原实现运行时出现了将复数转为整数并舍弃虚部的警告，两次均保留作者原实现。其阶段输出依然重复一致。默认输入有较大的原片偏移，后续扩散色表需要抵消这项偏移。单独关闭预处理也不能证明完整模型可过线。两个人像方向是既有开发对照，未做真实视频或自然评价。

## 权重与本机运行边界

作者仓库的训练说明指定电影片段及 100 份选用色表，推理配置还依赖 Stable Diffusion 1.5、CLIP、GS-Extractor 和 L-Diffuser。另一个独立维护者将四组权重合并为 Safetensors，托管页标记 Apache 2.0。这里只检查该第三方转包的结构与可取得性，未核对其参数值与作者发布是否完全同一。该文件不进入软件或 Git。[原始训练与推理说明](https://github.com/seunghyuns98/VideoColorGrading/blob/9d58052bd70eb68ed684003619e62795825f346d/README.md) · [固定转包](https://huggingface.co/Kijai/VCG_comfy/blob/13229792541f7361072a6eefb21402e9e4cc4d68/checkpoints/vcg_combined_fp16.safetensors)

服务器返回 HTTP 206，实际取得首 1,048,576 字节。头部为 236,904 字节，索引含 1,796 个张量，逐项检查形状、类型和连续偏移，可复算完整文件 4,119,430,034 字节。L-Diffuser 与 ReferenceNet 各约 1.72 GB，CLIP 和 VAE 分别约 351 MB 和 335 MB。文件方登记的完整 SHA-256 为 `788ee27da6d2459f867238488055d32ebb79f50c1988c3362efc91c25240221d`，本轮只核对已下载的范围。主研究环境当前没有直接可导入的 Diffusers 与 ComfyUI，检查时约有 2.8 GiB 空闲内存和 6.5 GiB 空闲显存。权重容量不等于推理峰值。全量下载前，先复用已存在的依赖验证逐组件映射与卸载，随后在同批目标上分别核对默认及关闭预处理的真实生成色表。阶段结果支持选择这个对照，不能代替完整模型的效果判断。

## 复现

从固定源码获取 `util.py`，用 HTTP Range 取得固定转包的 0–1048575 字节，分别保存为同一目录下的 `util.py` 和 `header.bin`。按 [Gros Film B 环境](2026-11-21-gros-film-b-cross.md) 运行 `experimental/adaptive-reference-20261113/probe_vcg_preprocess.py <本机来源目录> <新输出目录>`。已有本地来源为 `outputs/adaptive-reference-20261113/vcg-source-intake-v1`。两遍各约 5 秒，16 张 PNG 和报告逐个校对，另复算 20 项判定和权重字节数。试验成片只保留本机 `outputs/adaptive-reference-20261113/vcg-preprocess-v1`。[来源哈希、分项评分和索引](2026-11-21-vcg-preprocess-evidence.json)
