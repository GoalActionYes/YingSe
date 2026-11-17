# StyleGallery：资产与最小导入已验证，真实推理待执行

本项是新路线的必要准备，不是画质升级。已完成公开固定资产下载和完整身份核验、隔离依赖导入；没有加载模型参数、生成图片、训练或改动产品。下一步直接执行固定样例，不再增加一轮泛泛模型调查。

## 为何值得一次实际筛查

[StyleGallery作者仓库](https://github.com/iiiiiiiword/StyleGallery)提出扩散特征分区、语义匹配和区域风格约束，区别于已停止的直方图条件、区域小头和ModFlows默认全局颜色流。它针对风格迁移，不保证自然调色、肤色或结构保持适合本项目。没有把论文展示当作本机收益。

此前ICTone／SAGE所需受限底模没有可用访问条件；这条路线使用可匿名获取的[SD1.5底模](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)，并需要DINOv2-base与DepthAnythingV2-Large。这里只证明另一条实际获取路径存在，不推翻此前访问结果，不断言大模型必然更好。

## 固定身份与体积

| 资产 | revision | 下载字节（含所需配置等） |
|---|---|---:|
| SD1.5标准FP32 | 451f4fe16113bff5a5d2269ed5ad43b0592e9a14 | 4266680108 |
| DINOv2-base | f9e44c814b77203eaa57a6bdbbd535f21ede1415 | 346349943 |
| DepthAnythingV2-Large | cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9 | 1341397304 |
| 合计 | 固定20文件 | 5954427355 |

权重采用LFS SHA256验证，普通文本按Git blob验证。下载会话13000正常退出0，全部文件验证通过；不下载另一精度副本、加速LoRA或SAM。支持保留已验证文件与断点文件，不能覆盖失败证据。

StyleGallery源码固定95f1b125158d9c5ef4f6ba5e9f17bcb11820d6dc，核验8份必要文本；DepthAnything源码固定a561b849ebae10a6f5ef49e26c83cbbcd36c71bf，核验13份文本。完整清单、Git blob、SHA256、模型文件与依赖身份见[资产登记](2026-11-17-stylegallery-assets.json)。源码和权重留在.tools/StyleGallery-audit，不上传第三方完整程序或模型。

StyleGallery代码MIT，DepthAnything源码Apache2.0；模型分别为SD CreativeML OpenRAIL-M、DINOv2 Apache2.0、DepthAnythingV2-Large CC BY-NC4.0，沿用非商业课程研究范围，不将代码许可等同于所有模型许可。

## 已完成的兼容检查

仅增加隔离vendor中的7个固定wheel，合计44301237字节；复用ModFlows已核验的9个隔离wheel，没有更改原环境或执行整个requirements安装。使用Torch2.6.0+cu124、torchvision0.21.0+cu124、transformers4.57.1、accelerate1.11.0、diffusers0.36.0、OpenCV4.11.0。其中transformers与accelerate版本不同于作者requirements中的4.57.6／1.12.0，真实运行兼容性仍待验证。

首次导入发现缺httpx，随后只补齐该依赖链，未加载模型或运行图片。会话81049的独立准备核验退出0，全部21份源码、20份模型文件及依赖wheel身份核对通过，作者pipeline及所需模型类导入通过；xFormers和SAM缺失，作者存在相应可选回退路径。见[准备证据](2026-11-17-stylegallery-preparation-evidence.json)。

该次检查瞬时可用物理内存约2.80GB、CUDA可用约7.44GB；不把它们当固定资源，更不能从下载成功或体积推算完整推理一定可行。没有实测加载峰值、样例画质或速度。

## 下一步固定执行

遵循[8对必要条件筛查协议](../../experimental/adaptive-reference-20261113/stylegallery-screen-protocol.md)，按原顺序用已消耗开发素材，保持作者标准单参考配方，记录必要工程适配与输出尺寸。最多8对、每对10分钟、推理总预算45分钟；已有3例严重退化即无法达到完整64对最多2例的限制。该数学必要条件不能代替完整开发通过。

先读取本报告与协议，确认下载结果finished且全部文件身份一致，再执行真实模型加载／首例推理；不要重复下载或轮询已退出的13000／81049。内存调度或兼容问题要记录具体证据，不能因此将整个目标受阻，也不能悄悄更改算法配方。

fetch_stylegallery_weights.py只恢复固定模型资产；源码及隔离wheel须按登记放到规定目录，check_stylegallery_preparation.py核验并检查导入。所有媒体、权重、环境均不上传Git。完整目标未达成，稳定1.3／第10阶段保持。
