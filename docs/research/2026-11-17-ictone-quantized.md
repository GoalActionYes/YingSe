# ICTone 量化执行前置验证

## 运行前协议

问题：跨场景参考下的肤色／中性色错误尚未解决。ICTone 的任务训练与已失败的通用艺术风格优化不同，但标准 FLUX-Fill 链无法整组件驻留本机；公开 GGUF 是否提供值得实际筛查的执行路径？本轮只回答格式、LoRA 接入及资源前提，不声称画质收益。

已有证据：标准底模与 LoRA 合计约35.766GB；本机显存8188MiB。公开量化发布者声称由 FLUX-Fill 转换，尚未核对张量维度、LoRA 类型或本机接入行为。不能用普通 FLUX-dev 代替 Fill，也不能删除作者的非空提示词来省去文本编码器。

主要改动：仅尝试量化执行兼容性；不改 ICTone 任务、提示词、权重、调色强度或图像验收。比较对象为小型 GGUF 线性层在注入 LoRA 前后的输出及显式反量化计算，绝不把这个小测试当作整模型验证。

预算：元数据／文件头每响应最多2MB，最多3个模型仓库；新增隔离依赖合计上限5MB，小型 CPU／GPU 数值测试不超过60秒，不下载大权重。读取公开文件头必须限制读取量并关闭响应，不能因服务忽略 Range 而读完整文件。

通过条件：固定公开资源身份、Fill 输入维度与 LoRA 模块／秩／dtype；真实量化层注入非零 LoRA 后与显式计算相符，禁用后恢复；完整链资源缺项逐项明确。任何格式不符、数值错误或超过预算都停止对应步骤。即使小测试通过，也必须先解决 T5、VAE、卸载和峰值资源，再进入真实图片筛查；量化输出不能冒充作者原精度输出。

许可范围：公开可下载不等于可任意使用或交付。保留上游许可及来源，当前仅核查离线研究前提；不绕过访问控制，不上传权重，不据此认定最终产品授权已满足。

## 实际结果

[数值与身份摘要](2026-11-17-ictone-quantized-evidence.json)记录固定 revision、LFS 预期哈希、实际读取文件头哈希、依赖和错误收尾。只读了两个2MB前缀，不能声称完整权重已校验。三份仓库元数据均已保存本机。

- 公开 Fill Q3_K_S revision `78b83f1da140a4dcd6466580516544e1f6effe3e`，文件5238187296字节。780项 GGUF 张量经 Diffusers 原转换函数生成1160项 meta 张量，与显式配置的全部键和尺寸一致：输入384、输出64、19个双流块、38个单流块、guidance embedding 开启。这是形状兼容，不是转换数值或整图验证。
- ICTone 的686项 LoRA 张量均为BF16，343对 A/B 均为秩256，全部匹配上述 meta 模型对应线性层尺寸；参数927301632个。没有下载1.85GB权重。
- 实际 Diffusers 0.36／PEFT 0.18 的小型 Q3_K 层，CPU与CUDA均成功注入非零BF16 LoRA；两者对显式计算的误差均为0，禁用后精确恢复，底层压缩字节未变。没有自动扩成FP32适配器。测试用秩4／6×256合成层，不代表全部大层或整模型运行。
- Diffusers原生Q3_K反量化为FP16，NumPy GGUF路径为FP32；浮点值最大差0.00042725，满足预先明确的FP16舍入比较界限（rtol .002、atol .001）。LoRA输出和禁用恢复仍要求精确相等。
- 新依赖仅两个纯Python wheel合计652651字节，核对预期SHA256后解压至独立 `.tools/ICTone-quantized-audit/vendor`，未改主环境。GGUF为MIT，PEFT为Apache；模型许可单独处理。

首次数值测试在dtype比较处中止，尚未注入LoRA；修正为明确区分反量化舍入与LoRA精确检查。首轮meta模型误用默认输出384，和文件头64不符；按文件头改为64后全量尺寸通过。两次失败目录保留，不当作模型推理失败，也不把修正算画质收益。

## 资源与下一项

压缩底模张量实际存储5238132992字节，BF16 LoRA1854603264字节，仅两项合计7092736256字节（约6.61GiB），未含VAE、CLIP、激活、反量化临时量、CUDA上下文及其他开销。本轮实测瞬时空闲6505MiB，小于两项合计；因此不能按全驻留路径直接启动。该瞬时读数不是机器固定容量，也不能据此断言块级卸载不可行。

公开 T5 Q3_K_S 的元数据文件体积2099467456字节，不证明本地 Transformers 能保持它的压缩状态。本机 Transformers 4.57.1 的 `modeling_gguf_pytorch_utils.py:load_gguf_checkpoint` 在加载张量时逐项调用 `dequantize`，不能把GGUF文件体积当作该加载路径的驻留内存。作者使用非空triptych提示词，不能跳过T5或换成零向量冒充原方案。

下一项只核查完整非空提示词的T5量化执行／缓存与Transformer块级卸载：先取得固定配置及文件头，验证模块映射与小型数值行为，再测资源预算；未解决前不下载大权重、不启动整图。此项与旧标准全组件CPU卸载检查不同，新增的是压缩线性层及分块驻留的真实兼容证据；不重新启动已失败画质路线。当前没有整模型、图片、视频成绩，也没有可体验升级。

公开来源：[Fill量化发布页](https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF)、[ICTone LoRA](https://huggingface.co/ToneStyle/ICTone-Fill-LoRA)、[T5量化发布页](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)、[Diffusers 0.36 GGUF文档](https://huggingface.co/docs/diffusers/v0.36.0/en/quantization/gguf)。发布者声称的原模型来源不等于本轮完成了量化前后数值核验。

复现顺序：主环境Python运行 `audit_ictone_quantized.py --run 新目录`；仅在隔离依赖未准备时运行 `prepare_ictone_quantized_probe.py`；随后运行 `probe_ictone_gguf_lora.py --run 新目录` 和 `audit_ictone_gguf_shapes.py --metadata 元数据结果绝对路径 --run 新目录`。所有脚本位于 `experimental/adaptive-reference-20261113`，拒绝覆盖旧运行目录。前置两个已有研究vendor的版本见证据，安装脚本不会替代或更新它们。
