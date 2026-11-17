# 两图调色与局部编辑素材核查

接续问题是能否通过已调色参考和目标两张图，复现真实参考原图曾在雕塑样例中提供的收益，并进一步区分目标肤色、白衣和背景。固定核查公开元数据、论文及推理源码。只有能使用目标局部信息、且访问和依赖明确的候选才准备实图；素材需可核验跨内容同款、原图身份及局部处理的依据。未使用 PST50 第 16–50 组。见 [既有两图结果](2026-11-18-inretouch-inferred-original.md)和 [保留样张记录](2026-11-18-pst50-reserve-audit.md)。

## VeraRetouch 两图参考模式

作者给出的两图用法把目标图同时用作待渲染图和参考前图，已调色参考作为参考后图。入口默认选择 `RetouchRenderer_Resnet18Encoder_InputCatMixedCBAM`，配置选择 `ConditionalMLPDecoder` 和 `add` 条件方式。渲染器给整张目标使用一份编码，解码器将目标展开为 RGB 向量，在线性层、逐向量归一化及激活后加相同条件。固定这两张输入时，目标画面中相同 RGB 的像素经过同一映射。此处检查的是发布权重对应的默认路径，源码库包含其他配置。见 [作者两图用法](https://github.com/OpenVeraTeam/VeraRetouch/blob/fefe6c4408f00beae07e132fc404cae0efee0985/README.md)、[推理入口](https://github.com/OpenVeraTeam/VeraRetouch/blob/fefe6c4408f00beae07e132fc404cae0efee0985/infer_ref_retouch.py)、[配置](https://github.com/OpenVeraTeam/VeraRetouch/blob/fefe6c4408f00beae07e132fc404cae0efee0985/configs/renderer_config.py)、[渲染器](https://github.com/OpenVeraTeam/VeraRetouch/blob/fefe6c4408f00beae07e132fc404cae0efee0985/model/retouch_render.py)和 [解码器](https://github.com/OpenVeraTeam/VeraRetouch/blob/fefe6c4408f00beae07e132fc404cae0efee0985/model/colormlp_v2.py)。

编码器能根据图像选全图映射，所查解码器无法对处在不同区域却具有相同 RGB 的两个目标像素做不同调整。这是源码可确认的能力边界，实际整图调色质量未测。权重元数据为 182,507,490 字节，SHA256 为 `62457c0556a29b07d008774adb5fdd95be96fa8bc8539a034c5822afe7568d58`。托管页标记 Apache 2.0，作者许可证另行规定权重及衍生模型限学术研究。结论按作者的明确权重限制处理。当前任务需要目标局部信息，省去下载和接入。见 [权重元数据](https://huggingface.co/Gyh68/VeraRetouch.Encoder_Renderer/blob/main/encoder_renderer.pth)和 [作者许可证](https://github.com/OpenVeraTeam/VeraRetouch/blob/fefe6c4408f00beae07e132fc404cae0efee0985/LICENSE)。

## 局部编辑素材

- iRetouch 的补充材料说明有 500 组真实修图前后对，以及经过人工修订的文字说明，涵盖提亮主体和压暗背景。检查作者代码提交 `c5ed1dafc7af38f4af28138be1b9201fdd82e827` 的发布说明和目录，未核实这 500 组的下载入口、素材许可或不同原始内容之间的同款关系。作者仓库的下载请求针对约 20 万的训练集，不能作为这 500 组的发布状态证据。见 [补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Wu_InstantRetouch_Efficient_and_High-Fidelity_Instruction-Guided_Image_Retouching_with_Bilateral_Space_CVPR_2026_supplemental.pdf)和 [作者源码](https://github.com/OpenImagingLab/InstantRetouch/tree/c5ed1dafc7af38f4af28138be1b9201fdd82e827)。
- CRHD-3K 发布 3,022 对专业处理前后的服装照片，编辑意图是去除褶皱和瑕疵。文件分成 8.2 GB 原图和 9.1 GB 目标两包，限非商业研究。这项目标涉及纹理修整，不能验证白衣中性色和参考调色是否同时保留，无须下载。见 [作者发布说明](https://github.com/youngLBW/CRHD-3K/blob/main/README.md)。

本轮保留现有软件与已消费开发素材的分组，未下载新媒体或权重。继续研究所需的实图证据，是同一个明确调色意图在不同内容上的前后配对，并能核对人物与中性色受到的具体调整。现有独立验收门槛继续适用。
