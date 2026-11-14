# 现成参考调色方法的部署与表示边界核查

本轮没有启动训练、下载大模型、运行外部模型或上传图片。核查用于决定哪些路线值得继续试，不是新的效果评测。现有13号模型不因此更换。

## NeuralPreset：局部计算不等于整条链可离线

作者[论文](https://arxiv.org/abs/2303.13511)采用先归一化、后风格化的两阶段颜色映射；[仓库](https://github.com/ZHKKKe/NeuralPreset)主要公开评价实现，图片入口指向网页演示，视频演示标为TBD。

进一步读取作者网页公开的HTML、Webpack清单、主包及编辑器延迟包，共4份静态资源。编辑器代码的参数提取链为：缩小图像→`/npreset/upload`→`/npreset/process`→`/npreset/fetch`→取得预设参数；像素应用链加载dNCM／rNCM TensorFlow图，在浏览器分块执行。两个环节不能混为「完整模型都在浏览器」。[公开演示资源](https://zhkkke.github.io/NeuralPreset/static/js/0.a0c3843b41ad628bc3a0.js)

仅对公开模型元数据地址做了一次GET，在本机TLS握手失败，没有提交图像。这不能证明接口在所有地点不可用，也不能证明权重永远不公开。当前取得的资源没有提供完整的离线预设提取编码器，因此不把它列为已能离线运行的替代引擎。没有尝试绕过权限或恢复服务端代码。

本轮使用REA技能指导调查，但活动会话没有REA工具或现成CLI；实际证据来自普通HTML／JavaScript静态检查，没有生成REA Evidence，没有执行外部JavaScript或观察实际网络上传。报告不将其写成原生反编译或运行实证。

## VideoColorGrading：生成LUT仍受全局映射约束

[作者仓库](https://github.com/seunghyuns98/VideoColorGrading)提供扩散模型生成颜色LUT的路径。核查[实际推理代码](https://github.com/seunghyuns98/VideoColorGrading/blob/main/grading.py)及[管线](https://github.com/seunghyuns98/VideoColorGrading/blob/main/pipeline.py)：生成16级3D LUT并加恒等表，然后用同一颜色表逐帧应用。该机制可能改善映射估计，但最终单一全局LUT不能让同一RGB仅因人物／背景身份不同而输出不同颜色；这是表示能力边界，不是画质结论。

其[默认配置](https://github.com/seunghyuns98/VideoColorGrading/blob/main/configs/prompts/video_demo.yaml)需要SD1.5、CLIP、GS-Extractor及L-Diffuser等资产；本机当前研究环境缺少diffusers、einops、cv2。没有运行安装脚本，也没有执行模型或估算未经实测的显存／速度。源码存在不等于已具备本机部署条件，论文展示不等于我们的素材已经通过。

## 其他筛选信息与下一步

[ICTone作者说明](https://github.com/dengyuhai/ICTone)使用FLUX-Fill及LoRA，把两图作为联合生成上下文；它代表另一种信息利用方式，不是直接复用目前轻量网格的权重。当前未在本机测量其细节保持、速度、显存或视频时序，不据此宣布能够达到项目要求。

后续判断应同时检查：是否取得完整离线推理链；参考信息是否保留所需关系；输出表示能否表达区域差异；素材结构是否保持；实际运行是否足够快。不能为了避免训练而接入只有在线编码器的演示，也不能因为采用更大模型就忽略性能和结构风险。

原静态资产及8份文件的URL／字节数／SHA256保留在本机 `.tools/NeuralPreset-demo`；[公开数值证据](2026-11-15-pretrained-route-audit-evidence.json)仅保留清单、位置和限制，不分发作者完整程序。
