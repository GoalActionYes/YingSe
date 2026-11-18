# 局部调色监督来源与公开访问核对

白球服在给定编辑范围时可以保住人物颜色并给背景加冷色。当前缺的是可用于训练或开发检查的样本，既能标明编辑发生在哪里，也能证明同样的编辑意图如何作用于不同拍摄内容。本轮核对四条新线索的官方论文、源码树和访客可见元数据。见[核查规则](../../experimental/adaptive-reference-20261113/local-edit-source-access-protocol.md)及[发布身份与响应哈希](2026-11-18-local-edit-source-access-evidence.json)。

- **Adobe Artist-Retouched / PIH。** 论文和补充材料记录了 46,173 对人工修图前后图，以及每对图的前景蒙版。前景蒙版用来构造只保留前景或背景调色的合成图，无法确定原编辑中每笔操作的范围或跨场景同款标识。固定源码树的 51 个文件和公开 Releases 均未提供这批素材的下载入口。源码说明给出了准备目录和预训练模型入口。Apache-2.0 为代码仓库标识，素材授权仍待核实。见[论文补充材料](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wang_Semi-Supervised_Parametric_Real-World_CVPR_2023_supplemental.pdf)和[固定源码说明](https://github.com/adobe/PIH/blob/2823cccf0778c6ea213a3d366f03864ac8ab82e6/README.md)。
- **NTIRE 2026 摄影调色迁移。** 官方访客 API 可读，数据页提供开发测试、最终自动评价和最终主观评价三个云盘链接。规则说明挑战数据仅供挑战范围内的开放研究和教育，验证与最终测试不向参赛者提供目标答案。挑战算法依赖参考编辑前后两张图，再处理第三张目标。MetaDC-INR 固定源码树的公开样例列出了三张输入及该方法输出，缺少目标作者结果和编辑范围。此元数据不能替代素材授权，也不能支持正常两图成绩。见[官方访客元数据](https://www.codabench.org/api/competitions/12917/)、[挑战报告](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/html/Elezabi_Photography_Retouching_Transfer_NTIRE_2026_Challenge_Report_CVPRW_2026_paper.html)和[公开样例清单](https://github.com/birdortyedi/MetaDC-INR-retouching-transfer/tree/f9692d7cf83fc1e479a4bd72f5151a573765de8c/data/eval/sample107)。
- **MirrorPPR47M。** 固定版本的公开目录提供专业及模拟两部分，API 未列数据许可。论文所列专业部分 27 类操作涵盖五官、脸型和身体比例，可用于结构修饰研究。本轮需要的是肤色、中性色和背景的色彩编辑。见[作者论文](https://arxiv.org/html/2606.29308v1)和[固定数据版本](https://huggingface.co/datasets/SJTU-DENG-Lab/MirrorPPR47M/tree/178d715c4161ad30b0c983d863df763a50773a1e)。
- **GradeStudio。** 作者说明的两图模型使用语义匹配结果作为教师，49,000 对训练样本主要由内容邻居和固定配方构成。公开文档报告的参考匹配结果衡量与老师输出的差异。固定树未提供这批训练图，源码说明明确未授予使用、复制、修改或分发许可。保留为了解当前方法的线索。见[固定源码说明及许可](https://github.com/LivDevGit/GradeStudio/blob/5c10ccb685be7b58cdd88ec2f5e78a8975d4139e/README.md)。

直接核查保存 23 份文本或元数据响应，共 820,535 字节。逐一核对四份 README 的 Git blob、源码提交与树的关系、目录完整性及访客 API 页字段。初次 Hugging Face 连接超时，后来通过项目已有网络设置和字段筛选读取所需元数据。完整 API 请求超过该次读取上限，改用指定字段后通过。没有读取媒体或提交任何参赛资料。原响应保存在本机 `outputs/adaptive-reference-20261113/local-edit-source-access-v1`。

本轮没有符合条件的新训练来源。下一批局部监督应优先取得**可确认使用范围的两个以上拍摄内容，分别保存原图、同一全局调色版本，以及开启和关闭同一背景编辑的成片**。编辑范围或参数用于离线监督及评分，生产推理仍只读正常两张输入。单源仅有说明、主体蒙版或单张好图时，仍可作局部诊断，不能替代同款配对。已有背景物体色干预继续用作负例。课程交付保持现有软件入口，自动画质升级尚未通过。
