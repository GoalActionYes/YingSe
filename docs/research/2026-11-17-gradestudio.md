# 直接语义区域迁移：固定筛查失败

GradeStudio 的直接迁移器已在旧开发 64 对运行，未通过第一必要门槛，不接入软件。它不再学习此前失败的局部小头，而是通过内容特征对应构造区域 Lab 迁移和空间 LUT 混合；这种差别足以支持一次固定筛查，但不保证实际风格理解。

| 全强度 RGB8，开发 64 对 | 平均 RGB 误差 | 对现有基线收益 | 改善超过 0.1 | 严重退化 |
| --- | ---: | ---: | ---: | ---: |
| 现有 multiref-v1 | 12.41130 | — | — | — |
| 作者默认全局迁移 | 15.65584 | -26.14% | 14 | 19 |
| 作者默认语义区域迁移 | 15.56879 | -25.44% | 15 | 20 |

要求仍为均值改善至少 5%、至少 40/64 改善、严重退化最多两例，三项均未通过。严重退化指误差同时增加超过 5 和 25%。原图均值 19.18420；两种迁移确实改变画面并改善部分误差，不能解释为完全未工作，但未达到现有模型水平。语义版本比作者全局臂平均仅好 0.556%，不能据此宣称有效升级。

这是已消耗的配对开发集，不是独立自然场景验收。固定方案失败不证明作者方法对一切照片无效；不以该结果替代人类偏好或视频结果，也不调整区域数、温度、强度追分。

## 来源与实际范围

[作者源码](https://github.com/LivDevGit/GradeStudio/tree/5c10ccb685be7b58cdd88ec2f5e78a8975d4139e)固定六区域及默认参数，直接调用 `semantic_match` 和 `transfer_lut`。本机只下载 22 份源码／说明共 184709 字节，没有安装依赖或下载新权重。复用已有严格本地 DINO-small 模型实例，作者 448 长边预处理与迁移公式保持原样；未调用 README 的学生网络，也没有训练。不能把作者自报指标写成本项目测量结果。

该固定 Git 树没有 LICENSE 文件，查询无 Release 资产；这不证明其他渠道没有权重，但不具备将其包装为可分发产品的许可依据。本仓库仅保留自有桥接、协议和哈希，不转载作者源码。

SAGE 本机源码的 `model/README.md` 和 `scripts/download_required_models.sh` 显示需要受限访问的 SD3.5 Medium 及额外 DINO-large、SigLIP2、CleanDIFT 等模型，与[作者说明](https://github.com/chenxib/sage-color/blob/main/model/README.md)一致。本轮没有下载或运行这些模型，不能报告它们的速度、结构保持或效果。

## 核验与复现

`screen_gradestudio.py` 在读取 GT 前保存全部 192 份输出；身份 LUT、输出形状和有限值、非负且归一化的区域权重、首例精确重复均通过。`audit_gradestudio.py` 从原 GT 重算 192 份 PNG 误差、均值和退化数；64 份基线与前一轮零初始化输出逐像素相同。来源源码哈希也重新核对。[数值与哈希](2026-11-17-gradestudio-evidence.json)保留全部 64 对，没有删例。

研究调用语义臂中位约 84.07 ms，仅限本机 256×256 图、已驻留模型；不能当成完整视频吞吐。

在仓库根目录使用已有研究环境，依次运行以下脚本：

```powershell
& '.tools/vision-advisor-env/Scripts/python.exe' experimental/adaptive-reference-20261113/prepare_gradestudio_source.py
& '.tools/vision-advisor-env/Scripts/python.exe' experimental/adaptive-reference-20261113/screen_gradestudio.py
& '.tools/vision-advisor-env/Scripts/python.exe' experimental/adaptive-reference-20261113/audit_gradestudio.py
```

第二步需要本机已核验的原始数据、准备索引、DINO 和基线权重，输出目录必须不存在；脚本会拒绝覆盖已有结果，不应删除旧结果强行重跑。数据、权重、环境、第三方代码和图像不随 Git 发布。

## 后续边界

停止本次默认直接区域方案，稳定 1.3 与第 10 阶段保持。PETAL、SA-LUT 等已有运行结论也不重做。下一路线须先核验不同的联合迁移能力及可运行依据，明确成本和已有实验覆盖，再决定是否执行；不能将大型模型的 README 声称当作能力证据，也不能继续无限叠加旧修正头。独立画质、人类偏好和视频验收仍未完成，完整目标保持未达成。
