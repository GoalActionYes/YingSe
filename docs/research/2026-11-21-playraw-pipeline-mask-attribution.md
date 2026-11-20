# 使用相同蒙版后，RAW 与 sRGB 的改色差异仍超过 3 RGB8

此前关闭导入锐化已证明 sRGB 关态可以逐像素保持，但混合光照片的两个高权重区仍超线。本轮读取同一照片的 RAW 全覆盖编辑和关闭锐化的 sRGB 全覆盖编辑，用同一个作者蒙版、80% 强度和亮度保持方法合成。原 RAW 局部开态只用来评分。这样可比较外部套蒙版的近似误差，以及全覆盖操作在两条管线间的响应差异。[事前协议](../../experimental/adaptive-reference-20261113/playraw-pipeline-mask-attribution-protocol.md)

| 评分区 | RAW 合成对原局部成片 | sRGB 合成对原局部成片 | 两种合成的直接差 |
| --- | ---: | ---: | ---: |
| 满权重，621 像素 | 0.41 | 3.05 | 3.06 |
| 高权重留出折 0 | 0.52 | 3.25 | 3.37 |
| 高权重留出折 1 | 0.51 | 3.20 | 3.35 |
| 中权重，3,895 像素 | 0.92 | 1.63 | 1.68 |

均为色度 MAE，单位 RGB8。逐像素有符号残差可以相加，表内 MAE 不能相加。验证了原 sRGB 成片逐像素复现，两种合成在蒙版外均保持原像素。两份已缓存的 RAW 全覆盖版本均满足事前归因标准。此结论限于同一作者操作和一张照片。具体差异仍可能来自工作色彩空间、运算顺序或其他显影模块；这轮没有把它们单独识别。也没有产生普通两图候选。保留结果作为后续选择工作状态或操作数据的依据，固定 sRGB 直放路线维持结束。[本机对照](../../outputs/adaptive-reference-20261113/playraw-pipeline-mask-attribution-v1/0-comparison.png) · [数值、输入与成片哈希](2026-11-21-playraw-pipeline-mask-attribution-evidence.json)

用上一轮的研究 Python 环境及搜索路径运行评分脚本，入参为研究输出根目录及一个新目录，无需调用编辑器或映射盘符。两次各四张成片、两张对照图和报告逐件相同。报告 SHA-256 为 `b34c4481379fe4df3ac794f9faa41c2b3e1af101860fbeF40b116236da83b178`。[脚本](../../experimental/adaptive-reference-20261113/probe_playraw_pipeline_mask_attribution.py)
