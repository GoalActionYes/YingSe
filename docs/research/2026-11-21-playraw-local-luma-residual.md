# 局部明暗未解释同一作者暖调的跨照片差异

上一轮数字色卡的关闭态 TIFF 内嵌历史显示，导入时启用了自动锐化。项目已有[关闭锐化对照](2026-11-21-playraw-disable-import-sharpen.md)，确认关态可逐像素复原，但混光高权重区仍超线。[相同蒙版对照](2026-11-21-playraw-pipeline-mask-attribution.md)也发现，差异主要已出现在 RAW 与 sRGB 全覆盖操作的响应之间。这轮检验普通 RGB 图片中的局部明暗能否解释这部分残差。[事前协议](../../experimental/adaptive-reference-20261113/playraw-local-luma-residual-protocol.md)

用已归档混光原片的两套全覆盖成片，拟合 RAW 开关色度差减去 sRGB 开关色度差。每折只拟合一半 8×8 棋盘块，另外一半用来生成留出成片。对照只用每个像素的 RGB 及二次项，主臂额外读取该像素与近邻的亮度差。两个折分别使用 341,820 和 341,939 个合格训练像素，均来自同一幅照片。另将两折系数的均值应用到整个未参与拟合的《Cuckoo》内容。实际输出沿用已知编辑位置、80% 强度、同一色域处理和亮度约束。

| 区域 | 原 sRGB | 只按像素颜色修正 | 额外读取近邻明暗 |
| --- | ---: | ---: | ---: |
| 混光原片高权重区，留出折 0 | 3.248 | 1.436 | 1.446 |
| 混光原片高权重区，留出折 1 | 3.202 | 1.380 | 1.393 |
| 《Cuckoo》脸颊 | 0.567 | 0.912 | 0.872 |
| 《Cuckoo》胸口 | 0.626 | 1.271 | 1.293 |

表中为色度 MAE（RGB8），《Cuckoo》两套缓存数据相同。两个拟合臂在混光同一张照片的留出块上都降到原绝对门槛内。加入邻域明暗后两折的误差均微升，未达到预定的至少 0.25 改善。《Cuckoo》胸口仍在绝对门槛内，但两种修正分别比原 sRGB 恶化 0.645 和 0.666，均超过退化上限 0.5。所有保护范围外均逐像素保持，此项使用事前给定的区域。原 sRGB 成片三份均逐像素复现。[混光留块对照](../../outputs/adaptive-reference-20261113/playraw-local-luma-residual-v1/mixed/comparison.png) · [跨照片退化](../../outputs/adaptive-reference-20261113/playraw-local-luma-residual-v1/cuckoo-0/comparison.png)

结束固定近邻明暗和直接复用全部残差系数。现有对照说明 RGB 本身能解释这一张原片上部分颜色响应，但同一公式在第二张照片会把原本接近预期的肤色改远。留块不是独立内容，两份《Cuckoo》缓存也只代表一张照片。后续若提取真实作者操作，需有可验证的编辑工作状态或更多跨内容的同款对照，不能把一张照片拟合出的残差直接作为产品规则。两图输入的操作识别、正式独立画质、人类评价和视频仍待合格候选。

两遍各九张结果、三张对照及一份报告，13 个文件逐一哈希一致。另用实际 PNG 独立复算 36 项区域误差和投影。沿用[既有 RAW 与 sRGB 对照](2026-11-21-playraw-pipeline-mask-attribution.md)的视觉 Python 和 OpenCV 路径，运行 `experimental/adaptive-reference-20261113/probe_playraw_local_luma_residual.py <研究输出根目录> <新的输出目录>`。无需重新显影或映射盘符。报告 SHA-256 为 `9e8ee71cac96ec3cc8dcf9b0bb8820069b2c3fafbcc846170ee42b3ba0bd9820`。[数值与文件身份](2026-11-21-playraw-local-luma-residual-evidence.json)
