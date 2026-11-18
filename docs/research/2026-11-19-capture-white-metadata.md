# 这批 RAW 和 JPEG 的拍摄白平衡不足以判断调色方向

检查八张早前已核验的 SyncLight DNG 以及八组 FACD 原图和滤镜版本、九张 COCO 原始 JPEG。`AsShotNeutral` 记录拍摄时选定的白平衡中性色。EXIF `WhiteBalance` 则记录自动或手动模式，不含数值增益。见[事前协议](../../experimental/adaptive-reference-20261113/capture-white-metadata-protocol.md)、[DNG 标签定义](https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml)与 [CIPA EXIF 规范](https://www.cipa.jp/std/documents/e/DC-008-2012_E_C.pdf)。

| 输入 | 数值中性色 | 模式 | 观察 |
| --- | ---: | ---: | --- |
| SyncLight 两场景八张 DNG | 8/8 | 8/8 手动 | 八张中性色都是 `[0.422791, 1, 0.534029]`，四组差向量全为零 |
| FACD 32 张 JPEG | 0/32 | 12/32 | 有字段的三组各自四个版本模式一致，另五组四个版本均缺字段 |
| COCO 9 张 JPEG | 0/9 | 0/9 | 本次未找到两类字段 |

四组 DNG 每组抽样 11,844 个 Bayer 格。只看隔离补光原片，有 6 至 730 个格符合曝光范围；环境原片没有符合该范围的格，因此没有共同合格的近中性代理点。本轮没有可计分的真实像素方向或余弦。字段恒定与八张均为手动白平衡一致。这个对照不能判断其他相机的自动模式会如何随光照变化。见[原片与配对范围](2026-11-18-synclight-real-pixel.md)。

这批素材不支持拿拍摄字段直接选普通两图的换光方向。对于已有滤镜样片，模式标记也没有提供有意调色的变化。保留 RAW 字段作为来源核验；没有修改映色处理或自动赋方向。若进一步验证拍摄元数据，需要真正随灯变化重新测量的自动白平衡配对，并分别核查导出图是否保留数值。另一条独立证据路线是可核验的参考编辑前后配对。

两遍数值文件逐字节一致。独立读取 TIFF IFD0 复核八份中性色数值，并核对既有来源哈希。首次运行名字多拼了一段路径，读取第一组时即退出，修正后按同一方案执行。复现用现有视觉解释器和已安装的 RawPy 隔离目录运行 `probe_capture_white_metadata.py <新结果 JSON>`。见[数值证据](2026-11-19-capture-white-metadata-evidence.json)。
