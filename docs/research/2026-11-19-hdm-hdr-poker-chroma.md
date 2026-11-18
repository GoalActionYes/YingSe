# 扑克镜头的偏色在等亮度对照中仍然可见

13 号（80%）让扑克场景变亮，同时给浅色桌布带来更明显的暖色。图中第 2 列沿用原图色度，只把每个像素的亮度调到与第 3 列的 13 号成片相同。三行是同一原片的 0.5、1、2 倍曝光，第 4 列是作者成片的 SDR 预览。诊断图最大亮度量化误差小于 0.004（线性 Y，范围 0–1），测量像素未因制作第 2 列而压缩色度。[本机等亮度对照](../../outputs/adaptive-reference-20261113/hdm-hdr-poker-chroma-v1/comparison.png) · [事前协议](../../experimental/adaptive-reference-20261113/hdm-hdr-equal-luminance-protocol.md)

第一轮把亮度从度量里去掉。用 CIE 1976 u′v′ 检查三档原片共同满足近中性条件的 1,397 个像素，三档平均色度距离均上升。但排除三通道端点后只剩 16 个像素，低于 256 的门槛。原图片和 13 号在这批坐标没有端点，排除的 1,381 个像素全是作者 SDR 图红通道到达 255。因此这一轮没有通过事前的稳定性检查。[CIE 色度定义](https://cie.co.at/eilvterm/17-23-073) · [sRGB 转换](https://www.w3.org/TR/css-color-4/#color-conversion-code)

第二轮直接从原始作者 PQ 成片取显示裁切前的色度，保留同一坐标、原图与 13 号成片。三档的距离仍增加，全部超过事前 0.001 的门槛。量化前的作者参照让差距有所缩小，方向没有翻转。这个镜头里，整图误差下降与近中性色度退步可以同时发生。[裁切前协议](../../experimental/adaptive-reference-20261113/hdm-hdr-preclip-chroma-protocol.md) · [前轮整图测量](2026-11-19-hdm-hdr-display-results.md)

| 原片曝光 | 对作者 SDR 图距离，原图 → 13 号 | 对作者 PQ 裁切前距离，原图 → 13 号 |
| --- | ---: | ---: |
| 0.5 倍 | 0.00723 → 0.04402 | 0.01101 → 0.03851 |
| 1 倍 | 0.00720 → 0.03238 | 0.01100 → 0.02706 |
| 2 倍 | 0.00715 → 0.02338 | 0.01098 → 0.01845 |

作者论文明确说明扑克镜头里的白桌布由 HMI 灯照亮，也确认另一个 `Showgirl 2` 镜头有漫反射白羽毛饰物，人物经历钨丝灯到 HMI 的换光。论文未给这两个场景中白面的绝对亮度。此次像素按原图数值筛选，未对桌布或皮肤单独标注，也没有证实扑克与参考镜头共用调色意图。保留扑克为开发反例，结束对原来两帧的曝光试探。若用这套档案继续验证主体与中性色，优先登记有明确物体和灯光信息的区域，再核对同一材质换光前后的响应。13 号仍是当前体验版，开发反例不参加独立验收。[作者论文，第 3.4 节](https://www.hdm-stuttgart.de/~froehlichj/hdm-hdr-2014/Paper_Jan%20Froehlich_Creating%20Cinematic%20Wide%20Gamut%20HDR-Video%20for%20the%20Evaluation%20of%20Tone%20Mapping%20Operators%20and%20HDR-Displays_SPIE-EI_2014.pdf)

复现时用上一轮已存的准备图、13 号成片目录和原始档案目录，运行 `probe_hdm_hdr_equal_luminance.py <准备目录> <渲染目录> <新输出目录> --archive <原始档案目录>`。Python 环境需有 NumPy、Pillow 和 OpenCV。两次运行的四张图和 JSON 逐文件哈希一致。另核查全部 RGB8 值的编解码往返、D65 色度、整体亮度缩放的色度不变性及限域后亮度保持。媒体留在本机。[复现脚本](../../experimental/adaptive-reference-20261113/probe_hdm_hdr_equal_luminance.py) · [数值与哈希](2026-11-19-hdm-hdr-poker-chroma-evidence.json)
