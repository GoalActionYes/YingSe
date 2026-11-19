# 扑克镜头的偏色可定位到白桌布

13 号的扑克镜头偏色落在两块白桌布垂面上。对同一实拍源帧的 0.5、1、2 倍曝光版本，两块布料均比原图更偏离作者成片的色度，向内缩 2 像素后仍满足预先设定的误差增加门槛。只保留 13 号亮度、沿用目标原图色度的离线对照，三档均更接近作者成片的桌布色度。作者资料确认这是由 HMI 灯照明的白桌布。[本机选区与成片](../../outputs/adaptive-reference-20261113/hdm-poker-material-roi-v1/comparison.png) · [取样协议](../../experimental/adaptive-reference-20261113/hdm-poker-material-roi-protocol.md) · [拍摄资料，第 3.4 节](https://www.hdm-stuttgart.de/~froehlichj/hdm-hdr-2014/Paper_Jan%20Froehlich_Creating%20Cinematic%20Wide%20Gamut%20HDR-Video%20for%20the%20Evaluation%20of%20Tone%20Mapping%20Operators%20and%20HDR-Displays_SPIE-EI_2014.pdf)

| 1 倍曝光选区 | 像素数 | 原图 | 13 号 | 保持原色度、匹配 13 号亮度 |
| --- | ---: | ---: | ---: | ---: |
| 桌布左垂面 | 303 | 0.00646 | 0.02274 | 0.00643 |
| 桌布右垂面 | 625 | 0.01609 | 0.04096 | 0.01603 |
| 浅色衬衫 | 538 | 0.00868 | 0.01467 | 0.00869 |
| 额头 | 170 | 0.00832 | 0.02717 | 0.00830 |

表中为到作者 PQ 帧在 SDR 裁切前的平均 u′v′ 距离，越小越接近。图上作者列仅用于 SDR 观察。内缩后两块桌布还剩 142／373 像素，三档最小误差增量为 0.01018。浅色衬衫在 2 倍曝光时从 0.00865 降至 0.00814，保留为反例。额头三档的误差均增加，但这里只标了一小块，衬衫和额头也无物理颜色测量。区域仅用于离线测量，现有推理和成片未改。[上轮等亮度方法](2026-11-19-hdm-hdr-poker-chroma.md)

这次把原来的全图近中性色异常定位到了具体材质，并说明这个案例中保留亮度变化仍可减少桌布偏色。后续可将两块桌布作为开发回归项，同时检查候选是否保住其他样例中应当发生的局部冷暖变化。三档曝光源自同一帧，属于一个案例。无需按这张桌布定制规则或改变独立验收。

用既有 NumPy、Pillow 和 OpenCV 环境运行 `probe_hdm_poker_material_roi.py <本机研究输出根目录> <新输出目录>`。两次重测的 JSON 与对照图逐字节一致。用独立的 5×5 滑窗复核了所有内缩区域像素数，并核对色度差值。原片、掩膜对照和图片留本机。[测量脚本](../../experimental/adaptive-reference-20261113/probe_hdm_poker_material_roi.py) · [完整数值与哈希](2026-11-19-hdm-poker-material-roi-evidence.json)
