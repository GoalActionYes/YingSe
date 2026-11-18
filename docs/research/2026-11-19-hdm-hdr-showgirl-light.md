# 同一人物换光时，肤色和白色羽毛同时改变

在 HdM-HDR-2014 的 `Showgirl 2` 镜头选定两个时点。作者记录同一位演员从钨丝灯走到明亮 HMI 灯下，白色羽毛饰物用于提供漫反射白面。档案两条序列均为连续 341 帧，按事前固定的 68、272 号零起始序号，各取原片与作者成片，共 37,870,886 字节。两对亮度梯度零位相关为 0.697、0.780，峰值均在零位，通过匹配门槛。仍按序号认定帧关系，未取得原时间码。[作者论文，第 3.4 节](https://www.hdm-stuttgart.de/~froehlichj/hdm-hdr-2014/Paper_Jan%20Froehlich_Creating%20Cinematic%20Wide%20Gamut%20HDR-Video%20for%20the%20Evaluation%20of%20Tone%20Mapping%20Operators%20and%20HDR-Displays_SPIE-EI_2014.pdf) · [事前协议](../../experimental/adaptive-reference-20261113/hdm-hdr-showgirl-light-protocol.md)

只看原片缩图，先登记左侧下脸颊和右肩白羽毛的内部区域。第一帧两块区域为 48、56 像素，第二帧为 84、132 像素。换光同时伴随起身和转身，由镜中倒影变为直接拍摄。区域对应同一侧脸颊及服饰部位，单根羽毛、姿态和反射路径均无法锁定。显示缩图用于查看位置，以下 CIE 1976 u′v′ 距离在显示裁切前计算。[本机标注对照](../../outputs/adaptive-reference-20261113/hdm-showgirl-light-result-v1/comparison.png) · [测量区域](../../experimental/adaptive-reference-20261113/hdm-hdr-showgirl-light-rois.json) · [CIE 色度定义](https://cie.co.at/eilvterm/17-23-073)

| 对象 | 原片换光前后色度距离 | 作者成片两时点的色度距离 |
| --- | ---: | ---: |
| 左侧下脸颊 | 0.06709 | 0.07201 |
| 右肩白羽毛 | 0.09413 | 0.08460 |

白羽毛通过原片色度变化大于 0.01 的预设门槛。两个区域在两类图像中均向较低的 u′、v′ 移动，白羽毛的色度位移大于脸颊。亮度也变了：同一原片序列内，白羽毛区域线性 Y 中位数从 2.47 升至 27.59，作者 PQ 成片相应区域由 1762 降至 597 cd/m²。原片的线性值缺绝对光度标定，这两组量纲只在各自序列内比较。

这组素材提供了一个有明确材质和照明记录的开发对照。下一轮可固定同一目标与强度，仅把参考由同一帧原片换为作者成片，测量软件能否对真实编辑作出响应，同时保留目标画面原有的灯光差异。两帧同时改变了灯光、构图和反射路径，不能据此估计光源本身的色温，也不能直接作为独立画质验收。此次未运行或更换 13 号模型。

复现用现有取样脚本的 `--showgirl-light` 选项，随后运行 `probe_hdm_hdr_showgirl_light.py <本机档案目录> <ROI JSON> <新输出目录>`。复测两次的 PNG 和 JSON 逐文件哈希一致。媒体遵照作者许可留作本机学术研究。[测量脚本](../../experimental/adaptive-reference-20261113/probe_hdm_hdr_showgirl_light.py) · [来源哈希和测量结果](2026-11-19-hdm-hdr-showgirl-light-evidence.json)
