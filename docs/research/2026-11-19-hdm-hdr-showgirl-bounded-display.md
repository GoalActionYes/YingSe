# 去掉通道裁切后，HDR 白羽毛的编辑方向仍不稳定

沿用 `Showgirl 2` 两个固定时点及脸颊、白羽毛区域，只改变四张输入图的 SDR 显示转换。对原片与作者成片统一映射亮度，超出显示色域时沿中性轴压缩色度。原报表中四个作者目标区域全部有通道达到 RGB8 端点，新图各区端点比例均为 0。[事前协议](../../experimental/adaptive-reference-20261113/hdm-hdr-showgirl-bounded-display-protocol.md) · [本机四图](../../outputs/adaptive-reference-20261113/hdm-showgirl-bounded-prepared-v2/comparison.png)

| 对象 | SDR 与 HDR 裁切前的编辑方向余弦 | 方向检查 |
| --- | ---: | --- |
| 镜中暖光脸颊 | 0.788 | 通过 |
| 镜中暖光羽毛 | 0.986 | 通过 |
| 直拍冷光脸颊 | 0.961 | 通过 |
| 直拍冷光羽毛 | −0.99965 | 失败 |

直拍冷光段的成片羽毛在压缩中保留的线性色度系数平均仅 0.324，方向仍几乎完全反转。亮度与端点合格不足以保证编辑信号可信。四区必须全部通过，故在输入检查处停止，未生成新的 13 号成片。此前在自定显示下测得的脸颊反转、羽毛灯光差距缩小仍留作开发警示，其对显示转换的敏感性尚未解除。结束这四帧的显示修正及共享风格监督探索。[上一轮实际成片](2026-11-19-hdm-hdr-showgirl-reference.md) · [数值、输入和输出哈希](2026-11-19-hdm-hdr-showgirl-bounded-evidence.json)

## 原生 SDR 开发对照

另核查已配准的 FACD 七张照片与三款作者滤镜。冻结近中性色代理区域后，七组均有至少 256 像素可同时用于原片和三种滤镜的测量，保留率为 99.59%–100%。婚纱样例 9,999 个像素全部可用，最小组 270 个像素全部可用。接下来的候选对照可用这七组核对是否保留有意的调色，再用现有白球服换背景反例核对是否减轻无意串色。这组原生 SDR 检查确认测量覆盖，沿用原先完整结果，不作为新画质成绩，七组仍是已消费的开发样本。[端点协议](../../experimental/adaptive-reference-20261113/facd-endpoint-support-protocol.md) · [覆盖数值](2026-11-19-facd-endpoint-support-evidence.json) · [七组历史成片](2026-11-18-facd-two-image-baseline.md) · [白球服背景对照](2026-11-18-target-hue-retention.md)

复现 HDR 检查，在上轮 `prepare` 命令末尾加 `--bounded-display`。两遍四张输入 PNG、标注板与 JSON 哈希一致；默认转换的四张图和清单在旧、新脚本间也逐文件一致。另核查单调亮度、全部八个输入区域端点及失败时拦截渲染。FACD 在原有研究环境运行 `probe_facd_endpoint_support.py <配准目录> <配准证据 JSON> <新输出 JSON>`，两遍摘要字节一致。无产品改动或媒体上传。[HDR 脚本](../../experimental/adaptive-reference-20261113/probe_hdm_hdr_showgirl_reference.py) · [SDR 核查脚本](../../experimental/adaptive-reference-20261113/probe_facd_endpoint_support.py)
