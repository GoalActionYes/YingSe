# PST50 的 41 条视频均已核查封装，原镜头分组仍待确认

原核查有两条待定。`in42` 的 MOV 元数据超过当时的单条预算，`in46` 是扫描器未处理的 MXF。此次冻结原发布与两个编号，用每条不超过 512 KiB 的精确范围读取补齐。读取内容限 MOV 盒头及 `moov`、MXF 头分区包及该封装声明的头部元数据。没有读取视频、音频负载或保留集成片。[运行前协议](../../experimental/adaptive-reference-20261113/pst50-container-completion-protocol.md) · [此前 39 条封装核查](2026-11-19-pst50-video-family-metadata.md)

| 视频 | 本次核实内容 | 来源结论 |
| --- | --- | --- |
| `in42.mov` | `moov` 为 136,714 字节。时间码有一个 34 字节描述，没有扩展源名称 | 有计时信息，缺少可追溯的原镜头标识 |
| `in46.MXF` | 头部元数据为 523,776 字节，含 60 个 KLV 项，包含材料包和源包 | 两包都有 UMID，均无包名称，也无公开原片映射 |

两条合计读取 660,742 字节，逐段核对 206、范围、长度与 Xet 身份。新增两条均无可用于匹配已有来源的名称。41 条里可追溯到文件内非空时间码源标签的仍为 16 条。开发编号 1、3 的保守归组继续有效，未从保留段新增可确认的拍摄家族。原片与参考图片的拍摄组仍需上游项目／镜头清单或同等证据。封装 UMID 标识的是包，不能单凭它确认物理拍摄组。[数值与范围回执](2026-11-19-pst50-container-completion-evidence.json) · [FFmpeg 的 MXF 分区字段](https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/mxfdec.c)

首次取得 `in42` 的 `moov` 后，旧解析器将无扩展的 34 字节时间码描述当成非法。修正最小长度后，两次补查的三份文件逐字节一致。39 份旧 `moov` 重新解析仍与历史处理器及时间码记录匹配。独立检查 `in42` 的时间码长度、读取范围位于媒体载荷之前，及 `in46` 的全部 KLV 边界和两个包的标识哈希。

在现有研究 Python 环境运行 `audit_pst50_container_completion.py <固定 video.json> <旧全量 results.json> <新目录> [--proxy <本机代理>]`。本机结果保存在 `outputs/adaptive-reference-20261113/pst50-container-completion-v2` 和 `pst50-container-completion-repeat-v2`。保留集继续封存，当前没有独立画质或视频验收结论。
