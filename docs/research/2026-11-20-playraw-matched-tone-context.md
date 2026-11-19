# 补齐共同 RGB 后，单一曲线变化未重现实拍肤色偏差

混合光原片与《Cuckoo》缺少共同的肤色 RGB，无法直接比较同色响应。本轮生成 12 个肤色色块和 4 个中性灰色块。两条流程均使用 darktable 4.0.1 中同一项 RGB 色彩平衡操作。第二条额外在编辑前改变曲线、输出后逆转曲线。关态匹配到最多差 1 色阶，开态仍保留两条流程各自的响应。用第一条流程的实际色度变化重放第二条流程，检验单一显影曲线变化能否解释此前的偏色。[事前协议](../../experimental/adaptive-reference-20261113/playraw-matched-tone-context-protocol.md) · [实拍 RGB 覆盖缺口](2026-11-20-playraw-rgb-support.md)

| 预设区域 | 合格色块 | 跨流程色度 MAE 超过 3.00 | 最大 MAE（RGB8） |
| --- | ---: | ---: | ---: |
| 肤色色块 | 12／12 | 0／12 | 3.000 |
| 中性灰色块 | 4／4 | 2／4 | 4.565 |

两遍结果相同。所有合格肤色块在同流程重放时的色度误差均为 0。跨流程肤色色块误差为 0.50 至 3.00，未达到预定的至少 4 块超线门槛。中性灰的 144 和 192 码值色块，关态 RGB 完全一致，但跨流程色度误差分别为 3.57 和 4.57。两块都经过同一项有意覆盖全图的调色，该结果用于说明中性色的响应可受处理流程影响。

停止用这组曲线差异解释《Cuckoo》的肤色失败。当前数据支持的是一个受控结果，未证明实拍偏色由显影曲线造成，也没有建立普通两图下可自动识别的修正依据。合成色卡用于隔离变量，不计入自然照片的独立评估。实际产品版本未升级。

## 复现

设置仓库根目录为 `PYTHONPATH`、`OPENBLAS_NUM_THREADS=2`。在已有视觉环境中，用本机 ASCII 盘符运行 `python experimental/adaptive-reference-20261113/probe_playraw_matched_tone_context.py <研究输出根目录> <新输出目录>`。脚本核对既有原 XMP 和编辑器校验值，生成色卡并完成两遍各四次 TIFF 导出。两遍的表内指标和评分成片哈希相同。从落盘 PNG 用整数亮度权重独立复算每块的关态差、色度误差和投影。报告 SHA256 为 `509fa744b22b51c38dd35b0e50f27e1397b1ce38c5583fbe8a9fed1a3bcddb24`。[本机色卡、对照和成片](../../outputs/adaptive-reference-20261113/playraw-matched-tone-context-v1/0) · [复现脚本](../../experimental/adaptive-reference-20261113/probe_playraw_matched_tone_context.py)
