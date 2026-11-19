# 映色在实拍上未充分区分两种已知胶片风格

在同一个公开 LogC3 创意库中加入 `3111 Film B`，与已有的 `3110 Film A` 并列。用完全相同的色域和 SDR 视图处理扑克、实体色卡和焊工原片。三个原片的 A／B 成片差分别为 3.24、1.87、2.53 RGB8，均超过预设最低差，各自零位配准成功。这些样例现在既能检查有无风格，也能检查选择不同风格时是否传对方向。[运行前协议](../../experimental/adaptive-reference-20261113/arri-two-look-training-probe-protocol.md) · [ARRI 官方风格库](https://www.arri.com/en/software/previs-software/image-comparison-tool-looks)

固定目标原图，只在另一拍摄来源的三张对应参考之间切换，生成九张 13 号 80% 成片。

| 目标 ← 参考 | A 到 B 整图投影 | 共有近中性色投影 | 关风格参考的整图平均变化 |
| --- | ---: | ---: | ---: |
| 扑克 ← 色卡 | 0.194 | 0.263 | 7.86 RGB8 |
| 色卡 ← 焊工 | 0.308 | 0.350 | 27.73 RGB8 |
| 焊工 ← 扑克 | 0.184 | −0.127 | 23.08 RGB8 |

投影 1 表示完整复现已知 A 到 B 变化的幅度。整图和共有近中性色的三组比较均未同时满足预设的 0.75–1.25 与响应误差要求。焊工近中性色向相反方向变化。色卡原固定左脸颊的投影为 0.016，右脸颊为 0.820，后者通过增量门槛，但关风格时已存在 18.01 RGB8 的区域平均变化。此处色卡额外报告的是 RGB 差，不能和早先只度量色度的脸颊码值直接比较。[实体色卡三列对照](../../outputs/adaptive-reference-20261113/arri-two-looks-v2/hdr_testimage-comparison.png) · [焊工三列对照](../../outputs/adaptive-reference-20261113/arri-two-looks-v2/sparks_4592-comparison.png)

Film B 只读取发布 ZIP 的目录和单一成员，共 377,071 字节，核对稳定的 ETag、范围、CRC 与 SHA256。首次运行误指向早先截断的焊工下载，从同一帧的既有完整重试副本按原 SHA256 恢复，没有换帧。两遍 25 份制备图、成片、面板及报告逐字节一致，另独立复核三处风格差及六项 A／B 响应。[数值和哈希](2026-11-19-arri-two-looks-evidence.json)

三组实拍适合作为风格方向的开发控制。它们是对已知 LUT 作固定渲染的效果，整套只有三个源，且存在跨目标／参考复用。只凭这批数据不能拟合或评价开放风格，也不能验证背景、曝光和已知风格已经实现独立分离。本轮未训练或接入新软件。用已有环境运行 `fetch_arri_shared_look.py <新目录> --second-film`，将本机 OpenCV 和 OpenColorIO 路径加入 `PYTHONPATH` 后执行 `probe_arri_two_looks.py <本机研究输出根目录> <新输出目录>`。LUT 和媒体仅本机留存。[脚本](../../experimental/adaptive-reference-20261113/probe_arri_two_looks.py)
