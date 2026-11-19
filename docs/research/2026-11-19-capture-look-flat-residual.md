# 平坦块细尺度残差无法稳定识别已知调色

已验证在知晓 Film B 时可以准确重放部分人物成片。本轮单独检查普通参考成片能否提供识别线索。保持旧三态读出、同拍摄变体差分惩罚及 0.65 回退阈值，改为从较平坦图块的细尺度残差提取通道协方差。四个拍摄来源依次整组留出。JPEG 85 检查复用未二次压缩时拟合的模型。[事前协议](../../experimental/adaptive-reference-20261113/capture-look-flat-residual-protocol.md) · [全图统计的历史对照](2026-11-19-capture-invariant-look.md)

| 留出来源 | 旧全图统计正确数 | 细尺度特征，原参考 | 细尺度特征，JPEG 85 |
| --- | ---: | ---: | ---: |
| DAST 人物曝光 | 16/45 | 24/45 | 17/45 |
| MLS 实拍换灯 | 4/12 | 4/12 | 4/12 |
| Middlebury 相机色卡 | 2/6 | 2/6 | 2/6 |
| ARRI 路径实拍 | 3/9 | 3/9 | 3/9 |

DAST 关闭态保持由旧对照的 15/15 变为 14/15，JPEG 后只有 10/15。同款换曝光近中性色最坏差 6.28 → 8.20 RGB8。其余三个来源在新特征下均选择关闭，输出稳定但有意风格未迁移。两种压缩条件都无一组风格通过完整成片门槛，四域联合均为 0/4。DAST 识别数提高没能转化成可靠调色。结束使用这套固定残差特征和读出的路线。通道残差会同时受到材质细节、拍摄、缩放和压缩的影响，本试验无法单独归因。[本机 DAST 原参考成片](../../outputs/adaptive-reference-20261113/capture-look-flat-residual-v1/original/dast) · [JPEG 85 成片](../../outputs/adaptive-reference-20261113/capture-look-flat-residual-v1/jpeg85/dast)

复现使用已有研究环境，将仓库根目录和本机 OpenColorIO 加入 Python 模块路径，设 `OPENBLAS_NUM_THREADS=4`，运行 `python probe_capture_look_flat_residual.py <新输出目录>`。两遍的 144 张成片和报告逐件一致。另检查线性斜坡的滤波结果、全部成片的哈希及八套判定计数。仅验证两个已知风格，素材已用于开发。软件与独立验收仍未改变。[脚本](../../experimental/adaptive-reference-20261113/probe_capture_look_flat_residual.py) · [数值摘要](2026-11-19-capture-look-flat-residual-evidence.json)
