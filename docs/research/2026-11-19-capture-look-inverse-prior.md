# 能撤销已知风格，仍无法从单张参考选出该撤销的风格

固定两款候选风格及「未调色」解释。对单张参考尝试 Film A／B 的近似逆，用其它拍摄来源的未调色统计先验和逆重放误差评分。原来的四来源逐组留出、72 态及原成片门槛保持。未使用配对原图作预测。[事前协议](../../experimental/adaptive-reference-20261113/capture-look-inverse-prior-protocol.md) · [历史闭集对照](2026-11-19-capture-invariant-look.md)

| 来源 | 旧固定统计正确数 | 逆向先验正确数 | 已知正确操作时还原原片的平均 RGB8 MAE |
| --- | ---: | ---: | ---: |
| DAST 人物曝光 | 16/45 | 15/45 | 1.23 |
| MLS 实拍换灯 | 4/12 | 4/12 | 1.03 |
| Middlebury 相机色卡 | 2/6 | 2/6 | 1.28 |
| ARRI 路径实拍 | 3/9 | 3/9 | 1.34 |

48 个有意编辑的参考用已知正确的逆向时，全部变得更接近自己的原片，且逆重放误差都低于 2 RGB8。未知风格时，72 项中「未调色」的总代价全部最低，所有目标均保持原图。有意编辑全部漏判，整套为 0/4 域。作为定位失败的额外检查，从已有评分中扣除正确候选的重放误差项后，只有 22/48 个正确逆解释的统计先验代价低于「未调色」。该核查未重新选风格或生成候选。结束固定逆向先验路线，保留成片。已知风格可逆只是重放能力，不能用来推定普通参考的编辑历史。[本机成片](../../outputs/adaptive-reference-20261113/capture-look-inverse-prior-v1)

在既有研究环境中将仓库根目录、本机 OpenColorIO 和 SciPy 加入 Python 模块路径，设 `OPENBLAS_NUM_THREADS=4`，运行 `python probe_capture_look_inverse_prior.py <新输出目录>`。两遍的 72 张成片与报告逐件一致，另核对全部输出哈希、代价排序和 48 项正确逆向的误差。本轮闭集操作有限，照片已用于开发，未改软件或打开独立素材。[脚本](../../experimental/adaptive-reference-20261113/probe_capture_look_inverse_prior.py) · [数值摘要](2026-11-19-capture-look-inverse-prior-evidence.json)
