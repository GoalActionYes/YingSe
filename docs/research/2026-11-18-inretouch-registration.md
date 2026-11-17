# INRetouch 四对示例的布局与色彩描述核查

四对前后图在横纵各 ±3 像素的搜索范围内，都以零整数位移的梯度相关最高。不过只有两对达到事前的结构一致性门槛。人像参考和鸭子目标的逐像素误差宜作为描述性结果，不能精确量化单纯由调色变化造成的差异。见 [预先协议](../../experimental/adaptive-reference-20261113/inretouch-registration-protocol.md)和[数值证据](2026-11-18-inretouch-registration-evidence.json)。

| 前后图 | 全图零位梯度相关 | 全图峰值位移 | 结构核查 |
| --- | ---: | --- | --- |
| 红岩参考 Example1 | 0.9624 | 0，0 | 达到门槛 |
| 人像参考 Example2 | 0.7936 | 0，0 | 低于 0.90 |
| 鸭子目标 a0006 | 0.7479 | 0，0 | 低于 0.90 |
| 目标 a0007 | 0.9198 | 0，0 | 达到门槛 |

全图和固定四象限的 20 个峰值均在零位。另构造横向 2、纵向 −1 像素的内部检查，五个搜索区都恢复了预期位移。使用另一种归一化相关计算独立复核 1,225 个位移分数及所有峰值。八张源图均为 RGB、EXIF 显示方向为 1，均未嵌入 ICC profile。相同的元数据不能证明设备色彩空间或相机处理一致。

这次检查支持排除搜索范围内的简单整数平移。相关值较低的原因仍可能包括色调、压缩、局部细节、亚像素或其他几何差异。两张已调色参考导致 13 号生成不同结果这一观察仍成立，因为两份输出来自同一目标输入。前后图拟合误差和与作者目标的距离作为开发诊断保留。没有校正后择优重算、改变旧门槛、运行新候选或接入软件。

复现时从仓库根目录设置 `$env:PYTHONPATH = (Get-Location).Path` 和 `$env:OPENBLAS_NUM_THREADS = '2'`，运行 `python experimental/adaptive-reference-20261113/probe_inretouch_registration.py <原示例目录> <新输出目录>`。本机核查目录为 `outputs/adaptive-reference-20261113/inretouch-registration-v1`。
