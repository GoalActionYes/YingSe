# 有参考原片时，单参数能跨灯还原降饱和

上一轮用参考编辑前后图拟合完整 RGB 仿射变换。两份青蓝灯参考成功，两份暖灯参考的颜色不足以稳定估计完整变换。这里保持图像、布局、评分和强度 0.8，只把离线拟合限制为一个灰轴外饱和度增益。拟合器读取参考前后图，没有获得实际编辑系数、目标图或评分用真值。目标真值只用于比较输出。[预定协议](../../experimental/adaptive-reference-20261113/beyond-rgb-saturation-protocol.md) · [完整仿射对照](2026-11-20-beyond-rgb-paired-edit.md)

## 成片与负例

| 指标 | 完整仿射 | 单参数饱和度 |
| --- | ---: | ---: |
| 通过拟合的真实编辑前后图 | 2／4 | 4／4 |
| 跨场景整图合格 | 4／8 | 8／8 |
| 灰阶块合格 | 20／40 | 40／40 |
| 固定目标切换参考灯的最大整图差 | 11.878 RGB8 | 0.0079 RGB8 |
| 拒绝错误曝光配对 | 4／4 | 4／4 |
| 同图保持与错配回退 | 16／16 | 16／16 |

拟合得到的增益为 0.59957–0.59998，接近实际使用的 0.60。参考留出块误差为 0.299–0.317 RGB8，整图成片响应误差最高 0.0124 RGB8。四个固定目标的参考换灯差均低于 1 RGB8。另加四份全局对比度编辑，四份都因超出该拟合器的适用范围而被拒绝。对比度负例的留出亮度误差为 2.29–4.53 RGB8，超过预定 0.6 门槛。照片保持、错配拒绝和对比度拒绝只覆盖此次控制样本。[每次拟合、测量和成片哈希](2026-11-20-beyond-rgb-saturation-evidence.json)

## 结论与接续

在同一张参考有编辑前后图、且编辑属于全局饱和度变化时，完整仿射的暖灯回退可以通过限制待估参数来解决。这个结果不能外推到一般风格，普通两图也没有编辑前参考。结束这两组静物上的参数试验。后续用普通两图可取得的信息做对照，或将此组用作已知编辑的离线监督，必须另以未消费的拍摄家族核验。既有的拍摄元数据核验和固定白平衡先验已有失败记录，避免原样复跑。当前无可交付的自动画质升级。[拍摄元数据核验](2026-11-19-capture-white-metadata.md) · [白平衡先验的失败边界](2026-11-17-deepwb-prior.md)

## 复现

先准备[实拍输入](2026-11-20-beyond-rgb-reference-light.md#复现)和[已知编辑成片](2026-11-20-beyond-rgb-known-edit.md#复现)。从仓库根目录运行，复跑使用另一个输出目录。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_beyond_rgb_saturation.py "$base/beyond-rgb-transfer-headroom-input-v2" "$base/beyond-rgb-transfer-headroom-first-v2" "$base/beyond-rgb-edit-first-v1" "$base/beyond-rgb-saturation-replay"
~~~

单参数诊断复跑的两份报告与 28 张图逐字节相同。原完整仿射脚本用默认参数复跑，数值报告与成片也与前一提交一致。只有实验代码、数值与文档进入 Git。
