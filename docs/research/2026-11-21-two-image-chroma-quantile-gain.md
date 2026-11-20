# 两图的颜色分位匹配放大了真实换灯串色

先前离线用正确成片调节力度后，13 号已有的颜色方向可以覆盖 33 项染色检查。这轮只允许读普通目标、单张参考和由这两张图得到的 13 号成片。先估计成片相对目标的主要对数色度变化方向，再比较三张图沿该方向的 25%、50%、75% 分位位置，用一个 0 至 2 的力度拟合参考与目标的分位差。候选保留每个像素在原成片中的线性亮度及方向，改变色度力度。使用已归档的 80% 13 号成片，与新候选逐项成对比较。实拍／数字编辑标签、参考前图、目标正确成片和区域都不参与单次推理。[事前协议](../../experimental/adaptive-reference-20261113/two-image-chroma-quantile-gain-protocol.md)

## 画面与控制

| 目标状态 | 原 13 号整图换灯差 | 校准后整图换灯差 | 原 13 号数字风格通过区 | 校准后数字风格通过区 |
| --- | ---: | ---: | ---: | ---: |
| MIT 客厅 | 11.24 | 11.25 | 1／4 | 0／4 |
| Gros 人像 | 13.28 | 13.63 | 2／5 | 1／5 |
| WHITE_green 灯 1 | 10.80 | 20.46 | 0／6 | 0／6 |
| WHITE_green 灯 8 | 15.38 | 22.01 | 0／6 | 1／6 |
| BLUE_red 灯 1 | 10.74 | 16.97 | 1／6 | 1／6 |
| BLUE_red 灯 8 | 11.17 | 16.52 | 3／6 | 0／6 |

换灯差单位为 RGB8 MAE，表示使用实拍换灯参考时相对使用原参考的画面变化。所有构图仍无完整通过，原 13 号通过的 7 个数字风格区域均失去通过资格，新增的 3 项来自其他区域。人像右白幕的数字编辑误差从 5.98 降至 2.54，右脸颊却从 1.83 升至 6.44。MIT 客厅数字参考的力度被限为 0，左墙风格投影从 0.699 跌至 0.078。WHITE_green 灯 1 的确认实拍输入却获得 1.978 倍的色度力度。[原参考偏色、换灯差、校准倍率与成片哈希](2026-11-21-two-image-chroma-quantile-gain.json)

同图控制六张均逐像素保持。每张目标自身线性曝光乘 0.65 的六个控制中，额外校准都减少了相对目标的色度变化。另用合成纯色验证已知染色校准、同图和像素数量不足的原样回退。这些控制结果不足以覆盖跨场景实拍和数字编辑反例。

## 决策

结束用整体色度分位匹配来估计力度。它可以让成片的颜色分布靠近参考，却没有区分场景原色、实拍灯色和有意编辑所需的信息。后续若继续自动估计力度，需要先找到可交叉核验的局部证据，再评估在拍摄光色和数字编辑上是否分别表现正确。现有离线力度上限继续保留。软件、独立自然度和视频验收状态没有变动。

## 复现

使用 [MIT](2026-11-21-mid-matched-wall-cast.md) 和 [Beyond RGB](2026-11-20-beyond-rgb-matched-cast.md) 的固定输出及原 13 号本地检查点，选择尚未占用的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_two_image_chroma_quantile_gain.py outputs/adaptive-reference-20261113/two-image-chroma-quantile-gain-repro
~~~

两次各生成 30 张成片、6 张对照及数值报告，37 个文件逐一核对 SHA-256 一致。成片留在本机 outputs/adaptive-reference-20261113/two-image-chroma-quantile-gain-v1 及同级复跑目录。
