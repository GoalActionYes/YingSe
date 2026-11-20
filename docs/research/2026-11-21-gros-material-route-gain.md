# 部位对应后，增大力度可以救回部分肤色，但仍有方向限制

上轮把参考脸颊和白幕的编辑变化传到目标对应部位，白幕新增两项通过，关键暖灯右脸颊仍偏弱。本轮固定关态、目标作用域和 RGB 编辑方向，只允许每个像素在同一个方向上使用 0 至 4 倍力度。范围外保留原出图。离线读取目标正确成片，求使连续 RGB8 编辑误差最小的力度，然后生成量化成片。[上轮对照](2026-11-21-gros-material-route.md) · [事前协议](../../experimental/adaptive-reference-20261113/gros-material-route-gain-protocol.md)

原门槛为 RGB8 平均绝对误差 ≤3，风格投影 0.75 至 1.25。除生成图片外，还衡量任何窗口内的力度能达到的误差下界与投影区间。两者包括半码取整余量。下界仍超过 3，或投影区间完全落在门槛外时，才确定不可达。最小误差成片的投影不达标，而边界尚未排除时，保留为未证实。

| 目标 ← 参考 | 原区域风格通过 | 离线调力度后 | 有严格不可达界 |
| --- | ---: | ---: | ---: |
| 01 ← 02 | 1／10 | 5／10 | 2／10 |
| 01 ← 03 | 0／10 | 4／10 | 3／10 |
| 02 ← 01 | 0／10 | 3／10 | 5／10 |
| 02 ← 03 | 0／10 | 6／10 | 3／10 |
| 03 ← 01 | 0／10 | 3／10 | 2／10 |
| 03 ← 02 | 2／10 | 6／10 | 0／10 |
| 总计 | 3／60 | 27／60 | 15／60 |

01 ← 02 的暖灯右脸颊误差由 2.721 降为 0.973 RGB8，风格投影由 0.424 升为 0.909，力度中位数为 2.0 倍。上轮新增通过的 03 ← 02 的双灯右白幕均保住，力度中位数约为 1.11 倍；原冷灯右脸颊也仍过线。两张实际成片需要的局部力度不同。

15 项不可达中，9 项同时超出两种边界，3 项仅超误差界，3 项仅超投影界。01 ← 03 的冷灯右脸颊误差下界为 4.368 RGB8，投影最多 0.099。相比之下，01 ← 02 的暖灯左脸颊这次未通过，但区间仍包含原门槛。其余未通过而无严格不可达界的共有 18 项。[区域数值、成片和完整报告哈希](2026-11-21-gros-material-route-gain.json)

## 决策与复现

六组整图换灯差仍低于 3，范围外像素与原部位对应结果逐像素一致，完整方向仍为 0／6。对已有颜色方向足够的区域，值得检验能否用可信的参考差值估计力度。另一些部位或整图必须得到新的颜色方向或支持范围，不能只继续调这个倍数。这里的逐像素力度使用正确成片，仅说明表示能力，普通两图预测、独立画质、真实自然度和视频仍待完成。

准备[原部位对应成片及其依赖](2026-11-21-gros-material-route.md)，从仓库根目录运行，选择新的输出路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-material-route-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_material_route_gain.py "$base/gros-material-route-gain-repro" $sha
~~~

两轮各保存 12 张成片、6 张对照及 1 份报告，19 个文件逐件核对 SHA-256 一致。完整成片留本机 outputs/adaptive-reference-20261113/gros-material-route-gain-v1 和同级复跑目录。
