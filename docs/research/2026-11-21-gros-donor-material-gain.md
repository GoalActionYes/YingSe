# 用参考部位的实测编辑差估计目标力度

前一轮离线使用正确成片调节力度，恢复了白幕和部分脸颊。这轮去掉力度计算对目标答案的依赖。用同一参考人物的 Film B 开关图，测量对应部位的 RGB8 编辑中位差，按 80% 强度投影到目标部位的现有编辑方向，估计 0 至 4 倍力度。信号太小或方向相反时保持原力度。取用上一轮相同羽化范围，范围外逐像素保留，目标正确成片在新结果生成后才用于评分。[事前协议](../../experimental/adaptive-reference-20261113/gros-donor-material-gain-protocol.md)

| 目标 ← 参考 | 原区域风格通过 | 按参考部位估计后 |
| --- | ---: | ---: |
| 01 ← 02 | 1／10 | 5／10 |
| 01 ← 03 | 0／10 | 5／10 |
| 02 ← 01 | 0／10 | 2／10 |
| 02 ← 03 | 0／10 | 5／10 |
| 03 ← 01 | 0／10 | 4／10 |
| 03 ← 02 | 2／10 | 6／10 |
| 总计 | 3／60 | 27／60 |

新通过 25 项，原通过的 3 项中保住 2 项。六个方向的整图换灯差均低于 3 RGB8，完整方向仍为 0／6。另一个离线方法也通过 27 项，两者共同通过 22 项、各有 5 项只由其中一种方法通过。前一轮求的是逐像素 RGB8 误差最低的力度，风格方向为独立门槛。因此它的实际通关数并非所有候选的上限。[估计倍率、逐项分数与成片哈希](2026-11-21-gros-donor-material-gain.json)

01 ← 02 暖灯右脸颊估计为 2.53 倍，风格误差从 2.721 降到 1.560，投影为 1.097，通过门槛。冷灯右脸颊却估计为 0.60 倍，投影降到 0.641，失去原有通过资格。原本过线的两张右白幕继续通过，误差分别从 2.080 降至 1.569、从 1.978 降至 1.257 RGB8。

## 决策与复现

参考开关差提供了可用力度证据，并在本轮不读取目标答案的条件下恢复了多处肤色和白幕。但同一个固定规则仍会在两张人物照片的编辑幅度不同处误判，结束将参考中位差直接作为目标幅度的规则。后续需要验证目标材质和颜色对幅度的调制，同时保持原有照明和白幕门槛。当前可复现部分仍依赖参考开关对以及两端人工部位，未达到普通两图、独立画质、真实自然度或视频要求。

沿用[已登记部位的参考归档](2026-11-21-gros-material-route.md)及上游环境，在仓库根目录选择新的输出路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-material-route-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_donor_material_gain.py "$base/gros-donor-material-gain-repro" $sha
~~~

两轮各保存 12 张成片、6 张对照和 1 份数值报告，19 份文件 SHA-256 逐一一致。另核对已知倍率、上限、相反或零变化回退及羽化范围。完整成片和报告留在本机 outputs/adaptive-reference-20261113/gros-donor-material-gain-v1 和同级复跑目录。
