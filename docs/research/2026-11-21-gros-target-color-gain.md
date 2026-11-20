# 按目标原色取样未能同时保住冷灯和暖灯肤色

上一轮借用参考部位编辑前后的 RGB8 中位差，恢复多处暖灯脸颊和白幕，却削弱一处原本达标的冷灯脸颊。本轮只改变供样选择。每个目标核心像素在参考同部位寻找八种近似原色，第八近距离必须不超过 12 RGB8 欧氏距离单位。以有覆盖的目标像素重新估计编辑方向所需的倍率。达到 75% 覆盖且既有幅度和方向检查通过才用新倍率，否则复用原值。羽化范围、80% 强度、目标底图和区域门槛不变，目标正确成片在出图后用来评分。[事前协议](../../experimental/adaptive-reference-20261113/gros-target-color-gain-protocol.md) · [原倍率对照](2026-11-21-gros-donor-material-gain.md)

48 个局部条件中，36 个达到供样覆盖，34 个采用估出的新倍率。六组整图换灯差均低于 3 RGB8，而且修改范围外的像素与原倍率结果逐像素一致。60 项风格检查由 27 项通过降至 24 项，改进 1 项、掉项 4 项，六个方向都未完整通过。[逐区倍率和成片哈希](2026-11-21-gros-target-color-gain.json)

| 01 ← 02 右脸颊 | 旧倍率 → 新倍率 | 旧投影 → 新投影 | 结果 |
| --- | ---: | ---: | --- |
| 冷灯参考 | 0.60 → 0.84 | 0.641 → 0.856 | 恢复通过，近色覆盖 100% |
| 暖灯参考 | 2.53 → 3.09 | 1.097 → 1.272 | 失去通过，近色覆盖 99.68% |

风格投影通过范围为 0.75–1.25，MAE 上限为 3 RGB8。暖灯例的变化来自幅度过强，即使参考部位已有近似原色，也未提供可跨拍摄条件通用的倍率。其他掉项是 01 ← 02 冷灯左脸颊、03 ← 01 冷灯右脸颊及 03 ← 02 冷灯左脸颊。上轮通过的两项右白幕保持通过。

## 决策与复现

结束用同部位 RGB8 近邻筛选来修补固定力度。已有供样覆盖仍出现肤色失准，继续调近邻距离不足以解决表现差异。后续更有依据的方向是把已证实无法只靠倍率修复的颜色方向单独建模，并使用独立的实际编辑信息验证。当前近邻输入依赖已知参考开关图和人工登记部位，普通两图软件、独立画质、自然度和视频仍待完成。

首次运行因 SciPy 导入路径缺失，在出图前退出。复跑复用已有 SciPy 1.16.3 路径。准备原倍率对照的全部本地依赖，从仓库根目录运行，选择未占用的输出路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-donor-material-gain-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_target_color_gain.py "$base/gros-target-color-gain-repro" $sha
~~~

两次各生成 12 张成片、6 张对照及 1 份报告，19 个文件逐一核对 SHA-256 一致。另验证近色匹配、远色排除及不同 RGB 数量不足回退。完整逐区域报告及成片留在本机 outputs/adaptive-reference-20261113/gros-target-color-gain-v1 和同级复跑目录。
