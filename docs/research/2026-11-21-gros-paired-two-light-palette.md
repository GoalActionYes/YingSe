# 同一参考加入实际冷灯供样后，六个跨人物方向均通过受控门槛

上一轮只有暖灯参考开关可供校正，人物 01 作参考的两处目标右脸颊缺近色样本。本轮只向近色残差表加入同一参考人物的真实冷灯开关图。暖灯三次表、八种不同颜色、距离上限 6、权重衰减、强度和目标原图均保持原设置。目标答案及两端测量区域只用于评分。先复现暖灯样本单独的原输出与分数，再比较加样。三位人物的有效参考色数分别从 24,876、23,556、26,400 增到 45,375、41,834、47,595。[事前协议](../../experimental/adaptive-reference-20261113/gros-paired-two-light-palette-protocol.md) · [单灯对照](2026-11-21-gros-paired-palette-residual.md)

| 目标 ← 参考 | 暖灯右脸颊覆盖，原 → 补样 | 暖灯右脸颊投影，原 → 补样 | 暖灯整图风格 MAE，原 → 补样 |
| --- | ---: | ---: | ---: |
| 01 ← 02 | 0.81% → 76.79% | 0.938 → 0.972 | 0.735 → 0.706 |
| 01 ← 03 | 0.00% → 94.16% | 1.069 → 1.013 | 0.935 → 0.870 |
| 02 ← 01 | 0.16% → 99.68% | 1.294 → 1.017 | 0.146 → 0.049 |
| 02 ← 03 | 0.65% → 100.00% | 1.199 → 1.001 | 0.165 → 0.055 |
| 03 ← 01 | 6.91% → 100.00% | 1.348 → 1.000 | 0.312 → 0.173 |
| 03 ← 02 | 9.26% → 99.51% | 1.026 → 0.990 | 0.219 → 0.151 |

MAE 单位为 RGB8。补样后风格由 58／60 升至 60／60，完整方向由 4／6 升至 6／6。六组整图换灯差全部小于 3，整图关闭态均逐像素保持原图。相对单灯修正和此前较早的局部结果，所有旧通过项目均保住，旧通过区没有超过 1 RGB8 的误差增幅。两处失败在原色样本覆盖后恢复，支持供样不足是这两处失效的关键限制。[覆盖、全量风格摘要与哈希](2026-11-21-gros-paired-two-light-palette.json)

## 决策与复现

冻结为已知参考编辑开关条件下的正对照。六个方向来自三位已用开发人物，并依赖同一参考在两种实拍灯下分别提供编辑前后图。接下来应验证普通目标图与单张参考图提供的线索是否足够接近该结果，不继续堆已知开关样本来替代产品目标。稳定版功能、独立画质、真实自然度及视频验收仍待推进。

使用[既有参考开关和单灯成片](2026-11-21-gros-paired-palette-residual.md)，从仓库根目录选择未占用的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-paired-palette-residual-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_paired_two_light_palette.py "$base/gros-paired-two-light-palette-repro" $sha
~~~

两次各 6 张成片、6 张对照和 1 份报告，13 个文件逐一核对 SHA-256 一致。单灯残差成片、单灯评分及新输出中无近色支持的像素已逐项复核。本机完整结果位于 outputs/adaptive-reference-20261113/gros-paired-two-light-palette-v1 及同级复跑目录。
