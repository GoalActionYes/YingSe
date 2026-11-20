# 暖灯参考的近色残差降低整图误差，两处脸颊仍缺供样

上轮直接用暖灯原始开关图拟合，六个方向完整通过四个，余下两组都使用人物 01 的暖灯参考。本轮复现相同三次表及归档成片，仅加上同一参考开关图中的近色残差。对每种目标 RGB8 查询八种不同的参考原始 RGB，反距离平均它们的实际编辑减拟合编辑差。第八近距离在 3 至 6 时衰减，不足支持时保留原输出。算法对全图使用同一规则，脸颊框仅用于测量。冷灯、关闭态、强度和目标预期保持。[事前协议](../../experimental/adaptive-reference-20261113/gros-paired-palette-residual-protocol.md) · [原暖灯拟合](2026-11-21-gros-paired-raw-warm.md)

| 目标 ← 参考 | 暖灯整图误差，原 → 残差校正 | 右颊八邻覆盖 | 暖灯右颊投影，原 → 残差校正 |
| --- | ---: | ---: | ---: |
| 01 ← 02 | 0.915 → 0.735 | 0.81% | 0.938 → 0.938 |
| 01 ← 03 | 1.090 → 0.935 | 0.00% | 1.069 → 1.069 |
| 02 ← 01 | 0.548 → 0.146 | 0.16% | 1.294 → 1.294 |
| 02 ← 03 | 0.511 → 0.165 | 0.65% | 1.200 → 1.199 |
| 03 ← 01 | 0.662 → 0.312 | 6.91% | 1.353 → 1.348 |
| 03 ← 02 | 0.500 → 0.219 | 9.26% | 1.028 → 1.026 |

误差以 RGB8 MAE 计。余下两处右颊的第八近颜色距离中位数为 11.22 和 9.64，均超过上限 6。校正减小了六组的整图误差，且没有丢掉上轮通过项，也没有出现旧通过区误差增加超过 1 RGB8。整图风格仍为 12／12，区域风格仍为 58／60，完整方向保持 4／6。[逐方向覆盖及 SHA-256](2026-11-21-gros-paired-palette-residual.json)

## 决策与复现

两处失败拥有很低的直接供样覆盖，当前局部修正无法触达足够多的目标像素。它不能测出覆盖充分时原三次表在此区域会不会继续过调。下一轮优先借同一参考人物另一个真实灯位的未校正开关对补色，保持原近色门槛并检查全部旧通过项。配对开关仍为额外输入，普通两图和独立验收尚待完成。

沿用[原暖灯重放](2026-11-21-gros-paired-raw-warm.md)，从仓库根目录运行，选择新路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-paired-raw-warm-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_paired_palette_residual.py "$base/gros-paired-palette-residual-repro" $sha
~~~

两次各保存 6 张成片、6 张对照及 1 份报告，13 个文件逐一核对 SHA-256 一致。复现原三次表的六张暖灯成片以及无支持目标像素均逐像素一致。完整输出留在本机 outputs/adaptive-reference-20261113/gros-paired-palette-residual-v1 与同级复跑目录。
