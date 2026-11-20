# 正确部位对应让白幕风格通过，肤色力度仍不足

核对冻结的 13 号入口后，参考图只通过整图颜色直方图参与网格预测，参考部位的位置没有进入模型。已有水平翻转实验验证过这一点。这次使用新的参考信息，明确把参考左脸颊、右脸颊和两侧白幕与目标对应部位关联。[原参考位置诊断](2026-11-21-reference-mirror-chroma.md) · [事前协议](../../experimental/adaptive-reference-20261113/gros-material-route-protocol.md)

保持六个 Gros 跨人物方向、冷暖灯、Film B 开关和 80% 强度。冷灯用归档原参考，暖灯用上轮已限域校正的参考。目标关闭态沿用整图参考的成片。每处参考部位单独裁取开关图，用同一 13 号渲染目标，取两者差值替代对应目标部位的整图编辑差。周围按预设余弦权重过渡，范围外保留整图结果。目标答案仅参与评分。已知同一参考的开关对、参考冷暖校正和两端人工部位是诊断输入。

## 结果

- 60 项区域风格检查中，整图参考通过 1 项，提供部位对应后通过 3 项。新增两项均为 03 ← 02 的右白幕，冷灯风格误差从 3.813 降至 2.080，暖灯从 3.186 降至 1.978 RGB8，投影由 0.659、0.715 升至 0.810、0.816。原通过的冷灯右脸颊也保住。
- 01 ← 02 暖灯右脸颊的风格误差从 3.747 降至 2.721，但投影由 1.084 降至 0.424，低于 0.75 门槛。相对预期的差距缩小伴随风格力度不足。
- 六个方向的整图换灯差均低于 3 RGB8。完整方向数两臂均为 0／6。两灯的局部修改范围外均逐像素保持原整图参考结果。[全量分数和文件哈希](2026-11-21-gros-material-route.json)

## 决定与复现

正确的部位对应能带来白幕收益，但裁取参考再直接套用 13 号并不能提供足够的肤色风格幅度。下一项要区分力度不足来自裁取后的小区域颜色描述，还是这类颜色变化超出当前渲染方向。两步都应保留本轮白幕、换灯和范围外检查。继续保持为开发诊断，尚无普通两图候选或独立验收。

沿用[上轮限域参考及其依赖](2026-11-21-gros-bounded-material-field.md)，从仓库根目录选择新输出路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-bounded-material-field-v3/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_material_route.py "$base/gros-material-route-repro" $sha
~~~

两轮各生成 12 张成片、6 张对照和 1 份报告，19 个文件的 SHA-256 逐一相同。另用手算验证范围外、全权重、重叠和羽化四种合成情形。完整成片位于本机 outputs/adaptive-reference-20261113/gros-material-route-v1 和同级复跑目录。
