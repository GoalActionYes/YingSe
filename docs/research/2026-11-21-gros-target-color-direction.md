# 近色参考的颜色方向恢复五项区域风格

上一轮按目标颜色选择参考近邻估力度，部分脸颊仍失准。本轮用相同近邻和 75% 覆盖门槛，只改变颜色方向。取目标成片开关的 RGB8 中位变化，保留它的向量长度，再让方向与供样的编辑差一致。然后用原余弦羽化把二者的差加到开态。关闭态、未覆盖部位及范围外保持原成片，目标预期只用于评分。[事前协议](../../experimental/adaptive-reference-20261113/gros-target-color-direction-protocol.md) · [力度对照](2026-11-21-gros-target-color-gain.md)

60 项风格检查从 24 项通过提高到 29 项，新增 5 项且无旧通过掉项，旧通过区误差均未增加超过 1 RGB8。36 个部位使用方向修正，校正后的成片未出现额外裁切。六个方向整图换灯差仍低于 3 RGB8，完整方向仍为 0／6。[数值、哈希和逐区域通过项](2026-11-21-gros-target-color-direction.json)

- 02 ← 01 冷灯右白幕此前在原方向 0–4 倍范围内可证过不了。本轮改变颜色方向后，误差从上一轮的 4.083 降至 1.346 RGB8，风格投影为 0.916，达到门槛。
- 01 ← 03 冷灯右脸颊原来的不足也在严格不可达列表。换方向后误差由 6.150 降至 2.867，但投影为 0.400。02 ← 01、02 ← 03 的类似冷灯脸颊也降到 3 以下，投影仍偏弱。
- 其余四处新增通过包括 01 ← 02 暖灯右白幕、03 ← 01 暖灯右脸颊及 03 ← 02 冷灯两处脸颊。

## 决策与复现

相同部位的近色供样可以提供比原渲染更合适的颜色方向，这次也确实跨过了一项原倍率上限。另一些脸颊已改善误差但强度仍偏弱，下一轮应在这条有证据的方向上验证与目标颜色相关的幅度，继续保住新增项和原白幕。该诊断要读参考编辑开关与两端人工部位，尚未进入普通两图和独立验收。

沿用[原近色倍率对照](2026-11-21-gros-target-color-gain.md)，从仓库根目录运行，选择尚未使用的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-target-color-gain-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_target_color_direction.py "$base/gros-target-color-direction-repro" $sha
~~~

两遍各生成 12 张成片、6 张对照及 1 份报告，19 个文件逐一核对 SHA-256 一致。另核对符号方向、向量长度不变、无信号回退和羽化。完整分数和成片保留在 outputs/adaptive-reference-20261113/gros-target-color-direction-v1 和同级复跑目录。
