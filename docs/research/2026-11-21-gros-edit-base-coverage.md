# 恢复目标底色后，局部范围外的偏差仍足以让整图失败

上轮按部位校准颜色方向和力度，34／60 项风格通过，整图关闭态和风格均超线。本轮将两项限制拆开。先以目标原图替代存档关闭态，把原开态减关态作为编辑增量加回。再做一个诊断对照，在原羽化范围内直接换为已知正确成片，范围外保持重放后的像素。两臂沿用相同风格、两灯、六个跨人物方向与原投影、误差及换灯门槛。[事前协议](../../experimental/adaptive-reference-20261113/gros-edit-base-coverage-protocol.md) · [上轮成片](2026-11-21-gros-directed-chroma-strength.md)

将编辑增量放回目标原图后，12 个关态都逐像素保持原色，10 个开态到预期目标的整图误差下降，2 个上升。重放裁切最多占 0.040% 像素，区域风格仍为 34／60，六组整图换灯差低于 3 RGB8。12 项整图风格全部超出误差门槛。

| 目标 ← 参考 | 在目标原图重放后的整图误差，冷／暖 | 局部完全正确时的整图误差下界 |
| --- | ---: | ---: |
| 01 ← 02 | 5.153／5.893 | 4.939／5.627 |
| 01 ← 03 | 5.494／6.235 | 5.284／5.968 |
| 02 ← 01 | 3.246／3.827 | 3.082／3.585 |
| 02 ← 03 | 4.701／4.844 | 4.488／4.606 |
| 03 ← 01 | 4.850／4.315 | 4.629／4.069 |
| 03 ← 02 | 3.284／3.419 | 3.119／3.239 |

单位为 RGB8 MAE。局部范围占目标的 7.81%–8.25%。将范围外误差总量分摊到全图，就得到范围内完全正确时的整图误差下界。理想对照四处局部区域均过，合计 48／60，但 12 项整图下界均超过 3，完整方向仍是 0／6。硬边界替换只用于验证下界，边缘自然度未评估。[逐项下界、输出与报告哈希](2026-11-21-gros-edit-base-coverage.json)

## 决策与复现

结束在相同脸颊和白幕范围内追整图达标的尝试。恢复目标底色和重建整图风格都是必要条件。下一轮优先检验范围外的全图编辑表达，保留已取得的局部收益。重放使用参考前后开关及原部位输入，理想对照直接读取目标答案。普通两图候选、独立画质、真实自然度与视频仍未验收。

沿用[原力度成片及其上游](2026-11-21-gros-directed-chroma-strength.md)，从仓库根目录选择新的输出路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-directed-chroma-strength-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_edit_base_coverage.py "$base/gros-edit-base-coverage-repro" $sha
~~~

两轮各生成 24 张成片、6 张对照及 1 份报告，31 个文件逐一核对 SHA-256 一致。另验证了编辑增量的正负符号、色域边界和零编辑恒等。完整成片留本机 outputs/adaptive-reference-20261113/gros-edit-base-coverage-v1 和同级复跑目录。
