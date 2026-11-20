# 普通两图直接回归编辑系数，仍未保住原片与风格

把旧预设分数连续混合后仍会误调原片。本轮不输出预设概率，直接回归目标原图到有意调色结果的二次 RGB 系数。用已归档的 DAST、MLS、Middlebury、ARRI 四类资料生成 720 条参考状态与不同内容目标的训练组合。三位 Gros 人物未进入训练。两臂的输出函数、训练标签、归一化方法和正则相同，先比较只看参考的描述，再增加参考与目标的描述差。目标与单张参考进入正式推理，配对参考、灯位、人工区域及目标正确成片不进入系数预测。[事前协议](../../experimental/adaptive-reference-20261113/gros-two-image-edit-regression-protocol.md) · [四源训练资料](2026-11-19-capture-invariant-look.md)

另外用三张测试目标各自的原图和已知调色成片拟合二次颜色函数，验证这种颜色函数能否表达预期风格。这项离线检查不衡量普通两图。正对照对相同目标在各参考条件上复用，因此相同目标的正对照评分会重复出现。

| 方案 | 原片整图误差超过 3 | Film A 区域风格 | Film B 区域风格 | 完整方向 |
| --- | ---: | ---: | ---: | ---: |
| 只看参考描述 | 10／12 | 1／60 | 0／60 | 0／12 |
| 目标与参考描述 | 11／12 | 1／60 | 0／60 | 0／12 |
| 已知正确成片拟合 | 0／12 | 60／60 | 60／60 | 12／12 |

普通两图臂的原片整图误差为 2.85–9.58 RGB8。Film B 整图投影为 0.240–0.481，低于门槛 0.75，风格误差为 4.98–7.42 RGB8。其中冷灯和暖灯的变化程度不同。同图控制在六组均保持原图。两个实际预测臂的两款风格均无一项整图通过。[计数、逐方向结果与输出哈希](2026-11-21-gros-two-image-edit-regression.json)

## 决策与复现

停止使用这套固定描述直接做线性系数回归。已知目标成片的对照显示，二次函数能够表示这两款风格，现有描述在留出人物上却没有给出足够可靠的编辑系数。720 条训练组合来自少量反复使用的场景和两个操作，不能解释为 720 个独立拍摄场景。下一步需要更可靠的两图匹配证据或新增编辑监督，继续提高颜色表阶数或调正则缺乏依据。软件、独立画质、人工自然度及视频尚未验收。

准备[历史训练来源](2026-11-19-capture-invariant-look.md)与 [Gros 交叉参考](2026-11-21-gros-film-b-cross.md)。从仓库根目录选择新的输出路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-film-b-cross-v2/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_two_image_edit_regression.py "$base/gros-two-image-edit-regression-repro" $sha
~~~

两轮各保存 72 张实际预测成片、6 张对照和 1 份报告，79 个文件逐一核对 SHA-256 一致。另验证带正负增量的已知变化、80% 重放及零编辑恒等。完整成片留在本机 outputs/adaptive-reference-20261113/gros-two-image-edit-regression-v1 与同级复跑目录。
