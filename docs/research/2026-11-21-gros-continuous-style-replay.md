# 连续混合预设仍不能从普通两图保住原色和风格

已知多灯参考开关对可让 Gros 六个跨人物方向通过。普通参考没有编辑前图。之前只读当前参考的局部明暗和颜色模型在大部分情形均回退为保持原图。本轮复现同一个模型的全部分数。将三个分数截去负值后归一，用它们混合目标原图和目标上两项 80% 的固定调色。比较原 0.65 的硬选择。推理只读目标和单张参考，目标预期、灯位与编辑标签均未进入渲染选择。两种预设构成试验的有限操作范围。[事前协议](../../experimental/adaptive-reference-20261113/gros-continuous-style-replay-protocol.md) · [硬选择实验](2026-11-21-gros-local-tone-coupling.md)

| 编码／方式 | 原片整图误差高于 3 | Film A 区域风格 | Film B 区域风格 | 完整方向 |
| --- | ---: | ---: | ---: | ---: |
| 原样，硬选择 | 0／12 | 0／60 | 0／60 | 0／12 |
| 原样，连续混合 | 12／12 | 0／60 | 0／60 | 0／12 |
| JPEG 85，硬选择 | 0／12 | 0／60 | 10／60 | 0／12 |
| JPEG 85，连续混合 | 12／12 | 0／60 | 0／60 | 0／12 |

连续混合使原参考情况下的目标整图误差达到 5.69–7.31 RGB8。原样 Film B 的整图有意响应投影为 0.084–0.155，JPEG 85 后为 0.085–0.258，均低于 0.75。原方法在 JPEG 的单一冷灯参考判对 Film B，两个目标合计十个区域单项达标，但换到同一人的暖灯参考时漏掉编辑。连续混合失去这十项，仍未新增完整方向。[各项分数、权重及哈希](2026-11-21-gros-continuous-style-replay.json)

## 决策与复现

结束用这组旧分数产生软力度的路线。硬选择主要漏调，连续混合则同时带来原片误调和风格迁移不足。已知开关有足够颜色支持时的六方向结果继续作为开发对照。普通两图的下一候选需要有新增的拍摄与编辑区分依据，或由两图直接预测实际编辑并验证原片负例。当前软件未接入，独立画质、人工自然度及视频仍待完成。

使用[已归档的参考识别对照和依赖](2026-11-21-gros-local-tone-coupling.md)，在仓库根目录运行以下命令，选择新输出路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-local-tone-coupling-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_continuous_style_replay.py "$base/gros-continuous-style-replay-repro" $sha
~~~

两轮各保存 72 张成片、12 张对照及 1 份报告，85 个文件逐一核对 SHA-256 一致。另验证权重归一、零权重回退、原图端点和已知混合值。完整成片留在本机 outputs/adaptive-reference-20261113/gros-continuous-style-replay-v1 与同级复跑目录。
