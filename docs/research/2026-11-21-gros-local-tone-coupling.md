# 局部明暗与颜色关系未能跨冷暖灯识别预设

上一轮的真人交叉成片表明，同一个预设开关在换灯后，可能使目标脸颊过调而使白幕更接近预期。这轮尝试在单张参考中寻找独立于灯光的局部预设线索。先缩到长边 256 像素，再检查每个 16×16 图块内对数亮度与两个对数通道色比之间的相关。仅用未裁切、颜色和明暗均有足够波动的像素，再取所有有效块相关值的四分位。原全图统计留作对照。两者沿用相同的旧四来源拟合、拍摄差分惩罚以及 0.65 回退门槛。Gros 三人完全不进入拟合。[事前协议](../../experimental/adaptive-reference-20261113/gros-local-tone-coupling-protocol.md)

| 参考编码与统计方式 | 原片识别 | Film A 识别 | Film B 识别 | 完整跨人物方向，两个预设合计 |
| --- | ---: | ---: | ---: | ---: |
| 原样，全图 | 6／6 | 0／6 | 0／6 | 0／12 |
| 原样，局部 | 6／6 | 0／6 | 0／6 | 0／12 |
| JPEG 85，全图 | 6／6 | 0／6 | 0／6 | 0／12 |
| JPEG 85，局部 | 6／6 | 0／6 | 1／6 | 0／12 |

两个预设的全部结果均分别计分。局部统计在 JPEG 复存后只选中人物 01 冷灯的 Film B，暖灯同款参考依然回退。应用于人物 02 和 03 时，冷灯参考下的对应输出满足风格条件，换到暖灯则直接保持目标原色，整图换灯差为 9.902、9.872 RGB8，两处白幕的范围为 11.311–11.643。原片均保持原图。有意风格大范围漏判，不能用原片通过替代完整画质。[决策、逐方向摘要与全量报告哈希](2026-11-21-gros-local-tone-coupling.json)

## 决策与复现

结束这项局部相关描述。原样输入没有通过任何预设，唯一 JPEG 命中无法跨同一人物的实际换灯保持一致。已有全图、逆变换、局部纹理及局部颜色共变路线都尚未提供可靠的调色历史识别。下一项工作应使用新增且可单独核验的信息，或验证无需判断预设名称也能保住局部风格的渲染，而非继续放宽此处门槛。本轮只测有限的已知预设，三位人物均已用于开发，软件、独立画质、人工自然度和视频验收未推进到发布阶段。

沿用[历史训练来源](2026-11-19-capture-invariant-look.md)和 [Gros 交叉成片](2026-11-21-gros-film-b-cross.md)，从仓库根目录选择新的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$sha = (Get-FileHash outputs/adaptive-reference-20261113/gros-film-b-cross-v2/results.json).Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_local_tone_coupling.py outputs/adaptive-reference-20261113/gros-local-tone-coupling-repro $sha
~~~

两次各生成 144 张成片、24 张对照和 1 份全量报告，169 个文件逐一核对 SHA-256 一致。完整逐区域结果随本机成片留在 outputs/adaptive-reference-20261113/gros-local-tone-coupling-v1 及同级复跑目录。
