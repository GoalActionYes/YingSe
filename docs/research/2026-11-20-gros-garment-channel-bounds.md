# 分通道取值仍无法补足黑衣与红衣缺测

上一轮在黑衣和红衣上缺少三个通道都可靠的衣物像素。本轮分别核对 R/G、B/G。通道接近暗端或亮端时给出单边界，其余通道保留半码量化范围。逐区域求色比中位区间，再用参考前后作差并减去白幕变化。只有所有可能的衣物残差都满足原门槛时才改变判定，已经有充分证据的状态照旧。[事前协议](../../experimental/adaptive-reference-20261113/gros-garment-channel-bounds-protocol.md) · [原衣物试验](2026-11-20-gros-garment-capture-cue.md)

| 衣物缺测状态 | 新状态 R/G 双边有限像素 | 新状态 B/G 双边有限像素 | 判定 |
| --- | ---: | ---: | --- |
| 02 黑衣，实际暖灯 | 22.52% | 3.17% | 仍不确定 |
| 02 黑衣，匹配整图暖调 | 1.08% | 0% | 仍不确定 |
| 03 红衣，匹配整图暖调 | 36.96% | 0.039%（1 个） | 仍不确定 |

02 黑衣冷灯基准的两个色比也都没有上下界齐全的中位区间。03 红衣冷灯基准有完整区间，整图染色后 R/G 的中位数只有下界、B/G 两端均无法限定。三种状态的衣物残差最小可能范数均为零，最大范数无有限界，因此不能用这些范围证明共同变色，也不能证明衣物仍保持原色。五类有限及无界情况下的判定，以及一个通道缺测时另一个比值仍可读取的测试均通过。

候选 48 张成片和六张图板与上一轮逐字节一致。六组有意局部暖调仍过，四个换灯方向仍逐像素保持目标，完整联合仍 2／6。作者衣物框的羽化污染按原资格保留。停止在这三个缺测状态上继续放宽阈值。若继续用衣物旁证，后续需要在采集时增加未过暗、未饱和的材质参照，或另找与衣物通道无关的空间依据。此次依赖参考前后图及固定材料框，产品代码未改，独立画质仍待验证。[完整区间、支持率及哈希](2026-11-20-gros-garment-channel-bounds-evidence.json)

两遍各 48 张输出、六张图板和报告逐字节一致。已有材料的路线、成片和背景臂通过归档验证。从仓库根目录运行，在未使用的目录下存放产物。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_garment_channel_bounds.py "$base/gros-portrait-light-transfer-v1" "$base/gros-screen-chroma-v1" "$base/gros-garment-capture-cue-v1" "$base/gros-garment-channel-bounds-replay"
```
