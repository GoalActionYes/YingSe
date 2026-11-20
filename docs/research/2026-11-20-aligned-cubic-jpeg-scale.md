# 放大细节比较尺度只恢复一个 JPEG 重放正例

上一轮保留了八组无损同原片的 Film A 调色，两端存为 JPEG 后全部回退。本轮分别压缩参考，或同时压缩目标和参考，均固定 JPEG 85 与 4:2:0。与原 4×4 细节比较并行，候选先对每个不重叠 2×2 像素取平均，再比较细节，相当于在原始 8×8 区域统计。三次拟合、回退条件和相关阈值相同。每种编码都测原 47 个重拍方向和 14 个同原片编辑，其中八个已通过的无损对固定用来测保留情况。[事前协议](../../experimental/adaptive-reference-20261113/aligned-cubic-jpeg-scale-protocol.md)

| 编码条件 | 八个固定正例获得三次映射 | 实际重放且过成片门槛，原尺度／放大 | 47 个重拍方向误触，原尺度／放大 |
| --- | ---: | ---: | ---: |
| 只压缩参考 | 2 | 0／0 | 0／0 |
| 两端都压缩 | 3 | 0／1 | 0／0 |

双端 JPEG 的 Scene1_a 白灯 213，细节相关从 0.414 升到 0.755，恢复已知编辑重放。与正常 13 号相比，整图响应误差从 6.64 降到 2.29 RGB8，近中性色从 7.71 降到 2.29，投影分别为 0.861 和 0.821。两端压缩还让四个重拍方向拿到三次映射，粗细两臂仍全部回退到原 13 号。只压缩参考时，没有重拍拿到三次映射。[恢复的成片](../../outputs/adaptive-reference-20261113/aligned-cubic-jpeg-scale-v1/both_encoded/CHROMA-Scene1_a-white-213-comparison.png) · [双端重拍反例](../../outputs/adaptive-reference-20261113/aligned-cubic-jpeg-scale-v1/both_encoded/CHROMA-Scene1_a-white-241-to-anchor-comparison.png)

两种编码均低于至少六组的事前门槛。三次映射尚未生成时，后级细节判断没有机会恢复成片。停止单改后级尺度。若要继续，先用现有 JPEG 量化条件单独测前置三次拟合的误差是否由编码差别造成，再比较同原片编辑与同机位重拍的误触。此次用的是已有开发图生成的压缩版本，真实 JPEG、不同尺寸、跨场景及独立画质仍未验证。产品代码未改。[逐对决策、缺测、成片哈希](2026-11-20-aligned-cubic-jpeg-scale-evidence.json)

两遍各 33 张处理结果（含常规 13 号）、11 张对照图及报告逐字节一致。完全同图的原身份旁路保持。用已有研究环境，从仓库根目录运行，目标目录须未占用。

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '4'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_aligned_cubic_jpeg_scale.py outputs/adaptive-reference-20261113/aligned-cubic-jpeg-scale-replay
```
