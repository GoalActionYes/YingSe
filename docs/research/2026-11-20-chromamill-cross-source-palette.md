# 外部实拍补样改善台面，未覆盖两处白灯彩色物体

上一轮两处白灯彩色物体缺少近色样本。这轮追加 MLS 两处布景在标准灯、弱蓝灯下的四组实拍配对。MLS 和 CHROMA-MILL 的编辑使用同一份 Film A 和同一 SDR 显示转换。另用每个 RGB 通道 64 档的均匀色样经同一 Film A 计算前后差，作为已知编辑的正对照。两臂都保留原三次拟合、八近邻补偿及 80% 强度，不以目标答案建色样。追加后，实拍臂每折约 18.8 万种不同输入颜色，均匀臂每折约 26.9 万种。[事前协议](../../experimental/adaptive-reference-20261113/chromamill-cross-source-palette-protocol.md)

| Scene1_a 彩色控制 | 原局部修正投影 | 追加 MLS 实拍投影 | 追加均匀色样投影 |
| --- | ---: | ---: | ---: |
| 白灯 213 | 0.436 | 0.436 | 0.699 |
| 白灯 241 | 0.693 | 0.693 | 0.835 |

投影通过区间为 0.75–1.25。MLS 在两处的第八近邻覆盖都仍为零，两个区域逐像素沿用原三次结果。均匀色样在两处的第八近邻覆盖均为 100%，平均补偿权重分别为 0.421 和 0.353。白灯 213 的响应误差从 2.376 降到 1.267 RGB8，投影仍低于门槛。白灯 241 的误差从 1.671 降到 1.105，响应通过。完整协议实拍臂漏两项，均匀正对照漏一项，两臂皆未通过。[Scene1_a 对照图](../../outputs/adaptive-reference-20261113/chromamill-cross-source-palette-v1/Scene1_a/comparison.png)

两臂均保留八组整图和浅色台面通过，且保留四个蓝灯目标相对白灯训练臂的收益。MLS 的整图误差范围为 0.95–1.33 RGB8，均匀臂为 0.62–0.92；整图和三块台面通过不能覆盖上述彩色控制失败。两臂关态均逐像素保持，无权重的位置与原混灯映射完全一致。[Scene2_a 对照图](../../outputs/adaptive-reference-20261113/chromamill-cross-source-palette-v1/Scene2_a/comparison.png) · [全部读数及哈希](2026-11-20-chromamill-cross-source-palette-evidence.json)

结束继续给这套固定回退做法添加同类实拍色样。颜色数量无法保证覆盖特定材料，而均匀臂说明有足够近邻也不保证修正幅度达标。后续保留已有完整 Film A 色表作成片端的已知编辑基准，将检查重点转回普通参考中是否存在可验证的编辑信息。此次只使用两套开发数据，无真人、独立白墙或自动识别结论，产品未改。

两遍各 16 张候选、两张并排图及报告逐字节一致。八张原局部候选逐像素复现。运行以下命令时，目标输出目录须不存在。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.extend([".tools/tf-convert-env/Lib/site-packages", ".tools/DeepWB-audit/vendor", "outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"]); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_chromamill_cross_source_palette.py "$base/chromamill-local-fit-residual-v1" "$base/chromamill-cross-source-palette-replay"
```
