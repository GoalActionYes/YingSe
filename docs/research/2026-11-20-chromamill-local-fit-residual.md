# 近色样本改善整体成片，两处白灯彩色区域没有足够覆盖

混灯三次映射修好了蓝灯浅色台面，仍把 Scene1_a 两档白灯下的彩色物体调得太轻。本轮保持同一批 8,192 个混灯样本，用邻近 RGB 样本纠正原映射的局部残差。这项修正只在目标原图的颜色附近有足够不同的供样颜色时生效。找不到足够近邻时，沿用原混灯成片。目标编辑后图、浅色台面和有色物体区域均只用于评分。[事前协议](../../experimental/adaptive-reference-20261113/chromamill-local-fit-residual-protocol.md) · [原混灯结果](2026-11-20-chromamill-mixed-light-fit.md)

| 白灯彩色控制 | 混灯样本中第一近邻距小于 6 的比例 | 白灯样本／混灯样本的第八近邻距离中位数 | 局部候选投影 |
| --- | ---: | ---: | ---: |
| Scene1_a 213 | 59.8% | 14.14／15.39 | 0.436 |
| Scene1_a 241 | 48.3% | 11.49／12.77 | 0.693 |

距离为 RGB8 欧氏距离。两项彩色区域在白灯和混灯两套样本中，第八近邻距小于 6 的像素均为零。因此局部候选在这两处逐像素沿用原混灯成片，两项风格投影仍低于 0.75。八个场景条件的彩色控制区都没有触发修正；有些的原成片已经过线。现有配对对浅色材料足够，对不同实景的彩色材料覆盖仍不均衡。试验不能排除三次拟合本身也低估了这些区域的变化。

全部八张候选的整图误差均低于原混灯成片，范围从 1.28–1.66 降至 0.99–1.38 RGB8。八组整图与浅色台面联合成绩保留，四组蓝灯仍比只用白灯样本降低至少 25%。八个关态负例逐像素保持，所有零权重位置也逐像素保留原混灯成片。完整协议因两项彩色控制失败。保留局部收益和反例，停止仅凭这一供样集合调整近邻回退阈值。后续若有独立多灯色样，可先固定覆盖办法，再验证是否补到了各类材料的颜色，并用整布景轮换检验新增颜色能否迁移。[完整读数与哈希](2026-11-20-chromamill-local-fit-residual-evidence.json) · [Scene1_a 并排图](../../outputs/adaptive-reference-20261113/chromamill-local-fit-residual-v1/Scene1_a/comparison.png) · [Scene2_a 并排图](../../outputs/adaptive-reference-20261113/chromamill-local-fit-residual-v1/Scene2_a/comparison.png)

使用的是已知 Film A 的额外前后图和已用开发场景。没有真人、独立白墙或双图自动风格成绩，未改产品。提取原试验的样本选择函数后，逐像素复现上轮两臂共 16 张归档成片；新候选两遍各八张成片、两张图板和报告逐字节一致。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.extend([".tools/tf-convert-env/Lib/site-packages", ".tools/DeepWB-audit/vendor"]); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_chromamill_local_fit_residual.py "$base/chromamill-mixed-light-fit-v1" "$base/chromamill-local-fit-residual-replay"
```
