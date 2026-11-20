# 多灯样本改善浅色台面，白灯彩色区域仍会漏调

此前给出逐像素照明比例后，直接换算 Film A 增量也无法保住浅色台面。这轮给同一个三次颜色拟合器同量的编辑前后样本。白灯臂只看不同强度的白灯，混灯臂拿一半样本换成蓝灯。两处布景互换，整个目标布景不参与拟合，两个灯色和两档强度一起出片。两臂均将已知编辑应用到目标原图，目标自己的成片只用于评分。[事前协议](../../experimental/adaptive-reference-20261113/chromamill-mixed-light-fit-protocol.md)

| 蓝灯目标 | 整图误差，白灯样本 → 混灯样本 | 三块浅色台面误差，白灯样本 → 混灯样本 |
| --- | ---: | ---: |
| Scene1_a 213 | 5.26 → 1.39 | 4.41–4.77 → 0.37–0.51 |
| Scene1_a 241 | 5.64 → 1.39 | 4.79–5.99 → 0.51–0.74 |
| Scene2_a 213 | 5.41 → 1.31 | 5.79–6.18 → 0.53–0.78 |
| Scene2_a 241 | 5.35 → 1.31 | 5.92–6.07 → 0.47–0.80 |

误差单位为 RGB8。四组蓝灯均超过预设 25% 降幅，整图与三块浅色台面的联合通过数由 4／8 到 8／8。八个关态负例逐像素保持。没有任何评分区的误差比白灯臂增加超过 0.5 RGB8。

**完整协议未通过。** Scene1_a 两档白灯下的彩色控制区改色幅度偏弱，投影分别从 0.621 降到 0.436、从 0.834 降到 0.693，低于预设的 0.75。后一个对照的平均误差虽然下降，原先合格的风格响应却退化。Scene2_a 两档蓝灯下，彩色控制区由反向变为同向并通过，须保留两类效果一起看。[Scene1_a 完整对照](../../outputs/adaptive-reference-20261113/chromamill-mixed-light-fit-v1/Scene1_a/comparison.png) · [Scene2_a 完整对照](../../outputs/adaptive-reference-20261113/chromamill-mixed-light-fit-v1/Scene2_a/comparison.png)

补充蓝灯样本能在当前开发布景修复大量浅色台面响应。接下来应先查白灯彩色区域是否缺少贴近目标的供样颜色，或是否是共享多项式同时拟合两种灯时压低了白灯响应。这两种解释需要不同的修复，不直接扩大同类配对数量。两臂都比正常产品多了明确的参考前后图，仍不知道怎样从单张参考识别真实意图。当前范围没有真人；浅色台面也不代表独立白墙。未改产品或启动正式验收。[逐区域数值、范围及哈希](2026-11-20-chromamill-mixed-light-fit-evidence.json)

## 复现

两遍各 16 张成片、两张并排图和结果 JSON 逐字节一致。使用已有研究环境和固定的 CHROMA-MILL 归档。运行以下命令，指定尚未使用的输出目录。

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_chromamill_mixed_light_fit.py outputs/adaptive-reference-20261113/chromamill-mixed-light-fit-replay
```
