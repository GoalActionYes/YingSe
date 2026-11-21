# 互换两套成片的明暗和色度，仍无法同时保住肤色、白幕和风格

同一组真人跨参考中，PPST 的原片残差合成能保住结构，整图 Film B 风格投影也更接近 1。13 号在部分肤色和白幕的预设关闭态更接近目标原图。为核对两项优势能否组合，在同一强度的冻结 PNG 上，分别使用 13 号的明暗和 PPST 的色度，以及 PPST 的明暗和 13 号的色度。两臂完整使用各自分量的原幅度。这里的明暗是 BT.709 系数加权的 RGB 码值亮度，色度是 RGB 码值扣除该亮度后的分量。[事前协议](../../experimental/adaptive-reference-20261113/gros-ppst-component-swap-protocol.md) · [PPST 原片合成](2026-11-21-ppst-detail-render.md) · [13 号工作台同组回归](2026-11-21-gros-workbench-identity.md)

| 方案 | 整图关态误差，冷／暖 | 整图风格投影，冷／暖 | 整图开态换灯差 | 五区联合通过 |
| --- | ---: | ---: | ---: | ---: |
| 冻结 13 号 | 3.60／5.72 | 0.447／0.735 | 4.28 | 0／5 |
| PPST 残差合成 | 13.43／8.57 | 1.125／0.903 | 5.18 | 0／5 |
| 13 号明暗 + PPST 色度 | 4.47／6.14 | 0.407／0.610 | 4.32 | 0／5 |
| PPST 明暗 + 13 号色度 | 13.25／8.32 | 1.165／1.027 | 5.08 | 0／5 |

误差用 RGB8 MAE。13 号明暗加 PPST 色度时，冷灯右颊关闭态误差为 1.36，低于两份原成片的 4.36 和 22.58，暖灯同一区却升至 13.77，高于 13 号的 6.25。PPST 明暗加 13 号色度时，整图投影两灯都达范围，但暖灯整图风格误差由 PPST 的 3.10 升至 3.69，右白幕的暖灯风格投影升至 1.433。两种组合的结构相关均为 0.9908–0.9942，通道裁切比例最高约 0.71%。[全分区成绩及文件哈希](2026-11-21-gros-ppst-component-swap-evidence.json)

按预设结束固定互换规则。此例中的明暗与色度由不同渲染器提供也无法同时解决偏色、换灯和有效风格。这个结论限于一组已用真人开发对照及两种既有成片。下一步应先核实新线索能否识别参考中真实编辑，再将它用于成片，避免用现有输出分量继续调比例。本轮未更改软件、权重或导出流程。

## 复现

使用已存在的 [Gros 开关成片](2026-11-21-gros-film-b-cross.md) 与 [PPST 残差成片](2026-11-21-ppst-detail-render.md)。本机结果位于 `outputs/adaptive-reference-20261113/gros-ppst-component-swap-v1`。两遍各 8 张成片、2 张对照板和 1 份报告，SHA-256 逐文件一致。用代数等价的 RGB 分量表达式复算 8 张成片，量化中间值的浮点舍入差不超过 1 码值。直接从实际 PNG 复算两个组合的 10 个区域，评分一致。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_ppst_component_swap.py outputs/adaptive-reference-20261113/gros-ppst-component-swap-repro
~~~
