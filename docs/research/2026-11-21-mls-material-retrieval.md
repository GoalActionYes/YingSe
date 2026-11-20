# 视觉特征在换灯时更稳，同材质匹配仍未达标

上轮证明区域调色需要明暗与色度，本轮先查区域对应本身。固定 MLS 两处场景中由数据集附表登记的两种相同表面，蓝纸棱柱和绿纸立方体。用标准灯原照片查询另一场景的四种参考状态，每幅参考的所有网格块都参与检索。比较平均 RGB 距离与冻结 DINOv2 特征余弦距离。评分才用登记材质范围，覆盖至少 75% 的块取得标签，每种材质每幅均有至少 19 个评分块。无登记近邻计为错误。[事前协议](../../experimental/adaptive-reference-20261113/mls-material-retrieval-protocol.md)

| 参考状态 | 目标 02，RGB／视觉特征 | 目标 03，RGB／视觉特征 |
| --- | ---: | ---: |
| 标准灯 | 93.45%／67.21% | 89.66%／70.86% |
| 标准灯加 Film A | 65.32%／71.19% | 39.14%／63.19% |
| 蓝灯 | 47.37%／72.47% | 25.86%／61.47% |
| 蓝灯加 Film A | 42.11%／75.10% | 0.00%／64.44% |

两类材质等权。视觉特征在换灯或调色的六项均优于 RGB，全八项平均提升 17.88 个百分点；同灯未编辑的两项则退化。八项中仅一项达到预设 75%。蓝纸棱柱每项召回至少 85%，主要错在绿纸立方体。以目标 03 的蓝灯加 Film A 为例，29 块绿纸中有 12 块匹配成功，17 块匹配到登记材质之外，没有被认成蓝纸。表中无登记不等同于背景或某种具体材质，现有标注只能确定它没有落到满足覆盖率的两类取样块。[逐状态混淆计数与输入哈希](2026-11-21-mls-material-retrieval-evidence.json)

这批匹配未达到进入出图的门槛，不将单个近邻直接用于区域调色。保留同灯 RGB 较可靠和变光视觉特征较稳定的对照。如果继续验证自动对应，需要额外证据区分有效候选范围，并检验支持不足时能否可靠地保留原图。素材只有两个场景、两种表面，不能推广到白墙、衣物或肤色，也没有变更产品或独立验收。

两次结果逐字节一致，SHA-256 为 `68e0ce014d03cb8c6bf942f024148bfb173541100985b1ce216e98fb578d90c7`。复用视觉研究 Python、原模型及已有两个辅助依赖目录，未下载或安装新依赖。

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:OPENBLAS_NUM_THREADS='2'
& .tools/vision-advisor-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.extend([".tools/tf-convert-env/Lib/site-packages",".tools/DeepWB-audit/vendor"]); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_mls_material_retrieval.py outputs/adaptive-reference-20261113/mls-material-retrieval-replay
```
