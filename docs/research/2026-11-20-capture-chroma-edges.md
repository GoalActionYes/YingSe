# 同机位看得见换灯差异，单张参考仍无法可靠判断

人物和近中性色的修正幅度无法从亮度排名确定。本轮换用相邻像素的颜色关系，比对相距四像素的线性 RGB 对数色比差。均匀数字染色在没有裁切和量化时会给对数色比加上同一个常数，求空间差时抵消。实拍灯色改变可能在不同材料、阴影和反光处产生不同响应。保留同一布景的白灯、实际换灯、整体色偏相近的数字染色，用三个状态都有效的边做配对测量。同一描述量另从单张图计算，逐布景留出，测试是否能用于软件只有一张参考图的情景。[事前协议](../../experimental/adaptive-reference-20261113/capture-chroma-edge-protocol.md)

| 开发资料 | 实拍边缘变化至少为数字染色两倍 | 单张正确识别换灯 | 白灯／数字染色误报 |
| --- | ---: | ---: | ---: |
| CHROMA-MILL 三布景双相机 | 6／6 | 4／6 | 0／12 |
| Beyond RGB 两布景 | 2／2 | 0／2 | 0／4 |

双相机的 Scene3_a 两台均漏判。Beyond RGB 两组的共同有效边均超过 156 万，单张各自的边保留率超过 99.9%，因此可用边少并不能解释其漏判。Beyond RGB 的真实换灯边缘变化中位数为 0.05849、0.07075，数字染色为 0.00866、0.00954。单位为线性色比的自然对数差，为本轮计算量。并排图以同一比例展示六幅图。[Scene3_a 尼康](../../outputs/adaptive-reference-20261113/capture-chroma-edges-v1/Scene3_a-nikon.png) · [Beyond RGB WHITE_green](../../outputs/adaptive-reference-20261113/capture-chroma-edges-v1/WHITE_green-beyond.png)

同机位配对时可以度量数字染色前后的局部颜色关系是否接近保留。单张图同时含有原本的物体边缘，四个差分统计无法可靠区分哪些差异来自换灯，故结束这套单图判别，未触发参考校正或目标出图。配对证据也不足以确定差异来自哪个材质或拍摄环节。后续若用双态数据提供监督，应检查模型是否仅靠场景身份记住差别。先前已验证准确还原参考后，原目标渲染器仍会丢失风格响应，故候选须同时改善目标端的实际迁移。[原目标风格响应反例](2026-11-20-chromamill-oracle-reference.md) · [读数、判定和哈希](2026-11-20-capture-chroma-edges-evidence.json)

这八套均为已使用的开发布景，不覆盖真人肤色或独立白墙。本轮未改产品。复现使用先前双相机和 Beyond RGB 归档，两遍各八幅对照和报告逐字节一致。图片留本机。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_capture_chroma_edges.py "$base/chromamill-camera-pair-first-v1" "$base/beyond-rgb-matched-cast-first-v1" "$base/capture-chroma-edges-replay"
```
