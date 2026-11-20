# 已知调色意图后整图暖调可复现，局部模式受人脸漏检限制

先前单看参考成片的脸颈色差，会把有意全局暖调判成拍摄偏色。本轮评估一种有额外输入的方式：给出同一参考的编辑前后两图，并直接指定保留目标光色、整图暖调或仅调脸部。软件根据参考前后的有效像素估计色比变化，局部模式只从原片自动定位人脸。评分用的人工区域和目标答案不进入渲染。这项对照检验额外信息的能力上限，用户不用处理真实灯光与编辑的隐含标签才能继续用现有软件。[事前协议](../../experimental/adaptive-reference-20261113/gros-declared-edit-protocol.md)

| 六个跨人物方向 | 开发结果 |
| --- | --- |
| 指定保留目标光色时，实拍暖灯参考逐像素保持目标原图 | 6／6 |
| 指定整图暖调时，四个评分区的响应与串色均过门槛 | 6／6，最坏区域响应 MAE 不超过 0.03 RGB8 |
| 指定只暖脸时，双颊响应及区外保持同时达标 | 2／6，成功方向的双颊投影约 0.999 |
| 综合门槛至少 5／6 | 实际 2／6，未过。13 号普通两图输出为 0／6，已知衣物证据为 2／6 |

同一人物 02 在 768×512 冷灯原片上漏检。其余两人都各检出一张脸。02 作为供样或目标时共有四方向，局部模式原样回退到该方向的 13 号成片，最差双颊响应误差为 18.33 RGB8。成功的两个方向均能区分实拍暖灯、整图暖调和只暖脸。无人、双脸负例均逐像素返回传入的回退图，参考前后相同的恒等负例逐像素保持原图。负例只测试了局部模式的拒绝与回退接线，不能覆盖人像漏检、多脸交互或一般无人照片的产品行为。[本机成功对照](../../outputs/adaptive-reference-20261113/gros-declared-edit-v1/01-from-03/comparison.png) · [本机漏检对照](../../outputs/adaptive-reference-20261113/gros-declared-edit-v1/01-from-02/comparison.png)

前三行收益来自额外给定的意图及同原片参考前后图，换灯保持是明确选择的结果。整图和局部编辑来自同一个已知色度函数。仍需用真实作者不同内容验证色度幅度，以及参考无原图时的拍摄偏色问题。结束这套固定 OpenCV 人脸定位与重放方案。其反例说明，已知调色意图还需要可靠的区域位置，不能仅靠固定检测器完成。目标导入、手动微调、预览及导出均未修改，尚无独立画质、自然评价或视频通过结论。

首次执行时，Windows 上 OpenCV 无法从项目中文路径打开附带的 XML。在临时目录以完全相同的字节加载成功，检测参数不变。经路径修复后，两次各 24 张成片、6 张并排图及报告逐件同哈希。报告 SHA-256 为 `b6ea7a97e1e94058b0f7c4afd4526931c89a8014105dc9b32e7c5d2c352f711d`。[逐项数值、模型与成片哈希](2026-11-20-gros-declared-edit-evidence.json)

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_declared_edit.py "$base/gros-portrait-light-transfer-v1" "$base/gros-screen-chroma-v1" "$base/gros-garment-capture-cue-v1" "$base/gros-declared-edit-replay"
```
