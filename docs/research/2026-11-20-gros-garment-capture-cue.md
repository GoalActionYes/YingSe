# 衣物旁证保住局部暖调，深色及极端颜色仍缺证据

用上一轮三个真人、六个跨人物方向，比较供样照片前后两态的脸颊、衣物和白幕色度。先用白幕差表示全图暖调，再用衣物和脸颊扣除白幕后方向是否一致，决定是否传递局部暖调。两臂共用参考编辑前图，目标的编辑答案只用于评分。和普通两图输入相比，本轮额外给了参考原图、衣物和脸部区域，只能作为离线诊断。[事前协议](../../experimental/adaptive-reference-20261113/gros-garment-capture-cue-protocol.md) · [区域登记](../../experimental/adaptive-reference-20261113/gros-garment-regions.json)

| 评估项 | 只用白幕变化 | 加入衣物变化 |
| --- | ---: | ---: |
| 保留只暖脸并保护羽化以外区域 | 0／6 | 6／6 |
| 实际换灯时逐像素保持目标原图 | 0／6 | 4／6 |
| 联合通过，含全部参考状态有足够证据 | 0／6 | 2／6 |

01 的蓝衣和 03 的红衣在真实换灯下，扣除白幕差后，与脸颊残差方向余弦分别为 0.854 和 0.895。两名供样各对应两个目标，四张候选精确回到原图。六组有意只暖脸均得到保留，两颊的投影约为 0.997–1.001。整图有意暖调的误差为 0.091–0.151 RGB8。另有 02 的黑衣在实拍暖灯只剩 80 个合格像素，匹配全图暖调时为零；03 的红衣在匹配全图暖调时只剩 1 个。在这些状态，衣物证据不足，候选沿用白幕分量，按事前门槛记为未通过。[同快门的黑衣参考对照](../../outputs/adaptive-reference-20261113/gros-garment-capture-cue-v1/01-from-02/comparison.png) · [蓝衣供样的可用结果](../../outputs/adaptive-reference-20261113/gros-garment-capture-cue-v1/02-from-01/comparison.png)

真实作者的编辑开关在孩子脸部的色度残差范数为 0.585，登记衣物为 0.029，被判作局部编辑。复核发现衣物框触及作者蒙版低权重羽化，最高 24／255，无高权重点。原协议要求衣物框零权重，因此这个对照未通过严格资格，不能作为真实作者编辑的通过成绩。该读数和图像均保留。[作者蒙版来源](2026-11-20-playraw-mixed-light-skin.md) · [分区结果和输出哈希](2026-11-20-gros-garment-capture-cue-evidence.json)

保留有可靠材料对照时的局部暖调收益。结束把固定三通道完整支持视为可靠自动判断的做法。下一项能区分原因的检验，是量化黑衣和红衣被排除的像素究竟含有哪些未饱和通道，判断是否还能给出有界的照明证据。此次未取得普通两图自动结果，白墙、独立自然评价和视频验收均无结论，产品未改。两遍各 48 张成片、6 张并排图和报告逐字节一致。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_garment_capture_cue.py "$base/gros-portrait-light-transfer-v1" "$base/gros-screen-chroma-v1" "$base/gros-garment-capture-cue-replay"
```
