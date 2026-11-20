# 区内暗亮色差减少误报，仍把真实作者编辑判作拍摄偏色

白幕对照已验证全图色度有可用线索，但只暖脸的成片完全漏调。本轮检查单张人像的脸部暗亮颜色关系。从有效脸颊像素取线性亮度的三等分，算亮档和暗档的两个对数色比中位数差。与只取脸颊整体色比中位数相比，分类器、拟合方式和留出参与者相同。实拍暖灯为正类，冷灯、同向全图数字染色和只暖脸为负类。[事前协议与作者区域修正](../../experimental/adaptive-reference-20261113/portrait-chroma-tone-cue-protocol.md)

| 指标 | 脸颊色比中位数 | 区内暗亮色比差 |
| --- | ---: | ---: |
| 暖灯实拍正确 | 3／3 | 3／3 |
| 冷灯和两种数字编辑错误判作纯拍摄偏色 | 6／9 | 3／9 |
| 真实作者脸部开态判作纯拍摄偏色 | 否 | 是 |

暗亮差把 01 和 02 的有意编辑与暖灯分开，但把留出的 03 冷灯原图及两种编辑全部判为拍摄偏色。对同一张 03 参考，亮暗两档的 `log(B/G)` 中位数差在冷灯、暖灯、数字染色时分别为 +0.027、−0.204、+0.031，这是同人同构图的实际变化。另一个人的皮肤与用光会改变这项统计，单张留出结果未达到零误报要求。

真实作者对照初拟了成人脸部和双手。蒙版复查发现三个框的高权重像素为零，已核验的单一开关实际集中在画面右侧孩子的脸部。追加的儿童面部框有 3,812 个高权重像素，有效覆盖在开关前后均超过 99.7%。分类读取框内 RGB，作者蒙版只确认编辑归属。暗亮差判别在作者开、关两态均标记为纯拍摄偏色，因此实际开态有被过滤掉的风险。成人脸和双手保留为未受该开关影响的对照，手部编辑缺测。这只是一个来自其他采集来源的真实编辑对照。[作者开关与本机成片](2026-11-20-playraw-mixed-light-skin.md) · [详细读数及输入哈希](2026-11-20-portrait-chroma-tone-cue-evidence.json)

两项判据均未过线，结束用固定暗亮统计直接决定是否压低局部色度。保留真实换灯与原作者开关成片，后续的证据必须能在留出参与者上同时识别实际拍摄变化和有意局部编辑。本轮只验证线索资格，未生成新候选成片或修改软件。首次三个成人框的读数留本机；追加儿童框前后，Gros 全部读数、留出判定和门槛均一致。最终两遍报告逐字节相同。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_portrait_chroma_tone_cue.py "$base/gros-portrait-light-transfer-v1" "$base/gros-screen-chroma-v1" "$base/portrait-chroma-tone-cue-replay"
```
