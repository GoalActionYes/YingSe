# 两处室外换光下的局部压暗开关

已有真人对照能复现脸颊局部提亮。本轮从另一份公开编辑设置取得局部压暗，用两处实拍场景检查它在光照变化后还能不能按预期编辑墙面或铺地。采集来自 WildRelight，沿用以前已核验的两处场景和两个时间点。RawTherapee 的设置来自 Martin Werner 给另一张原片做的局部压暗。用开源编辑器重放该设置时，本项目按原本登记的墙面和铺地矩形移动选区、缩放半径，保留作者该工具组的其他参数。这里形成的是项目制作的开发对照，拍摄者没有参与重新放置选区。[事前协议](../../experimental/adaptive-reference-20261113/wildrelight-authored-darken-protocol.md) · [采集报告](2026-11-21-wildrelight-wall-cross.md) · [WildRelight 作者固定发布](https://huggingface.co/datasets/Lez/wildrelight/tree/90ab579145da9f3ea998d706c8defd0d4b87641f/small-aligned) · [作者局部编辑](https://discuss.pixls.us/t/vignette-correction/48970/2)

| 被编辑区域 | 首时段的变化 | 末时段的变化 | 区域完成 |
| --- | ---: | ---: | ---: |
| 建筑浅墙 | −2.00 | −3.00 | 1／2 |
| 钢柱旁铺地 | −5.00 | −6.00 | 2／2 |

表中数值为开启后在选定区域测得的 Y′ 变化中位数，单位为 8 位码值。原门槛为至少压暗 3 个码值，四张须同时达标。全部关闭图与同一准备图逐像素一致。开启后，墙面另一区域、钢柱及排除框外的各自保护取样均逐像素保持，四张图都没有新增截断像素。首时段墙面只压暗 2 个码值，留下一个暗化不足的反例。按照原门槛结束本次重放，未把三张合格图单独送进 13 号以追求通过。两次真实拍摄时的亮度、阴影和光色一起变化，这轮未进一步判定哪一项造成幅度差。[开关成片](../../outputs/adaptive-reference-20261113/wildrelight-authored-darken-run-v1/editor-comparison.png) · [逐图数据](2026-11-24-wildrelight-authored-darken-evidence.json)

公开材料补查中，Aalborg 实验确实逐项开关眩光源，但官方照片归档仍需申请。MIST 的照明状态由合成渲染得到。两者均没有用作本轮实拍替代。[Aalborg 发布记录](https://zenodo.org/records/6895080) · [实验采集说明](https://orbit.dtu.dk/files/265981334/Johra_2021_J._Phys._Conf._Ser._2042_012008.pdf) · [MIST 官方数据说明](https://www2.cs.sfu.ca/research/groups/Vision/data2/MIST-Dataset/)

## 复现与核验

使用既有自适应 Python 和 RawTherapee 5.13，按[原户外采集记录](2026-11-21-wildrelight-wall-cross.md#复现)和[原局部编辑记录](2026-11-20-rawtherapee-sky-pair.md)准备本机固定材料。从项目根目录运行下方命令。首次映射到磁盘根仍让 RawTherapee 收到含中文的路径，导出停止。将临时盘符直接映射到项目根后，以映射盘运行，完成后解除。已有失败目录保留。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$root = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_wildrelight_authored_darken.py $root $rawTherapeeCli "$root/wildrelight-authored-darken-new"
~~~

两次有效执行合计 14.0 秒，两份报告逐字节一致，SHA-256 为 `adecf1b3507b5e99fb94da910e33b25862017da30a2987ab07d95a67b405c2ec`。另从两遍共 16 张编辑器成片复算，并核对选择位置之外的局部设置字段、实际开关及 ICC。未新增训练数据、运行产品推理或更改软件。成片和侧车留本机。这两处都已用于开发，本轮过线项只保留为开发对照。
