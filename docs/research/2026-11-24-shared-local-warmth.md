# 用同一项局部暖色补上肤色与浅色表面的开发正例

室内外工作台对照确认了误传真实灯光和漏掉有意提亮。若下一版保护人物和浅色表面时将有意暖色一并去掉，用户仍得不到参考风格。本轮用已有 RawTherapee 5.13 的局部「Color & Light」作可复现的编辑工具，固定暖色参数，仅随已登记位置平移和缩放编辑区域。项目定义此编辑，原拍摄者没有制作这些开关。来自三个采集来源的实拍和所有开关成片均作开发材料。[事前协议](../../experimental/adaptive-reference-20261113/shared-local-warmth-protocol.md) · [编辑器源码](https://github.com/RawTherapee/RawTherapee/blob/5.13/rtengine/iplocallab.cc#L19190)

## 成片与判定

| 采集来源 | 局部暖色开关通过 | 可与有意暖色全 RGB 匹配的真换光方向 |
| --- | ---: | ---: |
| MIT 两个房间 | 14／16 | 1／112 |
| WildRelight 浅墙和铺地 | 8／8 | 0／24 |
| Gros 三位真人 | 6／6 | 0／6 |

28 张合格开关图中，编辑测区红减蓝的中位变化为 11–30 RGB8，亮度中位变化绝对值不超过 2.24 RGB8。全部 30 张关闭图逐像素保持；所有已登记保护区 RGB8 差 P95 为 0。两张未达标的图都来自 MIT 灯位 24，有效像素分别为 76.89% 和 0%，客厅同时超过亮度上限。保留两张及失败理由。[逐档与逐组证据](2026-11-24-shared-local-warmth-evidence.json)

复印室左墙由灯位 8 换到 0，三通道中位变化是 15、−1、−9。同一个灯位 8 下打开有意暖色则是 10、−3、−13，两向量余弦 0.9242，幅度比 0.9516，符合原联合线。这两张参考对同一区域产生相近变化，可以检查是否保留这项有意暖色并避免误传参考灯光。Gros 第 2 包从冷灯到暖灯的脸颊方向也相近，余弦 0.9936，但有意编辑的变化只有实际换光的 30.85%，不计入匹配。户外 24 个有序换光方向均未达到方向匹配线。[本机复印室开关对照](../../outputs/adaptive-reference-20261113/shared-local-warmth-v2/mit/14n_copyroom1/comparison.png) · [本机户外浅墙开关对照](../../outputs/adaptive-reference-20261113/shared-local-warmth-v2/wildrelight/wall/comparison.png)

把 28 张作为开发正例，要求候选保留已明确意图的肤色及浅墙暖色。仅有一个采集来源满足有意暖色和换光近似，不能进入既定的跨来源暖色训练比较。下一项素材应提供同机位、原拍摄条件可追溯、较小色度差的实拍光照，继续用固定暖色参数和原门槛。保留已经补齐的局部提亮跨来源回归作为独立的开发检查。影像效果改善、真人评价和视频验证仍待候选。[EGM 继续条件](2026-11-24-spatial-condition-design.md)

## 复现

沿用前一阶段的冻结原图、相同研究 Python 和本机 RawTherapee 5.13。在空闲的 ASCII 映射盘根运行。首跑的核对只接受 LF，原侧车包含 CRLF，改为按字段核对两项原始值后重跑，色度参数和门槛保持固定。两遍完整出图分别耗时 55.4 秒和 54.7 秒，机时合计约 1 分 50 秒。另从磁盘用独立计算核验共 180 张 PNG（每遍 30 张输入、60 张开关成片）、28 份处理设置、14 张图板，以及两遍全部 284 个有序换光比较。两遍报告逐字节一致，SHA-256 为 `0e040a54c3de7b6f3124664748cf592e7d3913705cc7c5610e074c081354c02c`。原图、完整出图、编辑器及运行环境保留本机。[执行脚本](../../experimental/adaptive-reference-20261113/probe_shared_local_warmth.py)

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$root = 'V:/outputs/adaptive-reference-20261113'
$cli = "$root/rawtherapee-sky-pair-v1/runtime/RawTherapee_5.13_win64_x86_64_release/rawtherapee-cli.exe"
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_shared_local_warmth.py $root $cli "$root/shared-local-warmth-new"
~~~
