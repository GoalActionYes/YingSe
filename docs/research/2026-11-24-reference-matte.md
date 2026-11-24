# 截图黑条会改变调色，自动去边未过线

在参考内容逐像素一致时，只加上下或左右的黑条，就足以让 13 号明显改变另一个场景的肤色和浅色表面。排除合成黑条后，所有成片恢复为使用干净参考时的结果。虽然个别脸颊的误差显著下降，另有白幕、墙面和灰球反向退化。自动去边按事前条件停止，保留正常工作台和主研究门槛。[事前协议](../../experimental/adaptive-reference-20261113/reference-matte-protocol.md)

## 如何检查

使用冻结 13 号及 80% 强度。室内外各取两个场景，按已有 18 个跨来源方向读取原参考、实拍换光参考和局部提亮参考。人像用三位真人的六个有序配对，包含冷暖灯和 Film B 开关。两张黑条参考分别沿上下和左右成对填充。两端各为对应内容边长的八分之一，颜色为 RGB 全零。候选只把满足严格对称检查的黑条排除在参考分析外，渲染仍使用原目标像素。各场景的编辑答案和评价区域只在线下计分。

EGM 第 5 阶段由独立审阅者核对协议与代码。审阅确认旧 FACD 裁边是为已知配对做对齐，不能直接充当普通参考的裁边规则。审阅还要求明确以原图或有意编辑为对照，分别报告收益、退化和停止线，以及限制“恢复干净参考”这一结论的范围。这里直接调用冻结渲染器，干净参考成片与旧档案逐像素相同。脚本内有同图对照，工作台导入、同画面拟合、保存与恢复不在本试验范围。

## 结果

| 开发对照 | 局部误差下降至少 3 RGB8 的测量项 | 误差上升超过 3 RGB8 的测量项 | 干净参考与恢复后的完整通过方向 |
| --- | ---: | ---: | ---: |
| 室内外浅墙与铺地 | 72 | 140 | 均为 0／18 |
| 真人双颊与白幕 | 104 | 40 | 均为 0／6 |

测量项按方向、灯档或开关、黑条位置和区域计数。不同方向会复用成片，计数不代表独立人员或独立照片。一个跨人物暖灯关闭态的左脸颊误差从 27.61 降到 4.15 RGB8。另一跨人物暖灯关闭态的右白幕误差从 3.83 升到 14.72。复印室的灰球在一张室外实拍换光参考下由 18.45 升到 39.89。这里的下降来自移除黑条引入的影响，恢复后的 13 号仍会把干净参考的实际照明和调色混在一起。[人像前后对照](../../outputs/adaptive-reference-20261113/reference-matte-v2/portraits/03-from-02/comparison.png) · [白幕反例](../../outputs/adaptive-reference-20261113/reference-matte-v2/portraits/03-from-01/comparison.png) · [室内反例](../../outputs/adaptive-reference-20261113/reference-matte-v2/walls/14n_copyroom1/comparison.png)

七项合成负例和探针同图保持通过。在 32 张未额外添加黑条的 FACD 文件中，八张原照片均保留，四个照片组各有三张已知滤镜成片触发，合计 12 张。被裁的是作者发布的真实边框。单张 RGB 图无法说明这种边框是交付构图还是外部播放器添加，规则不能替用户作出选择。

## 视频截图的实际编码检查

使用已有 FFmpeg 7.1 构建，把两张已有实拍人像的成片加上下各 64 像素黑条，编码为 H.264、YUV420P 后抽帧。FFmpeg 的 `cropdetect` 在两张测试中均测得 768 × 512、左上角为（0，64）。精确 RGB 全零检查在一张中得到相同边界，另一张上下各多留了两行，边缘最多为 2 RGB8。后者从带边输出恢复后，对正确裁边输出的左脸颊平均差仍有 0.60 RGB8。另一张的左脸颊偏差下降，白幕误差却从 1.50 升到 8.15。这里使用的是受控 H.264 回路，真实播放器和其他编码格式仍未验证。[FFmpeg 官方滤镜说明](https://ffmpeg.org/ffmpeg-filters.html#cropdetect)

停止把这一固定裁边规则接入自动调色。截图黑条属于已验证的额外干扰；它无法解释干净照片中的偏色，也无法验证灯光与有意风格的区别。后续跨场景候选仍需在干净参考上同时改善肤色、浅色表面和有意调色，并通过原来的独立、人评和视频要求。

## 复现

使用已有研究环境，从项目根运行，输出目录使用新名称。启动预检遇到实验子目录不存在，修正目录创建后执行正式对照，输入、评分和阈值均按原协议。两次完整运行分别约 134 秒和 120 秒。报告及每次 423 张 PNG 的哈希一致，独立从成片重算 600 项区域记录。媒体及完整逐区域数据保留本机。[数值摘要与哈希](2026-11-24-reference-matte-evidence.json)

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$root = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_reference_matte.py $root "$root/reference-matte-new"
~~~

编码检查从同一来源选 `01-warm-on.png` 和 `02-cold-off.png`。依次生成独立片段和帧文件，下例展示前者。

~~~powershell
$ff = './.deps/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
$in = "$root/gros-film-b-cross-v2/01-warm-on.png"
& $ff -nostdin -loop 1 -framerate 24 -i $in -vf 'pad=iw:ih+128:0:64:color=black,format=yuv420p' -frames:v 12 -c:v libx264 -crf 18 -threads 2 -n "$root/matte-new.mp4"
& $ff -nostdin -i "$root/matte-new.mp4" -frames:v 1 -threads 1 -n "$root/matte-new.png"
& $ff -nostdin -i "$root/matte-new.mp4" -vf 'cropdetect=limit=24:round=2:skip=0' -frames:v 1 -an -f null -
~~~
