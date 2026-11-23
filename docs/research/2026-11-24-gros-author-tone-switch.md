# 用 RawTherapee 制作三档实拍下的局部提亮对照

把 Martin Werner 公开的 RawTherapee 局部提亮参数用在另外两位人物的实拍照片上。原来的测试用固定半档提亮，这次换成编辑器里已有的局部工具，检验它在参考灯光变化时能否产生稳定的右颊提亮，同时保留白幕。这一步借用成熟编辑器生成离线测试答案，映色仍用自己的工作台从两张普通图片推断如何调色。[事前协议](../../experimental/adaptive-reference-20261113/gros-author-tone-switch-protocol.md) · [作者操作及原片](2026-11-24-playraw-flash-room.md) · [RawTherapee 5.13 源码](https://github.com/RawTherapee/RawTherapee/blob/5.13/rtgui/tools/locallab.cc)

从作者配置取局部工具组，按事前登记把选区移到两位人物的画面右颊。该段只有三项几何字段改变，提亮参数保持一致。两位各有低、中、高三档参考灯和一张冷灯目标，每张分别导出关闭与开启。关闭图与输入逐像素一致。开启后，两处白幕取样及脸部登记椭圆外的像素均保持。最坏一张新增 147 个通道触及 0 或 255 的像素，占画面 0.021%。八组均过素材门槛。[开关成片](../../outputs/adaptive-reference-20261113/gros-author-tone-switch-v1/editor-comparison.png) · [数值与哈希](2026-11-24-gros-author-tone-switch-evidence.json)

| 人像编号 | 冷灯目标右颊 | 低档参考右颊 | 中档参考右颊 | 高档参考右颊 |
| --- | ---: | ---: | ---: | ---: |
| 2 | +16.14 | +14.35 | +18.28 | +17.57 |
| 4 | +14.35 | +11.14 | +16.57 | +12.43 |

表内为开启后亮度变化的中位数，单位为 8 位 Y′ 码值。原门槛为 +3。

## 现有 13 号工作台的表现

将两位人物交叉作为目标和参考。工作台每次读取目标原图和一张普通参考 PNG，强度固定为 80%。离线再与目标的编辑答案比对。共取得 12 张实际成片。两个人物方向、六档参考均漏掉绝大部分右颊提亮，一档的响应略微反向。只换参考灯光时，白幕最坏变化为 12.33 RGB8。六张关闭态和此前相同灯档的 13 号成片逐像素一致。[人物 2 接收人物 4 的风格](../../outputs/adaptive-reference-20261113/gros-author-tone-switch-v1/2-from-4-comparison.png) · [人物 4 接收人物 2 的风格](../../outputs/adaptive-reference-20261113/gros-author-tone-switch-v1/4-from-2-comparison.png)

| 目标与参考 | 低、中、高档分别保留的编辑量 | 右颊响应误差范围 | 只换参考灯档的白幕最大差异 |
| --- | --- | ---: | ---: |
| 目标 2，参考 4 | 1.56%、−0.59%、0.09% | 12.52–12.85 RGB8 | 7.00 RGB8 |
| 目标 4，参考 2 | 5.67%、2.43%、0.46% | 10.16–10.80 RGB8 | 12.33 RGB8 |

原门槛要求编辑响应在期望的 75%–125% 内，右颊误差及白幕跨档变化各不超过 3 RGB8。两个方向均未达到。这里的响应幅度按三通道投影计算，和上表的亮度码值衡量不同的量。白幕误差及局部提亮遗漏都保留为回归。

## 用途与复现

本轮验证了用开源编辑工具，在三档实际拍摄灯光上快速制作一组有明确答案的局部提亮测试。两位人物来自已用拍摄库，选区由团队依据现有坐标放置。将这批素材固定为开发回归。扩充数据时优先换拍摄来源，并取得可核验的实拍灯档和局部编辑记录。本轮维持 1.3 和 13 号的原有版本，未训练、未新增产品依赖。正式独立画质、人类评价及视频验收仍待合格候选。

使用已有自适应研究环境和 RawTherapee 5.13 CLI。按原室内混光案例准备作者的 PP3，保留三档换灯和冷灯目标的原研究目录，并在一个新目录运行下方命令。非 ASCII 工作路径若影响便携工具读写，可用临时盘符映射项目所在磁盘，在映射路径运行并在结束后解除。源文件、侧车和成片留在本机。[原灯档和工作台](2026-11-24-gros-mixed-light-transfer.md) · [研究脚本](../../experimental/adaptive-reference-20261113/probe_gros_author_tone_switch.py)

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
& ./.tools/adaptive-env/Scripts/python.exe `
  experimental/adaptive-reference-20261113/probe_gros_author_tone_switch.py `
  outputs/adaptive-reference-20261113 $rawTherapeeCli `
  outputs/adaptive-reference-20261113/gros-author-tone-switch-new
~~~

首次机器执行约 32 秒，使用现有本机工具，无需下载或付费。两遍报告 SHA-256 均为 `6b66892e5966ab36920c89da11696e1a2067bb43fee13db4b608b5c22c596cd5`。用两遍各 16 张编辑器成片、12 张工作台成片逐张核对哈希，并独立复算开关信号、保护区域、端点、两个人物三档的响应和跨档差异。
