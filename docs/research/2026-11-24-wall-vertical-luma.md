# 同一墙的亮度起伏无法单独判断局部提亮

固定两间房已有的 12 个合格换灯与编辑方向。在每张参考成片上，从同一块墙取上、中、下三框。对照只读中间亮度，另一种测量用中间亮度减去上下两框的平均值。后一项能抵消上下方向上的线性明暗坡度。位置在关态原图上核对，三框都取同一墙面，保留原纹理、接缝和阴影。开关状态与原片不进入逐图取值，只有离线门槛判断使用这些信息。[事前协议](../../experimental/adaptive-reference-20261113/wall-vertical-luma-protocol.md) · [原出片对照](2026-11-24-matched-wall-lightening-workbench.md)

| 房间 | 同图开关时中心相对周围额外变亮 | 未编辑图的最高读数 | 编辑图的最低读数 |
| --- | ---: | ---: | ---: |
| 复印室 | +15.82 至 +17.64 | 25.49，20 号灯 | 19.50，16 号灯 |
| 客厅 | +16.18 至 +21.32 | 42.25，0 号灯 | 9.24，20 号灯 |

数值为 8 位亮度码值。共有 23 张不同的参考成片，全部测区有效覆盖达到 80%。在一个房间学习阈值前，未修图的最高读数必须比修过的最低读数低。两个房间的两种测法都发生重叠，均未产生阈值。按原协议，留出预测全部回退到关闭，12 个方向的有意提亮因此都漏掉。严格按唯一图计为 13 张关闭图保持、10 张开启图漏判。客厅未编辑的 0 号灯仍产生典型热点，仅读取中间相对周围亮多少无法区分拍摄灯和该局部操作。这次只排除固定亮度规则。[逐图读数和哈希](2026-11-24-wall-vertical-luma-evidence.json)

这项诊断与旧两颊比值都在连接近似材料时测试单一亮度读数。测量改用已知的同一面墙，照明造成的局部起伏仍然和编辑重叠。停止以这种固定起伏决定调色开关。把这些图加入不同照明成因的开发反例，保持受控编辑、训练用配对和产品两图输入的区分。人工墙面坐标从未进入软件推理。本轮未改工作台、未运行新候选，真实人评及最终独立素材仍封存。[旧脸颊反例](2026-11-24-gros-cheek-luminance-ratio.md) · [EGM 审查后的成片门槛](2026-11-24-spatial-condition-design.md)

## 复现与核验

沿用[已准备的室内成片和研究环境](2026-11-24-shared-authored-tone-admission.md#复核与复现)。下面的命令从项目根运行，输出目录须是新的。两次运行分别耗时 4.2 秒、2.9 秒，报告逐字节一致。另用实际 PNG 和整数亮度权重复算 23 张图的 69 项区域读数、阈值和逐方向判定。报告 SHA-256 为 `a5d3d3b3b59c2b07e08d4aca35869296c24c6e57ef8d8d570c9a799d5ab9ccf3`。[执行脚本](../../experimental/adaptive-reference-20261113/probe_wall_vertical_luma.py)

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$root = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_wall_vertical_luma.py $root "$root/wall-vertical-luma-new"
~~~
