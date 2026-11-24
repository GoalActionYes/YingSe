# OpenCL 恢复作者暖墙开关，新增端点超线

此前同一富士 RAW 在 Windows darktable 5.0.0 的 CPU 去马赛克处中断。源日志与作者侧车均使用 Markesteijn 1-pass。该版本有同一方法的 OpenCL 实现。这次沿用同版程序、原片和完整侧车，只换出图后端，实际取得了两对可复查的开关。作者的暖墙和脸部颜色变化可见，前景桌面和杯身保持。开启后 2.316% 的像素新增通道端点，超过事前 2% 的准入线，保留为开发反例。[事前协议](../../experimental/adaptive-reference-20261113/playraw-terry-opencl-protocol.md) · [GPU 去马赛克源码](https://github.com/darktable-org/darktable/blob/release-5.0.0/src/iop/demosaic.c#L719-L830) · [此前 CPU 诊断](2026-11-24-terry-wall-switch.md)

## 实际出图与工程边界

四次退出码均为 0。每份日志均记录 RTX 4060 Laptop GPU 在 CL0 完成原尺寸 6240 × 4156 去马赛克，然后缩放为 1024 × 682 导出 RGB8 TIFF。开、关态共享模块的设备和顺序相同。开启态另有 Color Zones，原侧车完全保留。关闭态只变更该模块启用位的一字节。开关重复像素分别完全相同，导出蒙版也完全相同。ICC 为 sRGB，四份配置的字节差只在创建时间。首次包含本机设备内核编译，日志中导出约 69.9 秒。复用该实验自己的内核缓存后，三次为 3.30、3.65、3.67 秒。这里解决的是这张 RAF 的备用出图路径，旧 CPU 访问异常的具体根因仍待确认。[数值与哈希](2026-11-24-terry-opencl-evidence.json)

EGM 独立审阅要求用每次实际执行日志证明 GPU 路径，沿用暖色色度的变化方向、桌面保护与端点要求。测区在看开态前从关态固定，蒙版页从 UTF-8 原始标签核对为 Color Zones，尺寸与成片一致。缩小导出时，日志记录将原尺寸蒙版变换到输出尺寸，蒙版只做离线检查。[测区登记](../../experimental/adaptive-reference-20261113/playraw-terry-opencl-regions.json)

## 成片与准入

| 关态预先登记的区域 | 开启后的暖色轴中位变化，RGB8 | RGB8 平均绝对变化 | 逐像素保持 |
| --- | ---: | ---: | --- |
| 左侧受灯照的远墙 | +16.0 | 11.41 | 否 |
| 右侧墙面 | +11.5 | 8.52 | 否 |
| 人脸左侧脸颊 | +14.0 | 12.01 | 否 |
| 桌面 | 0 | 0 | 是 |
| 杯身 | 0 | 0 | 是 |

暖色轴以红蓝增量差的一半计算。左墙亮度中位下降 2.68 Y′。保留作者有意调整脸部这一事实，不能把脸颊的可见变化全部记为偏色。16,174 个像素进入了此前未达到的 0 或 255 端点，全部在作者蒙版非零范围内。红通道新达 255 的有 10,154 处，蓝通道新达 0 的有 5,961 处。蓝通道抵达 0 的像素在关态中位值为 25。标记集中在两个灯周围和部分脸部，桌面没有新增端点。[本机开关、作者蒙版和端点图](../../outputs/adaptive-reference-20261113/playraw-terry-opencl-audit-v1/comparison.png)

这一组外部作者操作可作为有意暖墙、桌面保护和端点退化的开发反例。按事前规则，此处停止两版工作台出片及训练准入。该拍摄仍只有一档已核实实拍光照，不能补足跨灯档交叉监督，也不能作为正式独立验收。没有修改稳定版或研究版的推理逻辑。

## 复现

沿用原准入的本机 RAW、作者 XMP 及 5.0.0 官方运行目录。从项目根映射空闲 ASCII 盘符，下例为 V 盘，给导出使用隔离目录。跑出关态后核对登记的像素哈希，再执行其他三个导出和审计。必要时检查该机器的设备选择及缓存。两遍审计产生逐字节相同的报告与三个 PNG。另直接从四张 TIFF 重算像素、单字节开关、增量和 16,174 个端点。[出图程序](../../experimental/adaptive-reference-20261113/probe_playraw_terry_opencl.py) · [审计程序](../../experimental/adaptive-reference-20261113/audit_playraw_terry_opencl.py)

~~~powershell
$env:PYTHONPATH = 'V:/;V:/.tools/tf-convert-env/Lib/site-packages'
$root = 'V:/outputs/adaptive-reference-20261113'
$out = "$root/playraw-terry-opencl-new"
$raw = "$root/playraw-flash-room-intake-v1"
$cli = "$root/playraw-terry-wall-switch-v1/runtime/bin/darktable-cli.exe"
$code = 'V:/experimental/adaptive-reference-20261113'
$python = 'V:/.tools/adaptive-env/Scripts/python.exe'
New-Item -ItemType Directory $out
& $python "$code/probe_playraw_terry_opencl.py" $raw $cli "$out/off" off
& $python "$code/probe_playraw_terry_opencl.py" $raw $cli "$out/off-repeat" off --cache-dir "$out/off/cache"
& $python "$code/probe_playraw_terry_opencl.py" $raw $cli "$out/on" on --cache-dir "$out/off/cache"
& $python "$code/probe_playraw_terry_opencl.py" $raw $cli "$out/on-repeat" on --cache-dir "$out/off/cache"
& $python "$code/audit_playraw_terry_opencl.py" $out $raw "$out-audit"
~~~
