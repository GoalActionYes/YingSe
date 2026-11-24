# 16 位导出仍未消除作者暖墙开关的新增端点

同一份 Terry 作者暖墙设置在 8 位 sRGB 成片中产生了 16,174 个新增端点像素，超过预定 2%。本轮只把 TIFF 输出改为 16 位整数，保留同一 RAW、侧车、sRGB、尺寸和 OpenCL 渲染设置。原生 16 位仍有 15,664 个新增端点像素，占 2.243%。结束通过提高位深恢复本例的方案。[事前协议](../../experimental/adaptive-reference-20261113/playraw-terry-depth-protocol.md) · [原始作者开关](2026-11-24-terry-opencl.md)

## 同一步编辑的位深对照

16 位的两份关闭和两份开启分别逐像素一致，蒙版重复一致。四张均为 1024 × 682 的 sRGB RGB16。与原 RGB8 成对核对，去除创建时间后的 ICC 逐字节一致，OpenCL 模块顺序、后端和原尺寸去马赛克路径也一致，所有日志均未执行可按位深改变计算的抖动模块。16 位到 8 位按最近整数回量化后，四张图和蒙版均逐像素复现已有 8 位数据。EGM 以 darktable 5.0.0 源码核对了这些条件，整数端点仍无法单独定位最早达到边界的处理步骤。[导出位深的管线传递](https://github.com/darktable-org/darktable/blob/release-5.0.0/src/imageio/imageio.c#L1046-L1051) · [抖动模块](https://github.com/darktable-org/darktable/blob/release-5.0.0/src/iop/dither.c#L326-L348)

| 核对项 | 像素数 | 含义 |
| --- | ---: | --- |
| 原 8 位新增端点 | 16,174 | 关态未达到该通道端点，开态达到 |
| 同一通道在 16 位仍为新端点 | 15,440 | 约占原新增端点的 95.46% |
| 16 位内部值，回量化后成为旧端点 | 734 | 原新增端点中与 8 位取整一致的部分 |
| 不能归因 | 0 | 所有原端点通道都能在两态匹配 |
| 原生 16 位新增端点 | 15,664 | 含 224 个关、开两态在 8 位已落在同一端点的像素 |

同一画幅的 2% 上限为 13,967 个像素。仅追踪到持续存在的 15,440 个旧端点，已经高于这个上限。开启后的左墙暖色轴中位变化约为 +15.80 个等效 RGB8 码值。16 位精度下桌面和杯身分别有 4 和 3 个像素发生变化，每个通道至多一个 16 位码值，回到 8 位时完全保持。未改变任何评分区、暖墙门槛或实际 RGB8 准入规则。[开关、持续端点与取整位置](../../outputs/adaptive-reference-20261113/playraw-terry-depth-audit-v1/comparison.png)

这组作者操作继续保留为有意暖墙、保护桌面和端点退化的开发反例。现有工作台的普通图片使用 RGB8，这轮没有合格的 8 位比较输入。此场景也只有一档已核实实拍光照，跨来源训练和完整调色验收另按原标准。

## 复现

复用同一官方 darktable 5.0.0、本机出图环境和原 8 位报告。按[原 OpenCL 出图步骤](2026-11-24-terry-opencl.md#复现)使用新的目录，对四态各加 `--bits 16`。可以复用实验独立内核缓存。出图日志的耗时分别约 27.4、3.14、3.29、3.26 秒。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$root = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/audit_playraw_terry_depth.py $root "$root/playraw-terry-depth-v1" "$root/playraw-terry-depth-audit-new"
~~~

两遍报告和图板逐字节一致。另用 FFmpeg 7.1 解码为 RGB48LE，核对四张图与 OpenCV 的全精度读数一致，从独立解码重新计算全部端点数和保护区差。TIFF、图板和日志保留本机。[数值与哈希](2026-11-24-terry-depth-evidence.json)
