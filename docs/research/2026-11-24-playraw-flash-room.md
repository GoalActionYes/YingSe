# 室内混光下的真实局部提亮开关

已有混光开发对照使用程序施加半档提亮。本次想核实，真实编辑者选定位置和强度的操作，能否用公开设置重现并提供逐像素对照。选择 DanielLikesDT 公开的混光人像 `DSCF8379.raf`，取 Martin Werner 的局部提亮设置和 Terry Pinfold 的墙面暖色设置。原帖只提供一张 RAW，未取得同场景的第二个实拍灯档。原片按作者声明采用 CC BY-SA 4.0，论坛回复及其附加设置按论坛条款仅作本机课程研究。原片、侧车和成片保留本机。[原帖](https://discuss.pixls.us/t/playing-with-different-light-colors/48178/1) · [Martin 的编辑](https://discuss.pixls.us/t/playing-with-different-light-colors/48178/20) · [Terry 的编辑](https://discuss.pixls.us/t/playing-with-different-light-colors/48178/13) · [论坛条款](https://discuss.pixls.us/tos)

## 实际成片

按[预先备案的方法](../../experimental/adaptive-reference-20261113/playraw-flash-room-protocol.md)，RawTherapee 5.13 只切换原配置 `[Locallab]` 下的 `Enabled`。原配置来自较早的开发版，其余总体曝光、白平衡和渐变保持一致。关闭与开启各导出两遍，同状态重复渲染的成片逐像素一致，四张图尺寸均为 1870 × 1247，ICC 数据完全一致。这里验证的是本机重放。[关闭局部提亮](../../outputs/adaptive-reference-20261113/playraw-flash-room-rawtherapee-v1/off/image.png) · [开启局部提亮](../../outputs/adaptive-reference-20261113/playraw-flash-room-rawtherapee-v1/on/image.png) · [取样坐标](../../experimental/adaptive-reference-20261113/playraw-flash-room-regions.json)

| 在关闭图上预先选定的区域 | 像素数 | 亮度变化中位数（RGB8 码值） | 发生 RGB 变化的像素数 |
| --- | ---: | ---: | ---: |
| 画面右侧脸颊 | 1054 | +12.93 | 1054 |
| 远处墙面 | 5600 | 0.00 | 0 |
| 浅色杯身 | 1482 | 0.00 | 0 |

亮度变化按 Rec. 709 系数从成片 RGB8 计算。预选区域满足既定门槛。整张图有 48,651 个变化像素，两处保护取样只证明取样范围保持原像素。结果保留为开发对照。[数值和文件哈希](2026-11-24-playraw-flash-room-evidence.json)

## 墙面调色的重放结果

Terry 的目标是强调墙面的暖黄色。darktable 5.4.1 在关闭第 16 项 `colorzones` 后导出时触发 Windows 访问异常。去掉整份 XMP、单独打开同一 RAW 也报 `0xC0000005`，均无输出 TIFF。只翻转单个设置的做法在文本中可以核对，本机运行尚未核实墙面效果。去掉 XMP 的日志 SHA-256 为 `92296cde61758cf9b7039f68e19e525e2d03f09d4691a719d33b595ab87d7e26`。

## 后续用途

这组素材增加一个真实混光下的局部提亮正例，适用于检查自动调色能否保留有意提亮、同时避免把无关墙面和杯身一并调亮。开发中已查看这组素材。验证同一有意编辑能否在不同实拍灯档下被正确识别，仍需要多灯档及与之配对的编辑记录。研究版与产品的训练准入和验收维持既有要求。

复现时按协议取得作者附件，用[重放与评分脚本](../../experimental/adaptive-reference-20261113/probe_playraw_flash_room.py)分别向 `off`、`on`、`off-repeat` 和 `on-repeat` 四个新目录执行 `render <附件目录> <编辑器 CLI> <新目录> rawtherapee <off 或 on>`，再对共同父目录运行 `audit <父目录> <新报告.json> rawtherapee --regions <取样坐标.json>`。工具及附件身份见数值记录。
