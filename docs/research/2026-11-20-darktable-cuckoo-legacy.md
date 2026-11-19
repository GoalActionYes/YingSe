# 同代 darktable 成功复现局部去绿开关，胸口变化未达门槛

上一轮用 darktable 5.4.1 导出这份 2022 年作者 XMP 时发生访问冲突。XMP 的 RGB 色彩平衡模块参数版本为 4，混合参数版本为 11，与 3.8.1 源码声明相同。因此使用同年 2 月发布的 Windows 3.8.1 隔离目录，只更换重放工具。官方 x64 包为 118,978,495 字节，SHA256 与发布说明中的 `9ed13cc0972c4473c7e5ac69b3393e6636ba9e2a2cf8b8b9e2a51882288235d1` 相符。用已有 7-Zip 解包，未安装。CLI 确认版本为 3.8.1。[上一轮失败记录](2026-11-20-darktable-cuckoo-local.md) · [官方版本和哈希](https://www.darktable.org/2022/02/darktable-3.8.1-released/) · [模块版本源码](https://raw.githubusercontent.com/darktable-org/darktable/release-3.8.1/src/iop/colorbalancergb.c) · [混合版本源码](https://raw.githubusercontent.com/darktable-org/darktable/release-3.8.1/src/develop/blend.h)

以同一原片、原样 XMP 和仅关闭一项的 XMP 分别导出 1024 × 683、带 sRGB ICC 的 RGB8 PNG。作者先用白平衡调整脸部，随后用基于胸口色相的蒙版处理残留偏绿。这次胸口是预设主区域，脸颊只报告附加读数。先从关态固定四个测量框，再读取开态像素。[编辑者说明](https://discuss.pixls.us/t/cuckoo-problems-with-color-balance/29959) · [事前协议](../../experimental/adaptive-reference-20261113/darktable-cuckoo-legacy-protocol.md) · [测量区域](../../experimental/adaptive-reference-20261113/darktable-cuckoo-legacy-regions.json)

附加核查按 3.8.1 结构读取这一步的二进制参数。混合模式启用了参数蒙版，按输入 `hz` 色相通道筛选，未启用手绘区域。模块记录的色相旋转为约 −25 度，这是内部调整参数。它不表示最终成片或胸口每个像素都旋转了相同角度。[混合与色相通道定义](https://raw.githubusercontent.com/darktable-org/darktable/release-3.8.1/src/develop/blend.h)

| 区域 | 色度差中位数 | 色度差 P95 | RGB 有变化的像素 |
| --- | ---: | ---: | ---: |
| 胸口 | 0.00 | 0.50 | 326 / 1,056 |
| 脸颊 | 0.00 | 0.72 | 82 / 272 |
| 浅灰树皮 | 0.00 | 0.50 | 201 / 837 |
| 草地 | 0.00 | 0.00 | 0 / 1,984 |

色度差是逐像素红、蓝去亮度分量的平均绝对变化，单位为 RGB8 码值。胸口未达到预先设定的 2 码值，两处环境区域都在 1.5 的保护上限以内。整图有 18,311 个像素的这一读数至少为 2，占 2.62%，均在四个登记区域以外。这项补充读数说明开关生效，无法证明变动位置符合编辑意图。保留这个实拍开关作为弱响应反例。它仍与全图去绿共用同一原片，不能计为独立场景、自然度或跨场景迁移结果。[本机关态](../../outputs/adaptive-reference-20261113/darktable-cuckoo-legacy-v1/off.png) · [本机开态](../../outputs/adaptive-reference-20261113/darktable-cuckoo-legacy-v1/on.png)

用复现脚本重新导出后，开态和关态分别有 23、9 个通道值与首遍相差 1 码值，四处区域的全部表内读数一致。整图超过或达到 2 的像素计数复跑为 18,310。四份 ICC 除创建日期字段之外逐字节一致。复跑初次调用用了反斜杠，旧版 CLI 误解析输出路径。改回首遍的正斜杠并在独立目录完成复跑。复核脚本首次比较时把日期也纳入 ICC 等同，随后单独核对字段；另用 64 位整数复核色度门槛，排除了初次定点计数的溢出。两遍正式数值报告 SHA256 分别为 `1ed05801ae697d80b2d40e611e006684f95b196e0a334bfe2f7d6dda7973bdf6` 和 `0339f55729bbb30f1894ac60033ccc5896af34759136ca9288b8bd2e0c86cafd`。

用 [PowerShell 导出脚本](../../experimental/adaptive-reference-20261113/render_darktable_cuckoo_legacy.ps1) 传入 `-Cli`、`-Raw`、`-SidecarDirectory`、`-OutputDirectory`、`-State`。输入与输出通过临时 ASCII 盘符访问，侧车来自上一轮本机档案，导出状态为 `on`、`off`、`on-repeat`、`off-repeat`。四态都用隔离的配置和缓存。旧运行时、本机素材和成片均未纳入 Git。
