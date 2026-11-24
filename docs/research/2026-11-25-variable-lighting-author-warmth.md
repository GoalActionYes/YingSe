# P1020943 日照区域加暖的作者记录核验

Todd Prior 发布的早晚两个处理版本同时改变了全图 AgX 和一项带蒙版的色调均衡。两份 JPEG 提供了可比较的编辑历史，但未发现作者加暖可单独开关的颜色实例。这组素材停在元数据核验，未下载原片、重放调色、测量成片或训练模型。[原帖及作者说明](https://discuss.pixls.us/t/variable-lighting-workflow/53701/5) · [检查协议](../../experimental/adaptive-reference-20261113/variable-lighting-author-warmth-protocol.md)

## 为什么查这组

已有整帧实验表明，给定正确编辑后，调色网格能把暖色留在墙面。普通跨场景参考仍缺乏证据来判断墙面暖色是灯光造成的，还是创作者希望迁移的调整。先前已核对这个主题中只有一份 RAW。本轮新增的是作者每张实际导出中的编辑历史核验。若能隔离单项局部调色，可补一条外部作者的暖色与保护区回归。[网格实验](2026-11-24-matched-wall-warmth-grid.md) · [旧的主题清单](2026-11-24-public-capture-edit-check.md)

## 实际结果

| 来源 | 历史终点 | 实例数 | 核验结果 |
| --- | ---: | ---: | --- |
| Todd 下图，作者的第一版 | 21 | 21 | AgX 最终为第 19 步，全图混合，无新增色调均衡实例 |
| Todd 上图，作者的第二版 | 27 | 22 | AgX 最终为第 26 步，全图混合且模块参数有变化。新增已启用的第 25 步色调均衡，混合掩码值为 7 |
| Todd 随帖 XMP | 27 | 22 | 与上图的最终实例参数、开关、混合参数和蒙版历史逐项相同 |
| kofa 随帖 XMP | 32 | 21 | 局部白平衡的两个色彩校准实例最终为第 15、14 步，掩码值为 5、9 |

Todd 两张成片的 RGB 色彩平衡和色彩校准均未变化。色调均衡主要改变不同亮度范围的曝光，AgX 是全图处理。只关掉色调均衡时仍保留新版 AgX，因而得不到作者发布的第一版，也无法单独测量作者意图中的加暖。kofa 按建筑受光面与阴影路面分别取白平衡，记录了另一种同场景处理。两位作者只使用同一原片，应按一个拍摄来源登记。[darktable 色调均衡说明](https://docs.darktable.org/usermanual/development/en/module-reference/processing-modules/tone-equalizer/) · [kofa 的处理说明](https://discuss.pixls.us/t/variable-lighting-workflow/53701/7)

## EGM 审查与决定

独立审查要求把作者意图关联到明确的导出和有效实例，核验标准 XMP 与可能的扩展依赖，并注明新增回归的具体价值。比较后已把 Todd 随帖侧车关联到上图。两包均可解码且未声明续包。未遍历至 JPEG 首个扫描，后续独立段和蒙版的实际像素作用范围仍未核验。已确认两个实例发生变化，现有证据足以停止此案的单项操作验证。保留这条混合编辑反例，四图配对准入和产品画质结论沿用现有记录。

## 耗时与复现

初筛预估 10–15 分钟。两份侧车共 65,812 字节。从两张 JPEG 首批前缀中识别出上图 XMP 跨界，仅续取声明的 8,328 字节。两条原文件的三次范围请求均返回 206，总计读取 139,400 字节前缀。侧车首次落盘到双跑报告生成约 7 分钟，合计接收 205,212 字节。无下载 RAW、渲染和训练耗时。

复现时，从 [Todd 原帖](https://discuss.pixls.us/t/variable-lighting-workflow/53701/5) 和 [kofa 原帖](https://discuss.pixls.us/t/variable-lighting-workflow/53701/7) 下载两份 XMP 至本机证据目录，分别命名为 `todd.xmp` 和 `kofa.xmp`。对原始 JPEG［[上图](https://d2x313g9lpht1q.cloudfront.net/original/3X/c/b/cb74fe4d9a05388ffc9fc1e8602aa0c414de2326.jpeg)、[下图](https://d2x313g9lpht1q.cloudfront.net/original/3X/4/e/4e61799c38022f9334d9645a14ad123852560707.jpeg)］用 `curl.exe --range` 取下表范围，用 `-D` 保留响应头，用 `-o` 保存响应体。

| 文件 | 范围（含两端） | 响应体 | 响应头 |
| --- | --- | --- | --- |
| 上图 | 0–65,535 | `upper.head` | `upper.headers` |
| 上图 | 65,536–73,863 | `upper.tail` | `upper-tail.headers` |
| 下图 | 0–65,535 | `lower.head` | `lower.headers` |

在仓库根目录运行下列命令。脚本校验 HTTP 响应头、JPEG 标准 XMP 段和 darktable 混合参数版本，以历史终点之前每个模块的最后一条记录为准，比较解码后的参数。两次输出字节一致，另用固定段偏移读取 XML 并独立比较了解码参数，结论一致。[机器可读证据](2026-11-25-variable-lighting-author-warmth-evidence.json)

```powershell
$inputDir = 'outputs/adaptive-reference-20261113/variable-lighting-author-warmth-v1'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/audit_variable_lighting_author_warmth.py $inputDir "$inputDir/report.json"
```
