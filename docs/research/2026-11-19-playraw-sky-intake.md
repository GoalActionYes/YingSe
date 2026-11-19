# 两张实拍保留了可复原的天空压暗操作

在两个摄影讨论中找到原作者授权的 RAW 入口，以及编辑者提交的 darktable XMP。此次只读取四份 XMP，共 71,687 字节。两组对应不同原片。第二帖的三个版本都来自同一张照片，不能增计场景。[天空选择帖](https://discuss.pixls.us/t/for-something-different-sky-selection/56371) · [蒙版压暗天空帖](https://discuss.pixls.us/t/play-raw-mask-and-darken-sky/43881) · [事前规则](../../experimental/adaptive-reference-20261113/playraw-sky-intake-protocol.md)

| 帖子与回复 | 第二曝光实例最后状态 | EV | 蒙版 |
| --- | --- | ---: | --- |
| 天空选择，第 9 楼 | 关闭，历史第 13 项开启，第 16 项关闭 | −3.00 | 绘制和参数 |
| 压暗天空，第 4 楼 | 开启，第 19 项 | −2.28 | 绘制和参数 |
| 压暗天空，第 7 楼 | 开启，第 28 项 | −2.00 | 绘制和参数 |
| 压暗天空，第 18 楼 | 关闭，第 45 项。第 35 项启用过纯参数蒙版 | −1.75 | 最后保存为绘制和参数 |

天空选择帖的第 13、16 项有逐字节相同的曝光参数和混合参数，启用位不同。下载的最终 XMP 应视为关闭状态，不能直接当作帖子预览图所称的开启成片。原作者还指出开启预览在山脊边缘有明显蓝边，应在重渲染时单独测量。第二张原片已确认有最终启用的候选，优先保留第 7 楼，其作者明确使用基础管线和一个带蒙版的曝光实例。两图的曝光量和配置不同，目前证明的是相同的局部意图。[编辑者的说明与附件](https://discuss.pixls.us/t/for-something-different-sky-selection/56371/9) · [第二张的编辑说明与附件](https://discuss.pixls.us/t/play-raw-mask-and-darken-sky/43881/7)

下一项可以在固定 darktable 5.4.0 中，对每张实拍分别保持其余设置，只开关这一曝光实例。从 RAW 与编辑历史复原后，再检查天空确实压暗、山脊边缘是否受损、非天空区域是否保持。只有成片确立局部作用，才拿来比较两图软件；保留幅度不同这一限制。原始作者授权针对帖子中的 RAW。编辑记录和派生图留在本机，当前未取得任何媒体，也未安装编辑器或验证像素。肤色、白墙和正式独立验收仍需各自证据。

按官方 [5.4.0 曝光模块](https://github.com/darktable-org/darktable/blob/release-5.4.0/src/iop/exposure.c)、[混合位定义](https://github.com/darktable-org/darktable/blob/release-5.4.0/src/develop/blend.h)和[混合版本迁移](https://github.com/darktable-org/darktable/blob/release-5.4.0/src/develop/blend.c)核对模块版本 6／7、混合版本 13／14。两次命令报告逐字节一致。另外独立遍历 XML、解码并核对四份副本的非全局曝光历史行。Web 阅读器拒绝二进制 XMP，改用每份上限 64 KiB 的 HTTPS 读取。

| XMP | SHA256 |
| --- | --- |
| `sky-A.xmp` | `2c40ede1b4e183bd2c534d4b6783804ebf68a6ff7a81fb8be101475cd77d59fb` |
| `sky-B1.xmp` | `d523246d781339abb2835d7e17e7d6bb19a9fd77774920b9bff368dbfd2a051e` |
| `sky-B2.xmp` | `ed9da91b50e550b5faa838735a4dead00f00eec8f4742f261371cd2173a03d90` |
| `sky-B3.xmp` | `c5d12529ef71a43322939960f0348528bb40d560d4e2895775a55a00468e9e94` |

本机复现使用 `audit_playraw_sky_xmp.py outputs/adaptive-reference-20261113/playraw-sky-intake-v1`。四个文件名按上表顺序对应协议列出的附件。
