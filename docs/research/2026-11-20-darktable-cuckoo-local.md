# 原片可默认导出，蒙版去绿的作者设置未能出图

上一轮取得同一原片的 RawTherapee 全图色彩平衡反例。这轮核对 Todd Prior 的 darktable 设置，尝试隔离局部去绿。编辑者说明，白平衡先处理了脸部，随后用胸口的色相范围建立参数蒙版，调整仍偏绿的皮肤。这里需要保留原白平衡与其它修改，只开关这项 RGB 色彩平衡。[原片与第 11 楼作者说明](https://discuss.pixls.us/t/cuckoo-problems-with-color-balance/29959) · [事前协议](../../experimental/adaptive-reference-20261113/darktable-cuckoo-local-protocol.md)

重用已核验的 `DSC00178.ARW`，只取 8,825 字节的 XMP，SHA1 与发布端文件标识相符。XMP 为第 4 版格式，有 16 个历史记录，编号 10 的 `colorbalancergb` 是唯一启用的同名步骤，混合参数有值。复制原设置后，只把该步骤的 `enabled` 从 1 改为 0。逐字节检查只有一字节改变，再用 XML 解析确认其它步骤及混合参数一致。原样和关闭配置均保存在本机，没有上传作者文件。

| darktable 5.4.1 导出条件 | 实际结果 |
| --- | --- |
| 常规项目路径，原样作者 XMP | `0xC0000005`，无 PNG |
| ASCII 映射路径，无 XMP | 退出 0，1024 × 682 的 RGB PNG 解码通过 |
| 同一 ASCII 路径，原样作者 XMP | `0xC0000005`，无 PNG |

对这份 ARW，默认处理能够出图，作者配置在两个路径下均无法出图。日志报告访问冲突，具体模块或根因仍未知。默认图不作为局部操作关态，关闭配置未进入渲染和区域评分。这套 darktable CLI 路线在此结束。后续若有兼容渲染条件或编辑者提供的精确开关成片，可继续使用已冻结的原件。现有 RawTherapee 对照仍只证明全图调整会影响环境，不能代替局部去绿结果。[已完成的全图对照](2026-11-20-rawtherapee-portrait-toning.md)

复现时使用上一轮核验过的 Windows darktable 5.4.1，复用 1024 长边、CPU 双线程、关闭 OpenCL、禁用用户预设和 sRGB8 PNG 选项。分别运行无 XMP 与原作者 XMP 命令，配置和缓存目录分开。素材与日志位于 `outputs/adaptive-reference-20261113/darktable-cuckoo-local-v1`。作者 XMP SHA256 为 `578f776ad24571aded0878d286319b85c4f290b398ff18141f277858607332bb`，只关一项的 XMP 为 `ef95b1097a27ae604c82914a9ae2419fe9321fec837b6fb5138ab95c954686a8`。默认导出 PNG 为 `03143a08eee835d16946a29dfc176d7377653a8d6ae4dc65d0e5ac65c13f2b33`。[编辑器与既有命令选项](2026-11-19-playraw-sky-toggle.md)
