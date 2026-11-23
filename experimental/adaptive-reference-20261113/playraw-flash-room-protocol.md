# 同一闪光与室内混光原片的两项真实局部编辑

开发对照已经能测试只换灯档和程序半档提亮。现在补一组真正由不同编辑者选择范围和强度的开关。DanielLikesDT 公开同一张混合光人像 RAW，声明 CC BY-SA 4.0。Terry Pinfold 描述在墙面和部分肤色范围使用带手绘蒙版的色彩区域工具；Martin Werner 描述局部提亮脸部及另外的整体调整。两个侧车均仅供本机课程研究，按论坛用户内容许可处理。该帖子只发布一张原片，两份操作仍归为同一拍摄内容。[原片与拍摄说明](https://discuss.pixls.us/t/playing-with-different-light-colors/48178/1) · [Terry 的编辑](https://discuss.pixls.us/t/playing-with-different-light-colors/48178/13) · [Martin 的编辑](https://discuss.pixls.us/t/playing-with-different-light-colors/48178/20) · [论坛条款](https://discuss.pixls.us/tos)

已在侧车源文确认两项操作。Terry 的 darktable XMP 只翻转第 16 项、已启用的 `colorzones`，其蒙版模式解码为 3。Martin 的 RawTherapee 配置只切换 `[Locallab]` 的 `Enabled`，保留同一配置中整体曝光、渐变、白平衡和其它步骤。配置只有一个局部斑点。RawTherapee 保留作者配置，角色内所需外置校准文件若不可用，从日志记录并只在原配置不能解码时做额外可审计对照。先核对作者发布的 RAW 长度、Discourse SHA-1 和编辑器版本，再用本机隔离环境各导出关闭和开启状态，重复两次。预算 20–40 分钟，原片允许最多 23 MiB、两份设置合计少于 128 KiB，不安装编辑器，不更新产品或权重。两份作者配置针对不同程序，评分各自只比同一程序的开关。

初步资格要求保持尺寸和 ICC，编辑的预期区域至少 128 像素，所需亮度或色度变化中位数至少 2 RGB8，受保护区域的变化 P95 不大于 1.5 RGB8。darktable 优先读取导出的实际蒙版页，高权重与零权重分别评分。RawTherapee 先仅从关闭态与作者说明登记保守的脸部及区外两处取样，再查看开态；没有实际蒙版页时不声称精确区外覆盖。命令未完成、效果不足、边界溢出均保留，不改变原编辑者操作强度重试。两项实验只检查源图能否提供可追溯编辑监督，不把同场景多名编辑者当成跨场景同款，也不声称取得多档实拍配对或训练资格。
