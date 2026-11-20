# ICC 解码工程候选的本地 Windows 便携包

以候选提交 `b5e6b70` 构建稳定工作台的独立便携包。包内实际包含 Display P3、Adobe RGB 及带匹配 CMYK 配置的输入转换。此次验证 Display P3 与 CMYK 两种，Adobe RGB 沿用候选源码验证记录。便携包和此处的 MP4、CUBE 都对应稳定工作台。13 号空间调色仍从原研究入口启动，空间效果无法完整表达为普通 CUBE。[候选范围与既有源码验证](2026-11-20-image-icc-candidate.md)

本机包位于本工作树的 `outputs/icc-portable-candidate-local-v1/reference-color-20261120-145438.zip`，62,857,300 字节。完整解压到可写目录，再双击内层的 `ReferenceColor.exe`。包内 Python、NumPy、Pillow 和 FFmpeg 随包载入，无需另装 Python。启动器仍依赖原便携版的 Windows 和浏览器环境。运行中的候选数据放在解压目录的 `data` 下。需要回退时用本包 `ReferenceColor.exe --stop` 停止后台，再使用原稳定版启动入口或旧课程包。当前候选没有覆盖主工作树或旧发布指针。

| 实测项 | 结果 |
| --- | --- |
| 包完整性 | 25 个应用文件与候选源码一致，清单全部 SHA256 通过 |
| 实际启动后测试 | 启动、重复复用、停止重启以及缺失 FFmpeg 提示通过 |
| 现有输出 | 手动基础、HSL 和曲线微调下的 PNG、普通 CUBE 和 MP4 导出通过；MP4 完整解码成功 |
| 场景图片解码 | 带 Display P3、带配置 CMYK 分别作目标和参考，四种 PNG 输出均与各自正确转换的 sRGB 基准逐像素一致 |
| 额外检查 | ICC 四种输出的预览与导出相同；核对依赖从解压包内加载；测试后台已停止 |

未标记控制相对正确转换的 sRGB 成片确有差异：Display P3 作目标与参考的 RGB8 平均误差分别为 4.311、1.138，CMYK 分别为 14.354、6.186。此对照衡量文件色彩配置引入的偏差。它没有验证自然跨场景配色、白墙保护、视频遮挡或完整时序稳定。本轮输出视频是既有内置合成动图。本轮还没有在第二台 Windows 机器或独立可见窗口完成交互回归。[包哈希及数值证据](2026-11-20-image-icc-portable-evidence.json)

使用已有 Python 3.12 便携运行时执行 `desktop/reference_color/build.py`，以 `YINGSE_RELEASE_ROOT` 指向数值记录中的内容目录。用包内 Python `-I` 运行随包的 `desktop/reference_color/verify.py`，传入全新解压目录。ICC 复核在第二份解压包上启动 `ReferenceColor.exe --check`，使用现有 `probe_external_icc_images.py` 的请求与预览／导出比较过程，CMYK 使用已归档的三态文件。包、相片和出图仅留本机。
