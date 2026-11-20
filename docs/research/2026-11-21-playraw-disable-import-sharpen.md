# 自动锐化造成了关闭态偏差，局部显影差异仍在

上轮的已导出 sRGB 试验，作者模块关闭时仍出现 64–71 RGB8 的单通道偏差。以同一输入在关闭态追踪编辑器，日志显示导入色彩配置、锐化、导出色彩配置及显示编码均有执行。固定输入、作者开关、导出设置、强度及合成方式，仅为实验 CLI 设置 `plugins/darkroom/sharpen/auto_apply=false`。关闭态追踪中锐化消失，其余三项保留。[事前协议](../../experimental/adaptive-reference-20261113/playraw-disable-import-sharpen-protocol.md)

| 检查 | 原流程 | 关闭自动锐化 |
| --- | ---: | ---: |
| 混合光关闭态最大通道偏差 | 64 | 0 |
| 《Cuckoo》两份缓存关闭态最大通道偏差 | 71 | 0 |
| 混合光两折高权重区色度 MAE | 3.294／3.266 | 3.248／3.202 |
| 《Cuckoo》脸颊／胸口色度 MAE | 0.559／0.608 | 0.567／0.626 |

关闭态均逐像素还原。误差单位为 RGB8，预定肤色 MAE 限值为 3。《Cuckoo》两份缓存来自同一内容，均保住两处肤色响应和预登记保护区。混合光两折仍超线，所需暖调方向投影仅为约 0.824。区外均逐像素保持，区内量化 Y′ 偏差最大为 0.4934 RGB8。该试验以已知作者参数和已知区域为前提。

自动锐化可以解释这三份输入的关闭态偏差。去掉它后，混合光的残余误差仍表明，直接在已导出 sRGB 上重放本例不能替代原 RAW 显影操作。尚未分离 RAW 工作空间、运算顺序与蒙版混合的各自作用。按原门槛结束固定的 sRGB 端点方案，保留《Cuckoo》的实际改善。映色稳定版及已有功能未变。下一项若继续利用作者信息，需验证能直接读出的操作或工作状态对不同内容是否有预测能力，普通两图仍需独立验证。[《Cuckoo》对照](../../outputs/adaptive-reference-20261113/playraw-disable-import-sharpen-v2/cuckoo-0/comparison.png) · [混合光反例](../../outputs/adaptive-reference-20261113/playraw-disable-import-sharpen-v2/mixed/comparison.png)

首次批处理的 Windows 路径分隔符导致输出名异常，改用原实验的正斜杠路径后，两遍各三个成片、三个对照图、六个中间导出 RGB 哈希和报告一致。报告 SHA-256 为 `a28dc25d4ddb88cead6b24e0229130914e7fb6920f0e0710a890abc3c47d3814`。[数值与哈希](2026-11-21-playraw-disable-import-sharpen-evidence.json)

复现使用上一轮的 ASCII 映射盘符和既有研究 Python 环境。先运行 [导出脚本](../../experimental/adaptive-reference-20261113/render_playraw_disable_import_sharpen.ps1)，入参为研究输出根目录和新输出目录。再用同样两个参数运行 [评分脚本](../../experimental/adaptive-reference-20261113/probe_playraw_disable_import_sharpen.py)，沿用[上一轮的 Python 搜索路径](2026-11-20-playraw-rendered-input-operator.md)。
