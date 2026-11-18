# GIER 找到两条人物提亮线索，仍缺同款操作依据

已取得 GIER 作者公开的任务索引，4,996,658 字节。读取全部 6,179 条记录，只查操作和区域标注。它补充的是用户请求、实际编辑结果的索引以及候选区域，能用于找真实局部意图。[发布索引](https://sites.google.com/view/gierdataset/home) · [作者论文](https://openaccess.thecvf.com/content/ACCV2020/papers/Shi_A_Benchmark_and_Baseline_for_Language-Driven_Image_Editing_ACCV_2020_paper.pdf) · [事前协议](../../experimental/adaptive-reference-20261113/gier-local-edits-protocol.md)

| 调整 | 有局部区域引用 | 只有这一项操作且有局部区域引用 |
| --- | ---: | ---: |
| 亮度 | 236 | 7 |
| 对比度 | 146 | 1 |
| 色相 | 237 | 6 |
| 饱和度 | 197 | 4 |
| 色偏 | 136 | 3 |
| 黑白 | 12 | 5 |
| 曝光与明度 | 155 | 0 |

26 条单项记录的原图 URL 都非空且各不相同，仍需检查原片和拍摄组。逐条读最初请求，有两条明确要求提亮人物，其中一条还要求保留背景。另有保留彩色主体、其余区域转黑白的相近意图，但主体不同，个别只保留对象的一部分。中性色线索涉及去除硬币蓝偏、减淡墙面色偏等不同操作。部分所谓单项记录的请求还同时提到修改其他对象或改变画质，说明标注项数不能证明编辑只有一步。

公开操作对象只有 `local` 和 `ids`，无法从索引核实实际编辑参数或开关状态。作者论文说明局部区域由标注者从预分割结果中选择，无法从这些区域编号推断编辑器的真实笔刷。原图来自外部社区，作者页面与代码库未核实到这批候选照片可供本项目使用的材料许可。停止凭 GIER 任务标签配对，没有下载媒体或训练。保留其用途为查找局部编辑意图。[官方代码库](https://github.com/jaxfield/LDIE_ACCV)

两遍 Python 报告逐字节一致，PowerShell 独立重算八个操作的总数、有效区域引用数、单项数及 26 个不同 URL。索引、原始链接和请求文本留在本机。用 `audit_gier_local_edits.py <本机 GIER.json> <新 JSON 路径>` 复现。[聚合数值与索引 SHA256](2026-11-19-gier-local-edits-evidence.json)
