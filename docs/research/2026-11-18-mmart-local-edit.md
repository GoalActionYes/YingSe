# MMArt 有真实修图对和区域评价蒙版，缺同款局部操作记录

为判断能否取得跨场景局部监督，本轮核对 JarvisArt 的两份公开发布。作者论文描述了真实照片修图与区域评价，论文所述 MMArt-Bench 实验为 200 个样本。当前固定发布包的元数据比论文实验描述多，以下按实际发布内容计数。见[作者论文](https://papers.neurips.cc/paper_files/paper/2025/file/4ac4365b98bc242acd5ab974a05c68a8-Paper-Conference.pdf)和[事前核查规则](../../experimental/adaptive-reference-20261113/mmart-local-edit-protocol.md)。

## MMArt-Bench 实际发布内容

固定版本公开提供 1,439,614,823 字节的 ZIP。四次边界检查通过的区间请求共读取 167,294 字节，只包含 ZIP 目录及一份数据索引。两次读取的目录和索引字节一致。独立解析器复核 563 个 ZIP 记录、558 个实际文件和索引 CRC32。见[固定发布](https://huggingface.co/datasets/JarvisArt/MMArt-Bench/tree/54b33d56f8dabcc2065670a3060063d743f69e65)和[数值与身份](2026-11-18-mmart-local-edit-evidence.json)。

| 核对项 | 实际结果 |
| --- | ---: |
| 原图与成片的配对路径均存在 | 253/253 |
| 带区域框且按文件名后缀唯一对应蒙版 | 51 |
| 归一化后各不相同的文字指令 | 253 |
| 重复的源成员字节数与 CRC32 组合 | 0 |

索引只有原图与成片路径、输出位置、裁切模式、指令和可选区域框。没有编辑前后开关、编辑参数、跨图片同款标识或拍摄家族字段。文字不同不能证明编辑互不相同，文件来源也没有经过解码后的近重复筛查。51 张蒙版的文件名都可唯一与区域框样本关联，论文将其用于人像区域质量评价。该证据无法确认每张成片中具体哪一种调色操作作用在蒙版里。见[作者区域评价程序](https://github.com/LYL1015/JarvisArt/blob/32a7e5845ba635a7caba480197c61df91df03796/src/evaluation/test_local_metrics.py)。

## MMArt-PPR10k 与既有素材的关系

固定版本的唯一数据顶层目录为 `global`。进一步核对已知含局部笔刷的三个上游 PPR10K 案例，两个对应目录返回 404，另一个 1645_1 含 XMP、Lua 和 JPEG。仅取该公开 XMP 并核对 Git blob；它与已有上游专家 A 的 XMP 字节不同。原版检测到一项启用的笔刷调色，公开配置中未检测到同项或其他检查范围内的活动局部项。不能由目录名推断全部样本的编辑历史，也不能确认公开成片是否反映原版笔刷。见[固定发布目录](https://huggingface.co/datasets/JarvisArt/MMArt-PPR10k/tree/8c36f7408ca5b4e0e9ec92a3fe115191d3beca61)及[上游笔刷复核](2026-11-18-ppr-explicit-local.md)。

两份发布的数据卡均标注 Apache-2.0。这里只读取公开元数据，上游照片的具体使用范围尚需按来源核实。MMArt-Bench 可作为将来评估人物区域的素材线索，当前证据不足以批准用它训练或评估两图同款空间迁移。此次不继续扩大下载。新训练仍优先找能确认操作与作用范围的成组实图。

复现时在仓库根目录设置 `$env:PYTHONPATH = (Get-Location).Path`，用现有研究 Python 运行 `experimental/adaptive-reference-20261113/audit_mmart_local_edit.py <新输出目录>`，网络需要时传入 `--proxy`。本机保留区间结果、索引和 XMP。课程软件未修改，未读取发布包里的照片、成片或蒙版。
