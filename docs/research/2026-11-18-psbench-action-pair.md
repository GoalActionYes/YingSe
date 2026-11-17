# 操作视频核查与跨内容同教程调色

102 和 105 的文字都描述冷环境、暖肤色。发布视频显示两项实际处理不同。核对固定版本两份 MP4，共 25,217,426 字节，均匹配发布 LFS SHA256。各按 10 秒输出画面序列，再抽取有明确操作或画面变化的 18 帧。见 [固定发布](https://huggingface.co/datasets/zyn1216/PSBench/tree/929d0710eb7c86a988f8a69a74619052b60ac293)、[核查协议](../../experimental/adaptive-reference-20261113/psbench-workflow-protocol.md)和[文件及帧哈希](2026-11-18-psbench-action-pair-evidence.json)。

| 任务 | 视频中可核对的操作 |
| --- | --- |
| 102 窗边人像 | 0:55 使用 Camera Raw 基础调整，1:50 可见分离色调，2:35 已能看到墙面转青。2:55 导入独立光轨素材，3:50 为叠加图层加蒙版，4:10 可见画笔操作。最终图含光轨叠加。 |
| 105 人像和铁塔 | 0:23 至 3:05 先在台阶人像上制作通道混合、可选颜色、色阶、渐变映射和曲线。4:04 展示加载 Gold & Teal Photoshop 动作，4:14 到 4:24 铁塔从单一底图变为含多层调整的结果。 |

105 发布的左右图片分别对应教程中的台阶人像和铁塔，视频显示两张原始照片是在不同文档中处理的。该例有实际跨内容动作的证据。未读取动作文件或核对全部参数，也未确认它使用主体蒙版。102 与 105 整体不具备同一局部调色流程，102 的光轨和空间叠加也不能等同于目标物体改色。

## 拆开两张场景之后

去掉拼图中线两侧各 16 像素，得到两张 704×1080 的画面，前后粗结构检查均通过。13 号仅取得本侧原图与对侧已调色图，固定 80% 强度；两张结果落盘后再按本侧作者结果的 80% 混合评分。两次运行的 PNG、结果 JSON 哈希及数值一致。见 [复查协议](../../experimental/adaptive-reference-20261113/psbench-action-pair-protocol.md)。

| 目标 ← 参考 | 整图误差，保留原图 → 13 号 | 近中性色误差，保留原图 → 13 号 | 整图响应投影 |
| --- | ---: | ---: | ---: |
| 台阶人像 ← 铁塔 | 9.39 → 15.46 | 7.60 → 16.77 | −0.155 |
| 铁塔 ← 台阶人像 | 11.39 → 15.03 | 11.93 → 15.91 | 1.129 |

两方向整图均退化。人像的白台阶被染冷，衣服仍保留原图的紫色。铁塔和树林得到了暖色，云层与作者结果仍有可见差异。投影为正只表示有沿作者改动方向的变化，铁塔的总误差仍增加。结果支持把此例作为开发时的风格迁移反例。近中性色区域由目标原图计算，未单独量化肤色，未证明问题来自某种分割错误。

独立复算首次用了不同亮度系数，近中性色像素数校验未通过。统一到既有定义后，按保存的图片复算通过全部 12 项数值、输出尺寸与哈希。实验输出及评分未改。

复现时用已有研究 Python，在仓库根目录设置 `$env:PYTHONPATH = (Get-Location).Path`，再运行 `experimental/adaptive-reference-20261113/probe_psbench_action_pair.py <105 素材目录> <新输出目录>`。用现有 FFmpeg 执行 `ffmpeg -ss <秒> -i <MP4> -map 0:v:0 -frames:v 1 <PNG>` 可核查表中操作。两个源视频、操作帧及三列对照只保留本机 `outputs/adaptive-reference-20261113/psbench-shared-local-v1/workflow-v1`。对照图第三列显示作者全强度成片，数值比较统一使用 80%。实验输出在同一素材目录的 `action-pair-v1` 和 `action-pair-repeat-v1`。

下一阶段需要能核对主体／背景操作的跨内容样例。TOP-Bench-X 有跨内容编辑分组，但论文列出的 25 类未明确给出人像肤色和背景的协同调色，数据卡也把许可确认指向原作者，本轮未下载其图片。当前无新画质候选，课程交付继续用现有版本。见 [TOP-Bench-X 数据卡](https://huggingface.co/datasets/Aleksandar/Top-Bench-X/blob/ff92a06acdbcbd61aef546c64a98b9b5c2bdc102/README.md)和 [分组表](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Wang_EditCLIP_Representation_Learning_ICCV_2025_supplemental.pdf#page=8)。
