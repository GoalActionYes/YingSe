# 成对监督的视觉空间特征未跨来源区分换灯与调色

单张参考的平滑区明暗色比在四来源留出失败。本轮换用冻结 DINOv2 的相邻图块差描述空间结构。先用每个训练场景的实拍向量减去原图与合成编辑的平均，按拍摄来源等权拟合一个方向。各折均只给留出来源的一张参考做判别。相同原始内容和两台相机始终同折。半数阈值来自训练的两个投影中心的中点，没有在留出结果中选择。[事前协议](../../experimental/adaptive-reference-20261113/capture-dense-paired-structure-protocol.md) · [上轮固定统计](2026-11-21-capture-tile-tone-color.md)

| 留出来源 | 实拍识别 | 原参考误触 | 数字编辑误触 |
| --- | ---: | ---: | ---: |
| CHROMA-MILL | 6／6 | 6／6 | 6／6 |
| Beyond RGB | 2／2 | 2／2 | 2／2 |
| Gros 人像 | 2／3 | 2／3 | 3／3 |
| MIT 复印室 | 0／1 | 0／1 | 0／1 |

无缺测。实拍识别为 10／12，较旧固定统计的 5／12 日志更高；数字编辑误触由 5／12 增至 11／12，原参考误触为 10／12。两个来源的三态均被判成实拍。当前结果表明该训练判别未能移除来源差异，不能依据高命中率引入保护。复印室三个判定均未触发，客厅和人像的六张成片与原 13 号逐像素一致，联合 0／2。墙面串色仍为 15.14／15.73 RGB8，双颊仍为 12.38／9.58。[客厅成片](../../outputs/adaptive-reference-20261113/capture-dense-paired-structure-v1/living-comparison.png) · [人物成片](../../outputs/adaptive-reference-20261113/capture-dense-paired-structure-v1/portrait-comparison.png)

结束固定分档与冻结视觉描述上的两轮跨来源判断，不再为同样的参考去搜索特征层或开关阈值。当前试验缺能独立确认编辑范围、真实光照或风格意图的信号。后续若引入更多数据，应改变可观测证据，并预先检查实际风格响应和误触。没有修改软件、启动新训练或使用独立验收素材。

两次各六张成片、两张对照和报告一致，SHA-256 为 `c87aebc771b0ca827f46bc27528ac2e004ced6feed8d17d24b79e3d808200f33`。用已准备的 vision-advisor-env、既有 OpenCV 和本机 SciPy 辅助目录运行脚本，传入新输出目录。[脚本](../../experimental/adaptive-reference-20261113/probe_capture_dense_paired_structure.py) · [逐案分数与文件身份](2026-11-21-capture-dense-paired-structure-evidence.json)
