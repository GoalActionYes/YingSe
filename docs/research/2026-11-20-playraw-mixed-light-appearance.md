# 扩展模型的参考外观分支未能传递真人局部改色

13 号在作者真人照片的局部颜色开关上响应不足。本轮复用扩展模型第 4 轮的固定权重，只切换已有参考外观分支。三种参考构图、跨场景白衣负例、80% 强度和全部评分区沿用原检查。模型只读目标图和参考成片，作者范围仅用于评分。[事前协议](../../experimental/adaptive-reference-20261113/playraw-mixed-light-appearance-protocol.md) · [13 号正常入口结果](2026-11-20-playraw-mixed-light-transfer.md)

| 测量项 | 13 号 | 扩展模型仅颜色统计 | 扩展模型加参考外观 |
| --- | ---: | ---: | ---: |
| 原画幅作者区投影 | 0.102 | 0.112 | 0.026 |
| 镜像作者区投影 | 0.012 | 0.011 | 0.011 |
| 裁切作者区投影 | 0.014 | 0.009 | 0.009 |
| 原画幅作者区色度误差（RGB8） | 16.65 | 16.75 | 17.93 |
| 原画幅区外近中性色差 P95（RGB8） | 2.00 | 1.86 | 2.72 |
| 跨场景白衣区色度差 P95（RGB8） | 0.72 | 0.72 | 0.72 |

开启外观分支没有恢复任何构图的作者区响应，原画幅幅度下降，区外近中性色差反而增大。镜像及裁切保护部分降到约 0.50–0.72 RGB8，仍伴随明显的风格不足。跨场景白衣保护三臂均通过，联合门槛仍失败。此前受控换位置只测得外观信号，本次真实手绘局部颜色开关也未看到有效迁移。收尾把当前固定外观分支用于此类局部编辑的路线，保留失败成片用于后续方案对照。[受控位置互换的历史结果](2026-11-19-appearance-layout-response.md)

## 复现

原视觉环境设置仓库根目录为 `PYTHONPATH`、`OPENBLAS_NUM_THREADS=2`，运行 `python experimental/adaptive-reference-20261113/probe_playraw_mixed_light_appearance.py <新的输出目录>`。13 号评分逐项复现，两个扩展臂同图逐像素保持。两遍各 16 张成片及报告逐件一致，另用整数权重从落盘图复算 20 项评分。报告 SHA256 为 `99cd4c5af5803fa44f4d92b8416cca3c3e5a54a85f8b7672cca076a02913ca6f`。媒体保存在本机 `outputs/adaptive-reference-20261113/playraw-mixed-light-appearance-v1`。现有产品、模型和独立验收未改。[脚本](../../experimental/adaptive-reference-20261113/probe_playraw_mixed_light_appearance.py)
