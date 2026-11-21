# 标定视差能恢复部分风格，Dolls 的近中性色仍失真

走廊和床头缺少经过标定的跨视角位置。这里换用 Middlebury 2005 的 Art、Dolls 两个实拍场景，固定光照和曝光，使用发布的左右视差回查共同可见表面。继续沿用上一轮的逐通道仿射模型、Film B 和 80% 混合。对同一目标分别用视差配对、同坐标配对、右图自身前后配对以及冻结 13 号出图。两幅原片保留在本机，视差只作为离线诊断信息。[拍摄与视差说明](https://vision.middlebury.edu/stereo/data/scenes2005/) · [事前协议](../../experimental/adaptive-reference-20261113/middlebury-stereo-grade-protocol.md)

视差回查及邻域筛选后，分别保留 61.69% 和 74.47% 的左图像素，在固定棋盘格内用 51,577 和 64,626 点拟合。未用于拟合的对应区域上，原片 RGB8 差由 47.49 降至 2.98、由 49.37 降至 3.11。视差回查减少了取样错位。两类值均用相同的留出位置计算，仍属同场景诊断。

| 场景 | 方法 | 整图风格投影 | 整图关闭／开启误差 | 近中性色投影 | 近中性色开启误差 | 通过区域 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Art | 13 号 | 0.341 | 2.44／4.50 | 0.352 | 4.91 | 0／3 |
| Art | 同坐标 | 0.514 | 23.92／27.16 | 0.455 | 41.97 | 0／3 |
| Art | 视差对齐 | 0.850 | 1.11／2.50 | 0.857 | 2.994 | 3／3 |
| Art | 同机位前后图 | 0.878 | 0／2.11 | 0.895 | 2.75 | 3／3 |
| Dolls | 13 号 | 0.364 | 2.64／4.13 | 0.325 | 5.53 | 0／3 |
| Dolls | 同坐标 | 0.404 | 30.17／34.07 | 0.312 | 38.28 | 0／3 |
| Dolls | 视差对齐 | 0.817 | 1.08／2.68 | 0.661 | 3.93 | 2／3 |
| Dolls | 同机位前后图 | 0.792 | 0／2.54 | 0.650 | 3.60 | 2／3 |

误差单位为 RGB8，具体通过规则包含响应误差与绝对误差。Art 的近中性色开启误差只略低于门槛。Dolls 近中性色的响应误差，视差和同机位前后图分别为 3.62 和 3.60，且风格响应均不足。真值视差提供了可测的整图收益。同机位控制仍有相同的失败，说明现有取样色域或拟合表达存在限制，不能把这一失败都归到匹配。预设的两场全通过门槛未满足，因此未进入自动匹配或产品接入。可保留已校正的实拍对应点，用于区分后续的拟合表达限制和对齐限制。数据仅覆盖两个物体场景、单一灯光和 Film B，不能验证人像、换灯或普通跨场景输入。

## 复现与成片

两个下载文件为 `Art-2views.zip` 和 `Dolls-2views.zip`，均取自 [ThirdSize 发布目录](https://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/zip-2views/)。本机素材为 `outputs/adaptive-reference-20261113/middlebury-stereo-grade-input-v1`。按 [已有 Film B 运行环境](2026-11-21-synclight-lamp-edit-order.md) 配置仓库、OpenCV 和 OpenColorIO 路径，运行 `probe_middlebury_stereo_grade.py <素材目录> <新输出目录>`。两次运行各约 5 秒，逐个核对 18 张 PNG、报告的 SHA-256，并独立复算 24 项评分判定。成片在本机 `outputs/adaptive-reference-20261113/middlebury-stereo-grade-v1`。[数值、参数与文件身份](2026-11-21-middlebury-stereo-grade-evidence.json)

实拍与视差由 Middlebury Stereo 发布。引用 D. Scharstein 与 C. Pal，Learning Conditional Random Fields for Stereo，CVPR 2007。[论文与许可](https://vision.middlebury.edu/stereo/data/)
