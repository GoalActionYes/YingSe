# 更精细地匹配全图颜色分布，人物偏色仍未改善

比较两种只读目标和参考的全图方法。线性对照匹配均值与协方差；试验方法在八组正交基上迭代匹配分位数，再按同一 80% 强度出图。两臂都对相同 RGB 使用相同映射，无法单独识别人物或背景。实现借用 [Pitié 等的颜色分布迁移思路](https://mural.maynoothuniversity.ie/id/eprint/15125/)，不含原论文的梯度后处理。[事前协议](../../experimental/adaptive-reference-20261113/sliced-color-distribution-protocol.md)

| 既有开发对照 | 线性匹配 | 分位数匹配 |
| --- | ---: | ---: |
| 背景变化与有意风格联合通过 | 0/4 | 0/4 |
| 背景组最大保护区色差，RGB8 | 41.31 | 44.46 |
| 白球服组最大保护区色差，RGB8 | 9.84 | 12.79 |
| FACD 联合通过 | 0/7 | 0/7 |
| FACD 相对 13 号的最大近中性色误差增量，RGB8 | 62.32 | 71.58 |
| ARRI Film B 联合通过 | 0/6 | 0/6 |
| 色卡达到响应要求的区域 | 1 | 1 |

背景组在分析图上的平均投影分位数距离，原图到参考为 59.07，线性匹配为 26.01，分位数匹配仅 1.86（归一化后换算为 8 位码值尺度，测于 80% 混合前）。分位数匹配的裁切像素比例更低，保护区成片反而更差。白球服的有意明暗与色度响应有所提高，但仍达不到联合要求。说明这两套全图方法的问题不只在匹配能力；增加对参考全图分布的贴合度也会复制场景差异。停止采用此固定全图分位数匹配，不调整迭代或强度追回分数。

两次运行的 144 张成片、36 张对照板及报告共 181 件逐件一致，核对全部图像收据。同图保持通过，已知统一颜色平移和重复 RGB 检查通过。未使用测试目标答案求映射，未切换软件或打开独立验收。见[数值与哈希](2026-11-19-sliced-color-distribution-evidence.json)与[白球服本机对照](../../outputs/adaptive-reference-20261113/sliced-color-distribution-v1/project/background/872-1268/comparison.png)。

复现：在已有研究环境设置仓库根目录为 `PYTHONPATH`、`OPENBLAS_NUM_THREADS=1`，执行 `python experimental/adaptive-reference-20261113/probe_sliced_color_distribution.py <新输出目录>`。
