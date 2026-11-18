# 同一相机风格切换在不同白平衡下的变化

从作者固定发布取得同一实拍原片的十张渲染图。五种白平衡设定都有 AS 和 CS 两个配置。用 D 的 `CS − AS` 作为固定残差，逐像素加到其余四种白平衡的 AS 成片上，比较作者发布的 CS。这个离线实验能判断固定残差是否可跨白平衡复用，用到了产品两图模式没有的编辑对。样片用于本机非商业研究。未运行作者模型或映色模型。[作者固定发布](https://github.com/mahmoudnafifi/Deep_White_Balance/tree/d70afa227c94152fd8421db75b3ee6ac06a417dc/dataset) · [许可](https://github.com/mahmoudnafifi/Deep_White_Balance/blob/d70afa227c94152fd8421db75b3ee6ac06a417dc/LICENSE.md) · [事前协议](../../experimental/adaptive-reference-20261113/rendered-wb-profile-cross-protocol.md)

十张皆为原发布的 813 × 541 RGB8 PNG，无嵌入 ICC。D 的 AS 与余下九张的亮度梯度零位相关为 0.9951–0.9990，±3 像素内均在零位最优。公开清单核对 10 个 Git blob，共 5,645,323 字节。以下按发布 RGB 码值测量。

| 留出的白平衡 | 整图误差，原图 → 复用 | 共同近中性色误差，原图 → 复用 | 联合通过 |
| --- | ---: | ---: | --- |
| C | 14.35 → 0.85 | 15.50 → 0.78 | 是 |
| F | 12.20 → 2.56 | 15.84 → 2.48 | 否 |
| S | 14.76 → 1.64 | 10.64 → 1.11 | 否 |
| T | 12.29 → 3.75 | 9.39 → 7.19 | 否 |

只有 C 同时低于 1.5 码值门槛。T 明显失利，近中性色仍剩 7.19 码值误差。已发布样片有一块黑色遮盖区域。另行剔除十张全部为纯黑的 63,936 个像素，占 14.54%，四个留出设定的整图误差仍分别为 1.00、3.00、1.92 和 4.39，结论一致。该复算保留原先门槛与共同近中性色统计。照片无可量化肤色，浅墙也无作者中性标定。[T 的本机出图](../../outputs/adaptive-reference-20261113/rendered-wb-profile-cross/T-comparison.png)

同一照片的独立渲染已足以说明，这种固定相加的风格残差对部分白平衡设置会失效。停止用同一残差为其他白平衡制作伪标签。素材只有一个拍摄内容，保留为相机渲染诊断，画质主线仍需要跨拍摄组的可核验编辑和肤色、白墙标记。

两次离线计算的八对 PNG 及结果 JSON 完全一致。另从落盘图按像素重新计算四项整图误差。复现时将仓库根目录纳入 `PYTHONPATH`，用现有研究解释器执行 `probe_rendered_wb_profile_cross.py <新素材目录> <新输出目录> --fetch`，需要时用 `--proxy` 指定本机代理。复跑已有素材省略 `--fetch`。[脚本](../../experimental/adaptive-reference-20261113/probe_rendered_wb_profile_cross.py) · [数值与文件哈希](2026-11-19-rendered-wb-profile-cross-evidence.json)
