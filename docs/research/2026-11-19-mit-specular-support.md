# 已知真实反光位置后，亮边交点仍不可靠

用 MIT Intrinsic Images 的两只青蛙模型、一个杯子和一张皱纸验证。作者用交叉偏振分别拍摄反光最大和最小的图，再由两图差计算镜面分量。取作者图集 16 张 PNG，共 1,192,242 字节，四组的原图、漫反射图、镜面分量和掩膜尺寸一致。掩膜内的 `原图 − 漫反射 − 2 × 镜面分量` 的平均绝对残差为 0.39–2.12 个满幅 255 码值，均通过 3 码值门槛。图集无色彩配置文件，反光色只在发布的相机通道空间比较。[作者论文，第 3 节](https://people.csail.mit.edu/billf/publications/Ground_Truth_Dataset.pdf) · [作者图集](https://www.cs.toronto.edu/~rgrosse/intrinsic/gallery.html) · [事前协议](../../experimental/adaptive-reference-20261113/mit-specular-support-protocol.md)

两边都先限制在同一物体里，左侧用旧方法取得的全部亮暗对，右侧用独立偏振分量筛出有足够反光增量的点对。只改取样范围，交点算法和门槛相同。[本机两臂定位对照](../../outputs/adaptive-reference-20261113/mit-specular-support-v1/comparison.png)

| 场景 | 物体内亮暗对 | 通过偏振反光条件 | 筛选后结论 |
| --- | ---: | ---: | --- |
| 青蛙 1 | 2,075 | 148（7.13%） | 块均衡后剩 120 对，低于 128 |
| 青蛙 2 | 3,885 | 430（11.07%） | 剩 395 对，前向一致率 19.43%，低于 55% |
| 杯子 | 953 | 29（3.04%） | 点对分散，无合格块 |
| 皱纸对照 | 5,393 | 278（5.15%） | 支持方向过于集中 |

三件物体为 0/3，未达到预设的至少两件。青蛙 2 的偏振筛选提供了足够点对，交点仍与高光增强的方向不一致。杯子原来许多采样点在印花边缘。由此结束固定亮边交点和围绕它补自动定位的方向。结果针对这套交点方法，已经测得的反光分量仍可供离线诊断。没有依据将真实反光标签当作参考风格标签，未更改软件或读取独立验收素材。

复现先运行 `fetch_mit_specular_samples.py <本机输入目录>`，再用原研究环境设置仓库根目录为 `PYTHONPATH` 并运行 `probe_mit_specular_support.py <本机输入目录> <新输出目录>`。两遍的定位图和报告逐字节一致；共用拟合函数抽出后，上一轮四张定位图与 JSON 也逐字节复现。媒体仅本机学术使用，作者未给出明确分发许可。[脚本](../../experimental/adaptive-reference-20261113/probe_mit_specular_support.py) · [来源、数值与哈希](2026-11-19-mit-specular-support-evidence.json)
