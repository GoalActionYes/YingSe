# 从已调色参考估计底图的两图试验未通过

上一轮在红岩参考与雕塑目标 `a0007-IMG_2480` 上取得改善，但空间拟合需要参考原图。本轮用本机已有的 CanonCGT E2E 归一化模块从已调色参考估计底图，沿用空间网络、1000 步拟合和 80% 混合。CanonCGT 的归一化目标是统一颜色域。估计底图、目标输出及哈希均在读取真实参考原图和目标作者效果之前保存。见 [试验协议](../../experimental/adaptive-reference-20261113/inretouch-inferred-original-protocol.md)和 [CanonCGT 作者仓库](https://github.com/Jinwon-Ko/CanonCGT)。

以下为相对作者调色结果的 RGB8 平均绝对误差，单位为码值，越低越接近。近中性色沿用上一轮的近中性色代理区域。

| 处理方式 | 整图 | 近中性色 |
| --- | ---: | ---: |
| 13 号研究版 | 12.764 | 13.010 |
| 真实参考前后图，全局二次拟合 | 13.352 | 13.802 |
| 真实参考前后图，空间拟合 | 6.969 | 7.083 |
| 真实参考前后图，复跑空间拟合 | 7.036 | 7.176 |
| 已调色参考与估计底图，空间拟合 | 19.152 | 20.125 |

两图方案未通过预设门槛。估计底图与真实参考原图相差 20.716 码值。用估计底图把红岩参考拟合回已调色参考时，参考自身的重建误差只有 0.803，真实原图方案为 4.122，但估计底图方案迁移到雕塑场景后，石墙明显偏冷。参考自身的拟合误差无法预测这组跨场景迁移的表现。

独立复算了 10 项误差并核对了 7 份输出文件的哈希。默认路径使用同一随机种子复跑，仍通过原单例门槛，两次目标输出之间的 RGB 平均绝对差为 1.090，未达到逐像素复现。本机两图方案的 9.75 秒覆盖拟合、渲染、写盘和评分，未计归一化与输入准备。见 [数值与哈希](2026-11-18-inretouch-inferred-original-evidence.json)。

结束这套固定的归一化底图方案。此样例用于开发诊断，未验证人物肤色、其他场景、视频或完整独立评估。课程演示继续使用现有版本。见 [空间拟合正对照](2026-11-18-inretouch-spatial-fit.md)和 [课程交付入口](../../00-从这里开始/0-体验与课程交付.md)。

复现使用既有作者示例、固定网络文件及本机 CanonCGT 权重。在项目根目录设置 `$env:PYTHONPATH = (Get-Location).Path`，用 `.tools/vision-advisor-env/Scripts/python.exe` 运行 `experimental/adaptive-reference-20261113/probe_inretouch_spatial_fit.py <既有示例目录> <固定作者网络文件> <新输出目录> --inferred-original`。移除 `--inferred-original` 可复跑真实参考原图对照。两图输出在 `outputs/adaptive-reference-20261113/inretouch-inferred-original-v1`，对照复跑在 `outputs/adaptive-reference-20261113/inretouch-spatial-fit-default-regression-v1`。输入、生成图片和权重保留在本机。
