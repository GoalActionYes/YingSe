# 实体灰卡检查止于显影素材门槛

已确认原生网格能画出这两处暖色。此阶段找可见的现场光色依据，先核对真实灰卡到手后能否用于当前普通成片。Beyond RGB 白蓝场两档都有作者发布坐标，但灯 7 的第 20 号灰块只有 62.59% 的有效样本，低于事前 80% 门槛。因此此轮停止在灰卡素材筛查，没有计算灯 7 的整卡增益或生成白平衡候选。[事前配对和门槛](../../experimental/adaptive-reference-20261113/beyond-rgb-chart-anchor-protocol.md) · [白蓝场原片核验](2026-11-24-beyond-rgb-near-white-match.md)

## 开源资料的适用范围

VLM-CC 对 RAW 反复应用白平衡并检查临时渲染后剩余的色偏。公开仓库说明它先提供简化评估流程。GCC 用扩散模型在图里补一个色卡，为相机光色估计提供先验，其原实验对线性相机图像采用显影前的输入处理。两篇公开评估关注照明估计。未见两者在同一区域中同时压住真实换灯变化、保住有意局部暖色的对照。这类方法可能为后续提供先验，RAW、临时显影或生成色卡的效果还需要在已经拍摄与调色过的 RGB 参考上单独验证。本轮没有下载权重或调整现有环境。[VLM-CC 作者论文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_White-Balance_First_Adjust_Later_Cross-Camera_Color_Constancy_via_Vision-Language_Evaluation_CVPR_2026_paper.html) · [VLM-CC 作者实现](https://github.com/NothingIknow/VLM-CC) · [GCC 作者论文](https://openaccess.thecvf.com/content/CVPR2025/html/Chang_GCC_Generative_Color_Constancy_via_Diffusing_a_Color_Checker_CVPR_2025_paper.html) · [GCC 作者实现](https://github.com/chenwei891213/GCC-official)

## 已核验的灰块

| 实拍及编辑状态 | 灰块 20 有效覆盖 | 超出有效范围的样本 | 灰块 21–23 |
| --- | ---: | ---: | ---: |
| 灯 7，局部暖色关／开 | 62.59%／62.59% | 165／165，均为红通道大于 251 | 四态各 100% |
| 灯 1，局部暖色关／开 | 99.32%／99.32% | 3／3，均为红通道大于 251 | 四态各 100% |

每块按原取样网格评估 441 个坐标，取整后同一像素可以重复。灰块 20 在灯 7 有 360 个不同位置，在灯 1 有 358 个。独立回读线性显影裁切掩膜，灯 7 的 165 个无效样本中，31 个在重建的线性 sRGB 上已越界，44 个在最终 PNG 中达到 255。其余超过有效上限的样本仍保留。此处能确认部分高红值在显影中已经丢失余量，不能判断原始传感器或其未公布处理链是否裁切。[全部四态结果](2026-11-24-beyond-rgb-chart-anchor-evidence.json)

EGM 复核要求核对同灯开关的灰卡像素再讨论编辑是否影响增益。复查发现两档分别在四个固定灰块的开和关取样逐值相同。灯 7 的失败与局部暖色开关无关。原协议规定四块全过，本轮不更换色块、不调整门槛。两档共用白点能否在浅色背景上抑制换灯又保住有意暖色，仍未得到成片。当前检查只约束这套显影的白蓝场。灰卡定位来自作者记录，普通无卡参考和真人都还缺验证。未来若源素材过线，需同时报告原图、固定增益、逐图增益三者在共同有效区域的误差，才能判断逐图校正有没有独立收益。

## 复现

沿用研究环境与此前核验过的原四态 PNG。用新目录运行。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
& '.tools/adaptive-env/Scripts/python.exe' 'experimental/adaptive-reference-20261113/probe_beyond_rgb_chart_anchor.py' $base "$base/beyond-rgb-chart-anchor-replay"
~~~

前两遍主测约 0.14 和 0.12 秒，报告逐字节一致。独立用另一种双线性坐标写法，从落盘 RGB、作者色卡发布记录和显影裁切掩膜复算 16 条。首遍试运行曾在第一个失败灰块抛错，改为完成四张的全项诊断并写出素材失败报告，门槛没有调整。结果供素材与成像范围诊断，未新增训练、产品候选或独立验收数据。
