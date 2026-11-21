# 两场实拍强色灯中，普通 CUBE 无法恢复部分环境光颜色

映色先前的局部调色试验能保住目标原片的细节，但人像肤色和白幕偏色仍未解决。为判断整图调色的表达能力，选取 CL3AN 发布的前两场实拍，每场各取第 1 张直接色灯、第 1 张直接白灯和对应的环境光照片。六张原图为 6000 × 4000 RGB8 PNG。原定用环境光照片估计全图通道增益，但三态有效像素交集分别只有 0.15% 和 23.69%，都低于事前规定的 50%。该拟合保留为未判定，没有替换样本或放宽门槛。[事前协议](../../experimental/adaptive-reference-20261113/cl3an-real-light-global-fit-protocol.md) · [作者论文](https://openaccess.thecvf.com/content/ICCV2025/html/Vasluianu_After_the_Party_Navigating_the_Mapping_From_Color_to_Ambient_ICCV_2025_paper.html)

由于强色灯图出现大量重复边界码值，补测固定色表所能达到的误差下界。每场按 5 像素间隔取 96 万个原始像素，把至少两个通道等于 0 或 255、且三个输入码值完全相同的位置归组。普通 CUBE 必须把同组位置映射到同一个颜色。即使事先知道全部环境光答案，每组各通道的中位数也已经是绝对误差最小的统一输出。其余像素都按零误差计入整图分母。

| 场景与输入灯 | 重复边界像素比例 | 重复边界像素误差下界 | 整张采样图误差下界 |
| --- | ---: | ---: | ---: |
| 1，色灯 | 45.16% | 22.91 | 10.35 |
| 1，白灯 | 0.21% | 6.19 | 0.013 |
| 2，色灯 | 21.91% | 22.95 | 5.03 |
| 2，白灯 | 0.03% | 18.67 | 0.006 |

误差单位为 RGB8。第一场的原图零基坐标 (3410, 450) 和 (2275, 440) 在色灯下都为 (0, 75, 0)，环境光下分别为 (28, 42, 15) 和 (224, 223, 217)。任何同一张逐像素色表都得给这两个位置相同输出。两个色灯样本的整图下界已超过项目原有 3 RGB8 参考线，可以排除用普通 CUBE 单独完成这两场的环境光复原。白灯组的下界很低，只说明本测量给出的约束弱。例中亮度及颜色差混合了直接光和阴影变化，不能据此推断普通偏色、人像肤色或实际参考风格也出现相同程度的丢失。[数值与源图哈希](2026-11-21-cl3an-pointwise-bound-evidence.json)

这条结果给主体、背景与空间调色的后续研究提供了实际约束。位置、邻域及材质线索可以区分部分 RGB 相同的位置。现在的证据只证明逐像素映射缺乏这种表达能力，仍需用局部方法的实际成片验证能否恢复颜色并保留应迁移的风格。稳定版、候选软件和独立验收集均未参与改动。

## 复现

从[作者测试目录](https://drive.google.com/drive/folders/1CKVX9z09lD4W5jxlEqWVe4eRz3-qCxhF?usp=drive_link) 获取 GT 中的 1_GT.png 和 2_GT.png，IN_CR 中每场第 1 张 IN，IN_SH 中每场第 1 张 SH_IN。将文件按原名保存到本机输入目录。六张原图共 189,695,450 字节，仅用于项目开发诊断。作者仓库列出的两个权重资源直链在本次核对中跳转登录页，实验没有使用任何模型权重。[作者仓库](https://github.com/fvasluianu97/RLN2)

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
& .tools/vision-advisor-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_cl3an_real_light_global_fit.py "$base/cl3an-real-light-input" "$base/cl3an-real-light-global-fit-repro"
~~~

使用 Python 3.12.14、Pillow 13.0.0 和 NumPy 2.5.4。重复运行约 4 秒，六张本地预览及 JSON 的 SHA-256 逐文件一致。另按码值逐组统计目标通道直方图，独立复核四个整图下界。合成恒等输入通过全图增益分支的自检。原片和预览仅保存在本机。成片目录为 `outputs/adaptive-reference-20261113/cl3an-real-light-global-fit-v2`。
