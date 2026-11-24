# CL3AN 两组同标称色温的布光对照

Beyond RGB 剩余三个测光区间再次超时，本机也没有独立新试拍。本次改用已核验的 CL3AN 原图作一个容易区分的对照。作者用固定 Canon R6 Mark II 拍摄，在同一布景分别开启白色软箱漫射光和白色直射光。两套均标称 6400 K，经色卡校准。这里比较的是灯具和布光共同变化产生的实际效果。灯位、阴影、亮度与光谱均可能不同，结论无法归因到单一因素。过去用这两场的强 RGB 彩灯测过普通 CUBE 的重建下界，本轮用两档白光比较局部暖色，回答的是不同问题。[旧取件与下界](2026-11-21-cl3an-pointwise-bound.md) · [作者论文第 3.1 节和表 1](https://openaccess.thecvf.com/content/ICCV2025/papers/Vasluianu_After_the_Party_Navigating_the_Mapping_From_Color_to_Ambient_ICCV_2025_paper.pdf) · [事前协议](../../experimental/adaptive-reference-20261113/cl3an-neutral-warmth-protocol.md)

四张源 PNG 均为 6000 × 4000 RGB，逐一核对已有哈希。四图具有相同的 PNG 色度和伽马标记，无逐档 EXIF 或 ICC。一律沿用 sRGB 显示假定和以前的 BOX 缩小方式到 1200 × 800，保持相对曝光。先只看漫射光图固定同一浅灰桌面、保护区及附近两个地标，再查看直射光。在第一场对照了布偶上缘与木板右边角，第二场对照了画册边缘的两个角。两档的材质和对应位置可辨。桌面的阴影变化属于布光差。两场整图最优位移均为零，零位梯度相关为 0.709 和 0.781。[冻结坐标](../../experimental/adaptive-reference-20261113/cl3an-neutral-warmth-regions.json)

| 场景 | 漫射换直射，测区 RGB8 变化 | 同位置有意暖色，RGB8 变化 | 余弦 | 幅度比 |
| --- | --- | --- | ---: | ---: |
| 1 | [−56, −59, −60] | [13, −3, −14] | 0.148 | 0.191 |
| 2 | [−67, −70, −68] | [13, −3, −14] | 0.127 | 0.163 |

原阈值要求三态共同有效覆盖至少 0.80、余弦至少 0.90、幅度比 0.75–1.25。四张的暖色开关均通过素材线，四个方向的共同覆盖均为 1.00。正反向合计 0／4 匹配，四个方向都在方向和幅度两项失败。布光带来的三通道变化大致同向，前向登记区的亮度变化约为 −58 和 −69 RGB8。保护区同时发生不同程度的光照变化。此结果适用于两个固定桌面区域，源于同一个实验室。这两组增加了可用的局部暖色开关，未补齐暖墙匹配实拍，也没有改变产品。作者提供公开研究文件，项目保留媒体在本机学术诊断，未核实产品训练授权。[冻结数值与身份](2026-11-24-cl3an-neutral-warmth-evidence.json) · [完整暖色门槛](../../experimental/adaptive-reference-20261113/shared-local-warmth-protocol.md) · [作者发布与许可说明](https://github.com/fvasluianu97/RLN2#license)

EGM 复核确认两档各自的素材线通过后才比较三态，并建议把中性桌面作为独立场景类型保留。即使数字通过，也需与现有 MIT 暖墙对照核实材质与来源，才讨论跨来源开发。目前保留失败方向，结束这两场取样。[本机 EGM 报告](../../outputs/adaptive-reference-20261113/cl3an-neutral-warmth-egm.md)

## 复现

将四张来源按旧证据放进项目已有的 `cl3an-real-light-input`，使用现有暖色预检的研究 Python、RawTherapee 5.13 与 ASCII 映射盘。参照[预检的本机路径配置](2026-11-24-warm-capture-preflight.md)，在同一映射盘用以下命令和一个新的输出目录。

~~~powershell
& V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/probe_cl3an_neutral_warmth.py 'V:/outputs/adaptive-reference-20261113/cl3an-neutral-warmth-new'
~~~

两次从 Python 启动到报告写完分别为 29.6 秒、16.1 秒。两遍各 26 张 PNG、4 份 PP3 和 3 份 JSON 哈希一致。两遍共八个方向均从落盘成片独立复算，核对有效覆盖、三通道方向和幅度，以及输出 ICC。[本机第一场成片](../../outputs/adaptive-reference-20261113/cl3an-neutral-warmth-run-1/1/render/comparison.png) · [本机第二场成片](../../outputs/adaptive-reference-20261113/cl3an-neutral-warmth-run-1/2/render/comparison.png)
