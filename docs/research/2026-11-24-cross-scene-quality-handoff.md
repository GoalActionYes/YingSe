# 映色跨场景画质研究阶段交接

截至研究提交 `946a377`，稳定版 1.3 和自适应研究版 13 可供课程演示。原目标仍是让用户各给一张普通参考和一张目标，软件减少跨场景肤色、白墙和其他中性色偏色，同时保留有意迁移的色彩、明暗和饱和度。现有跨场景候选未过开发门槛。当前交接保留软件、成片、失败记录和恢复条件。[课程体验入口](../../00-从这里开始/0-体验与课程交付.md) · [研究进度](../RESEARCH_PROGRESS.md)

## 可以怎么汇报

我们从参考调色的全局变换做起，后来加入让人物和背景各处能用不同颜色变换的空间渲染。现在能做自动参考调色、预览、手动微调、撤销重做和图片或视频导出。同原片的研究候选还能根据共同细节重放局部编辑，且有实际重拍和异图回退记录。跨场景还要判断参考中的肤色和墙面变化究竟是摄影现场的灯光，还是作者想迁移的风格。这一步仍会犯错。普通 CUBE 保存全局颜色映射，依赖位置的完整效果需要软件渲染。[发展过程](2026-11-17-project-story.md) · [同原片候选](https://github.com/GoalActionYes/YingSe/blob/3ea0143552abc0977b2df0af2966b3b47cb3b4b5/docs/research/2026-11-24-same-capture-workbench.md) · [导出边界](../ROUTE_DECISIONS.md)

可用两张真实对照说明难点。MIT 复印室同一块墙，实拍换灯和有意加暖在墙面上的变化接近。把两种参考交给两版软件，稳定版 1.3 与 13 号都只有 0／2 个暖墙目标通过，3 个人像保护对照也均未通过。13 号只恢复了 5.50%–6.66% 的期望暖色，某处脸颊使用换灯参考后相对目标原图偏差达到 24.43 RGB8。同图对照在两版逐像素保持。当前本机启动文件、便携 ZIP 与这两张图板已核对存在，实测范围沿用各原报告。[两版实际工作台结果](2026-11-24-matched-wall-warmth-workbench.md) · [本机墙面图板](../../outputs/adaptive-reference-20261113/matched-wall-warmth-workbench-v1/spatial13/wildrelight-wall/comparison.png) · [本机人像图板](../../outputs/adaptive-reference-20261113/matched-wall-warmth-workbench-v1/spatial13/gros-2/comparison.png)

## 哪些方向结束了

- 把前景和背景分开、固定去色或调弱强度，可以降低某些串色，也会压掉有意调色。加入参考局部位置的两来源亮度训练降低了部分换光误差，但 18 个留出方向均未恢复需要的提亮。单张参考的冻结特征读出也未过门槛，不能当成自动开关。[阶段路线复盘](2026-11-18-route-handoff.md) · [位置训练出图](2026-11-24-positional-lightness.md) · [单张参考读出](2026-11-24-reference-signal-readout.md)
- 暖色路线已有 MIT 的匹配参考，多个目标共用一个匹配来源。LSMI、FAU、CL3AN、Ribeira 等可出片的真实条件均未补齐第二个方向和幅度达标的来源。最近一次 Ribeira 的墙亮度升约 7.8%，第二档的有效测区只有 33.81%。[LSMI](2026-11-24-lsmi-daylight-switch.md) · [FAU](2026-11-24-fau-shared-warmth.md) · [CL3AN](2026-11-24-cl3an-neutral-warmth.md) · [Ribeira](2026-11-24-ribeira-daylight.md)
- SFU 缺所选浅色容器；CML 实景墙面缺两档相机曝光记录。VisibleFixture-60 固定机位但使用自动曝光，121 个公开 EXR 头未保留所需参数。本次复查作者给出的原件目录仍返回 0 个对象。剑桥多光谱项目的固定版本发布了色卡测量与计算代码，Laposi 的相机研究以单一固定 3200 K 布光比较不同相机。这些来源本次均无法提供所需配对。[SFU 与 CML](2026-11-24-warm-white-source-intake.md) · [VisibleFixture-60](2026-11-24-visiblefixture60-exr-intake.md) · [剑桥发布](https://github.com/gfxdisp/multispectral_color_correction/tree/922108056ebc907868ff44c9158c10989ab2fcbb) · [Laposi 拍摄说明](https://davidlaposi.com/camera-study/)

## 恢复条件与剩余验收

EGM 复核确认，第二个匹配来源约束的是这条跨来源监督路线。适用的最小新增输入是另一独立拍摄来源的同一浅色墙面两档实拍。固定曝光、白平衡和机位，只改变有记录的暖白补光档位，保留原件和共同显影。可记录等效曝光校正的原始文件也可先做来源核验。优先使用能核对这些条件的真实补拍，或取得已定位的 CML 两档同曝光原件及记录。首组到件后约 30–60 分钟可按已实现的预检制作四图，检查方向、幅度、保护区和端点。只有两档开关、共同有效区及配对门槛均满足，才进入训练前小测。[已有预检](2026-11-24-warm-capture-preflight.md) · [完整研究设计](2026-11-24-spatial-condition-design.md)

现阶段课程仍可按已有软件和实图交付，跨场景画质研究等待上述新输入。训练路线恢复后，先测步速及显存，首轮两臂训练、出图和核验合计最多 4 小时机器时间。开发候选须同时通过肤色、浅墙、参考响应、明显退化和关键负例。之后才做封存独立图像、真人自然评价及含运动、遮挡和切镜的完整视频处理验证。通过后再接入可回退候选和提交 PR。当前没有达到这些条件的跨场景候选，未进行新的正式真人或视频验收。[工期与评审计划](2026-11-21-phase-cost-and-gates.md) · [原开发交叉门槛](2026-11-19-frozen-candidate-development-crosscheck.md)
