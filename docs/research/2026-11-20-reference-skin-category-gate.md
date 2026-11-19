# 肤色偏差不会因移除分类第一名限制而改善

同一参考人物换背景后，两组目标肤色曾因借用另一张参考的选区而改善。把定位输入灰度化则导致漏检。本轮检查另一个可能的损失来源：现有代码先合计脸部和身体皮肤分数，随后只保留单类首选为皮肤的像素。四组各两背景、两编辑状态，固定原彩色参考和目标选区，单独移除这项参考端限制。用原置信度曲线、平滑、最小覆盖、回退和 80% 强度出图。每个原选区和对照成片都先逐值复现。[事前协议](../../experimental/adaptive-reference-20261113/reference-skin-category-gate-protocol.md) · [灰度实验](2026-11-20-reference-skin-gray-locator.md)

| 编辑打开时的目标区域 | 原彩色定位 | 移除第一名限制 |
| --- | ---: | ---: |
| 785 脸部 | 6.750 | 6.750 |
| 9483 前额 | 6.667 | 6.625 |
| 9483 手背 | 7.071 | 7.054 |
| 白球服场景的人物手臂 | 1.161 | 1.181 |
| 8021 投影幕中的脸 | 3.370 | 3.379 |

单位为 RGB8 色度码值，度量同一参考人物更换背景后目标区域的变化。重点六项改善 0/6，最大下降 0.042 RGB8。其余没有越过 0.25 RGB8 的退化限。整组 16 张图共 256 个像素与原支路不同。八组换背景权重平均绝对差最多只移动了 0.016 个百分点，远小于上轮重点组 7.34–9.76 个百分点的原差异。结论限于所测第一名限制，保留其余阈值及处理影响的未知项。

Google 文档列出身体皮肤和脸部皮肤两类，没有拍摄归属类别。本地原图结果符合这一任务边界：9483 手背 28/28、白球服场景前景人物手臂 35/35 的登记像素首选身体皮肤；投影幕脸部 110/110 首选脸部皮肤，平均脸部皮肤分数 0.874，落在两处直接拍摄脸部小框的 0.826 与 0.907 之间。远处脸颊 28 个像素中，12 个首选脸部皮肤，13 个首选身体皮肤，3 个首选背景。只留脸部类别会牺牲手背和手臂，也保留这个屏幕反例。[MediaPipe 类别与输出说明](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter#multi-class_selfie_segmentation_model) · [Google 模型卡](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf) · [既有屏幕及人物上下文对照](2026-11-19-skin-person-context.md)

停止通过改变现有定位器输入或移除类别门槛获得稳定肤色的尝试。下一步若继续做区域归属，需要此前缺少的显示平面及前景证据，或者能从同一输入中单独获得可验证的调色意图。现有四组只作开发对照。原有软件未改，人评、独立画质及视频仍待验证。

两个独立进程的 16 张成片及报告逐件一致。同图保持和纯色无人回退通过。报告 SHA256 为 `5756e4fe861424db907ff46a3e357a6a701c67fc3e33f50b63ce10a339123449`。在仓库根目录设定 `PYTHONPATH`，用现有 `.tools/vision-advisor-env/Scripts/python.exe` 运行脚本，传入新输出目录和 `experimental/skin-protection-20261019/runtime/Lib/site-packages`。[复现脚本](../../experimental/adaptive-reference-20261113/probe_reference_skin_category_gate.py)
