# 按 ICC 读取图片可以消除一类参考调色偏差

工程接入核对发现，实际 13 号工作台已有独立严格 ICC 转换。本试验为控制输入直接调用共用解码与 13 号渲染，反映的是共用解码缺口。后续候选修复的是稳定工作台，13 号保持原策略。[工程验证](https://github.com/GoalActionYes/YingSe/blob/736a32dfcc8b2a2efac5f2a6577c8c2eb7aef7b7/docs/research/2026-11-20-image-icc-candidate.md)

现有图片解码会校验位深、按 EXIF 摆正图片并转换到 Pillow 的 RGB 模式，但未使用嵌入的色域定义。这意味着同一种显示颜色用 sRGB 或 Display P3 保存，映色可能读取出不同的颜色。这轮把四组实拍的 16 张参考从冻结的 sRGB 工作编码改存为带 Display P3 标签的 RGB8 PNG。固定目标、13 号、80% 强度和既有编辑状态，比较直接读取 P3 数值与按嵌入配置换回 sRGB 的成片。所有原 sRGB 成片逐像素复现。[事前协议](../../experimental/adaptive-reference-20261113/reference-icc-invariance-protocol.md) · [Pillow ICC 转换接口](https://pillow.readthedocs.io/en/stable/reference/ImageCms.html#PIL.ImageCms.profileToProfile)

| 只换参考文件编码时的成片区域 | 忽略 ICC | 按 ICC 转换 |
| --- | ---: | ---: |
| 白球服场景，前景人物手臂 | 2.04 | 0.06 |
| 同场景，后方人物脸颊 | 2.45 | 0.04 |
| 同场景，白球服 | 0.42 | 0.01 |
| 9483，另一背景参考下的手背 | 3.45 | 0.05 |
| 9483，同一参考下的浅墙 | 0.40 | 0.03 |
| 8021，投影幕中人脸 | 8.56 | 0.08 |

表中为编辑开启时相对原 sRGB 成片的局部色度误差，单位为 RGB8。16 张中，13 张旧路径的整图平均绝对误差达到 0.5；这 13 张全都在转换后至少减半并低于 1。全部成片转换后的最大整图误差为 0.125，参考本身因往返量化引入的最大平均误差为 0.155。白球服场景 C1 的有意编辑投影，冻结原结果 0.223，忽略 ICC 时 0.203，按 ICC 转换后 0.222。这里恢复的是原工作空间下的反应，未改善既有响应不足。[本机白球服成片](../../outputs/adaptive-reference-20261113/reference-icc-invariance-v1/872-1268/comparison.png) · [本机浅墙与手背](../../outputs/adaptive-reference-20261113/reference-icc-invariance-v1/9483-19432/comparison.png)

正确标记的 sRGB 和未嵌入 ICC 的控制逐像素保持。实验只改变参考文件编码，P3 参考由同一实拍转换而来，不能外推为所有来源、所有配置或宽色域导出的验证。下一阶段可在独立分支做图片解码候选，将分析、预览和导出纳入端到端核验，并验证没有配置或遇到不可用配置时的现有兼容性。该工程候选不能冒充开放风格迁移通过。本轮现有软件功能保留，尚未验证真实第三方宽色域照片、目标图片的同类处理或视频结果。

DisplayP3-v4 配置 480 字节，Git blob 为 `4ee40fb5eeb6dfde851c989ecc7c3ebd496ba9fc`，SHA256 为 `cb51de38e482ee974c0c76b9689e16aad04bad16e226fed2f30c842d15ff3a3d`，本机存放在 `outputs/adaptive-reference-20261113/icc-reference-assets-v1/DisplayP3-v4.icc`。首跑因为当前 Pillow 不接受 `Path` 参数，在生成任何成片前退出。改用同一配置的内存字节后，两遍各 16 张编码参考、32 张成片、四张图板、两个控制和报告逐件同哈希。报告 SHA256 为 `341a4947f76555e8523cd6942b2641bd6e34783382aa6e74b147ff3b3e9c22ec`。用既有视觉环境，设仓库根目录为 `PYTHONPATH` 和 `OPENBLAS_NUM_THREADS=2`、`OMP_NUM_THREADS=2`，运行脚本并传入新输出目录。[固定配置及 CC0 许可](https://github.com/saucecontrol/Compact-ICC-Profiles/tree/bdd84663061bc4ae95ca70decff54f581e27f702) · [复现脚本](../../experimental/adaptive-reference-20261113/probe_reference_icc_invariance.py)
