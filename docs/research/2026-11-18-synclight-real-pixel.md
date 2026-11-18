# 两个实拍房间的冷灯迁移基线

从已固定的 SyncLight 发布按成员取出八份 DNG，共 106,973,902 字节。每个房间用视角 1 作参考、视角 2 作目标，把实拍单灯的线性 RGB 乘以同一个冷灯系数。每场用同一白平衡与曝光。13 号分别用原灯版和冷灯版作参考出图，测量两份输出的增量。目标自身合成的冷灯增量用于评分，强度同为 80%。见[事前协议](../../experimental/adaptive-reference-20261113/synclight-real-pixel-protocol.md)和[作者发布](https://huggingface.co/datasets/davidserra9/synclight/tree/8067588dc4ee901b805c3813d234a661973f8e71)。

八份原片均为 iPhone 13，RAW 可见区域为 4032×3024。每个同机位配对的快门、光圈、ISO、黑白电平、Bayer 排列与转换矩阵一致。四个配对的梯度相关为 0.529–0.685，最佳平移都在零位，通过事前粗配准门槛。首次读取遇到 DNG 白平衡记录的第二个绿色通道为 0，按同一绿色系数补齐。另确认这批 DNG 的 `rgb_xyz_matrix` 为空，读取非零的 `color_matrix` 做同场一致性核对。两次诊断都在有效出图前结束，最终没有改曝光、灯色或评分门槛。

| 场景与测量区域 | 合格像素 | 冷灯方向投影 | 13 号增量误差 | 不迁移误差 | 通过 |
| --- | ---: | ---: | ---: | ---: | --- |
| scene35 走廊，整图 | 196,608 | 0.870 | 5.96 | 8.43 | 是 |
| scene35 走廊，近中性色 | 115 | — | — | — | 覆盖不足 |
| scene25 床头，整图 | 196,608 | 0.493 | 2.42 | 5.04 | 否 |
| scene25 床头，近中性色 | 23,741 | 0.608 | 1.46 | 3.41 | 否 |

误差单位为 RGB8 码值。床头近中性色虽然比完全不迁移更接近预期，冷灯响应仍低于 0.75 门槛。两组低受光区均为零像素，强受光区覆盖几乎全图，所以本批不能检验局部光色隔离。需有同时包含受光和较少受光区域的已知变化，才能验证软件是否把冷色局限在该灯的范围。当前两组可以作为跨视角换灯强度的开发基线，未验证肤色、作者风格或独立画质。

两遍 12 张 PNG、两份受光比例数组及报告哈希均一致。独立从 PNG 与比例数组复算八项状态或评分，并逐一复核下载成员 SHA-256。本机图板在 `outputs/adaptive-reference-20261113/synclight-real-pixel-v3/overview.png`。见[文件身份与评分](2026-11-18-synclight-real-pixel-evidence.json)。

复现用[取样脚本](../../experimental/adaptive-reference-20261113/fetch_synclight_real_pixels.py)传入固定的准入证据和新素材目录。使用从 [PyPI 固定发布](https://pypi.org/project/rawpy/0.28.0/)核对 SHA-256 的 RawPy 0.28.0 Windows CPython 3.12 wheel，仅装到隔离目录，复用研究环境已有的 NumPy 和渲染依赖。然后以研究 Python 执行[出图脚本](../../experimental/adaptive-reference-20261113/probe_synclight_real_pixels.py)，依次传入隔离解码器目录、原片目录及新输出目录。原片、wheel、模型和成片留在本机。
