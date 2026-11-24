# 用 SIDD 原始 DNG 核验同场景色温对照

现有跨场景工作台会把参考拍摄的暖色错带到目标浅色表面，也会漏掉有意编辑。FAU 两灯位扫描已停止。本轮只问能否取得一组可核验、跨色温的同机位拍摄用于后续开发预审。SIDD 独立相机档案提供物理灯档、标注曝光和原始 DNG，允许用统一显影保留实际变化。SIDD 的设置只保证同一连拍内固定，跨连拍需要从原片复核。[FAU 收尾](../../docs/research/2026-11-24-fau-shared-warmth.md) · [SIDD 采集论文，第 3.1 节](https://openaccess.thecvf.com/content_cvpr_2018/papers/Abdelhamed_A_High-Quality_Denoising_CVPR_2018_paper.pdf) · [SIDD 原始索引](https://abdokamel.github.io/sidd/files/SIDD_URLs.txt)

选定原始场景 001 的 iPhone 7、ISO 100、1/160 秒、正常亮度两组。实例 0028 为 5500 K，0033 为 3200 K。选这组是因为同一可访问发布同时提供两档，且同类 iPhone 高 ISO 组合缺一档。Samsung 候选不混进本轮。每档从 150 个以文件名排序的 DNG 中取第 1、76 张，固定之前不读照片。四张兼作候选和同灯复拍检查。使用 SIDD-CC-DNG 版本 6d0169e354f11ce3d98c1ffa214f9acf9d373155 发布的原始 DNG 重分发。数据卡说明须遵从 SIDD 条款，原作者公开页列为 MIT。素材仅保留本机，记录压缩包身份、成员 CRC 与本地 SHA-256。下载上限 64 MiB、出图和核验预算 20–35 分钟。只接受精确 HTTP 206 区间。任一档无法核验时保留已取得的事实，结束本轮，不取全档或改用现成的自动白平衡预览。[原始许可](https://abdokamel.github.io/sidd/#license) · [DNG 发布说明](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC-DNG)

先检查原始 EXIF 机型、快门、光圈、ISO、焦距、方向和显影所需的传感器几何，记录各档相机白平衡元数据。沿用既有 RawPy 回放，以 0028 第一张的白平衡固定四张输出。关自动白平衡和自动提亮，保持线性 sRGB 后按原函数编码。复用原轮廓配准，最多 1 个输出像素的位移，零位梯度相关至少 0.50；无论是否通过均记录。对照同灯复拍的差异及跨灯信号。未证明同场景对应或相机设置一致时不进入暖色开关预审。现有全部画质、素材和匹配门槛继续适用。即使物理对照过线，也只取一个静物场景供开发，后续在锚图固定测区、按既有开关和比例门槛决定是否值得进入工作台。[现有 DNG 处理与配准](probe_synclight_real_pixels.py) · [有意暖色匹配协议](shared-local-warmth-protocol.md)
