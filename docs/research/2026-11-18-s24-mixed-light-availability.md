# S24 有混合光区域，仍缺次光源颜色标注

上一篇 S24 试验的六张图蒙版全空。本轮固定同一发布，只读训练段前 100 个编号的目录，排除已知空蒙版哈希，按编号先后取六份蒙版和对应 JSON，合计 99,237 字节，包含两份索引。没有取得这一批原始照片或预设成片。见[运行前协议](../../experimental/adaptive-reference-20261113/s24-mixed-light-availability-protocol.md)和[固定衍生发布](https://huggingface.co/datasets/Lukhausen/S24-Raw-sRGB-2k/tree/c233b3ab441d24b8fd57c33b0ab99a205fc8b84b)。

六张蒙版均为 2000×1500，非零部分的覆盖率依次为 0.82%、43.05%、0.40%、1.97%、1.85% 和 4.02%。`img_0002` 与 `img_0014` 达到覆盖面积及相邻非零像素条件。六份 JSON 均只有主光源的中性白平衡和偏好白平衡，以及相机估计和校准字段。没有用于给次光源区域定色的另一组照明真值。上游也明确区分了二值区域标注与用于评估全图白平衡的主光源。因此满足全部预先条件的仍为 0/6，保留区域线索，结束本来源作为带独立次光源颜色真值的小样。见[上游字段说明](https://github.com/SamsungLabs/time-aware-awb/blob/55d193db82fe0d92738a22833657c3cfa58468ab/s24-raw-srgb-dataset/README.md)和[校验记录](2026-11-18-s24-mixed-light-evidence.json)。

12 份对象均重核发布长度与 Git 或 LFS 哈希，两遍结果逐字节相同，并以 Pillow 直方图独立复核六张蒙版的像素数。独立复核最初假定源文件全部为 8 位灰度，遇到 1 位蒙版后改用灰度转换再统计，没有改变蒙版判定或源文件。使用[核验脚本](../../experimental/adaptive-reference-20261113/audit_s24_mixed_light_availability.py)传入本机下载目录和新 JSON 路径可复核。原蒙版与完整标签仅留本机。

## 可继续的实拍来源

FAU 的双光源数据同时提供像素级照明真值和两个光源的相对影响。对作者公开的存档端点做了只读 HEAD，返回 200、42,311,680 字节，声明支持字节范围，尚未取档案。作者区分实验室真值与外景近似值，并要求引用来源。后续可先检查存档里小量实验室场景的图、标签和值域是否一一对应，再决定是否用于白墙偏色诊断。这类资料检验物体本色与局部光照，不直接衡量风格迁移。见[FAU 发布页](https://www5.cs.fau.de/research/data/two-illuminant-dataset-with-computed-ground-truth/)。

另一套 MIMI 提供色卡白点，但作者公布的是需填表获取的大归档。这轮未提交表单或下载。软件保持现有版本，未新增画质候选。见[MIMI 数据说明](https://github.com/Spectricity/MIMI-dataset)。
