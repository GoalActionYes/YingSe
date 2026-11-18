# 新增两间受光训练房间，尚未达到跨房间拟合门槛

固定从上游实拍训练目录按包长选择四个未使用的归档，只取视角 1 的环境光与可见单灯 DNG。两场通过同机位配准和强弱受光覆盖。事前要求至少三场才拟合，因此本轮完成了素材准备，未启动训练或给留出房间打分。走廊与床头仍保留为已消费的开发留出。见[固定协议](../../experimental/adaptive-reference-20261113/synclight-multi-room-protocol.md)与[固定发布](https://huggingface.co/datasets/davidserra9/synclight/tree/8067588dc4ee901b805c3813d234a661973f8e71)。

| 房间 | 同机位梯度相关 | 弱／强受光覆盖 | 处理 |
| --- | ---: | ---: | --- |
| scene22 | 0.657 | 5.70%／10.31% | 保留 |
| scene18 | — | — | 发布原片为 CR2，超出本轮固定 DNG 范围 |
| scene37 | 0.513 | 5.26%／21.14% | 保留 |
| scene24 | 0.405 | — | 低于 0.50 配准门槛 |

实际取六份 DNG，共 81,361,888 字节。两遍准备报告、两张 RGB8 训练图和两份受光真值数组哈希一致。另逐项复核原片哈希、两处覆盖值，以及新增与既有八份 DNG 的精确哈希无交集。尚无新预测或调色结论。见[数据身份与核验](2026-11-18-synclight-multi-room-evidence.json)。

本轮为了复用已验证的 DNG 解码，事先排除了其他 RAW 格式。该决定使便宜的 scene18 无法计入可用训练场景；它仍可单独作为以后核查 CR2 解码的线索。后续若补场景，需要固定独立房间、格式和配准规则，并继续保留原两组开发留出。

用[定向取样](../../experimental/adaptive-reference-20261113/prepare_synclight_multi_room.py)读取本机原发布元数据，再用[预处理与拟合脚本](../../experimental/adaptive-reference-20261113/probe_synclight_multi_room.py)的 `prepare` 子命令传入隔离解码器目录、原片目录和新输出目录。少于三场时，`fit` 会退出。取样、照片、解码器及图板保存在本机 `outputs/adaptive-reference-20261113/synclight-multi-room-*`。
