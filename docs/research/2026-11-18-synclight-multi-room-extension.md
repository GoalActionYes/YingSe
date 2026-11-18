# 三个补充房间均未进入训练

复用上一轮合格的 scene22 和 scene37。固定补查此前未读取的 CR2 场景 scene18，以及按发布体积预选的 scene23 和 scene36。按 TAR 成员取六份原片，共 93,930,594 字节。CR2 可用现有 RawPy 解码，三个场景的同机位元数据检查均通过。固有配准与覆盖问题发生在格式读取之后。见[事前协议](../../experimental/adaptive-reference-20261113/synclight-multi-room-extension-protocol.md)和[上一轮素材记录](2026-11-18-synclight-multi-room.md)。

| 新场景 | 同机位梯度相关，须 ≥0.50 | 弱／强受光覆盖，各须 ≥5% | 判定 |
| --- | ---: | ---: | --- |
| scene18，CR2 | 0.421 | 未继续合成 | 配准检查未过 |
| scene23，DNG | 0.688 | 6.07%／0.89% | 强受光覆盖不足 |
| scene36，DNG | 0.369 | 未继续合成 | 配准检查未过 |

scene18 和 scene36 的梯度相关峰值都在零位。低相关只说明当前粗配准认证没有通过，不能据此判定镜头实际挪动。scene23 能取得受光标签，但强区太少。合格训练房间仍是两间，低于三个的预设门槛。既有走廊和床头没有用于拟合，也没有产生新预测。若继续扩展，应先验证场景的强弱信号和对齐依据，优先查明确可改变资格的证据。

两遍准备报告及六份预处理文件的哈希一致。逐一复核六份原片哈希，与此前十余份原片无字节重复。另实测两间时拟合入口在创建结果前退出。见[原片身份与核验](2026-11-18-synclight-multi-room-extension-evidence.json)。本轮未降门槛或改特征。

复现用[定向取样](../../experimental/adaptive-reference-20261113/prepare_synclight_multi_room.py)的 `--extend`，再按[上轮准备入口](2026-11-18-synclight-multi-room.md)运行并加 `--previous <上轮准备目录>`。原片与预处理结果留本机。
