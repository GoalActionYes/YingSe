# 固定参考明暗压低整体差异，局部偏色和风格问题仍在

上一轮两台相机拍同一张实体色卡并套用同一 Film A，参考相机一换，目标白桌布的风格响应就改变。本轮固定两张目标原图、13 号和 80% 强度，将两台相机的参考 Y′ 与 Cb／Cr 直方图做交叉组合。关闭与开启 Film A 分别使用匹配状态的参考统计。原组合八张成片复现相同，新组合增加八张成片。目标原图、已知 Film A 和评分范围沿用上轮，后两者未参与推理。[事前协议](../../experimental/adaptive-reference-20261113/middlebury-camera-channel-replay-protocol.md) · [原相机对照](2026-11-19-middlebury-camera-film-cross.md)

| 目标共同近中性色 | 原换相机 RGB 差，关／开风格 | 固定 Canon 明暗，关／开风格 | 固定 Nikon 明暗，关／开风格 |
| --- | ---: | ---: | ---: |
| 扑克 | 24.77／27.68 | 3.65／3.30 | 3.36／3.31 |
| 实体色卡人物 | 29.83／31.60 | 3.83／3.40 | 5.77／6.28 |

八项固定明暗的近中性色 RGB8 差均至少减半，满足本轮的主导判断。在同一检查点上固定色度、只切换参考明暗时，这些范围仍变化 25.69–31.61 RGB8。RGB 有明暗成分，色度有不同结果。例如实体色卡近中性色在开风格时，原换相机色度差为 3.86，固定 Nikon 明暗后升至 6.34；扑克关风格时，原值为 3.01，固定 Canon 明暗后升至 3.62。

固定 Canon 明暗、只换成 Nikon 色度，左白桌布的风格投影由 0.884 降为 0.091，左脸颊由 0.355 变为 −0.724。固定 Nikon 明暗、换用 Canon 色度，左白桌布虽然由 −0.706 改善到 −0.140，仍反向。两种固定明暗在整图和共同近中性色的有意风格响应均未过联合条件。参考明暗项、色度项存在交互。报告保留各项及交互误差，不能把两条路径简单相加来预测结果。[左桌布，固定 Canon 明暗](../../outputs/adaptive-reference-20261113/middlebury-camera-channel-replay-v1/poker_fullshot-canon-y-nikon-chroma-on.png) · [左脸颊，固定 Canon 明暗](../../outputs/adaptive-reference-20261113/middlebury-camera-channel-replay-v1/hdr_testimage-canon-y-nikon-chroma-on.png)

停止直接复用某一台相机的明暗作为修复。这里比较的是同一色卡的两台相机成像，且离线取到了两套参考统计。普通两图产品没有第二台相机作为校准输入，也未验证真实视频和更广泛的风格。固定明暗大幅减少了场景亮度干扰，剩余问题需要保留独立的色度与有意风格门槛。

两次运行的八张成片和报告 SHA-256 逐件一致。另从本机成片独立重算 96 个 RGB、亮度与色度对照，以及 32 项风格投影／响应误差记录。复现使用既有研究环境，把仓库根目录和 OpenCV 依赖放入 `PYTHONPATH`，运行 `probe_middlebury_camera_channel_replay.py <本机研究输出根目录> <新输出目录>`。媒体和权重留本机，产品入口未改。[脚本](../../experimental/adaptive-reference-20261113/probe_middlebury_camera_channel_replay.py) · [数值证据](2026-11-19-middlebury-camera-channel-replay-evidence.json)
