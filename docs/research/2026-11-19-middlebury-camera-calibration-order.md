# 色卡校准缩小相机差异，两种处理顺序仍未保住完整风格

用 Canon 与 Nikon 未调色的 24 格色卡拟合一个固定 RGB 仿射变换。每格取中央区域，按棋盘位置选 12 格拟合，另 12 格只用于检验。固定同一个变换，分别先校准后加 Film A，或先加 Film A 再校准。后者模拟拿到已调色参考才进行相机校准。使用三张校准后的参考图在扑克、实体色卡人物上生成六张 13 号成片，强度 80%。目标真值和测量范围仅用于离线计分。[事前协议](../../experimental/adaptive-reference-20261113/middlebury-camera-calibration-order-protocol.md) · [原相机对照](2026-11-19-middlebury-camera-film-cross.md)

未调色参考在留出色格上的 RGB8 MAE 从 50.68 降到 8.82，超色域像素仅 0.004%，达到本轮输入门槛。先加 Film A 后再校准，有 8.24% 的参考像素须限幅。两种顺序生成的参考相差 9.07 RGB8，相对 Canon 已调色参考的误差分别为 7.90 和 16.05。这是同一张色卡内的留格验证，素材仍属于开发诊断。

| 目标范围 | 原参考换机 RGB8 差，关／开 | 先校准后调色，关／开 | 先调色后校准，关／开 | 风格投影，先校准／后校准 |
| --- | ---: | ---: | ---: | ---: |
| 扑克，共同近中性色 | 24.77／27.68 | 2.59／1.72 | 2.59／2.62 | 0.319／0.444 |
| 扑克，左白桌布 | 9.08／14.08 | 4.17／3.77 | 4.17／5.56 | 0.697／1.145 |
| 实体色卡，共同近中性色 | 29.83／31.60 | 2.80／1.74 | 2.80／1.88 | 0.355／0.374 |
| 实体色卡，左脸颊 | 24.28／23.59 | 4.17／2.82 | 4.17／1.55 | 0.584／0.435 |

两种顺序都降低了共同近中性色的换机 RGB 差，效果也随区域而异。左桌布先调色再校准时，风格投影与响应误差分别为 1.145 和 2.94，虽达到该局部的风格线，开风格后的换机差仍有 5.56。左脸颊相反，先校准的风格响应误差为 1.63，先调色为 3.09。两种顺序都未通过整图、近中性色及换机差的联合门槛。结束用此固定校准直接处理已有调色的参考。[左桌布，先校准](../../outputs/adaptive-reference-20261113/middlebury-camera-calibration-order-v1/poker_fullshot-calibrate_first.png) · [左桌布，后校准](../../outputs/adaptive-reference-20261113/middlebury-camera-calibration-order-v1/poker_fullshot-style_first.png)

两遍各 10 份产物的 SHA-256 逐件一致。用独立的正规方程、留出格和 PNG 复算校准、裁切、风格投影及换机误差。复现使用已有研究环境，将仓库根目录及 OpenCV、OpenColorIO 依赖加入 `PYTHONPATH`，运行 `probe_middlebury_camera_calibration_order.py <本机研究输出根目录> <新输出目录>`。两张参考原片用于离线校准；未引入产品入口，未读取独立验收集。[复现脚本](../../experimental/adaptive-reference-20261113/probe_middlebury_camera_calibration_order.py) · [数值证据](2026-11-19-middlebury-camera-calibration-order-evidence.json)
