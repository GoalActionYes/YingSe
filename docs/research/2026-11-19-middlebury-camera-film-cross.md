# 同一实体色卡换参考相机，白桌布的风格响应发生反转

Middlebury 用不同相机拍摄同一 24 格实体色卡，固定钨丝灯档、3200K 布光和标称零档曝光。本轮取 Canon EOS-1Ds Mark II、Nikon D70 的相机 JPEG 派生平滑 PNG。两张重排后的色卡均为 390×260，未嵌入 ICC，按 sRGB 处理。两张实拍参考的 RGB8 MAE 为 49.48，分离后明暗与色度 MAE 为 51.55 和 6.30。名称相同的曝光档位没有保证成片在码值上具有相同曝光响应。[发布说明与使用许可](https://vision.middlebury.edu/color/data/) · [事前协议](../../experimental/adaptive-reference-20261113/middlebury-camera-film-cross-protocol.md)

保持两张跨内容目标不变。两台参考相机各开关同一已知 Film A，再让 13 号以 80% 强度生成八张成片。开关只改数字编辑，开启后两张参考各改变 9.62／9.28 RGB8；无编辑显示往返精确保持。目标已知 Film A、桌布和脸颊区域只用于计分。[已有已知风格基线](2026-11-19-arri-two-looks.md)

| 目标范围 | 仅换参考相机的 RGB8 差，关／开风格 | 其中色度差，关／开风格 | 风格方向投影，Canon／Nikon |
| --- | ---: | ---: | ---: |
| 扑克，共同近中性色 | 24.77／27.68 | 3.01／6.19 | 0.387／0.157 |
| 扑克，左白桌布 | 9.08／14.08 | 2.40／1.31 | 0.884／−0.706 |
| 扑克，右白桌布 | 15.84／21.26 | 2.08／1.98 | 0.456／−0.343 |
| 实体色卡，共同近中性色 | 29.83／31.60 | 3.80／3.86 | 0.337／0.022 |
| 实体色卡，左脸颊 | 24.28／23.59 | 7.16／6.73 | 0.355／−0.026 |
| 实体色卡，右脸颊 | 24.32／25.98 | 5.53／3.14 | 0.203／0.027 |

方向投影 1 表示复制目标已知 Film A 的变化。左桌布用 Canon 参考时投影为 0.884，响应误差为 1.58 RGB8，两项均过诊断线；用 Nikon 参考时投影反向，误差为 3.96。两目标的整图／共同近中性色均未同时通过固定风格门槛，换参考相机也超过 3 RGB8 限制。RGB 差主要来自明暗，表中另列的色度差可避免把全部 RGB 变化当作偏色。该对照控制了参考的物理内容、布光和标称设置，未分离相机内部渲染与实际曝光响应，也未比较相机质量。[左桌布 Canon 成片](../../outputs/adaptive-reference-20261113/middlebury-camera-film-cross-v1/poker_fullshot-canon-on.png) · [左桌布 Nikon 成片](../../outputs/adaptive-reference-20261113/middlebury-camera-film-cross-v1/poker_fullshot-nikon-on.png)

ZIP 分段实际读取 291,934 字节，两个成员匹配发布尺寸并通过 CRC，另记录 SHA-256。两次运行的 11 份产物逐件匹配，另独立复核两张相机输入及八个区域的风格响应、换相机 RGB、明暗与色度数据。已有研究环境设置仓库根目录、OpenCV 和 OpenColorIO 到 `PYTHONPATH`。运行 `probe_middlebury_camera_film_cross.py fetch <新素材目录>`，再运行同脚本 `run <素材目录> <研究输出根目录> <新输出目录>`。本机相机图、权重和成片未上传，没有产品候选变更。[脚本](../../experimental/adaptive-reference-20261113/probe_middlebury_camera_film_cross.py) · [数值和身份](2026-11-19-middlebury-camera-film-cross-evidence.json)
