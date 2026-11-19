# 新两组实拍可分别检验参考灯色、灯强和有意风格

从 CHROMA-MILL 的固定公开发布取手机训练分区 Scene1_a、Scene2_a 两个场景，各有白／蓝灯和 B_213／B_241 两档。八张线性 PNG 合计 18,155,429 字节，逐件匹配公开的 LFS 长度及 SHA-256。数据卡标 MIT。作者采集过程控制相机设置和 LED 参数，两个 B 码值是灯强设定，相同设定的白、蓝灯画面仍有明暗差。固定使用 sRGB 编码曲线将线性 RGB 制备为最长边 512，未按单图校曝光。[作者论文](https://arxiv.org/abs/2611.04092) · [固定数据卡](https://huggingface.co/datasets/priubrogent/CHROMA-MILL/blob/fcb84de6a849931cc7d1583ebcdce3ca2006d2fd/README.md) · [事前协议](../../experimental/adaptive-reference-20261113/chromamill-capture-cross-protocol.md)

两个场景内各自四幅照片的最优整数位移均为零，零位梯度相关 0.922–1.000。两个白光目标的共同近中性色分别有 13,365／20,090 个像素。用另一场景的四种拍摄状态作为参考，各自开关同一 Film A，固定 13 号 80% 生成 16 张目标图。目标自身的已知 Film A 仅用于评分。

| 目标 ← 参考 | 白灯风格投影，213／241 | 蓝灯风格投影，213／241 | 开风格后换灯色的 RGB8 差，213／241 | 开风格后换强度的 RGB8 差，白／蓝 |
| --- | ---: | ---: | ---: | ---: |
| Scene1_a ← Scene2_a | 0.279／0.733 | 0.107／0.175 | 32.64／35.84 | 9.99／3.12 |
| Scene2_a ← Scene1_a | 0.589／0.727 | −0.117／−0.085 | 31.55／36.25 | 10.74／2.43 |

上表均测共同近中性色。Scene2_a 在蓝灯参考下两档风格增量都反向。两目标开风格后的近中性色换灯色差中，色度分量为 21.45–24.44，表中 RGB8 差还包含明暗。开风格后的灯色与灯强交互项分别为 6.87／8.43 RGB8，可见单独测一个档位可能漏掉另一档的影响。两组均未通过风格和稳定联合门槛。[Scene2_a 白灯成片](../../outputs/adaptive-reference-20261113/chromamill-capture-cross-v1/Scene2_a-white-241-on-13.png) · [Scene2_a 蓝灯成片](../../outputs/adaptive-reference-20261113/chromamill-capture-cross-v1/Scene2_a-blue-241-on-13.png)

保留这块可用于学习和诊断的交叉输入。两个拍摄场景无人，最终肤色和自然画质尚待验证。其余场景及数据集验证、测试分区未下载，同场景原有变体不能再算独立数据。本轮未训练或改软件。

两遍的 16 张准备图、16 张 13 号成片及报告哈希逐件一致，另从落盘图独立复算源哈希、区域支持、16 项风格响应和 40 组 RGB／明暗／色度对照。用已有研究环境设置仓库根目录、OpenCV 和 OpenColorIO，运行 `probe_chromamill_capture_cross.py fetch <本机八张源图目录>`，再运行 `run <源图目录> <本机研究输出根目录> <新输出目录>`。媒体留本机。[复现脚本](../../experimental/adaptive-reference-20261113/probe_chromamill_capture_cross.py) · [数值与哈希](2026-11-19-chromamill-capture-cross-evidence.json)
