# 近景行人真实穿越一项回归，候选输出与原版一致

上一段人物实拍的遮挡较小。本轮从 Xiph 原始发布取固定机位的街道片段，使用一名近景行人从右向中央经过的 72 个源帧。分别从右侧重新显露的阶段和左侧始终可见的建筑位置选定窗口。来源是 Taurus Media Technik 的 `pedestrian_area`，未重新构造遮挡。原候选、固定参考、13 号权重和导出设置保持原样。[原始发布](https://media.xiph.org/video/derf/) · [来源与 CC0 声明](https://commons.wikimedia.org/wiki/File:Video_Codec_Test_pedestrian_area_1080p25.y4m.webm) · [事前协议](../../experimental/adaptive-reference-20261113/spatial-street-occlusion-protocol.md)

| 预定测量位置 | 遮挡期可评分相邻帧 | 其中帧率转换重复帧 | 普通版／候选最大编码前波动（RGB8） | 最大编码后波动（RGB8） |
| --- | ---: | ---: | ---: | ---: |
| 左上招牌 | 8／45 | 8 | 0.974／0.974 | 1.552 |
| 左上建筑 | 8／45 | 8 | 0.970／0.970 | 1.222 |

候选没有一次暂缓。对 86 帧每对相邻图独立复算，未出现达到 0.10 变化量的图块，规则要求至少四个。两版的全部编码前抽取帧及完整导出解码数据一致。输入按原工作台规则转为 30 fps，两处通过同坐标稳定线的相邻帧全部是重复帧，低于预设 32 对，不能用这些数字判断独立拍摄帧的原版波动或遮挡收益。继续沿用原阈值，保留本片作为不会对这类连续明显运动错误暂缓的回归。

后续若继续验证行人遮挡，应先在不同原始帧上确认保护区域可做时间对齐，并在模型对照前登记评价。该片没有证明真实遮挡收益，候选尚未进入默认软件。素材和导出留本机。

## 复现

从 Xiph 的 `y4m/pedestrian_area_1080p25.y4m` 读取 0–35 的 36 字节头，再读取 802,484,784–1,026,434,015 字节的连续范围。可拆成每 8 帧一段的请求，核对 HTTP 206、`Content-Range`、字节数，每帧均须带 `FRAME` 头。原对象总长度为 1,166,402,286 字节。将两个范围串接后，本机 Y4M 长 223,949,268 字节，SHA256 为 `026fc4a382e7a3897753f8e1263f2602f352864d274a95ff3c79fc5fcf4425bd`。使用项目同一 FFmpeg 运行 `ffmpeg -v error -i <截取Y4M> -vf fps=30,scale=640:360 -pix_fmt rgb24 -c:v libx264rgb -crf 0 -preset fast <新MP4>`。MP4 SHA256 为 `aeba09546f01df7d14964f809592fb0d603cdf1e593eff6b35cdc4f3e9bf73e5`。

使用已有视觉环境，设置仓库根目录为 `PYTHONPATH`、`OPENBLAS_NUM_THREADS=2`，运行 `python experimental/adaptive-reference-20261113/probe_spatial_real_occlusion.py outputs <本机截取MP4> <新输出目录> --profile street`。两遍各两份 MP4 的完整解码帧签名相同，全部预定像素评分及重复帧独立复核。报告 SHA256 为 `7af5208010d95fcac70a17170feaf3f8d39f6a45b64ccfb7acdec6d5fc008ec6`。[复现脚本](../../experimental/adaptive-reference-20261113/probe_spatial_real_occlusion.py)
