# 真实人物相互遮挡未触发候选，保留为正常运动回归

受控测试中的矩形遮挡面积较大，不能说明面对实际人物交错时仍能改善。本轮挑选 SAM 2 作者公开的 `bedroom.mp4`，保留原始 960×540、30 fps、200 帧。输入中两人有互相遮挡和先后分开，右部墙面仍可见。只看输入固定测量窗和背景矩形，再用同一 13 号模型和参考进行两遍普通导出与候选导出。没有使用 SAM 2 模型或分割标签。[素材来源](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/notebooks/videos/bedroom.mp4) · [事前协议](../../experimental/adaptive-reference-20261113/spatial-real-occlusion-protocol.md)

| 遮挡期间测量区 | 可评分相邻帧 | 普通模式最大调色波动（RGB8） | 候选最大调色波动（RGB8） | H.264 后最大波动（RGB8） |
| --- | ---: | ---: | ---: | ---: |
| 墙面 | 16／63 | 0.201 | 0.201 | 0.971 |
| 右侧 | 60／63 | 0.581 | 0.581 | 0.966 |

普通模式两区的编码前最大波动都低于事前规定的 1.00 改善资格线。墙面可评分帧数也低于 32，解除阶段没有相邻输入满足稳定要求，保留缺测。另一处解除后的编码前最大波动为 0.413，候选没有增加。完整 200 帧里，候选没有一次暂缓。逐帧缩图复算最多只有一个图块达到进入所需的变化阈值，规则要求至少四个。两臂实际导出的解码数据一致。

这段片未重现原受控遮挡导致的大面积串扰。它可防止候选干扰类似正常运动，还不能验证真实遮挡有画质收益。后续需预先取得背景稳定、遮挡物大范围进入并可确认解除的实际片段。未改变模型参数、遮挡阈值或应用默认配置。

## 复现

将作者仓库固定提交的 `bedroom.mp4` 保存在本机。源文件 2,380,401 字节，Git blob 为 `50a9349cc82e227917ac39e9ebc04d153bbcf6c0`，SHA256 为 `1be76d5d19b066e8ad7c565d88a98e11a8f8d456a707508a7aa35390def70e30`。网络取样方面，尝试的 Wikimedia Commons 行人素材直连超时，配置代理后返回 429，未下载。当前选用作者官方仓库中可取得的小样，仅留本机验证。[行人素材发布页](https://commons.wikimedia.org/wiki/File:Video_Codec_Test_pedestrian_area_1080p25.y4m.webm)

使用已有视觉环境，设置仓库根目录为 `PYTHONPATH`、`OPENBLAS_NUM_THREADS=2`，运行 `python experimental/adaptive-reference-20261113/probe_spatial_real_occlusion.py outputs <本机原视频> <新输出目录>`。首遍两臂完成导出，评分遇到无合格相邻帧时中止。修正为保留缺测后，两个新目录完成正式复跑。四个 MP4 的完整解码帧和指定 PNG 逐项一致，全部窗口的像素读数独立复算。报告 SHA256 为 `4ebe5f62866463b530e66d20664d9d5f811f21efbd01663392ffa1273a277c6c`。素材、画面和成片均留本机。[复现脚本](../../experimental/adaptive-reference-20261113/probe_spatial_real_occlusion.py)
