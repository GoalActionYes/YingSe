# 两组真实自然光对照重现浅色表面偏色和风格不足

主线已经确认，软件会把参考背景或灯光的变化带进目标肤色和浅色表面。此前的实拍以室内或影棚为主。这轮加入 WildRelight 的墙面和柱旁铺地，两处场景各取首末时段。原作者提供配准后的不同时间 HDR 照片与同时记录的环境光图，包含实际光量、光色和阴影改变。环境光图用于核查采集，调色只接收普通目标和单张参考。两组已登记为开发素材。[事前协议](../../experimental/adaptive-reference-20261113/wildrelight-wall-cross-protocol.md) · [原作者数据](https://huggingface.co/datasets/Lez/wildrelight/tree/90ab579145da9f3ea998d706c8defd0d4b87641f/small-aligned) · [作者论文](https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_WildRelight_A_Real-World_Dataset_and_Benchmark_for_Single-Image_Relighting_CVPRF_2026_paper.html)

核对四张 HDR 照片、四张环境光图和两份元数据，共 72,967,852 字节，来源许可为 CC BY 4.0。每场景只以首时段决定曝光，再用同一映射制作两帧 RGB8。OpenEXR 文件头未见色度坐标，诊断采用固定线性 RGB 到 sRGB 显示编码；不能据此认定文件具有经校准的 sRGB 原色。本文中的强弱变化包含这条统一显影路径。渲染前查看成片，固定一段无窗墙面和柱右铺地。两个区域分别有 7,480 与 9,350 像素，两时段水平梯度相关为 0.798 和 0.814，可见区域内容相符且光照有变。铺地拍摄时明显受蓝色照明，登记为浅色材料，未假定其真实反射色为纯灰。

以另一场景的两时段照片作参考，分别关、开同一 Film B，使用 13 号的 80% 强度出片。换时段差衡量同一开关下两张成片的 RGB8 MAE。风格投影 1 表示与同一时段的开关预期幅度一致，原门槛为 0.75–1.25，同时要求响应误差、换时段差和关闭态偏差不超过 3 RGB8。

| 目标范围 | 参考换时段差，风格关／开 | 风格投影，首／末时段 | 风格误差，首／末时段 |
| --- | ---: | ---: | ---: |
| 墙面整图 | 26.05／28.69 | 0.025／−0.345 | 8.95／12.05 |
| 登记墙面 | 22.57／23.66 | 0.230／0.019 | 6.16／7.69 |
| 柱旁整图 | 15.70／20.17 | 0.396／0.355 | 5.58／4.77 |
| 登记铺地 | 17.51／21.07 | 0.905／0.457 | 0.91／4.36 |

两方向均未同时满足换光保护与风格。登记铺地在首时段参考下保留有效变化。同一款 Film B 换成末时段参考，投影仅剩 0.457。同一场景其他位置、联合条件均继续保留失败。此处无法区分是光源不同、阴影移动还是输入亮度跨度单独造成了成片的变化。该批补充了自然光变化下的真实开发对照。继续按原标准推进肤色、浅色材料和风格联合检查，独立画质、真人评价和视频仍待合格候选。[墙面本机对照](../../outputs/adaptive-reference-20261113/wildrelight-wall-cross-v1/wall-comparison.png) · [铺地本机对照](../../outputs/adaptive-reference-20261113/wildrelight-wall-cross-v1/pillar-comparison.png)

## 复现

按[固定发布文件清单](../../experimental/adaptive-reference-20261113/wildrelight-wall-release.json)取用十个文件，保留其中的相对目录。将清单复制到输入根目录的 `release.json`，若 Git 已改为 LF，复制时恢复 CRLF，确保 SHA-256 为 `91d49cdfa7bad201ecb7038a6030237a3331a7afd9ba52931281d89a05d0ef99`。脚本核对 EXR 的 LFS SHA-256 和元数据 Git blob。沿用[研究 Python 和运行参数](2026-11-21-gros-grid-expression.md)，另设置 `OPENCV_IO_ENABLE_OPENEXR=1`。执行 `probe_wildrelight_wall_cross.py prepare <输入根目录> <新显影目录>`，再执行 `probe_wildrelight_wall_cross.py run <显影目录> <新出图目录> <inputs.json的SHA> <区域JSON的SHA>`。原照片、环境光、元数据正文及完整成片留本机。

两遍显影的七个产物、两遍渲染的十一个产物均逐文件相同。另从 PNG 独立复算四处区域的偏差、风格投影和误差。完整报告 SHA-256 为 `b0b75d2c38dd7b83736d917d0ba00597d4e2085d37cf70f4baf72d7f69be47c5`。[数值、许可及文件身份](2026-11-21-wildrelight-wall-cross-evidence.json)
