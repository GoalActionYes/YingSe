# 参考近黑色度减轻部分人物偏色，白衣和雪地仍受背景干扰

此前按固定规则减少整体偏色，会一起削弱有意的胶片色。这轮检验参考最暗且未裁切的像素能否为共同色度提供更可靠的依据。普通两图推理在参考与目标原图中分别选取亮度 4–32 RGB8 的未裁切像素。将 13 号成片在目标暗区的色度中位数向参考暗区中位数移动，单轴移动限于 ±8 RGB8。整个校正沿用 13 号成片的亮度和既有色域处理。有意编辑的预期图和皮肤区域只用于评分。[事前协议](../../experimental/adaptive-reference-20261113/reference-black-anchor-protocol.md) · [此前的整体减色实验](2026-11-19-common-chroma-offset.md)

| 有意编辑打开，换背景时的目标区域 | 13 号 | 近黑锚定 |
| --- | ---: | ---: |
| 白球服场景，手臂 | 3.17 | 1.66 |
| 白球服场景，后方人物脸颊 | 3.18 | 1.77 |
| 白球服本身 | 1.41 | 2.22 |
| 785 脸部 | 3.01 | 4.48 |
| 785 雪地 | 1.33 | 2.45 |
| 9483 浅墙 | 0.89 | 0.89 |

以上是目标的局部换背景色度差，单位为 RGB8。预先登记的六项重点肤色测量均未达到改善门槛。白球服和雪地各有两个状态退化超限。72 张成片中 70 张启用锚点，另两张因暗区样本不足逐像素回退，故失败不能解释为全量回退。[本机白球服对照](../../outputs/adaptive-reference-20261113/reference-black-anchor-v1/background/872-1268/comparison.png) · [本机雪地反例](../../outputs/adaptive-reference-20261113/reference-black-anchor-v1/background/785-3553/comparison.png)

四套原有开发门槛均未通过：换背景 0/4，FACD 0/7，Film B 0/6，色卡 1/7。扑克以色卡作参考时 Film B 的整图响应投影从 0.338 升到 0.712，共同近中性色比预期作者成片仍差 12.54 RGB8。焊工以色卡作参考时，整图反向由 −0.325 减至 −0.139，共同近中性色仍反向，为 −0.338。这说明暗区共同色度会改变迁移方向和幅度，但不能稳定指导如何保色及迁移。关闭该近黑先验，不继续调暗部区间寻优。后续应优先使用能够单独检验物体颜色与编辑变化的配对证据，继续沿用四套检查。现有软件功能保留，未进入独立画质、人评或视频验收。[扑克 Film B 成片](../../outputs/adaptive-reference-20261113/reference-black-anchor-v1/film_b/poker_fullshot-from-hdr_testimage-comparison.png) · [焊工反例](../../outputs/adaptive-reference-20261113/reference-black-anchor-v1/film_b/sparks_4592-from-hdr_testimage-comparison.png)

两个独立进程各生成 72 张成片、18 张图板和报告，逐件哈希一致。验证了同图保持、平坦图回退及量化后的亮度误差上限 0.5 RGB8。报告 SHA256 为 `0218f8beb0727663d54afceb8a6fbb32fb8f6fc4ada3509c61288bac6eaf938c`。在已有视觉环境设定仓库根目录 `PYTHONPATH`、`OPENBLAS_NUM_THREADS=2` 及 `OMP_NUM_THREADS=2`，运行脚本并传入新输出目录。[复现脚本](../../experimental/adaptive-reference-20261113/probe_reference_black_anchor.py)
