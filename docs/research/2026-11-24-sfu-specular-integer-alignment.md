# SFU 整数配准后比较实拍换光和暖色编辑

此前原位检查在两组换灯中均测出 1 像素横移，相机原值与显影图支持同一方向。这轮在已有图上只测试固定整数裁切能否修正几何，未改变相机变换、曝光、暖色编辑或验收条件。通过几何审计的一组实际制作了两档的编辑前后四图。换光和有意暖色的幅度相近，方向未匹配。所有输入与定点区域来自前一轮，不算独立产品验收。[既有实拍与位移证据](2026-11-24-sfu-specular-solux.md) · [事前协议](../../experimental/adaptive-reference-20261113/sfu-specular-integer-alignment-protocol.md)

三档按 3500、4100、4700 K 分别从横坐标 2、1、0 开始复制 635 × 468 个原像素。两遍均验证裁切等于对源数组取对应区间。两组重新检查整图、原两处标签地标以及左容器标签、右容器握把两处未用于估计位移的地标。4700→4100 K 五项均通过，整图零位相关 0.8940，新地标为 0.9460、0.8429。人工从裁切片查看，测区未见新的反光边缘、阴影边缘或遮挡。4100→3500 K 的右握把仍失败，零位相关 0.4860，最大相关出现在纵向 −1 像素，增益 0.0488。保留这一失败，未对其做更复杂变换或暖色出图。[配准逐项结果](2026-11-24-sfu-specular-integer-alignment-evidence.json) · [本机测区核对](../../outputs/adaptive-reference-20261113/sfu-specular-integer-alignment-v1/material-review.png)

4700 K 和 4100 K 的两档编辑均通过素材线。关态与输入逐像素相同，两保护区及编辑外部的 RGB8 差 P95 均为 0。两方向只有颜色方向一项未通过。前向三态共同有效覆盖为 98.81%，反向为 81.11%。前向编辑与实拍向量的幅度比约为 0.928，处在 0.75–1.25 内。实拍换灯中红通道增加得更多，绿色略升，蓝色下降少于手工编辑。数值只适用于这个容器和先前确定的近似显影，前向 0.779 的余弦低于 0.90 要求。保留两张同款有意暖色的开发正例和一组实际换光不同向的反例。[原读数门槛](../../experimental/adaptive-reference-20261113/shared-local-warmth-protocol.md) · [本机四图](../../outputs/adaptive-reference-20261113/sfu-specular-integer-warm-preflight-v1/render/comparison.png)

| 方向 | 实拍变化 RGB8 | 有意暖色 RGB8 | 余弦 | 幅度比 |
| --- | --- | --- | ---: | ---: |
| 4700 → 4100 K | `[17, 4, -7]` | `[10, -3, -14]` | 0.779 | 0.928 |
| 4100 → 4700 K | `[-17, -4, 7]` | `[10, -3, -14]` | −0.779 | 0.928 |

保存的未裁切 XYZ 中，同一测区的 Y 中位数由 0.01107 升到 0.01198，裁切前两档均无出界像素。由此可记录该显影条件下的相对亮度变化，结合作者用灯距控制裁切的说明，不能只归因于光谱。几何通过也无法把浅色容器直接视为墙漆。当前第二个暖墙匹配来源仍待补齐，本轮没有产品改动、训练或候选验收。[相机转换及局限](2026-11-24-sfu-specular-solux.md) · [匹配及训练条件](2026-11-24-spatial-condition-design.md)

EGM 独立复核了源列映射、裁切边界和坐标移位，指出两个新地标仅独立于估位移，仍属于同一组实拍。复核也明确了近阈值时的标定限制和跨来源训练边界。落实意见时补明文档口径，执行用的图、坐标和门槛保持原设定。[协议与审阅边界](../../experimental/adaptive-reference-20261113/sfu-specular-integer-alignment-protocol.md)

## 复现

按前一份报告为三个灯档生成同曝光 PNG，保留原文件。以协议表中左端坐标 `left` 调用 Pillow 的 `Image.crop((left, 0, left + 635, 468))` 并保存成对应灯档的 PNG。把四个 4100 K 地标的横坐标统一减 1，再对全图及各块调用原 `probe_inretouch_registration.py` 中的 `gradient`、`shifts`，取第一个返回项。仅在审计合格的固定配对上，用原暖色预检配方复现。运行时需空闲的 ASCII 映射盘及此前准备的本机记录。[旧预检的路径配置](2026-11-24-warm-capture-preflight.md)

~~~powershell
$base = 'V:/outputs/adaptive-reference-20261113'
$captureDir = "$base/sfu-specular-integer-alignment-v1"
& V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/warm_capture_preflight.py `
    "$captureDir/4700.png" "$captureDir/4100.png" "$captureDir/capture.md" "$base/sfu-specular-integer-warm-preflight-new" `
    --edit 392 371 462 389 --guard 109 192 169 243 --guard 79 424 150 441
~~~

本机两遍的三张裁切图与几何报告逐文件一致。暖色预检用时 7.61 秒和 3.76 秒，各生成 11 张 PNG、2 份 PP3 和 1 份 JSON，文件哈希一致。两遍共四个方向从落盘图重新计算通道中位差、覆盖、余弦和幅度比。原片、结果图和拍摄备忘留在本机。[几何、暖色及输出哈希](2026-11-24-sfu-specular-integer-alignment-evidence.json)
