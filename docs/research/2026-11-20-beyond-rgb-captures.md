# Beyond RGB 换灯原片的可比性

同一组物体换灯，可以直接看到环境光对中性色的影响。上一轮为 MIXED_organic 和 WHITE_rich 分别选中相同的灯号 6 与 13，测光照度差约 6%。这轮取出固定的四张 Oppo 带色卡原片，核对色卡位置、曝光和中性色通道响应。[仪器筛选](2026-11-20-beyond-rgb-light-pairs.md) · [预定核对方法](../../experimental/adaptive-reference-20261113/beyond-rgb-capture-protocol.md)

## 实拍结果

四张图均来自 OPPO Find X5 Pro，画幅 4096 × 3072，标签为 1/100 秒、f/1.7、相同焦距。四份色卡 JSON 的宽高字段均与相机标签颠倒，按 HDF5 与相机标签对齐，保留发布值。检查预览中的六个圆标，各自落在灰阶色块中央。以下 ISO 均来自拍摄标签，色卡位移用 24 块色卡中心的最大位移衡量。

| 场景 | 最大位移（画面对角线比例） | ISO（灯 6 → 灯 13） | 设置曝光差 | 六个灰阶色块的归一化 RGB 距离 |
| --- | ---: | ---: | ---: | ---: |
| MIXED_organic | 0.042% | 249 → 272 | 0.127 EV | 0.612–0.760 |
| WHITE_rich | 0.153% | 306 → 228 | 0.424 EV | 0.623–0.698 |

两组机位均符合 1% 门槛，曝光均超过 0.1 EV 门槛。四张图的六个灰阶色块中央，采样点没有达到发布 RGB 的上界。两档灯光下都可采到从白到黑的色卡响应，两场景各六个色块均达到 0.015 的色度筛查门槛。在浅色场景的白色卡，相机 RGB 中红通道占比由 0.564 到 0.084，蓝通道由 0.069 到 0.463。这个变化是肉眼预览中暖黄与青蓝差别的一部分。实际测量用发布的相机 RGB，预览只应用固定 1/2.2 幂次映射，白平衡与色彩空间转换均未应用。[逐块数值与文件哈希](2026-11-20-beyond-rgb-captures-evidence.json) · [作者的原片读取示例](https://github.com/shirawerman/Beyond-RGB/blob/0a3ed93e37f9f1b67fb23781c5e6ab34631020af/example%20code/H5Image.py)

## 结论与用途

两组素材可供明显光色变化的诊断。相机增益不同，明暗结果会混入曝光变化。HDF5 已重建为三通道，传感器原始裁切未知。后续可固定参考与调色强度，把两档换灯画面分别送入同一调色流程，比较中性色响应，并以色卡为离线测量依据。该开发展示应保留暖色、冷色及过度抵消的反例。真人肤色、独立白墙及视频仍需对应素材与验收。本轮没有运行调色候选。

## 复现

来源为 Beyond RGB 正式发布，许可 CC BY 4.0。四次范围请求共 69,993,128 字节，12 个成员通过 ZIP 成员名、解压长度及 CRC32 检查。使用本机已有 Python 环境中的 h5py 3.16.0、NumPy 2.5.4 和 Pillow，无需修改产品环境。以下命令从仓库根目录执行，报告和预览文件使用新文件名。[官方发布](https://zenodo.org/records/16848482)

~~~powershell
$src = 'experimental/adaptive-reference-20261113'
$run = 'outputs/adaptive-reference-20261113/beyond-rgb-captures-v1'
$python = '.tools/tf-convert-env/Scripts/python.exe'
& $python "$src/fetch_beyond_rgb_captures.py" 'docs/research/2026-11-20-beyond-rgb-light-pairs-evidence.json' $run
& $python "$src/audit_beyond_rgb_captures.py" $run "$run/replay.json" "$run/replay.png"
~~~

两次测量的 JSON 与预览分别逐字节相同。JSON 的 SHA256 为 21560ea26bfcfa90adcb5dd1bcdad326ea7f3ec1e9f889274c1a16ee9435fd75，预览为 bae31f666c5df1e893d645be0da3c8924ffec9ec2fb3a331c7c9948a6f8bc31f。图像及原始发布标签均留本机。
