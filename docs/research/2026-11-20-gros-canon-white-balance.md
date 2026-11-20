# Canon 原始 JPEG 有数值白平衡，导出后仍可能沿用旧值

六张 Gros 原始 JPEG 都能读取 Rebel T2i 的 Canon ColorData4。拍摄和自动两组 RGGB 及温度在六张中分别一致。读取器校验型号、16 位条目结构、1,338 项长度、版本及偏移，只保存需核验的白平衡数值。RGGB 是拍摄通道的增益，不能直接接到 sRGB 调色。[事前协议](../../experimental/adaptive-reference-20261113/gros-canon-white-balance-protocol.md) · [字段解释与机型](https://exiftool.org/TagNames/Canon.html) · [固定版本的字段定义](https://github.com/exiftool/exiftool/blob/13.64/lib/Image/ExifTool/Canon.pm)

| 参与者 | 冷灯／暖灯的相机估计温度 | 暖灯相对冷灯的 log(R/G)、log(B/G) 增益差 | 完整方向门槛 |
| --- | ---: | ---: | --- |
| 01 | 5225／5077 K | 0.00000、+0.03474 | 未过 |
| 02，同快门 | 5238／5067 K | −0.00381、+0.03667 | 通过 |
| 03 | 5212／5077 K | +0.00191、+0.03469 | 未过 |

三个暖灯的相机估计温度均下降，蓝增益均上升；01 的红增益未变，03 的红增益反向。因此按完整门槛只通过 1／3。相机的估计温度下降 135–171 K，采集说明的灯档为 3200 K 和 5500 K，两者含义不同，尤其这里前景和背景分开受光。[原片及灯档说明](2026-11-20-gros-portrait-light-intake.md) · [作者采集与测光表](https://pmc.ncbi.nlm.nih.gov/articles/PMC6321972/)

六张同时制作有意脸部暖调，并分别保留和移除原 EXIF。相对同设置重编码的未编辑图，脸部变化为 11.24–15.01 RGB8。六对编辑导出各自的解码像素逐值一致。保留 EXIF 的六张仍报告逐值相同的拍摄和自动增益，移除 EXIF 的六张完全缺这些字段。相机记录可以随已编辑图片保留，也可以随导出消失。这里是两个明确的导出控制，不能外推所有软件的默认保留方式。[逐张数据、来源和导出哈希](2026-11-20-gros-canon-white-balance-evidence.json)

保留三组实拍白平衡读数，停止把这组字段单独用于是否取消局部暖调的产品决策。独立的人像和白墙验收仍待候选，产品无改动。两遍的 12 张导出及报告逐字节一致，图像和原始 EXIF 仅存本机。复现使用现有 Pillow 与研究解释器，无需下载 ExifTool 可执行文件。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_canon_white_balance.py "$base/gros-portrait-light-input-v3" "$base/gros-portrait-light-transfer-v1" "$base/gros-canon-white-balance-replay"
```
