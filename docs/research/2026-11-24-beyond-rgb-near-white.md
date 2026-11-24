# Beyond RGB 相近灯档取件前筛查

后续已补齐三份测光，完成筛查及唯一合格配对的取件、出片和双向比较。[续查结果](2026-11-24-beyond-rgb-near-white-match.md)

现有调色工作台在匹配暖墙对照中漏掉局部暖色，并把部分现场光色带到肤色。以前在 Beyond RGB 的灯光实验有意选择较大的色度差。为补一个变化较小的实拍对照，这轮先用仪器记录筛光色和照度，再看相机标签，只对都过线的配对取原片。场景固定为四个白背景静物，灯档固定为 1、7、9。测光和相机标签均来自作者发布。[事前协议](../../experimental/adaptive-reference-20261113/beyond-rgb-near-white-protocol.md) · [原曝光实验](2026-11-20-beyond-rgb-batch-intake.md) · [作者论文](https://openaccess.thecvf.com/content/WACV2024/html/Glatt_Beyond_RGB_A_Real_World_Dataset_for_Multispectral_Imaging_in_Mobile_Devices_WACV_2024_paper.html) · [正式数据发布](https://zenodo.org/records/16848482)

完整 ZIP 目录确认 12 档都有测光和 Oppo 标签。WHITE_red 的灯 7 缺 Oppo 色卡定位记录。成功校验了 9 份测光记录，均通过 HTTP 区间、成员名、长度及 CRC。6 份沿用本机缓存，新取 3 份共 6,144 字节。官方文件和 API 的区间尝试、直接连接及最后低并发重试后，仍缺 WHITE_blue 的灯 7、9 和 WHITE_red 的灯 9。流量统计只含成功区间。

| 场景 | 已核验测光档 | 可计算方向 | 照度通过 |
| --- | ---: | ---: | ---: |
| WHITE_blue | 1／3 | 0／3 | 待核 |
| WHITE_green | 3／3 | 3／3 | 0 |
| WHITE_red | 2／3 | 1／3 | 0 |
| WHITE_rich | 3／3 | 3／3 | 0 |

七个可计算方向都超过照度比上限 1.150。最接近的是 WHITE_rich 和 WHITE_green 的灯 1 与 7，色度距离分别为 0.01256、0.01188，照度比分别为 1.15757、1.15936。另五个方向待核。没有达到取件条件的配对，未下载相机标签和原片，固定暖色出片也未开始。这些数字只说明本次已读配对的探头读数，像素效果与相机固定曝光尚未核实。暖墙跨来源开发仍需要合格实拍，继续沿用第一组有原件的补拍流程。若剩下三个小区间恢复访问，可补齐本次仪器筛查。[逐档结果与文件哈希](2026-11-24-beyond-rgb-near-white-evidence.json) · [暖白灯试拍](2026-11-24-warm-capture-preflight.md)

## 复现

在仓库根目录用现有研究 Python 和已记录的中央目录运行下列命令。带 `--cached-only` 时只读取本机已校验范围，报告保留缺失项。区间恢复后可移除该选项并更换输出名。只有完整结果才会登记原片选择。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$run = "$base/beyond-rgb-near-white-v1"
& '.tools/adaptive-env/Scripts/python.exe' experimental/adaptive-reference-20261113/audit_beyond_rgb_near_white.py "$base/beyond-rgb-index-v1/central.bin" "$run/cache" "$run/replay-new.json" --cached-only
~~~

缓存复跑逐字节一致，另从落盘范围重新解压所有九份记录并重算七个方向。共享目录读取的旧曝光入口复跑与原报告逐字节一致。源资料、下载日志与缓存保留本机。
