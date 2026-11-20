# Beyond RGB 静物换灯配对

上一轮随手定位的两档灯光实测照度约相差四倍。当整幅画面亮度也大幅变化时，仅凭调色结果难以区分照明色变化和亮度变化的影响。本轮先固定静物场景，按实测照度和色度筛选原片，作为之后观察主体与背景如何随照明变化的开发素材。[上一轮索引诊断](2026-11-20-beyond-rgb-lab-index.md)

## 选择依据

每个场景取拍摄顺序前十二次，跨场景按实际灯号对齐。拍摄顺序和灯号各自独立。目录预检发现两个场景都缺灯号 10、11，因此在读取新增仪器数据前，按拍摄顺序明确取样范围。高低照度比至多 1.15，CIE xy 色度距离至少 0.025。若两场景均有合格的同灯号配对，按两场景中较小的色度距离选最大的一组。[冻结协议](../../experimental/adaptive-reference-20261113/beyond-rgb-light-pair-protocol.md)

## 筛选结果

两个场景各有 6 对合格，其中 4 对灯号共同。灯号 6 与 13 在两个场景中都具备最大色度距离，因此两场景均选这一对。

| 场景 | 灯号 6 照度 | 灯号 13 照度 | 高低照度比 | CIE xy 距离 |
| --- | ---: | ---: | ---: | ---: |
| MIXED_organic | 350.126 lx | 372.114 lx | 1.0628 | 0.42227 |
| WHITE_rich | 430.284 lx | 456.645 lx | 1.0613 | 0.42332 |

核对了全部 24 条 CRI 仪器记录的 HTTP 字节范围、ZIP 成员名、解压长度及 CRC32。复用 4 个范围，从发布源补取 20 个范围，共新增 40 KiB 仪器文件片段。从本地缓存复跑，结果字节一致，SHA256 为 1225b2d32061758f0de18a050dec2e061ce172501accab0d5de2123d1bfbc741。[完整读数、配对与 ZIP 成员位置](2026-11-20-beyond-rgb-light-pairs-evidence.json) · [Beyond RGB 官方发布](https://zenodo.org/records/16848482)

## 复现与后续用途

从仓库根目录运行。可把上一轮的四个测光片段及 HTTP 响应头按实际灯号预填至缓存，也可由脚本从官方发布源取得相同片段。

~~~powershell
$index = 'outputs/adaptive-reference-20261113/beyond-rgb-index-v1'
$run = 'outputs/adaptive-reference-20261113/beyond-rgb-light-pairs-v1'
New-Item -ItemType Directory -Force $run | Out-Null
python experimental/adaptive-reference-20261113/audit_beyond_rgb_light_pairs.py "$index/central.bin" "$run/cache" "$run/replay.json" --proxy http://127.0.0.1:7897
Get-FileHash "$run/replay.json" -Algorithm SHA256
~~~

选中的四份 Oppo 带色卡原片和标签已定位，压缩后约 70 MB。照度只代表测光探头所在位置。下一步先核对同一机位的画面几何、相机曝光、色卡像素及高光裁切，确认这些图像适合比较。此次只检验了仪器配对。静物可以辅助研究中性色与环境光，真人肤色、自然场景白墙和视频仍需要各自的独立验证。
