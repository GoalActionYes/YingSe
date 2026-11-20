# 白背景静物的换灯取件前核对

先前取到的两组 Beyond RGB 灯光高度饱和，显示转换需压缩整个色域，相机曝光也不同。这次按上一轮仪器读数固定灯号 1 与 8，并选用未出图的 WHITE_green 和 WHITE_red。先核对实际测光和两台相机的标签，小文件通过后才考虑取图，减少下载和显影成本。事前要求色度距离为 0.025–0.20、照度比不超过 1.15、同一相机在两组中的设置曝光差都不超过 0.10 EV。[预定范围](../../experimental/adaptive-reference-20261113/beyond-rgb-exposure-precheck-protocol.md) · [官方档案](https://zenodo.org/records/16848482)

## 资格结果

| 场景 | 灯 1／8 照度 | 照度比 | 色度距离 | Oppo 设置差 | 三星设置差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WHITE_green | 797.49／770.64 lx | 1.0348 | 0.1792 | 0.000 EV | 0.411 EV |
| WHITE_red | 858.15／730.22 lx | 1.1752 | 0.1800 | 小于 0.000001 EV | 0.242 EV |

两场景都具有预定范围内的实测色度差。Oppo 的 ISO 为 100、光圈 f/1.7，快门约 1/100 秒，两个灯号的曝光设置相同或在标签舍入范围内。WHITE_green 的照度也通过。WHITE_red 的照度差 17.5%，超过预定上限。三星两个场景均超过相机曝光差门槛。根据两组均通过方取图的条件，本轮没有合格的双场景组合，也没有下载原片。保留 WHITE_green 的 Oppo 配对作为此后其他研究的候选。照度是探头位置读数，曝光设置资格不代表像素无裁切或场景各区域照度相同。[测光、曝光与 ZIP 范围清单](2026-11-20-beyond-rgb-exposure-precheck-evidence.json)

核对期间复查了 4,654,010 字节原始中央目录以及固定 SHA256。原报告把每场景记录数写成 93，13 个场景的已提交目录解析结果均为 12。已更正报告正文和研究入口。目录哈希、成员定位和前序出图不受计数文字错误影响。[更正后的原索引报告](2026-11-20-beyond-rgb-lab-index.md)

取件共读 4 个测光和 8 个拍摄标签的范围，响应体合计 40,960 字节。核对 HTTP 206、范围、解压长度与 CRC32，缓存复跑报告逐字节相同。此前的真实浅色区域筛查和本轮均未得到可直接用于固定比较的两组素材。下一步可按仪器和相机参数预筛多个场景并在读图前固定配对，避免继续逐个取样。若没有满足同灯、同曝光范围的双场景数据，则应转向有明确采集标注的不同来源。真实人像和最终独立白墙仍另行保留。

## 复现

沿用[分卷原始索引](2026-11-20-beyond-rgb-lab-index.md)。从仓库根目录运行，输出报告使用未占用的文件名。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$run = "$base/beyond-rgb-exposure-v1"
New-Item -ItemType Directory -Force $run | Out-Null
& .tools/tf-convert-env/Scripts/python.exe experimental/adaptive-reference-20261113/audit_beyond_rgb_exposure.py "$base/beyond-rgb-index-v1/central.bin" "$run/cache" "$run/replay.json"
~~~
