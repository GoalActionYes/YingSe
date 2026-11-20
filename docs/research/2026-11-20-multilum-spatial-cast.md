# 参考图内的空间色偏未通过换灯与有意染色对照

固定 MIT Multi-Illumination 两个新地点的样例，以单张参考 RGB 检查局部色度分布。厨房取作者示例的 elm_revis_kitchen14，大厅取作者地点索引列出的 everett_lobby1。仓库既有两组使用不同地点和场景。本次沿用 14、24 灯位，场景在取片前确定。两张实拍控制了机位和画面内容，灯位变化也会影响阴影及照度。参考图独立计算 8 × 8 网格内近灰像素的代表色和网格间离散程度。对照图用统一通道增益，把原图的整体色偏拉近换灯实拍，同时保留线性亮度。[实验协议](../../experimental/adaptive-reference-20261113/multilum-spatial-cast-protocol.md) · [MIT 数据发布与许可](https://projects.csail.mit.edu/illumination/) · [作者场景索引](https://projects.csail.mit.edu/illumination/databrowser/)

低分辨率 RSR 的作者下载入口跳到机构登录，原分辨率入口也出现同样跳转。本轮采用官方能按场景小批访问、标为 CC-BY 的 MIT 素材。厨房完整归档超过预设下载预算，实际只读两份 ZIP 目录尾段和四个指定 JPG 的字节范围，共 1,360,987 字节。每项均核对 HTTP 206、成员名、长度及 CRC32，未下载整包。[RSR 作者入口](https://color-cvc.github.io/relighting_from_image/#rsr-real-scene-relighting-dataset)

## 结果

| 场景 | 实拍整体色偏幅度 | 与数字染色的整体色偏差 | 空间离散度：原图／实拍换灯／数字染色 | 空间判别 |
| --- | ---: | ---: | ---: | --- |
| 厨房 | 0.0164 | 0.0164 | 0.1092／0.1113／0.1093 | 两项差都不足 0.03 |
| 大厅 | 0.0606 | 0.0083 | 0.0858／0.1354／0.0820 | 两项差均超过 0.03，整体信号不足 |

实拍整体色偏幅度以对数色度范数计算，预设至少 0.08。两组的共同有效像素覆盖分别为 80.56% 和 95.02%，数字染色限域分别为 0.207% 和 0.019%，可测网格为 62–64。两组都因实拍整体色偏不足而未获正式比较资格。大厅实拍的空间离散度比数字染色高 0.0534，可以在图上看到左侧衣物受光更明显。数字图保持原线性亮度，因此该变化也可能混入局部亮度、阴影和反射变化。厨房两项空间差约 0.002。两组均保留成片和数值。[各灯位哈希、测量与资格](2026-11-20-multilum-spatial-cast-evidence.json)

## 决定与复现

结束这条固定空间离散度筛查，不更换灯位或样例追过线。两组均未持有足够整体色偏，结果既不足以支持该线索接入产品，也没有完成对明显色偏前提下的检验。若以后有新采集证据，应先用拍摄条件和局部受光差选定可比较的控制，再考虑单图线索。厨房和大厅素材属于本项目已使用的开发样本，不能进入独立验收。本轮未运行调色候选，产品版本与真实肤色、白墙和视频验收状态未改变。

从仓库根目录运行已有研究环境。复跑可复用范围缓存。两遍各四张新图与报告逐字节一致。实际照片和新图只留本机。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$base = 'outputs/adaptive-reference-20261113'
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_multilum_spatial_cast.py "$base/multilum-spatial-cast-input-v1" "$base/multilum-spatial-cast-replay"
~~~
