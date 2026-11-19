# 含实体灰阶与肤色的实拍场景未能稳定跟随同款风格

用焊工原片和同帧 Film A 版本分别作为参考，给一张同时含实体灰阶与两位人物的实拍原图调色。选取色卡下排五块灰阶和两处脸颊，标注只用于离线评分。作者的拍摄资料确认画面含色卡、透射灰阶和不同肤色的人物。两个脸颊未做仪器实测，比较对象是直接在同一原片上施加已知 Film A 所得到的 80% 预期效果。[本机四列成片](../../outputs/adaptive-reference-20261113/hdm-chart-pilot-v1/comparison.png) · [事前协议](../../experimental/adaptive-reference-20261113/hdm-chart-pilot-protocol.md) · [拍摄论文第 3.5 节](https://www.hdm-stuttgart.de/~froehlichj/hdm-hdr-2014/Paper_Jan%20Froehlich_Creating%20Cinematic%20Wide%20Gamut%20HDR-Video%20for%20the%20Evaluation%20of%20Tone%20Mapping%20Operators%20and%20HDR-Displays_SPIE-EI_2014.pdf)

| 原图区域 | 有效像素 | 关闭参考风格仍产生的色度误差 | 打开风格的响应投影／色度误差 |
| --- | ---: | ---: | ---: |
| 灰阶 1 | 110 | 4.21 | −0.051／2.25 |
| 灰阶 2 | 121 | 4.95 | −0.376／1.92 |
| 灰阶 3 | 121 | 4.92 | 0.888／1.78，通过 |
| 灰阶 4 | 121 | 4.82 | 0.629／1.34 |
| 灰阶 5 | 110 | 3.54 | 0.243／6.25 |
| 左侧脸颊 | 124 | 6.87 | 0.535／1.18 |
| 右侧脸颊 | 122 | 7.51 | 1.245／1.37，通过 |

误差单位为 RGB8 色度码值，使用 `R−Y′` 和 `B−Y′`，投影 1 代表完整的预期响应。两块较亮灰阶的风格响应反向，五块灰阶的明暗顺序保持。右侧脸颊虽然通过已知风格的增量检查，官方风格原本在该区的期望变化仅 0.38 码值，而未调色参考已使其改变 7.51 码值。因此局部风格响应通过并不能抵消基础偏色。本组沿用同一参考与同一目标，只作为开发诊断。[完整分项数值](2026-11-19-hdm-chart-pilot-evidence.json)

论文标记 481 帧，公开目录的连续原片为 480 帧、作者成片 479 帧。固定发布原片中位位置第 1260 帧，只下载这份 6,869,862 字节的 1920×1080 EXR。未将作者档案成片与原片强行配对。沿用已核验的 LogC3 与 ACES 2.0 SDR 流程，LogC3 超出常规 LUT 输入范围的通道占 0.17%，制备的开关图零位梯度相关 0.994。各区用目标两种风格状态共同未触及端点的像素评分。[校方项目与学术使用说明](https://hdm-stuttgart.de/vmlab/hdm-hdr-2014/) · [已有 Film A 跨实拍对照](2026-11-19-arri-shared-look.md)

两遍准备图、输出与报告逐字节一致，独立复算七区分数及灰阶顺序。用既有 OpenCV 和 OpenColorIO 2.5.2 环境运行 `fetch_hdm_hdr_paired_stills.py <新目录> --hdr-chart`，随后运行 `probe_hdm_chart.py prepare <档案目录> <Film A CUBE> <新准备目录>` 和 `probe_hdm_chart.py render <准备目录> <既有 arri-shared-look-cross-source-v1> <选区 JSON> <新输出目录>`。原片、成片和模型仅留本机。未修改软件或触碰独立保留集。[测量脚本](../../experimental/adaptive-reference-20261113/probe_hdm_chart.py) · [选区坐标](../../experimental/adaptive-reference-20261113/hdm-chart-pilot-regions.json)
