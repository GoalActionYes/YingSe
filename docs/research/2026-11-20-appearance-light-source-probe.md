# 冻结参考外观特征未能分开实拍与数字染色

参考外观分支能改变成片，但同时减轻串色、损伤有意风格。本轮检验其已有编码器是否能提供可单独使用的判别信息。两组判别器用相同线性岭回归，比较全图 RGB 统计，以及同一统计加 128 维冻结参考外观均值。运行时每个预测只读参考 RGB。训练配对、照明身份和数字操作只用于离线制作及评分。场景层面的角色在运行前固定。[事前协议](../../experimental/adaptive-reference-20261113/appearance-light-source-probe-protocol.md) · [外观开关成片](2026-11-20-beyond-rgb-appearance-ablation.md)

四个 CHROMA-MILL 训练场景各有两档实拍白／蓝灯。对白灯图施加按真实蓝灯整体色偏匹配的数字染色，配成 8 对。其中 Scene3_a 的高档因裁切超过 5% 排除，剩余 7 对来自原定四个场景。留出覆盖两个 CHROMA-MILL 场景四档已核验的对照条件和两个 Beyond RGB 场景，两来源均已在本项目用作其他开发诊断。[原实拍训练与成片](2026-11-19-pair-supervised-reference-correction.md) · [Beyond RGB 匹配对照](2026-11-20-beyond-rgb-matched-cast.md)

| 检查 | 全图统计 | 加冻结外观 |
| --- | ---: | ---: |
| 合格训练配对同时判对 | 7／7 | 7／7 |
| CHROMA-MILL 留出配对同时判对 | 0／4 | 0／4 |
| Beyond RGB 留出配对同时判对 | 0／2 | 0／2 |
| 六张白灯或未编辑参考误触发 | 2／6 | 0／6 |

两臂都把四张留出 CHROMA-MILL 实拍蓝灯预测为未换灯，尽管这些样例中真实蓝灯的得分高于匹配的有意染色。加入外观后，Beyond RGB 的 WHITE_green 实拍灯色和数字染色都判为换灯，BLUE_red 两份都判为未换灯。不能把原图误触减少解释为有意风格保护或串色改善，因为没有留出配对能同时得出两个正确动作。[每张输入的分数与哈希](2026-11-20-appearance-light-source-probe-evidence.json)

结束冻结均值外观与固定线性读出的组合，保留六组误判，不调整阈值、正则或权重追分。最初运行在导入旧脚本时缺少 requests，尚未拟合或生成结果。改为复用原来的线性 RGB 编解码和已安装的 OpenCV 后，两次报告逐字节一致。此次没有调色新成片、候选接入或独立验收。

## 复现

使用已有自适应环境与扩展第 4 轮权重。从仓库根目录运行，OpenCV 路径指向本机已有的对应 Python 版本。输出目录应未占用。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_appearance_light_source.py $base "$base/appearance-light-source-replay" --opencv-site .tools/tf-convert-env/Lib/site-packages
~~~
