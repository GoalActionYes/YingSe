# 两版在实拍换灯和有意提亮近似时仍会漏迁移墙面编辑

取复印室与客厅已经通过匹配的 12 个方向。每个方向固定另一间房的 0 号灯场景作目标，依次给两款工作台看原参考、真换灯参考和有意提亮参考。原片、实际换灯图、局部编辑图均通过上传、分析、异步 PNG 导出，统一使用 80% 强度。软件仅收到普通目标图和一张普通参考。目标自身的 RawTherapee 开态只在线下评分时使用。[事前协议](../../experimental/adaptive-reference-20261113/matched-wall-lightening-workbench-protocol.md) · [两种参考在左墙的可比条件](2026-11-24-shared-authored-tone-admission.md)

| 检查 | 稳定 1.3 | 13 号 |
| --- | ---: | ---: |
| 完整方向通过 | 0／12 | 0／12 |
| 目标左墙的提亮投影，期望 0.75–1.25 | −0.711 至 0.006 | −0.046 至 0.151 |
| 仅换参考实拍灯位的目标左墙变化 | 0.47–47.84 RGB8 | 1.61–46.42 RGB8 |
| 右墙、灰球和远处测区的最大换灯差 | 50.29 RGB8 | 43.89 RGB8 |
| 完全相同的目标和参考，逐像素保持 | 2／2 | 2／2 |

提亮投影度量实际编辑相对预期的方向和幅度，达到 1 表示该方向幅度相当。保护和误差的上限仍为 3 RGB8。两版的真实换灯差都在部分方向超过上限，且没有一项左墙提亮达到最低投影要求。固定只看原参考，目标左墙本身也分别有 3.08–34.50 与 8.87–25.43 RGB8 偏差。保留这些错误，供后续同时检查关闭态、应有风格和真实换灯。[逐区核验及文件身份](2026-11-24-matched-wall-lightening-workbench-evidence.json)

例如复印室 4 至 20 的两种参考变化在原左墙方向和幅度接近。给 13 号作参考后，实际换灯令目标客厅左墙变化 27.45 RGB8，右墙变化 27.67 RGB8。有意提亮的目标左墙只保留预期反应的 5.37%。稳定版在另一对复印室 14 至 20 的局部编辑里出现反向响应。两款工作台各保留全部结果。这一对照确认现有两版的难点包括漏掉局部提亮和误传参考灯光。两种参考仅在已登记左墙测区匹配，全图仍有可见光线与阴影差异。[13 号客厅成片对照](../../outputs/adaptive-reference-20261113/matched-wall-lightening-workbench-v1/spatial13/everett_living2/comparison.png) · [稳定版反例](../../outputs/adaptive-reference-20261113/matched-wall-lightening-workbench-v1/stable/everett_living2/comparison.png)

## 复现

沿用[准备好的室内开关成片和研究环境](2026-11-24-shared-authored-tone-admission.md#复核与复现)，从项目根运行脚本，传研究根目录和一个新输出目录。每遍共 46 张不重复的跨场景导出、4 张同图保持对照和 4 张图板。实际两遍耗时 99.7 秒和 98.1 秒。报告逐字节一致，SHA-256 为 `f439abb2f8fced434d1101845a94e0071a7ec27b350ee06a2928ba6c52461641`。另逐张核对两遍共 100 张工作台成片和 8 张图板，重算分区偏色、投影、关态和保护指标。室内样本属于同一个来源的开发组，已加入回归。未训练、未改动产品，独立评估保持封存。[执行脚本](../../experimental/adaptive-reference-20261113/probe_matched_wall_lightening_workbench.py)

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$root = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_matched_wall_lightening_workbench.py $root "$root/matched-wall-lightening-workbench-new"
~~~
