# 两个参考来源的工作台暖色回归

把参考从 MIT 复印室换成 Beyond RGB，稳定版 1.3 和研究版 13 都复现了漏迁移与误传。在应继承暖色的两张墙面上，两版仍少加暖；在应保护的三位人像上，换灯参考所得肤色及白幕出现偏差。分开人与背景能为局部调色提供位置，现有工作台仍需要判断参考中哪一部分变化来自现场灯光，哪一部分是作者有意的编辑。[两个成因的参考四图](2026-11-24-beyond-rgb-near-white-match.md) · [MIT 工作台基线](2026-11-24-matched-wall-warmth-workbench.md)

两组使用同一两张浅墙、同一三位冷灯人像、同一 80% 强度和评分区域。固定 Beyond RGB 的灯 7 关态、灯 1 关态和灯 7 局部暖色开态，分别作为基准、换灯和编辑参考。每次让软件实际导入一个目标和一张参考，再分析并导出 PNG。目标的局部编辑开关、区域和照明标签均用于离线判分。[事前比较协议](../../experimental/adaptive-reference-20261113/beyond-rgb-warmth-workbench-protocol.md)

## 实际导出结果

| 参考来源 | 软件 | 暖墙完整通过 | 人像肤色与白幕完整保持 | 暖色响应投影 | 换灯参考成片相对原图最大脸颊偏差 | 同图逐像素保持 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MIT 复印室 | 稳定 1.3 | 0／2 | 0／3 | 4.49%–8.03% | 14.40 RGB8 | 5／5 |
| MIT 复印室 | 研究版 13 | 0／2 | 0／3 | 5.50%–6.66% | 24.43 RGB8 | 5／5 |
| Beyond RGB | 稳定 1.3 | 0／2 | 0／3 | 1.09%–1.17% | 49.92 RGB8 | 5／5 |
| Beyond RGB | 研究版 13 | 0／2 | 0／3 | 0.09%–0.36% | 23.66 RGB8 | 5／5 |

暖色投影的合格区间为 75%–125%。以 13 号的室外浅墙为例，单独打开参考的编辑只留约 0.09% 的预期响应。把参考换成实际另一个灯档，则令同一区域相对基准参考成片变化 7.14 RGB8，高于 3 RGB8 上限。13 号最坏人脸为 Gros 2 的右颊，基准参考成片相对目标原图偏差为 20.17 RGB8，换灯参考成片为 23.66 RGB8。两张成片的右颊之间相差 3.56 RGB8，单独度量了切换参考灯档带来的差异。[完整指标与哈希](2026-11-24-beyond-rgb-warmth-workbench-evidence.json) · [室外浅墙图板](../../outputs/adaptive-reference-20261113/beyond-rgb-warmth-workbench-first/spatial13/wildrelight-wall/comparison.png) · [人物图板](../../outputs/adaptive-reference-20261113/beyond-rgb-warmth-workbench-first/spatial13/gros-2/comparison.png)

EGM Stage 5 核对了参考角色、旧目标与评分复用，以及产品只收到目标和参考的输入边界。两套参考还同时改变了材料、构图和相机，且 Beyond RGB 的匹配局限于实验箱右下白背景。同一参考的其他位置也会响应换灯，所以结果仅用于判断两种失效是否在同一批目标中复现。复核要求先过同图完整性再解释画质，并分开报告相对原图的基准偏色、真实换灯差和有意编辑响应。两遍同图完整性均已通过。两组结果供候选回归，原独立素材、真人有意暖肤正例、自然评价及完整视频验证另按原门槛执行。

## 复现与下一步

沿用已核验的 Beyond RGB 四图、旧目标集、稳定和研究版依赖。用新目录执行命令。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
& './.tools/adaptive-env/Scripts/python.exe' 'experimental/adaptive-reference-20261113/probe_beyond_rgb_warmth_workbench.py' $base "$base/beyond-rgb-warmth-workbench-replay"
~~~

两遍从导入到报告落盘分别耗时 80.70 和 77.24 秒。每遍 40 张实际导出、10 张图板和一份结果 JSON，51 个文件逐字节一致。两遍均从落盘 PNG 独立复算，每遍核对 44 个区域。三张参考的完整文件哈希与原登记一致，两版的应用、空间渲染和共用评分代码与 MIT 基线提交 `490e124` 相同。本次基准图、完整成片、视频素材和权重均保留本机。

两组数据和两版成片齐备，下一步可以做原有跨来源小测的输入预检。按完整拍摄来源隔离拟合和留出，先确认监督数据、位置条件、同规模池化对照及旧负例实际接得上。先测 15–20 分钟的实际显存与步速，再决定是否启动总上限 4 小时的首轮两臂运行。只有改善至少一个新的完整留出方向且未丢失对照已经通过的方向，才继续候选全套开发评分。[既有候选设计与进退标准](2026-11-24-spatial-condition-design.md)
