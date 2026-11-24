# 两版未能分开跨室内外的实拍换光和局部提亮

以 WildRelight 浅墙、柱旁铺地和 MIT 复印室、客厅固定互配。每个方向使用原参考、同场景实拍换光参考、有意局部提亮参考。前一轮已核对后两种变化在参考的测区内方向和幅度接近。这次让稳定 1.3 和研究版 13 当作普通目标图和参考图进行导入、分析和 PNG 导出。目标自己的手工提亮只在线下按 80% 强度制作比较答案。参考与目标来自两个采集来源，仍属于已用开发素材。[事前协议](../../experimental/adaptive-reference-20261113/cross-source-wall-lightening-workbench-protocol.md) · [户外配对](2026-11-24-wildrelight-shared-tone.md) · [室内配对](2026-11-24-shared-authored-tone-admission.md)

## 实际成片

| 检查 | 稳定 1.3 | 研究版 13 |
| --- | ---: | ---: |
| 完整方向通过 | 0／18 | 0／18 |
| 编辑区投影，原门槛 0.75–1.25 | −0.955 至 0.162 | −0.003 至 0.217 |
| 编辑区因参考换光而变动超过 3 RGB8 的方向 | 12／18 | 14／18 |
| 编辑区最大参考换光差 | 63.10 RGB8 | 46.89 RGB8 |
| 保护区最大参考换光差 | 52.60 RGB8 | 35.58 RGB8 |
| 同一图输入两次，逐像素保持 | 4／4 | 4／4 |

投影为 1 表示沿着预期提亮方向、幅度相当。13 号的最大换光差比稳定版低，超过原门槛的方向更多。只给未经局部编辑的原参考，各方向的目标编辑区与目标原图仍相差 15.00–59.79 RGB8 和 7.68–29.20 RGB8。这个测试同时记录误传灯光、基础偏差和漏迁移。[逐方向证据与哈希](2026-11-24-cross-source-wall-lightening-evidence.json)

以户外浅墙作参考时，13 号把复印室原有暖色明显拉向青灰。浅墙第二档的实拍换光令复印室编辑区变化 13.40 RGB8，灰球变化 16.07 RGB8。切换同款有意提亮，复印室编辑区仅出现 1.26% 的预期响应。另一个方向以户外铺地作参考，13 号在客厅左墙取得本轮最高的 21.71% 提亮响应，换光依然让右墙变化 14.34 RGB8。局部编辑的答案只定义登记的测区，不能从本轮判断其他合理的调色选择。[13 号复印室完整成片](../../outputs/adaptive-reference-20261113/cross-source-wall-lightening-workbench-v2/spatial13/14n_copyroom1/comparison.png) · [13 号客厅完整成片](../../outputs/adaptive-reference-20261113/cross-source-wall-lightening-workbench-v2/spatial13/everett_living2/comparison.png)

室内外两组同款提亮都出现上述问题。18 个方向加入下一候选的开发回归，考核编辑区提亮和其他区域保持。本轮没有人像，局部提亮测的是亮度编辑。下一阶段的完整候选验证仍需肤色、有意暖墙和独立画质材料，明确“应当调整浅色”的正例，防止把所有浅色一律保护。完整开发、人工自然评价和视频验收沿用既有门槛。[EGM 审过的继续条件](2026-11-24-spatial-condition-design.md) · [独立验收边界](2026-11-19-quality-handoff.md)

## 复现与核验

复用已经准备好的室内外 RGB 成片、原边界和研究环境。传研究输出根和一个新目录。首跑发现部分作用范围有小数及越出画面的坐标，改用上一轮以实际像素坐标比较的同一判定，留存中间导出。两次完整运行各约 128.8 秒，均导出 70 张不重复的跨来源 PNG、8 张同图 PNG 及 8 张对照板。报告逐字节相同，SHA-256 为 `d9caefeff6a92f7bbde83347aa687dc8959e4f51f75421792a6f0bbc6e33da3a`。两遍各从磁盘重新打开实际成片与输入，核验像素、ICC 或文件哈希，独立重算全部 240 个变体分区指标。媒体、模型及完整报告保留本机。[实验脚本](../../experimental/adaptive-reference-20261113/probe_cross_source_wall_lightening_workbench.py)

重建户外素材所用的发布清单按字节校验。在启用 `core.autocrlf=true` 的 Windows 上，新检出曾改变这份 JSON 的字节。仓库现在为该文件固定 LF，已在临时目录重现问题并验证修正后的新检出。输入文件内容和两版软件均沿用原版本。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$root = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_cross_source_wall_lightening_workbench.py $root "$root/cross-source-wall-lightening-workbench-new"
~~~
