# 作者参数改善了跨图局部改色，sRGB 端点直放仍未通过

上一轮从完成图拟合固定色差或通道比例，在另一张真实照片上都出现过调或不足。本轮使用同一份已归档的 darktable 作者参数，把已经导出的 sRGB 关态作为输入。分别运行开、关，并把两者的色度差按原范围与 80% 强度合成回输入图。实验只改变操作所在管线：以前的对照在 RAW 显影中应用，候选在 sRGB 导出图上应用。区域来自既有登记，原目标编辑答案仅用于评分。关闭模块后编辑器的输出也必须与输入相差不超过 1 RGB8。[事前协议](../../experimental/adaptive-reference-20261113/playraw-rendered-input-operator-protocol.md) · [RAW 编辑对照](2026-11-20-playraw-mixed-light-cross-editor.md)

| 检查 | 实际结果 |
| --- | --- |
| 《Cuckoo》脸颊色度 MAE，固定色差 → 作者参数 | 3.094 → 0.559 RGB8，投影 0.997 |
| 《Cuckoo》胸口色度 MAE，固定色差 → 作者参数 | 4.502 → 0.608 RGB8，投影 1.012 |
| 混合光原片两折高权重区 | 3.294／3.266 RGB8，超过 3 的限值 |
| 模块关闭时的最大 RGB8 偏差 | 混合光 64，《Cuckoo》两份缓存均为 71，未达标 |
| 区域外像素与区内亮度 | 三张合成结果区外逐像素保持，最大 Y′ 量化差 0.4934 RGB8 |

《Cuckoo》说明参数包含能随目标像素变化的有效信息。两份旧缓存的脸颊和胸口读数一致。两处浅色树皮及绿地保护区保持来自预先给定的编辑范围，不能代表自动保护。混合光高权重区仍超线，因此即使把关态偏差从候选合成里扣掉，也未达到预定画质要求。关闭时的误差说明这份最小编辑历史仍有从 sRGB 输入到输出的变化，现有实验未定位到具体模块或色彩转换。[本机《Cuckoo》对照](../../outputs/adaptive-reference-20261113/playraw-rendered-input-operator-v1/cuckoo-0/comparison.png) · [混合光反例](../../outputs/adaptive-reference-20261113/playraw-rendered-input-operator-v1/mixed/comparison.png)

停止采用这套 sRGB 端点直接重放。保留《Cuckoo》的局部改善证据与关闭态、混合光反例。下一步若继续使用作者参数，需要测明输入到模块之间的颜色变化，并核对明确的工作空间能否提供更可靠的目标响应。普通两图也仍缺从参考成片辨认作者操作的信息。稳定软件及功能未改动，未启动独立自然评价和视频验收。

使用现有 CPU 编辑器，两次各导出六张 TIFF。三张最终合成 PNG 和三张对照图逐件同哈希。混合光开态 TIFF 的 RGB 两次最多相差 1，涉及 17 像素，数值报告因此保留两种源开态哈希；其余编辑器输出的 RGB 一致。通过落盘图独立复算关态差和最终成片范围外保持。[完整读数和复跑差异](2026-11-20-playraw-rendered-input-operator-evidence.json)

在仓库根目录运行，先使用本机未占用的 ASCII 盘符映射该仓库。以下以 R 盘为例，释放映射只在该命令自行建立后执行。

```powershell
if (Test-Path 'R:/') { throw 'R: is in use' }
subst R: (Get-Location).Path
if ($LASTEXITCODE) { throw 'Could not map the workspace' }
try {
    $base = 'R:/outputs/adaptive-reference-20261113'
    $env:PYTHONPATH = (Get-Location).Path
    $env:OPENBLAS_NUM_THREADS = '2'
    & .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_playraw_rendered_input_operator.py $base "$base/playraw-rendered-input-operator-replay"
} finally {
    subst R: /D
}
```
