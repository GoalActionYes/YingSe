# 单张参考的暖色倍率补回了幅度，也放大了灯光偏色

在两组实拍参考和两个目标上，逐图试涂把八张有意暖色成片的幅度从预期的 44.78%–74.00% 提升至 77.70%–110.12%。按原画质标准，四组中有两组完整通过，与统一倍率通过的是同两组。狮子的红蓝灯参考即使未做暖色编辑，逐图倍率仍令两个目标的纯换灯偏差超过 3 RGB8。停止直接放大现有单图暖色估计的路线。[事前协议](../../experimental/adaptive-reference-20261113/warmth-single-reference-probe-protocol.md) · [狮子到 MIT 客厅的失败对照](../../outputs/adaptive-reference-20261113/warmth-single-reference-probe-a/lion/mit/comparison.png) · [杯子到 Beyond RGB 的通过对照](../../outputs/adaptive-reference-20261113/warmth-single-reference-probe-a/bmug/beyond_rgb/comparison.png)

## 为什么测量单张照片

已知参考的调色前后图时，绕开 RawTherapee 的参考侧色差衰减可以补足暖色。普通参考只有一张照片，因此本轮在同一张参考图上固定一笔轻微暖色，比较原敏感度和高敏感度下的实际显影的色差，再检验此倍率能否校正原有单图估计。原估计从同一表面的弱作用区拟合底色，再测量强作用区偏离拟合结果的色度。三种方法都沿用冻结的参考和目标位置、灰片权重、目标输入与 80% 重放强度。[已知开关时的结果](2026-11-25-warmth-color-scope.md) · [原单图评分和 FAU 换灯对照](2026-11-25-fau-single-reference-warmth.md)

EGM 独立审查提出增加统一倍率对照、复查 8 位量化、预先写明沿用的拟合准入条件，并单独统计纯换灯的退化。实现按这四项修订。统一倍率取八张匿名参考实测倍率的中位数 1.5935，只用于离线对照。逐图倍率只读当次参考、冻结位置和渲染设置，调过色和只换灯的参考均接受同样处理。先生成并登记全部 48 张原尺寸成片，随后才读取目标开态、开关标签和旧评分。

## 幅度、画质及纯换灯退化

表中各栏是白蓝灯与红蓝灯下，有意暖色达到预期的百分比。既有标准要求两档都在 75%–125%，各项 RGB8 误差不超过 3，并通过保护区和新增端点检查。最后一栏记录以红蓝灯纯换灯照片为参考时，成片相对未调色目标的核心区平均绝对误差。

| 参考 → 目标 | 原单图 | 统一倍率 | 逐图倍率 | 统一／逐图是否完整通过 | 红蓝灯关态原单图 → 逐图（RGB8） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 狮子 → MIT 客厅 | 48.14%／53.54% | 78.30%／74.58% | 88.99%／93.43% | 否／否 | 2.63 → 3.69 |
| 狮子 → Beyond RGB 白绿场 | 49.54%／44.78% | 72.31%／77.03% | 77.70%／96.21% | 否／否 | 2.64 → 3.57 |
| 杯子 → MIT 客厅 | 74.00%／69.83% | 106.61%／110.98% | 106.92%／110.12% | 是／是 | 未越线 |
| 杯子 → Beyond RGB 白绿场 | 63.00%／61.89% | 103.88%／102.06% | 104.35%／103.42% | 是／是 | 未越线 |

狮子红蓝灯的有意暖色成片也各有误差超线，MIT 客厅为 4.09 RGB8，Beyond RGB 白绿场为 3.63 RGB8。统一倍率同样放大了狮子的红蓝灯关态偏差，两个目标分别为 4.54 和 4.23 RGB8。两类倍率都保留了登记的区外保护；这次最明显的退化发生在候选编辑区域。这个结果说明，当同一个色度估计把场景灯色也算作要迁移的暖色时，倍率会连同灯色一起放大。[完整分区读数与文件哈希](2026-11-25-warmth-single-reference-probe-evidence.json)

八张参考在 8 位和 16 位均满足事前准入。两种精度使用相同核心像素，有效比例为 95.46%–96.24%。8 位倍率范围为 1.374–1.669，16 位为 1.527–1.785，同一参考的最大倍率差为 0.219。8 位读数用于正式比较，16 位复核了测量资格。两位深的倍率差表明，轻微试涂的具体倍率对量化敏感。

## 复核与继续条件

两遍机器运行分别耗时 41.62 和 38.53 秒。两遍共核对 134 份逐字节一致的 PNG、JSON 和 PP3 文件。独立复算核对了 32 项两位深试涂响应和 42 项区域评分，并确认归档中的原单图基线逐像素复现。第二遍文件访问审计确认全部 48 张输出与回执先写盘，才读取目标开态与完整旧评分。

下一步要验证单张参考是否还有独立于暖色幅度的线索，能在固定位置区分真实彩灯和有意调色。这批实拍灯光沿暖色方向的最高投影为预期编辑的 29.55%，低于先前登记的 50% 压力线。先核验更强的真实彩灯开关与对应编辑素材，再决定是否试验新线索。现有结果来自预给位置的静态开发案例。普通两图自动推断、肤色保护、独立真人评价和视频验收仍待验证。

## 复现

使用仓库现有的 RawTherapee 5.13、Python 环境及冻结输入。从仓库根执行，指定一个尚未存在的输出目录。

~~~powershell
if (Test-Path 'V:/') { throw 'V: is in use' }
subst.exe V: (Get-Location).Path
try {
    $base = 'V:/outputs/adaptive-reference-20261113'
    $env:PYTHONPATH = "V:/;V:/.tools/tf-convert-env/Lib/site-packages;$base/scipy-capacity-20261122/wheel;$base/colour-tools-20261122/colour_science-0.4.7-py3-none-any.whl"
    $env:OPENBLAS_NUM_THREADS = '2'
    & V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/probe_warmth_single_reference_probe.py $base "$base/paired-wall-warmth-coordinate-run-a" "$base/fau-single-reference-warmth-a" "$base/warmth-single-reference-probe-new"
    if ($LASTEXITCODE -ne 0) { throw "Warmth probe exited $LASTEXITCODE" }
} finally {
    subst.exe V: /D
}
~~~
