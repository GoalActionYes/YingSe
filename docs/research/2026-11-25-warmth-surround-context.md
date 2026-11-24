# 同样的核心像素，其暖色只在部分近处内容改变时增强

六张照片的局部开关重新出图后逐像素复现原先结果。将编辑范围外的画面换成核心底色，六张图的核心暖色读数完全不变。进一步只换编辑范围内、核心外的内容，杯子两档分别增加 1.79 和 1.51 Lab 单位。白蓝灯档向两个目标比较时均补回过半缺口，同一张杯子参考提供这两项结果。其余六个方向未达到预设的 50% 收窄线。[事前协议](../../experimental/adaptive-reference-20261113/warmth-surround-context-protocol.md) · [杯子原图开态](../../outputs/adaptive-reference-20261113/warmth-surround-context-a/bmug-base/original/on.png) · [杯子近处替换开态](../../outputs/adaptive-reference-20261113/warmth-surround-context-a/bmug-base/near_flat/on.png)

## 这次替换了哪里

按明度或底色单独计算的倍率都无法补足真实成片。本轮固定同一版 RawTherapee、PP3 和旧核心。远处替换只改编辑范围外。较近替换从远处替换的结果开始，区外逐像素保留，只再替换编辑范围内的非核心内容。核心周围保留至少 4 像素原始画面，边界用余弦羽化。两种替换均填原核心的 RGB8 中位颜色。三臂都使用原坐标、尺寸、ICC 和旧权重。[明度结果](2026-11-25-warmth-lightness-response.md) · [底色结果](2026-11-25-warmth-input-chroma.md)

远处替换令编辑范围外 96.11%–99.80% 的像素至少有一个通道改变 8 RGB8，编辑范围内原样保留。较近替换令编辑范围内 84.33%–92.17% 的像素达到同一变化线，区外与远处替换完全一致。每组先核对关态与输入逐像素相等、保留框和核心相同、编辑范围外保持及新增端点。六组共同核心覆盖为 95.46%–96.61%，18 对开关无新增端点。评分只在这些核验后进行。

## 核心暖色读数

从实际开关图取得 Lab 差，沿预定暖色轴报告。同一照片三臂都使用共同核心像素。

| 素材 | 原图 | 远处替换 | 接着替换较近内容 | 较近增量 |
| --- | ---: | ---: | ---: | ---: |
| 狮子，白蓝灯 | 5.94 | 5.94 | 5.97 | +0.03 |
| 狮子，红蓝灯 | 5.78 | 5.78 | 5.84 | +0.06 |
| 杯子，白蓝灯 | 5.47 | 5.47 | 7.26 | +1.79 |
| 杯子，红蓝灯 | 5.21 | 5.21 | 6.72 | +1.51 |
| MIT 客厅 | 8.73 | 8.73 | 8.73 | 0 |
| Beyond RGB 白绿场 | 8.74 | 8.74 | 8.74 | 0 |

较近替换使杯子白蓝灯的两个目标差分别缩小 54.75% 和 54.58%。红蓝灯的两项缩小 42.87% 和 42.75%。狮子四项只缩小约 1%–2%。远处替换在八个方向收窄均为零。所有这些比例衡量同一编辑器的核心响应差，完整实拍两图仍使用原画质门槛。[逐素材数值、分区和文件哈希](2026-11-25-warmth-surround-context-evidence.json)

## EGM 审查与决定

独立审查要求将两次替换改成嵌套对照，区分干预不足与有效阴性，并在计算暖色前完成个别素材资格核验。复核两遍分别耗时 18.24 和 20.17 秒。54 张 PNG、6 份资格回执、1 份结果文件逐字节一致。用保存的 PNG 独立核对保留区、实际改变比例、18 个核心读数和 8 个差值。六张原图的全部关态和开态也与各自旧存档的像素哈希一致。

结束用远处内容变化解释这批已知位置幅度差。近处内容只对杯子有明显影响，狮子和两个目标留下反例。下一步可继续区分杯子上的局部混合与原有颜色响应，先查这种变化能否由原图可取得的真实邻域解释，再用旧实拍评分衡量实际重放。当前证据来自同一编辑器的合成干预和六张已知位置图片。映色的普通两图输入、肤色、独立质量和完整视频验收仍按原任务。

## 复现

从仓库根目录运行，沿用锁定的旧原图、FAU 压力实验及平场权重。输出目录须不存在。

~~~powershell
if (Test-Path 'V:/') { throw 'V: is in use' }
subst.exe V: (Get-Location).Path
try {
    $base = 'V:/outputs/adaptive-reference-20261113'
    $env:PYTHONPATH = "V:/;V:/.tools/tf-convert-env/Lib/site-packages;$base/scipy-capacity-20261122/wheel;$base/colour-tools-20261122/colour_science-0.4.7-py3-none-any.whl"
    $env:OPENBLAS_NUM_THREADS = '2'
    & V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/probe_warmth_surround_context.py $base "$base/paired-wall-warmth-coordinate-run-a" "$base/fau-single-reference-warmth-a" "$base/warmth-surround-context-new"
    if ($LASTEXITCODE -ne 0) { throw "Context check exited $LASTEXITCODE" }
} finally {
    subst.exe V: /D
}
~~~
