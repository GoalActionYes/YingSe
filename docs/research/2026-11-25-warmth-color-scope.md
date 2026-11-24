# 色差衰减解释了这批配对参考的暖色不足

在完全相同的参考和采样像素上，绕开 RawTherapee 参考侧的色差衰减，再估计要迁移的色度偏移。**八张原尺寸重放全部过线，恢复 99.96%–110.62% 的预期暖色。** 原敏感度的同样本对照逐字节复现旧成片，暖色仍只有 59.08%–71.47%。两种重放都使用已知参考配方、开关和编辑位置。本结果提供可复现的离线幅度对照。[事前协议](../../experimental/adaptive-reference-20261113/warmth-color-scope-protocol.md) · [杯子 → Beyond RGB 的新成片](../../outputs/adaptive-reference-20261113/warmth-color-scope-a/bmug/beyond_rgb/compensated-captured.png) · [同输入的原敏感度对照](../../outputs/adaptive-reference-20261113/warmth-color-scope-a/same-support-control/bmug/beyond_rgb/compensated-captured.png)

## 为什么改这一项

先前只改变范围外的画面，暖色不动。同一杯子核心像素保留、只换较近内容时，暖色明显增强。RawTherapee 5.13 的实现将色彩与明暗工具的敏感度用于局部色差衰减，到 100 时该权重取 1，几何过渡仍保留。实验只把参考 PP3 的 `Sensi_0` 从 5 改到 100，其余字节均和原开关文件一致。两组重复都用实际编辑器重新出图。[邻域实测](2026-11-25-warmth-surround-context.md) · [参数读入](https://github.com/RawTherapee/RawTherapee/blob/5.13/rtengine/iplocallab.cc#L8719-L8723) · [衰减与几何混合](https://github.com/RawTherapee/RawTherapee/blob/5.13/rtengine/iplocallab.cc#L9014-L9068)

在杯子的白蓝灯和红蓝灯档，近处替换相对远处替换带来的暖色差，分别由 +1.79、+1.51 Lab 单位降到 0。原始杯子照片的核心响应分别从 5.47 增至 8.82、从 5.21 增至 8.79。狮子两档原本的邻域差仅为 0.03、0.06，切换后都为 0，原始照片响应却由约 5.8–5.9 增至约 9.0。两处 128 灰的开态像素与原敏感度完全相同，归一化权重差 P95 为 0，支持纯色均匀面的实际作用范围在本次校准中保持。

## 对真实重放的影响

参考两档敏感度的偏移均在同一组有效核心像素计算，保留比例为 95.46%–96.24%。两个目标的输入、强度 80%、冻结编辑位置、关态恒等和 RGB8 计分完全相同。高敏感度只用于生成参考偏移。测得的有效暖色目标区间为 75%–125%。

| 参考 → 目标 | 原敏感度重放 | 高敏感度重放 | 新成片最大核心误差（RGB8） |
| --- | ---: | ---: | ---: |
| 狮子 → MIT 客厅 | 71.47%／69.92% | 110.10%／110.62% | 1.53 |
| 狮子 → Beyond RGB 白绿场 | 68.41%／65.56% | 106.23%／105.89% | 1.09 |
| 杯子 → MIT 客厅 | 68.96%／63.67% | 108.44%／107.42% | 1.42 |
| 杯子 → Beyond RGB 白绿场 | 61.36%／59.08% | 100.65%／99.96% | 1.42 |

新成片与预期在登记保护区和另报的邻近保护带均逐像素相同，未增加通道端点。两个灯档之间最大的核心变化为 0.16 RGB8。这里的保护来自冻结位置和权重。皮肤的保护及独立自然评价仍待实测。[逐项数值与文件哈希](2026-11-25-warmth-color-scope-evidence.json)

## EGM 审查与下一步

独立审查要求固定参与取值的像素，检查灰片权重和绝对响应，并隔离历史评分。生成过程只读精简回执中的参考开关哈希、校准及冻结偏移。第二遍文件访问审计确认 16 张重放和哈希先写盘，随后才读取两个目标开态和完整旧评分。两遍分别耗时 22.51 和 22.94 秒。46 份 PNG、JSON 和 PP3 逐字节一致。另从实际 PNG 复算 24 项参考核心读数、两个灰片对照和 42 个分区结果。[精简输入回执](../../experimental/adaptive-reference-20261113/warmth-color-scope-inputs.json)

下一轮用每次只读取一张参考的方式试固定小幅编辑探针，核验它能否估计这类衰减，并校正已有的局部暖色读数。先固定候选位置以验证信息是否可取得，同时保留灯光假暖、色偏和正常风格的检查，随后才验证自动定位。这次只覆盖已知配方和两张静态目标，软件普通两图的完整验收仍待完成。

## 复现

从仓库根执行，沿用冻结 FAU 开关、嵌套背景替换和配对校准目录。输出目录须不存在。

~~~powershell
if (Test-Path 'V:/') { throw 'V: is in use' }
subst.exe V: (Get-Location).Path
try {
    $base = 'V:/outputs/adaptive-reference-20261113'
    $env:PYTHONPATH = "V:/;V:/.tools/tf-convert-env/Lib/site-packages;$base/scipy-capacity-20261122/wheel;$base/colour-tools-20261122/colour_science-0.4.7-py3-none-any.whl"
    $env:OPENBLAS_NUM_THREADS = '2'
    & V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/probe_warmth_color_scope.py $base "$base/paired-wall-warmth-coordinate-run-a" "$base/fau-single-reference-warmth-a" "$base/warmth-surround-context-a" "$base/warmth-color-scope-new"
    if ($LASTEXITCODE -ne 0) { throw "Scope replay exited $LASTEXITCODE" }
} finally {
    subst.exe V: /D
}
~~~
