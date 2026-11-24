# 固定明度的底色补偿使暖色缺口扩大

以同一明度显影七种底色后，按实测响应比重放局部暖色。八张原尺寸成片均低于原配对重放，只得到预期暖色的 **55.17%–70.03%**，低于 75% 门槛。结束用这条平场底色响应直接校正幅度的做法。[事前协议](../../experimental/adaptive-reference-20261113/warmth-input-chroma-protocol.md) · [本机成片与平场](../../outputs/adaptive-reference-20261113/warmth-input-chroma-a) · [逐项证据](2026-11-25-warmth-input-chroma-evidence.json)

## 为什么做这项检查

按明度缩放后，FAU 参考向 MIT 客厅和 Beyond RGB 白绿场重放的暖色仍不足。有些参考与目标的底色相差数十 Lab 单位，超出旧量化控制的范围。本轮用六组冻结关态的编辑核心色度和一个锚色，在固定 L*=60 情况下测量同一局部编辑的响应。保留两份冻结 PP3、尺寸、权重和原区域。两处 PP3 的区别只在位置及边界字段，暖色参数相同。[前一轮实拍与幅度](2026-11-25-fau-single-reference-warmth.md) · [明度补偿结果](2026-11-25-warmth-lightness-response.md)

每个颜色转换到浮点 sRGB 后先查色域，再统一舍入。实际七种 8 位输入互不重复，L* 为 59.90–60.07，极差 0.17。狮子位置和 MIT 位置的共同有效核心分别为 95.46% 和 96.61%，14 份平场开关均合格。两处实测的暖色轴响应逐色一致，在 8.87–9.26 Lab 单位间变化，极差约为锚色的 4.30%。变化方向与注册暖色轴的最小余弦为 0.9935。范围外保持且无新增端点。

## 原尺寸成片

倍率只从平场同一核心沿固定 Lab 轴的响应取得。真实成片仍用已锁定的参考配对偏移、目标区域、80% 强度和统一 RGB8 评分，亮度偏移固定为零。对照为底色和明度试验前冻结的原 FAU 配对结果。

| 参考 → 目标 | 原配对暖色幅度 | 底色补偿暖色幅度 | 两档倍率 |
| --- | ---: | ---: | ---: |
| 狮子 → MIT 客厅 | 71.47%／69.92% | 70.03%／69.81% | 0.995／0.980 |
| 狮子 → Beyond RGB 白绿场 | 68.41%／65.56% | 65.92%／63.28% | 0.977／0.962 |
| 杯子 → MIT 客厅 | 68.96%／63.67% | 64.26%／59.66% | 0.979／0.978 |
| 杯子 → Beyond RGB 白绿场 | 61.36%／59.08% | 57.35%／55.17% | 0.961／0.960 |

八个倍率都小于 1。按旧输出作线性近似，达到暖色下限至少需要约 1.049–1.269 倍。新成片在原有保护区仍达线，实际失败均在幅度项。补偿后幅度最低的是杯子到 Beyond RGB 的第二个灯档，暖色响应从 59.08% 降到 55.17%。[最低幅度的实际 PNG](../../outputs/adaptive-reference-20261113/warmth-input-chroma-a/bmug/beyond_rgb/compensated-captured.png)

## EGM 审查与后续依据

独立审查要求记录量化前后、平场的原幅度和真实成片的原评分单位，并明确缺测、幅度不足、画质失败的不同后果。第二遍文件访问检查确认八张主输出和哈希回执写盘后才读两张目标开态。两遍分别耗时 36.09 和 24.46 秒，52 份 PNG 和 JSON 逐字节一致。另从保存的 PNG 再测 14 个平场响应、8 个倍率和 28 项分区结果。

这轮排除的是用 L*=60 平场倍率修正这八张成片。真实素材的明度跨度约为 L*=33.33–78.43，色度与明度的交互仍未验证。下一项先把相同的编辑核心像素置于不同周围内容，在同一位置重跑开关，检查局部混合是否能解释剩余的幅度差。这将检验靠平场无法提供的场景信息。普通参考的意图和位置，以及软件的独立画质、肤色、真人和视频验证仍按原目标执行。

## 复现

从仓库根目录执行，沿用现有编辑器、环境和两个已核验的研究目录。输出目录须不存在。

~~~powershell
if (Test-Path 'V:/') { throw 'V: is in use' }
subst.exe V: (Get-Location).Path
try {
    $base = 'V:/outputs/adaptive-reference-20261113'
    $env:PYTHONPATH = "V:/;V:/.tools/tf-convert-env/Lib/site-packages;$base/scipy-capacity-20261122/wheel;$base/colour-tools-20261122/colour_science-0.4.7-py3-none-any.whl"
    $env:OPENBLAS_NUM_THREADS = '2'
    & V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/probe_warmth_input_chroma.py $base "$base/paired-wall-warmth-coordinate-run-a" "$base/fau-single-reference-warmth-a" "$base/warmth-input-chroma-new"
    if ($LASTEXITCODE -ne 0) { throw "Chroma check exited $LASTEXITCODE" }
} finally {
    subst.exe V: /D
}
~~~
