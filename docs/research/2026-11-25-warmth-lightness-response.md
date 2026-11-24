# 明度补偿仍无法恢复这组跨素材暖色

用同一局部编辑对七个明度档实际出图，得到明度响应曲线。把这条曲线用于之前偏弱的 FAU 配对参考，八张原尺寸重放只恢复了 **61.39%–69.85%** 的预期暖色，均低于 75% 下限。结束按这条明度曲线缩放暖色的做法。[事前协议](../../experimental/adaptive-reference-20261113/warmth-lightness-response-protocol.md) · [八张本机补偿成片](../../outputs/adaptive-reference-20261113/warmth-lightness-response-a) · [数值与哈希](2026-11-25-warmth-lightness-response-evidence.json)

## 要解释的现象

上一轮 FAU 参考的前后两张图都可取得，调色参数与 MIT 客厅、Beyond RGB 白绿场的目标设置相同，实际迁移仍只有 59.08%–71.47%。这轮只改变用于重放的强度倍率，检验编辑对不同明度的响应能否补齐不足。RawTherapee 5.13 会计算高低参数之间的差以形成直接明度项，两端相同使该项为零。端到端导出还含局部混合和颜色变换，因此以实际输出测量。[旧 FAU 配对成片](2026-11-25-fau-single-reference-warmth.md) · [RawTherapee 参数计算](https://github.com/RawTherapee/RawTherapee/blob/5.13/rtengine/iplocallab.cc#L19190-L19198) · [RawTherapee 调色实现](https://github.com/RawTherapee/RawTherapee/blob/5.13/rtengine/iplocallab.cc#L19638-L19647)

## 受控测量

冻结 FAU 狮子的木板与 MIT 客厅墙面两处编辑形状。两份 PP3 除编辑框位置和边界字段外逐行相同。以 D50 CIELAB a*=3、b*=9 制作从 L*=30 到 90 的七档常色。各档用同一色彩空间、相同关闭及开启流程，测量原登记核心。两处共有有效核心比例分别为 95.46% 与 96.61%。所有关闭图与输入逐像素相同，编辑范围外保持，无新增通道端点。

两处在七档中的暖色方向幅度完全一致，为 8.70–9.26 Lab 单位，随明度变化并非单调。最高和最低约相差以 L*=60 为基准的 6.05%。按 EGM 建议，在相同 L*=60 另出两个相近色度的控制样，实际明度差约 0.02，幅度为 8.97 和 9.19 Lab 单位。这个约 2.31% 的差异表明 8 位量化与小幅色度变化会影响读数。本轮结论以真实成片的补偿幅度为准。近邻色度控制仅覆盖窄范围，未直接解释真实素材之间的色度差距。

## 原尺寸重放

对每个目标与供体，按关态核心的实测 L* 插值，计算补偿倍率。仅缩放冻结的配对色度偏移，参考与目标的编辑位置、80% 效果强度、关闭图及色差门槛全部沿用。实际供体 L* 为 33.33–59.72，目标为 60.40、78.43，均在曲线支持内。所有真实成片都获得了有效系数。

| 参考 → 目标 | 旧暖色幅度 | 明度补偿后的暖色幅度 | 两档倍率 |
| --- | ---: | ---: | ---: |
| 狮子 → MIT 客厅 | 71.47%／69.92% | 69.85%／69.78% | 0.963／0.970 |
| 狮子 → Beyond RGB 白绿场 | 68.41%／65.56% | 68.25%／66.17% | 0.999／1.006 |
| 杯子 → MIT 客厅 | 68.96%／63.67% | 69.73%／64.93% | 1.013／1.011 |
| 杯子 → Beyond RGB 白绿场 | 61.36%／59.08% | 63.78%／61.39% | 1.051／1.048 |

以旧响应作线性近似，八张成片需要约 1.049–1.269 倍才能达到 75%。实际补偿只有 0.963–1.051 倍，各项是否达标以新 PNG 实测为准。四组都在暖色幅度项失败，原先通过的保护区域仍过线。

## 审查、复现和下一步

EGM 独立审查要求使用共同有效核心、补做固定明度的近邻色度控制、将显影覆盖不足与暖色不足分开。第二遍文件访问审计确认两个目标开态只在八张成片和哈希回执锁定后读取。两遍分别为 26.38、25.23 秒。64 份 PNG 或 JSON 字节一致。另从落盘图独立复算 18 次校准读数、8 个补偿倍率及 28 项原区域读数。

下一轮优先在固定明度下对照真实供体与目标的原有色度。Beyond RGB 目标核心的 b* 约为 46.52，FAU 杯子约为 3.96–3.99，远超过本轮近邻控制范围。先判断这种差距在独立平场上能否复现暖色幅度变化，再决定是否值得检查场景内容对局部混合的影响。MIT 校准位置同时用于其中一个目标，FAU 已在开发中消费。此轮结果适用于冻结编辑的已知位置。普通参考的识别与定位、肤色和自然度及视频验收仍待完成。

~~~powershell
if (Test-Path 'V:/') { throw 'V: is in use' }
subst.exe V: (Get-Location).Path
try {
    $base = 'V:/outputs/adaptive-reference-20261113'
    $env:PYTHONPATH = "V:/;V:/.tools/tf-convert-env/Lib/site-packages;$base/scipy-capacity-20261122/wheel;$base/colour-tools-20261122/colour_science-0.4.7-py3-none-any.whl"
    $env:OPENBLAS_NUM_THREADS = '2'
    & V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/probe_warmth_lightness_response.py $base "$base/paired-wall-warmth-coordinate-run-a" "$base/fau-single-reference-warmth-a" "$base/warmth-lightness-response-new"
    if ($LASTEXITCODE -ne 0) { throw "Lightness check exited $LASTEXITCODE" }
} finally {
    subst.exe V: /D
}
~~~
