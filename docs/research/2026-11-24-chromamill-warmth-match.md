# CHROMA-MILL 饱和彩灯没有补足局部暖色对照

现有工作台用复印室近似的真实换灯和有意暖墙做比较，两版均漏掉大部分有意暖色。下一步需要另一个来源的真实拍摄变化与同样的编辑相近。这轮先用本机缓存的蓝白两档灯，再用作者 GitHub 演示附带的红白单灯原始预览验证。两批都沿用同一个 RawTherapee 暖色开关和原门槛。[实际工作台失败](2026-11-24-matched-wall-warmth-workbench.md) · [FAU 对照](2026-11-24-fau-shared-warmth.md) · [原暖色门槛](../../experimental/adaptive-reference-20261113/shared-local-warmth-protocol.md)

## 取样与实际出图

完整红灯档案入口多次超时，项目发布页的数据按钮仍关闭。先复用已取的 8 张手机白／蓝灯线性 PNG 和登记台面坐标。照片来自两个固定布景，B_213 和 B_241 是灯强档位。相同 B 档并不等于相同照度。上一轮白绿灯数字染色和蓝灯 Film A 成片关注整图风格和校正。本次固定编辑范围与强度，只检查同一材料的实际换光和局部暖色能否接近。[缓存资料及采集说明](2026-11-19-chromamill-capture-cross.md) · [已登记台面](2026-11-20-chromamill-fixed-material.md) · [白绿灯旧对照](2026-11-20-chromamill-matched-supervision.md) · [白蓝协议](../../experimental/adaptive-reference-20261113/chromamill-blue-white-warmth-protocol.md)

通过作者 GitHub 演示另取三个布景的全部单档红、白灯文件。Git 树有 66 个文件，共 14,770,698 字节。逐文件核对长度与 Git blob 哈希，再解包内嵌的 600 × 400 RGB8 PNG。未使用网页的合成混色和插值。每个布景只看第 11 档白光准备图，固定充气球白带、杯内壁或布面作为浅色测区。另有远处保护区。每档用同一曲线和尺寸准备。公开的演示素材只有 8 位线性数据和灯档记录，无法逐张确认曝光、白平衡和机内预处理。不同布景还复用了部分道具，全部按同一采集来源归组。[固定 Git 树](https://github.com/priubrogent/chroma-mill/tree/62fe4a3a33cd50ff39d94d20111146e3bf4d2d88/static/mixer/linear_js) · [作者单灯与混色实现](https://github.com/priubrogent/chroma-mill/blob/62fe4a3a33cd50ff39d94d20111146e3bf4d2d88/static/js/mixer.js#L98-L148) · [红白协议](../../experimental/adaptive-reference-20261113/chromamill-demo-red-white-warmth-protocol.md)

## 结果

| 实拍比较 | 素材和整图对齐通过 | 灯色方向数 | 共同覆盖与信号前置条件通过 | 方向、幅度联合匹配 |
| --- | ---: | ---: | ---: | ---: |
| 缓存白蓝灯，2 个布景 | 8／8 | 同档 8，跨档 8 | 16 | 0 |
| 演示红白灯，3 个布景 | 43／66 | 同档 66，跨档 660 | 154 | 0 |

白蓝灯中所有开关都保留了指定暖色。8 个蓝到白方向的实际台面变化都明显增绿，而有意编辑略减绿。最接近的一组实拍和编辑的中位变化为 `[47.5, 53, −59]` 和 `[16, −4, −13]`。余弦 0.677，幅度比 0.227。全部颜色方向的最大幅度比仅 0.250。[本机台面成片](../../outputs/adaptive-reference-20261113/chromamill-blue-white-warmth-v2/Scene1_a/comparison.png)

红白灯中，33 张白灯和 10 张红灯通过。充气球与杯内壁的红灯照片未过素材线，主要失败在有意编辑改变亮度过多。剩下的布面组有 154 个可比方向，8 个方向余弦过线，幅度比上限 0.189，低于原定下限 0.75。最接近的一组由第 5 档白灯到第 6 档红灯，实拍变化 `[77, −53, −52]`，编辑变化 `[6, −2, −6]`，余弦 0.944，幅度比 0.081。同一档位和跨档位均无联合命中。这里只能说明现有端点灯色对照与固定编辑的差别。[本机红白布面成片](../../outputs/adaptive-reference-20261113/chromamill-demo-red-white-warmth-v1/Test17/comparison.png) · [数值与哈希](2026-11-24-chromamill-warmth-match-evidence.json)

EGM 核查指出，B 档变化可能带来亮度差，当初的台面测区也只在 241 档登记。全图梯度对齐可用于本轮初筛，局部错位和逐张拍摄参数还需单独核验。若以后出现数值命中，需在固定编辑区周边核对地标并确认相机记录，再决定是否进入工作台。此轮没有命中，保留 51 张开关开发正例与全部失败样本，结束这批高饱和彩灯取样。再投入一轮采集的条件是接近中性的暖白光小幅变化，固定相机设置并记录测区亮度与灯档，避开本轮已量出的方向或幅度差。原跨场景训练、独立画质、人评和视频门槛仍按已有计划执行。[可复核的补拍流程](2026-11-21-capture-edit-intake.md) · [工作台准入与成本](2026-11-24-spatial-condition-design.md)

## 复现

在项目根目录使用既有研究 Python。先取固定演示源，蓝白灯沿用本机先前已核验的 8 张缓存。原始图和完整对照保留本机。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$base = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/fetch_chromamill_single_light_demo.py "$base/chromamill-single-light-demo-v1"
~~~

需要代理时向取样脚本传入 `--proxy` 和当前环境的地址。正式导出使用未占用的 ASCII 映射盘。每次选未存在的输出目录。

~~~powershell
if (Test-Path 'V:\') { throw 'V: already in use' }
subst.exe V: (Get-Location).Path
try {
    $env:PYTHONPATH = 'V:/;V:/.tools/tf-convert-env/Lib/site-packages'
    $env:OPENBLAS_NUM_THREADS = '2'
    $base = 'V:/outputs/adaptive-reference-20261113'
    $cli = "$base/rawtherapee-sky-pair-v1/runtime/RawTherapee_5.13_win64_x86_64_release/rawtherapee-cli.exe"
    $exp = 'V:/experimental/adaptive-reference-20261113'
    & V:/.tools/adaptive-env/Scripts/python.exe "$exp/probe_chromamill_blue_white_warmth.py" $base $cli "$base/chromamill-blue-white-warmth-new"
    if ($LASTEXITCODE -ne 0) { throw "White-blue export returned $LASTEXITCODE" }
    & V:/.tools/adaptive-env/Scripts/python.exe "$exp/probe_chromamill_demo_red_white_warmth.py" $base $cli "$base/chromamill-demo-red-white-warmth-new"
    if ($LASTEXITCODE -ne 0) { throw "White-red export returned $LASTEXITCODE" }
} finally {
    subst.exe V: /D
}
~~~

白蓝最终两遍约 12.8、10.2 秒，红白两遍约 49.0、48.6 秒。每遍白蓝为 26 张 PNG、4 份 PP3 和 1 份 JSON，红白为 201 张 PNG、6 份 PP3 和 1 份 JSON。两遍各自逐文件一致。独立从落盘 PNG 复算 2,820 个方向，核对文件场景、哈希、ICC 及三帧共同有效测区。EGM 审阅后直接复用旧的固定坐标，白蓝早期与最终两遍的图像和参数相同。
