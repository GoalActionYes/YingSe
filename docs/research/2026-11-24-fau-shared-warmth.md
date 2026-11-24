# FAU 单灯位换色未补足有意暖色的实拍对照

映色需要能压住参考里由拍摄灯光带来的偏色，同时保留作者有意给主体或环境加上的暖色。此前相同的局部暖色设置只在 MIT 复印室找到一个实拍换光方向，其变化方向和幅度都接近有意编辑。这次用已有的 FAU 双灯档案补查另一机构的实拍，确认是否有依据投入跨来源比较和训练。先用 RawTherapee 重放已有暖色开关，计划用时 25–45 分钟，其中出图上限 8 分钟。实际两遍出图与计分分别约 50.3 秒、32.2 秒。本轮未新增下载或训练。[事前协议](../../experimental/adaptive-reference-20261113/fau-shared-warmth-protocol.md) · [既有暖色准入](2026-11-24-shared-local-warmth.md)

## 为什么用这套素材

FAU 实验室档案记录了左右两盏灯的白、红、蓝色组合，并给出暗区掩膜。旧 FAU 试验使用固定机位的红蓝互换。本次纳入全部 58 张实验室预览，检验只改变一个灯位的颜色能否与同一局部暖色编辑相似。按事前锁定的白蓝灯锚图选取一个浅色表面和远处保护区，跨灯档沿用坐标。测区含 7 处杯底，其余来自道具或台面。所有档位沿用相同的暖色参数 1600、2800 和力度 30，关闭态和开启态由同版 RawTherapee 实际导出。[Beigpour 等，原论文第 VII 节](https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2014/Beigpour14-MEW.pdf) · [FAU 档案](https://www5.cs.fau.de/research/data/two-illuminant-dataset-with-computed-ground-truth/) · [冻结测区](../../experimental/adaptive-reference-20261113/fau-shared-warmth-regions.json) · [旧 FAU 灯档比较](2026-11-18-fau-reference-light-setting.md)

EGM 独立审阅核对了论文、档案说明和 PNG 元数据。实际预览为 452 × 260，文件未嵌入 ICC 或 EXIF。档案无法逐张证实曝光、白平衡或像素对应。原始结果中的 `registered` 是旧评分接口供遍历使用的标记，本轮未执行独立配准确认。所有灯档来自一个采集来源，反向比较复用照片。即使某个方向命中，也只能进入开发预审，后续还需核查测区对应、采集记录和完整画面的效果。

## 实际出图和结果

| 比较范围 | 观察结果 |
| --- | --- |
| 58 张实际暖色开关 | 55 张通过原素材门槛，编辑区的红减蓝变化为 +4 至 +24 RGB8。同一布景剩余的 3 档在登记测区的暖色中位响应均为 0 |
| 282 个有序换灯方向 | 原方向与幅度联合门槛通过 0 个 |
| 114 个只改一侧灯色代码的方向 | 86 个通过双档素材、共同 RGB 支持和信号前置线。14 个幅度过线，方向过线 0 个 |
| 114 个方向用 RGB 与作者掩膜交集重算 | 最小共同覆盖 99.637%，联合通过仍为 0 个 |

信号前置线取三通道绝对变化的均值，最低 3 RGB8。最接近方向的是狮子道具的白蓝灯到红蓝灯，实拍和编辑的通道变化分别为 `[0, -6, -10]`、`[8, -2, -10]`。幅度比 1.111，在 0.75–1.25 区间内。方向余弦 0.741，低于 0.90。另一张适合展示的编辑正例是 `bmug` 的白蓝灯档。杯底的暖色变化为 +13 RGB8，亮度变化约 −0.31 RGB8，保护区 P95 为 0，未增加端点。[逐布景数值与版本哈希](2026-11-24-fau-shared-warmth-evidence.json)

55 张暖色开关可用于检查后续候选是否误抹有意暖色。现有 FAU 灯档未补足第二个暖色匹配来源，此档案取样结束。本轮使用外部编辑器验证素材，映色普通跨场景候选的画质仍待验证。

## 后续投入的条件

现有 15 号同原片候选可以演示从同一原片的调色版本重放局部编辑。普通跨场景用法还需要证实有意暖色和拍摄用光能分开处理。继续筛选相近灯色前，优先查找有记录的两档原片，或通过锁定曝光、白平衡和机位的方式获得小样。小样用于制作两档真实光照各自的编辑开关，并与已有 MIT 对照核验。既有预算为三组小样拍摄约 60–90 分钟、编辑导出约 1–2 小时、归组核验约半天，现场排期另计。取得合格小样后，先在现有工作台比较实际输出，再决定是否使用 GPU。问卷用于经过开发筛查的两份软件成片，比较肤色自然度与参考风格保留。[课程体验入口](2026-11-18-course-handoff.md) · [已验证的同原片候选](https://github.com/GoalActionYes/YingSe/blob/05da3b1112b8a33dbe165d25373758786b4ac55c/docs/research/2026-11-24-same-capture-resize.md) · [小样采集与预算](2026-11-21-capture-edit-feasibility.md) · [训练判定](2026-11-24-spatial-condition-design.md)

## 复现

在项目根目录执行，沿用既有 FAU 档案、RawTherapee 5.13、作者侧车模板及 Python 环境。以下将未占用的 V 盘临时指向项目根目录，供 RawTherapee 读取 ASCII 路径。每次使用尚不存在的输出目录。

~~~powershell
if (Test-Path 'V:\') { throw 'V: already in use' }
subst.exe V: (Get-Location).Path
try {
    $env:PYTHONPATH = 'V:/;V:/.tools/tf-convert-env/Lib/site-packages'
    $root = 'V:/outputs/adaptive-reference-20261113'
    $cli = "$root/rawtherapee-sky-pair-v1/runtime/RawTherapee_5.13_win64_x86_64_release/rawtherapee-cli.exe"
    & V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/probe_fau_shared_warmth.py $root $cli "$root/fau-shared-warmth-reproduce"
    if ($LASTEXITCODE -ne 0) { throw "FAU probe returned $LASTEXITCODE" }
} finally {
    subst.exe V: /D
}
~~~

两遍的 184 个 PNG、20 份 PP3 和 1 份结果 JSON 全部逐文件哈希一致。另从保存的 PNG 重算 282 个方向的三帧测区中位数，核对源图、掩膜哈希和开启、关闭态的像素与 ICC。掩膜重算在原 RGB 有效条件上叠加两张作者掩膜共同有效。原始压缩包、完整结果、对照图和编辑器运行环境保留本机。
