# 两颊亮度比例仍无法区分局部提亮与真实换灯

13 号容易把现场灯色传给目标，同时漏掉有意的局部提亮。这一轮用同一张参考图的左右脸颊互作对照。若真实光只让两个测区乘上同一个亮度倍率，亮度比会消除该倍率。先在已登记的双颊区域观察真实照片中还有多少灯光干扰，决定是否值得再加入自动取样和训练。[事前协议](../../experimental/adaptive-reference-20261113/gros-cheek-luminance-ratio-protocol.md) · [上一轮的编辑开关及实际工作台成片](2026-11-24-gros-author-tone-switch.md)

输入为两位人物的低、中、高灯档及冷灯目标，每张分别关、开原参数的局部提亮。另从原研究读取两张没提亮的暖灯图。合计 8 张开启、10 张关闭。先把 RGB8 反解成线性 RGB，以 Rec.709 系数计算各区的亮度中位数。对照只读右颊中位数，另一项读右颊与左颊的亮度比。只有单张图和登记坐标参与测量，开关身份和其它图用于离线检查。

| 人物 | 冷灯、关闭 | 强灯、开启 | 暖灯、关闭 |
| --- | ---: | ---: | ---: |
| 2 | 1.87 倍 | 1.70 倍 | 1.73 倍 |
| 4 | 3.22 倍 | 2.03 倍 | 2.31 倍 |

表中为右颊相对左颊的亮度。两位的八组同图开关中，提亮后该比值都上升。换到另一灯档后，两人未修图的暖灯比值均高过真的修过的强灯参考。对第 4 位，这个差在没修的冷灯下进一步增大。测试中登记的肤色范围包含拍摄角度、灯光分布以及相机处理的共同影响，不能仅据此将差异归给某一项。[开关对照图](../../outputs/adaptive-reference-20261113/gros-author-tone-switch-v1/editor-comparison.png) · [完整逐图读数](2026-11-24-gros-cheek-luminance-ratio-evidence.json)

事前定为单调阈值。每次先在一人身上找使所有开态得分都高于关态的阈值，再测试另一人。两种测量在两个人身上都无法选出这样的阈值，因此两折均回到全关闭。10 张未编辑图保留，8 张有意提亮全部漏掉。即使把两人放在一起，最弱开态的亮度比也只有 1.70，最强关态是 3.22。固定阈值认出前者时必然误认后者。按协议停止固定脸颊亮度比路线，保留正负例来检验未来能同时保住局部风格和避免肤色或白墙误调的候选。

## 核验与复现

两次运行合计约 5 秒，均无训练、下载或新增依赖。报告逐字节一致，SHA-256 为 `0588e2f8926fef8108c36b7d5078ff95e1c8ee98c71234afcd848d9b09a4e07a`。另从 18 张 RGB8 PNG 重算 36 项脸颊亮度，并重新核对两臂的留一判定。生产代码及稳定版未改动，本轮未启动实际工作台的新推理。两位都来自既有开发采集，评估范围限定在此编辑、灯档和标量判断，独立画质与视频指标维持原要求。

沿用[上一轮的固定文件和研究环境](2026-11-24-gros-author-tone-switch.md#用途与复现)。用本轮脚本，指定同一研究根目录和一个全新的输出目录。照片和成片保留本机。[复现脚本](../../experimental/adaptive-reference-20261113/probe_gros_cheek_luminance_ratio.py)

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$base = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_gros_cheek_luminance_ratio.py $base "$base/gros-cheek-luminance-ratio-new"
~~~
