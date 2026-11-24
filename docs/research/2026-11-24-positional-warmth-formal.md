# 暖色位置条件未恢复有意加暖

两折各用一个完整拍摄家族拟合，另一个家族同时留出参考和目标。每折比较 13 号、带坐标的局部块、同规模的池化参考。两种拟合均降低了部分底色偏差，但所有比较对象均为 0／2 个完整留出方向通过。局部臂还削弱了两位有意暖肤示例的响应。按事前停止条件保留反例和本机权重，本次位置条件路线结束。[素材和留出身份](2026-11-24-warmth-source-holdout.md) · [冻结评分和预算](../../experimental/adaptive-reference-20261113/positional-warmth-trial-protocol.md)

| 留出目标 | 比较对象 | 两档关态相对目标原图误差，RGB8 | 关／开各自跨参考灯档差，RGB8 | 两档有意暖色响应 |
| --- | --- | ---: | ---: | ---: |
| Beyond RGB 白绿背景 | 13 号 | 21.43／19.17 | 3.42／3.45 | 0.39%／0.79% |
| Beyond RGB 白绿背景 | 池化 | 10.64／7.82 | 2.98／2.99 | 0.14%／0.41% |
| Beyond RGB 白绿背景 | 局部位置 | 13.37／10.13 | 3.57／3.57 | 0.29%／0.50% |
| MIT 客厅浅墙 | 13 号 | 12.83／15.68 | 4.39／4.44 | 6.66%／4.48% |
| MIT 客厅浅墙 | 池化 | 5.62／10.09 | 5.04／4.97 | 5.96%／6.53% |
| MIT 客厅浅墙 | 局部位置 | 4.52／8.87 | 4.41／4.76 | 2.30%／6.36% |

区域误差每项上限为 3 RGB8，两档暖色响应各须达到目标的 75%–125%。Beyond RGB 上池化臂把两项跨灯差压到 3 以内，底色和编辑响应仍不合格。MIT 目标上，局部臂把关态误差从 12.83 和 15.68 降到 4.52 和 8.87，但仍漏掉绝大部分应迁移的暖色。两折的全部保护区、开态与响应误差另存数值。[可复核数值](2026-11-24-positional-warmth-formal-evidence.json) · [MIT 位置臂成片](../../outputs/adaptive-reference-20261113/positional-warmth-formal-v1/beyond_rgb/local/heldout/comparison.png) · [Beyond RGB 池化臂成片](../../outputs/adaptive-reference-20261113/positional-warmth-formal-v1/mit/pooled/heldout/comparison.png)

在有意给右颊加暖的独立开发对照中，第 2 位人物的 13 号响应为 21.35%，两折局部臂只有 18.63% 和 13.59%。第 4 位从 6.08% 降为 5.99% 和 5.07%。另一个人像保护例中，局部臂降低了参考成片相对原图的基准偏差，但第 2 位的右颊在换参考灯档时从相差 2.87 增至 3.51 RGB8，超过 3 的原上限。两种测量分开保留。[有意暖肤反例](../../outputs/adaptive-reference-20261113/positional-warmth-formal-v1/beyond_rgb/local/authored_faces/gros/2/comparison.png)

## 完整运行与复核

正式运行从固定 13 号和新初始化参考头开始，每臂 128 步，累计 512 步。实际训练优化约 13.24 秒，训练、原尺寸渲染、存盘与评分合计 60.72 秒。以相同权重只重跑出图和评分耗时 46.66 秒。两次各有 162 张渲染结果、24 份人像原图及期望图、42 张图板和 1 份结果 JSON，共 229 个文件逐字节一致。两遍均从磁盘成片复算，核对 42 条留出测区和 24 条有意暖肤测区。另核对 32 张 13 号外部结果与之前两版工作台的 13 号成片像素相同。同图 42／42 检查逐像素通过。

EGM 独立复核确认，四态测量、外部墙面及双向暖肤评分按照冻结协议运行，拟合时两角色均处于同一来源。审阅还指出两处报告限制。脚本的合并集合使用拟合家族标识，每折的实际留出家族列在 `heldout_family`。本次三个通过集合均为空，不影响判定。正式入口没有自动绑定预检回执，四小时是脚本检查上限。此次通过独立的预检回执和实际执行计时核验正式启动与完整落盘。回放只核查已经完整的首次运行，不作为第二次拟合或独立画质证据。

## 复现与后续

在仓库根目录沿用既有研究环境。首次训练和评分输出使用新目录。复核时复用本机留存的首轮权重，新目录保存重出的图。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$probe = 'experimental/adaptive-reference-20261113/evaluate_positional_warmth_trial.py'
& '.tools/adaptive-env/Scripts/python.exe' $probe $base "$base/positional-warmth-formal-new"
if ($LASTEXITCODE -ne 0) { throw 'Warm-color trial failed' }
& '.tools/adaptive-env/Scripts/python.exe' $probe $base "$base/positional-warmth-replay-new" --replay "$base/positional-warmth-formal-new"
~~~

训练结果和完整图像保留本机。短训练内有底色偏差改善，项目目标所需的参考暖色仍未恢复。前次提亮试验同样未过。停止追加此位置条件的参数、步数和种子。若要提出不同方案，需以新的可核查测量说明普通参考中哪项线索能跨拍摄来源区分现场光照与有意编辑。当前无通过开发门槛的候选，独立人评、视频验证和产品集成均待画质候选。
