# 两折暖色试跑的素材核验

两版软件在 MIT 复印室和 Beyond RGB 白蓝场两种参考下，均有浅墙漏迁移和人像偏色。要验证下一种算法能否利用真实换灯和有意调色的差别，拟合和留出各需一个匹配参考场景，以及同一来源里另一场景的目标开关。MIT 客厅已有合格目标。Beyond RGB 则补用缓存的白绿场 1 档。它的同款暖色开关通过原有素材线。两个来源现在各有四张参考和两张目标，可做两个角色都按来源隔离的开发试跑。[两版软件成片结果](2026-11-24-beyond-rgb-warmth-workbench.md) · [事前素材协议](../../experimental/adaptive-reference-20261113/warmth-source-holdout-intake-protocol.md)

## 白绿场实际出片

旧相机审计的 HDF5、标签和色卡哈希均匹配，并与白蓝场共用同一 D65 矩阵。固定 1/4 取样和 sRGB 编码后，白绿场整图约 14.85% 的像素至少一个通道在显影裁切范围外，冻结的下侧白背景区域无裁切。实际用 RawTherapee 5.13 的既有设置生成一对 PNG。编辑区的中位通道改变量为 [6, −3, −17]，红减蓝变化 24 RGB8，亮度变化 −2.10 RGB8。开关测试的有效像素为 100%，关态逐像素一致，保护区及范围外的误差 P95 为 0，全图新增端点为 0。全部数值过原素材线。这个目标只证明固定拍摄下的编辑被正确打开。真实换灯由白蓝场已核验的四张参考提供。[原相机审计](2026-11-20-beyond-rgb-batch-intake.md) · [目标开关图板](../../outputs/adaptive-reference-20261113/warmth-source-holdout-v1/target-first/render/comparison.png) · [逐文件身份和数值](2026-11-24-warmth-source-holdout-evidence.json)

| 家族 | 匹配参考 | 目标开关 |
| --- | --- | --- |
| MIT | 复印室，灯 8／0 的关态和开态 | 客厅，灯 0 关态和开态 |
| Beyond RGB | 白蓝场，灯 7／1 的关态和开态 | 白绿场，灯 1 关态和开态 |

两折各用一行拟合、另一行留出，互换一次。12 张已登记成片的像素哈希互不相同，逐张核对像素、ICC 和完整 PNG 哈希。同一场原片和全部衍生图按拍摄家族管理。EGM 复核强调，白蓝场和白绿场共享布景条件和部分道具，两个家族也共用项目定义的同款 RawTherapee 编辑。两折只回答在已限定角色的四图对照里，模型能否少传参考灯光并保留有意暖色。目标换灯、反向迁移、真实人像有意暖色和独立画质仍有对应验证要求。

## 复现

用既有环境和缓存，不需要安装或下载。与[暖白灯预检](2026-11-24-warm-capture-preflight.md)相同，先把仓库映射到空闲 ASCII 盘符。以 V 盘为例，输出使用新目录。

~~~powershell
$env:PYTHONPATH = 'V:/;V:/.tools/tf-convert-env/Lib/site-packages'
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'V:/outputs/adaptive-reference-20261113'
$run = "$base/warmth-source-holdout-replay"
New-Item -ItemType Directory $run | Out-Null
$probe = 'V:/experimental/adaptive-reference-20261113/prepare_warmth_holdout_target.py'
& V:/.tools/adaptive-env/Scripts/python.exe $probe display $base "$run/display"
if ($LASTEXITCODE -ne 0) { throw 'Display failed' }
& V:/.tools/adaptive-env/Scripts/python.exe $probe grade $base "$run/target" --source "$run/display" --edit 578 695 678 747 --guard 73 188 121 261
if ($LASTEXITCODE -ne 0) { throw 'Target grade failed' }
~~~

两遍显影的 3 个文件及两遍开关的 7 个成片、配置和报告文件逐字节一致。另从落盘 PNG 独立复算红减蓝变化、亮度、有效覆盖、保护区和新增端点。共用预检的 5 项测试通过。第一遍调用曾选了缺 Torch 的显影环境，随后改用项目现有的自适应研究环境并重跑了两遍。运行环境与照片均保留本机。

本轮未训练。预检时 RTX 4060 Laptop 有约 6 GiB 可用显存。下一阶段先锁定旧 13 号权重及数据来源，复用颜色输出、位置臂与同容量池化臂，同时验证训练顺序、家族留出、初始结果与 13 号一致。短预检预算 15–20 分钟，首轮两臂和完整成片的既定上限仍为 4 小时。两折只有各一个匹配参考方向，适合一次有停止条件的证伪。早先的局部提亮结果仍为失败，暖色出片须独立过原门槛后才进入后续开发回归。[既有设计与停止条件](2026-11-24-spatial-condition-design.md)
