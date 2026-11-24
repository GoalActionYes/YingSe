# 参考位置减少灯光误传，但未恢复有意局部调色

用真实拍摄换光和有意提亮配对重新训练两个同规模参考条件头。固定 13 号其余权重、分析和空间网格，只改变参考输入中是否保留 4×4 图块的位置。MIT 训练时只用其两个房间，测试使用 WildRelight 的两个场景。反向折只用 WildRelight 训练，评估 MIT 两个房间。每一折的目标与参考两侧都排除留出来源。这样能比较跨拍摄来源的泛化，主要成片仍是同一留出来源内的跨场景迁移。另用两个从未拟合的 Gros 人像互作有意暖色保护对照。[事前协议](../../experimental/adaptive-reference-20261113/positional-lightness-trial-protocol.md) · [EGM 设计](2026-11-24-spatial-condition-design.md)

两组在各折使用完全相同的场景、四张参考、目标翻转、步数和损失。每组 65,664 个可训练参数，每折各 128 步。初始抽检网格与关闭实验头的 13 号逐值一致。训练后，参考块换序对局部臂网格最大差为 0.00079 和 0.00157，池化臂逐值保持，验证了位置通路可影响网格。

## 原尺寸成片

| 留出来源、指标 | 13 号 | 局部位置 | 池化 |
| --- | ---: | ---: | ---: |
| WildRelight 完整方向通过 | 0／6 | 0／6 | 0／6 |
| WildRelight 编辑区最大换光差，RGB8 | 22.57 | 9.31 | 6.87 |
| WildRelight 最大提亮投影 | 0.310 | 0.196 | 0.168 |
| MIT 完整方向通过 | 0／12 | 0／12 | 0／12 |
| MIT 编辑区最大换光差，RGB8 | 46.42 | 14.15 | 16.66 |
| MIT 最大提亮投影 | 0.151 | 0.104 | 0.114 |

完整方向的提亮投影下限为 0.75，同时要求换光差、开关误差和保护区符合原线。局部与池化都降低了最坏灯光误传，部分编辑区域更接近目标底色。局部版在两组的最大提亮响应都低于冻结 13。WildRelight 上池化的换光差还低于局部版。两个脸颊暖色方向上，三个比较对象各折均未达标。用 WildRelight 拟合的局部版在 Gros 2 号脸颊的有意暖色投影为 0.111，冻结 13 为 0.214，出现有用色彩响应下降。全部同输入保持逐像素通过，来自明确的同图路径。[逐方向和人像结果](2026-11-24-positional-lightness-evidence.json) · [MIT 局部版原尺寸成片](../../outputs/adaptive-reference-20261113/positional-lightness-eval-v1/mit/local/everett_living2/comparison.png) · [Gros 有意暖色对照](../../outputs/adaptive-reference-20261113/positional-lightness-eval-v1/mit/local/gros/2/comparison.png)

本轮证明有限的跨亮度配对监督可以压低一些错误响应，仍缺能在参考图里保留有意编辑的信号。局部条件没有超过池化和现有 13 的联合门槛，这份权重保留为失败候选。在提出另一轮之前，先判定参考特征在换光和有意编辑的开关之间是否有可供跨场景识别的分量。定位原因前不延长同一训练或更换随机种子。真实暖色的第二个匹配来源、完整旧开发组，以及独立图片、真人和视频仍按原验收约定。[真实暖色材料边界](2026-11-24-mls-local-warmth.md)

## 复现

使用冻结的两批开关成片、已有研究运行时和 13 号权重。预检两步后执行正式试验，四臂共 512 个固定更新，GPU 优化累计约 23 秒；最大分配显存约 101 MB。两遍原尺寸出图约 123.2 秒和 65.8 秒，含本实验读图、渲染与存盘。两遍评分和输出的哈希一致，报告 SHA-256 为 `39d8a947d784034040c3e3b3e94c7b3854cef4674780c4ab4c3bdbc212a242b1`。独立核对 4 份权重、354 张 PNG 和 48 张图板，重算 516 项区域读数。MIT 留出的冻结 13 号另有 50 次逐图核对，与此前实际工作台导出逐像素一致。此次研究头未接入工作台，生产功能保持原入口。[训练脚本](../../experimental/adaptive-reference-20261113/train_positional_lightness.py) · [评分脚本](../../experimental/adaptive-reference-20261113/evaluate_positional_lightness.py)

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$root = 'outputs/adaptive-reference-20261113'
$python = './.tools/adaptive-env/Scripts/python.exe'
$code = 'experimental/adaptive-reference-20261113'
& $python "$code/train_positional_lightness.py" $root "$root/positional-lightness-training-new"
& $python "$code/evaluate_positional_lightness.py" $root "$root/positional-lightness-training-new" "$root/positional-lightness-eval-new"
~~~
