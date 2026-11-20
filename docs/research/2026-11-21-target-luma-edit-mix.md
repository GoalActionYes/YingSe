# 固定目标亮度混合未能兼顾整图与局部风格

上轮在客厅和人像上证明，给定正确的编辑方向并允许按答案为每个像素挑比例，两套现有渲染结果能合成出符合门槛的效果。本轮只改变混合比例的来源。由目标原图的 sRGB 亮度代理决定权重，越亮越偏向稳定版 1.3，越暗越偏向 13 号。选这个方向，是因为稳定版在旧对照的墙面和白幕上响应较足，13 号在客厅整图更接近预设风格。两者都使用原对照中 80% 强度、没有手动微调的成片。

使用已保存的 MIT 客厅、Gros 人像和 Beyond RGB 两个静物构图，每个静物构图包含两次实际拍摄状态，共六个目标。每个目标都比较稳定版编辑增量、13 号编辑增量、等量混合和固定目标亮度混合。固定亮度规则只根据目标原图算权重，真值及区域用于评分。编辑增量仍通过同一参考的编辑前后图取得，换灯和原参考由实验设置为保持原图。这一设置用于隔离目标侧的编辑保留问题。所有样本均为已用开发素材。[事前协议](../../experimental/adaptive-reference-20261113/target-luma-edit-mix-protocol.md)

## 实际成片

| 目标状态 | 稳定版通过区域 | 13 号通过区域 | 等量混合通过区域 | 固定亮度通过区域 |
| --- | ---: | ---: | ---: | ---: |
| MIT 客厅 | 2／4 | 1／4 | 3／4 | 3／4 |
| Gros 人像 | 5／5 | 2／5 | 2／5 | 2／5 |
| WHITE_green 灯 1 | 0／6 | 0／6 | 0／6 | 0／6 |
| WHITE_green 灯 8 | 1／6 | 0／6 | 5／6 | 5／6 |
| BLUE_red 灯 1 | 0／6 | 1／6 | 1／6 | 1／6 |
| BLUE_red 灯 8 | 0／6 | 3／6 | 1／6 | 2／6 |

同一静物构图的两次拍摄状态共同计组。固定亮度混合完整通过 0／4 组，稳定版 1／4，13 号和等量混合均为 0／4。各区域仍沿用平均绝对误差不超过 3 RGB8、风格响应投影 0.75 至 1.25 的门槛。

客厅灰球误差为 0.781 RGB8，两套端点分别为 2.644 和 2.803，但左墙的风格响应为 0.744，低于 0.75，稳定版原有的左墙效果丢失。Gros 人像的两侧白幕误差由稳定版的 2.556、2.574 增至 3.421、3.538 RGB8。静物 WHITE_green 灯 1 的六个区域均失败，BLUE_red 灯 8 则损失了 13 号原本通过的两个中性色卡区域。按协议计入所有七处退化，其中六处丢失原有通过区域，一处是仍过线但误差增加超过 1 RGB8。双零增量的身份保持检查在六张图上均通过。[区域分数及文件哈希](2026-11-21-target-luma-edit-mix.json)

## 决策与复现

结束这条固定亮度权重。墙面、白幕和灰球在同一条亮度规则下存在冲突。静物内容还有一项待区分的问题：两套现有增量的混合范围本身是否足以达到预设风格。先检查范围可以避免继续拟合一个即使比例选对也无法过线的规则。项目仍需真实两图画质、独立人工自然度和视频验证。

沿用 [MIT 对照](2026-11-21-mid-matched-wall-cast.md) 与 [Beyond RGB 对照](2026-11-20-beyond-rgb-matched-cast.md) 的已保存文件及哈希。在仓库根目录运行，使用一个新的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_target_luma_edit_mix.py outputs/adaptive-reference-20261113/target-luma-edit-mix-repro
~~~

两轮各生成 24 张成片、6 张对照图和 1 份报告，31 份文件逐一核对 SHA-256 一致。本机目录位于 outputs/adaptive-reference-20261113/target-luma-edit-mix-v1 和同级 target-luma-edit-mix-recheck。
