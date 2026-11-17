# 铁塔天空的规则方格来自颜色表切换

上一轮在相邻的 32×32 区域交替使用两张参考颜色表，连续回退后仍有方格。本轮将原有两张表分别应用到整个目标。两张表各自使用完全相同的原训练像素，参数、参考前后图、输出尺寸及 80% 强度均沿用。按旧分组把本轮的两张成片拼回去，在人像和铁塔上分别逐像素重现旧成片的 760,320 个像素。见 [事前协议](../../experimental/adaptive-reference-20261113/psbench-action-rgb-global-map-protocol.md)及[前一轮](2026-11-18-psbench-action-rgb-continuity.md)。

| 目标 | 整图误差，13 号／交替表／固定表 A／固定表 B | 近中性色误差，13 号／交替表／固定表 A／固定表 B |
| --- | ---: | ---: |
| 台阶人像 | 15.46／6.72／6.78／6.66 | 16.77／4.08／4.08／4.08 |
| 铁塔 | 15.03／5.32／5.01／5.63 | 15.91／6.28／6.22／6.33 |

误差为相对作者成片的 RGB8 码值。两个固定表在两个方向均通过预设误差和响应门槛。以旧插值的同一支持边界评分，铁塔第 95 百分位跳变从 9 降到 3；进一步只看同时跨越旧 32×32 分组的边，铁塔从 11 降到 3，人像从 8 降到 2，两张固定表均重现。边界用上轮的原蒙版与像素约束，分组按 `(y // 32 + x // 32) % 2` 计算。见 [数值和哈希](2026-11-18-psbench-action-rgb-global-map-evidence.json)。

查看两张固定表的实图，铁塔原有的规则方格均不再明显，天空仍比作者结果更蓝。人像的白台阶青偏改善保留，衣服的紫色仍未接近作者成片。逐像素拼接复现和两个固定表的表现支持将规则方格归因于实验分块切表。这不排除低支持颜色仍有连续但不够准确的变化。

两次运行的四张 PNG 和结果 JSON 哈希一致。独立从输出复算 80 项指标，核查旧边界与拼接复现。五列对照在本机 `outputs/adaptive-reference-20261113/psbench-shared-local-v1/action-global-map-v1/comparison.jpg`，不随 Git 上传。

额外参考原图提供了本例可迁移的颜色变化。固定表输出保留为离线渲染对照，这一组样张仍是开发素材。自动两图上线所缺的是从已调色参考和目标推断可靠的颜色变化，尤其是白墙等中性色与主体颜色承担不同作用时的判断。此前固定底图估计与原图分布辅助任务已经失败，保留原结论；下一项候选需要明确新的可观测信号再进入正常两图测试。见 [底图估计](2026-11-18-inretouch-inferred-original.md)及[原图分布辅助训练](2026-11-17-reference-normalization-training.md)。

复现时在仓库根目录设置 `$env:PYTHONPATH = (Get-Location).Path`，使用研究 Python 运行 `experimental/adaptive-reference-20261113/probe_psbench_action_rgb_global_map.py <素材目录> <新输出目录> <原分图证据 JSON> <硬切换证据 JSON> <连续过渡证据 JSON>`。素材目录与此前相同，三份证据位于 `docs/research`，分别为 `2026-11-18-psbench-action-pair-evidence.json`、`2026-11-18-psbench-action-rgb-interpolation-evidence.json` 与 `2026-11-18-psbench-action-rgb-continuity-evidence.json`。
