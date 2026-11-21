# 标准灯跨视角的 SIFT 候选覆盖不足

上一轮已经把参考中的冷灯影响压低，但 13 号面对标准灯参考仍只保留约一半 Film B 响应。这次利用两房间不同机位拍到的共享内容，先问固定 SIFT 能否找出足够的对应位置。如果两幅图中连候选点的分布上限都不足，就没有依据把这些点用于拟合整图的调色关系。[照明与迁移对照](2026-11-21-synclight-lamp-edit-order.md) · [事前协议](../../experimental/adaptive-reference-20261113/synclight-cross-view-coverage-protocol.md)

| 场景 | 参考风格 | 双向匹配 | 目标／参考覆盖上限 | 进入几何拟合 |
| --- | --- | ---: | ---: | --- |
| 走廊 | 关闭 | 15 | 4.54%／8.47% | 否 |
| 走廊 | 开启 | 11 | 1.67%／1.94% | 否 |
| 床头 | 关闭 | 6 | 4.97%／4.59% | 否 |
| 床头 | 开启 | 8 | 7.92%／12.05% | 否 |

沿用至少 12 个内点、内点率至少 50%、两图覆盖各至少 8% 的门槛。覆盖上限将所有双向匹配都视为正确，包含落到同一位置的重复点。任何只剔除匹配的几何筛选都无法扩大凸包。关风格的原片匹配数与此前报告的 15 和 6 一致。同机位开关 Film B 的 36 和 88 组匹配都通过，且两图覆盖均超过 40%。换房间分别只有 2 和 0 组匹配，两组均被拒绝。同机位控制只验证对已知重复内容能找到点，无法评估跨视角几何恢复的准确率。[分组结果与输入哈希](2026-11-21-synclight-cross-view-coverage-evidence.json)

结束从这批固定 SIFT 候选更换几何模型的路线。当前没有依据进入颜色拟合，也没有新成片。本轮只涉及两个已使用的房间，没有人像或独立验收。它回答的是匹配证据能否支持后续工作。即使主体与背景能分开，还需要识别同一材质或可信的局部颜色关系，才有依据判断该迁移什么。原先的跨场景肤色与中性色目标仍待解决。若后续考虑另一类特征，先用独立对应证据验证它确实找到了更多可信位置，再投入调色拟合。

## 复现

使用已有虚拟环境的 OpenCV 4.11.0。此前单应诊断记录为 OpenCV 5.0.0，本轮的上限和条件几何检查来自同一次 4.11.0 运行。固定读入原目标与参考，逐个核对 SHA-256。两次运行各少于一秒，报告逐字节一致。从仓库根目录运行，最后一个参数指向新的文件。

~~~powershell
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_synclight_cross_view_coverage.py outputs/adaptive-reference-20261113/synclight-cross-view-coverage-repro.json
~~~
