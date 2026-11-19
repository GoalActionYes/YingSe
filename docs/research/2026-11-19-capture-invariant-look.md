# 实拍变化约束减少了误调，仍不能可靠认出两种已知风格

把 24 张已核验的相机、曝光、换灯和跨场景照片制成关闭、Film A、Film B 三态，共 72 个参考。每次留出 DAST、MLS、Middlebury、ARRI 路径中的一个完整来源。两臂读同样的三个通道全图统计、使用同样的拟合样本和 0.65 回退规则。候选只增加同一内容、同数字风格、不同实际拍摄条件的成对分数差约束。预测仅输入一张已调色参考；渲染依据预测，用官方变换处理目标原图，再以固定 80% 混合。作者目标效果用于离线评分。[事前协议](../../experimental/adaptive-reference-20261113/capture-invariant-look-protocol.md)

| 整组留出来源 | 正确状态，对照 → 候选 | 关闭态保留原图，对照 → 候选 | 风格联合通过，对照 → 候选 |
| --- | ---: | ---: | ---: |
| DAST 实拍曝光 | 15/45 → 16/45 | 4/15 → 15/15 | 0/15 → 0/15 |
| MLS 实拍换灯 | 4/12 → 4/12 | 4/4 → 4/4 | 0/4 → 0/4 |
| Middlebury 相机色卡 | 2/6 → 2/6 | 0/2 → 0/2 | 0/2 → 0/2 |
| ARRI 路径的实拍 | 3/9 → 3/9 | 3/3 → 3/3 | 0/3 → 0/3 |

差分约束将关风格输入的正确状态从 11/24 提高到 22/24，但 Film A 只认对 1/24，Film B 仅 2/24。两台 Middlebury 参考的六种状态都被选作 Film B，同款换相机差为零，同时两张原片都被误调。DAST 在 15 张关风格参考上保持原图，仍有同一风格换曝光的近中性色差 6.28 RGB8。四个来源的联合门槛均未通过。较少误调或较低换图差仍可能伴随风格漏判。[Middlebury 原片误调](../../outputs/adaptive-reference-20261113/capture-invariant-look-v1/paired/middlebury/chart-canon-off.png) · [DAST Film B 对照](../../outputs/adaptive-reference-20261113/capture-invariant-look-v1/paired/dast/1-nominal-film_b.png)

停止这套固定全图统计和线性读出。它只测试已知三态，两种胶片风格和有限物理扰动不足以覆盖开放风格。整来源留出可避免同组变体进入拟合，但这些来源已用于大量开发，不能充当最终独立验收。真实肤色、白墙、自然评价、视频及软件候选仍待验证。

两遍的 144 张成片和全量报告逐件匹配。另从落盘图复核 144 个文件身份、决定和两类风格变更的整图／近中性色评分。复现设置已有研究环境的仓库根目录、OpenCV 与 OpenColorIO，执行 `probe_capture_invariant_look.py <本机研究输出根目录> <新输出目录>`。逐图得分和所有图留本机，Git 仅保存缩减数值和全量报告哈希。[脚本](../../experimental/adaptive-reference-20261113/probe_capture_invariant_look.py) · [可复核摘要](2026-11-19-capture-invariant-look-evidence.json)
