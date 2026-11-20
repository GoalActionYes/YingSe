# 只迁移局部色差未能改善肤色和中性色串色

不再先猜参考使用的预设。这轮直接从单张参考和普通 13 号结果各自计算 16×16 块内的颜色变化，去掉块平均色度，再用两套协方差把成片的局部色差向参考拉近。调节只施加在每块的中心化偏差上，量化和限色域前保留 13 号的块平均对数色比与逐像素线性亮度。这样能验证局部颜色关系是否提供可用风格，同时减少直接复制参考整图灯色的机会。原 13 号留作对照，人工区域和目标预期仅用于评分。[事前协议](../../experimental/adaptive-reference-20261113/local-chroma-contrast-transfer-protocol.md)

| 已用开发对照 | 原风格项目通过数 | 局部变换后通过数 | 完整联合通过 |
| --- | ---: | ---: | ---: |
| Gros 真人 Film B，六方向、双灯 | 4／60 | 3／60 | 两臂均 0／6 方向 |
| MIT／Gros／Beyond RGB 匹配染色，六目标状态 | 7／33 | 7／33 | 两臂均 0／4 构图 |

Film B 的项目为区域和参考灯光的组合。以 02 为参考、01 为目标，冷灯下原已通过的右脸颊与暖灯下的左白幕仍保留，但冷灯下左脸颊误差由 2.167 升为 4.077 RGB8。此处旧左脸颊未过风格投影，误差扩大构成额外反例。匹配染色的整图换灯差在六个状态中五个增加，唯一降低的人像为 13.283 到 13.280 RGB8。BLUE_red 灯 1 从 10.743 增至 10.968 RGB8。局部变换没有解决现有的大区域偏色。[结果摘要、成片与报告哈希](2026-11-21-local-chroma-contrast-transfer.json)

六个同图控制均保持输入，六个纯曝光控制附加色度 MAE 最高 0.220 RGB8，低于原约定上限 1。另检查块边界相对相邻非边界的 RGB8 差变化，最大绝对差 0.975。此项测量只看边界统计，不能证明没有可见接缝。用合成数据独立核对块均值为零、均匀色度平移在块内抵消及协方差变换公式。

## 决策与复现

结束固定局部色差对应。该步骤可以独立于预设识别运行，但保留下来的正是大区域已经偏离的平均颜色，局部调整也会损伤部分肤色。后续要检验能直接调整大区域平均颜色、且有可核查材质或多变体证据的候选。两个开发集、独立画质、人工自然度与视频的原门槛继续沿用。

准备 [Gros Film B](2026-11-21-gros-film-b-cross.md) 与[曝光控制和染色对照](2026-11-21-two-image-chroma-quantile-gain.md)的归档，从仓库根目录选择空的新输出路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$film = (Get-FileHash "$base/gros-film-b-cross-v2/results.json").Hash.ToLowerInvariant()
$exposure = (Get-FileHash "$base/two-image-chroma-quantile-gain-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_local_chroma_contrast_transfer.py "$base/local-chroma-contrast-transfer-repro" $film $exposure
~~~

两次各生成 48 张成片（含曝光控制）、12 张对照和 1 份报告，61 个文件逐一复核 SHA-256 一致。成片和完整逐区结果留在本机 outputs/adaptive-reference-20261113/local-chroma-contrast-transfer-v1 与同级复跑目录。
