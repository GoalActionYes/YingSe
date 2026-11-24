# 原生空间网格能表达两处小范围暖色

位置条件训练即使回到已经见过的样本，也漏掉有意暖色。这轮只查目标端的小面积表达。以前证实了局部肤色和提亮，MIT 这次的测区宽 42 像素，占 1500 像素画面宽度的 2.8%，相当于约 0.45 格网格宽。Beyond RGB 的测区宽 100 像素，约为 1.56 格。两处均给定真实目标的暖色开态。在原画质门槛下，两个目标的四张原生单网格完整帧均通过，全局共享的四张对照均未通过。[暖色拟合内成片](2026-11-24-positional-warmth-fitting.md) · [事前方案](../../experimental/adaptive-reference-20261113/matched-wall-warmth-grid-capacity.md)

| 目标与表示，留块拼图 | 有意暖色投影 | 编辑区误差，RGB8 | 编辑范围外一格保护带误差，RGB8 | 原分区全部通过 |
| --- | ---: | ---: | ---: | --- |
| MIT 客厅，全局共享 | 10.89% | 5.32 | 0.090 | 否 |
| MIT 客厅，原生网格 | 87.31% | 1.14 | 0.010 | 是 |
| Beyond RGB 白绿场，全局共享 | −2.64% | 6.32 | 0.561 | 否 |
| Beyond RGB 白绿场，原生网格 | 101.08% | 0.40 | 0.080 | 是 |

编辑区投影合格线为 75%–125%，误差上限 3 RGB8。保护带是编辑作用范围之外的一格宽区域，平均误差单独报告，未加入原门槛。MIT 保护带有 94,000 像素，Beyond RGB 有 67,584 像素。误差均值不能用来宣称每个边缘像素都无变化。[逐区数值及文件哈希](2026-11-24-matched-wall-warmth-grid-evidence.json) · [MIT 目标图板](../../outputs/adaptive-reference-20261113/matched-wall-warmth-grid-full-first/mit-comparison.png) · [Beyond RGB 目标图板](../../outputs/adaptive-reference-20261113/matched-wall-warmth-grid-full-first/beyond_rgb-comparison.png)

EGM 复核指出拼图来自两张网格，必须另外核对单张网格能否同时通过局部和保护区分数。原脚本将同张目标分成 16 像素方块，两套网格分别用一半方块上的固定间隔样本拟合。图板上排的拼图，每个像素取未用自己所属方块拟合的结果。下排的单网格完整帧包含各自参与拟合的方块。四张空间单网格的开态投影，MIT 为 87.32% 和 87.45%，Beyond RGB 为 101.23% 和 101.01%。四张保护带平均误差最高为 0.083 RGB8。求解参数、素材和阈值沿用原协议。原评估的两档参考在该实验中共享同一目标答案，关态以原图、开态以同一拟合图代入复用的四态门槛。零换灯差来自这个构造，普通两图推断仍须另证。这里的全局对照限制的是同样深度分层中的矩阵全图共享，结论限于该表示与这两个目标。[原生网格验证方法](2026-11-22-gros-local-grid-capacity.md)

## 复现

沿用本机 SciPy 1.18.1 研究目录、同一 13 号权重和冻结 PNG。输出使用新目录。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages;outputs/adaptive-reference-20261113/scipy-capacity-20261122/wheel'
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
& '.tools/adaptive-env/Scripts/python.exe' 'experimental/adaptive-reference-20261113/probe_matched_wall_warmth_grid_capacity.py' $base "$base/matched-wall-warmth-grid-replay"
~~~

复核版两遍分别耗时 24.03 和 21.36 秒。每遍 18 张 PNG 和 1 份报告逐字节一致，独立复算 42 条区域记录。四张留块图从两张现有单网格完整帧重拼，均与首轮实测逐像素相同。随机网格与原生浮点渲染最大差为 1.79 × 10⁻⁷，实际拟合各臂最大差为 4.77 × 10⁻⁷。空间系数求解达到旧 240 步上限，因此仅报告当前可达见证。输入只覆盖两个已知局部编辑，效果和真值都留存于开发对照。没有训练或接入产品。下一步只有在普通参考中出现可核验的新线索，才值得用现有渲染去检验迁移是否改善。
