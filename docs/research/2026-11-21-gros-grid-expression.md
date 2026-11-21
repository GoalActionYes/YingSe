# 13 号现有渲染器能表达这组 Film B 效果

上一轮平移调色网格后，脸颊串色加重。位置有用，但还需确认：现有渲染器如果知道该调成什么颜色，能否同时画对脸颊和白幕？这个问题决定应优先改渲染器，还是查输入参考时为什么没有得到合适的颜色变换。[原网格位置对照](2026-11-21-gros-shift-rendered-grid.md) · [事前协议](../../experimental/adaptive-reference-20261113/gros-grid-expression-protocol.md)

用已用于开发的三张冷灯人像。每张图从原片和 Film B 预期成片拟合渲染系数，屏蔽 16 像素棋盘块中的一半用于评分。两组沿用 13 号的颜色引导、渲染、量化和 80% 强度。取相同的 XY 网格系数，一组在所有像素共用一个颜色矩阵，另一组沿原有八层颜色引导插值。拟合时已给出目标预期成片，这是表达能力诊断。

| 目标 | 整图共用矩阵 | 保留八层颜色引导 |
| --- | ---: | ---: |
| 01 | 3／5 | 5／5 |
| 02 | 4／5 | 5／5 |
| 03 | 5／5 | 5／5 |
| 合计 | 12／15 | 15／15 |

两组用同一门槛，响应误差不超过 3 RGB8，风格投影在 0.75 至 1.25。增加深度后共用矩阵已通过的区域全部保留，15 个区域的风格误差均下降。共用矩阵漏掉的三处都是脸颊。例如 02 右颊的误差由 3.10 降到 0.90 RGB8，投影由 1.80 回到 0.98，说明原来的过调幅度收敛到预期附近。03 右颊投影从 1.09 降到 0.78，仍通过，但接近下限。原 13 号普通两图对照在相同留块上重算，六个跨人物方向均无区域同时通过换灯保护与风格传递。[全部留块分数和哈希](2026-11-21-gros-grid-expression-evidence.json) · [02 本机成片对照](../../outputs/adaptive-reference-20261113/gros-grid-expression-v2/02-comparison.png)

**路线决定。** 原有渲染能够表达这一项已知全局预设产生的局部效果差别。优先继续验证如何从普通两图估计颜色变化，把这组结果用作参考推断的对照。保持主体与背景协调的产品目标。识别出哪里是脸和白幕只解决了作用位置，还要判断各色域应该往哪里调、调多少。当前诊断无法证明模型能自动识别调色、拒绝实拍暖灯，或表达真正按位置编辑的预设。棋盘留块仍属于同一原图，独立画质、人工自然度和视频仍待验证。

## 复现

沿用 [Gros Film B 开关归档](2026-11-21-gros-film-b-cross.md)及[上一轮研究运行环境](2026-11-21-gros-shift-rendered-grid.md)，仓库根目录下选择空的新输出目录运行。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_grid_expression.py outputs/adaptive-reference-20261113/gros-grid-expression-repro
~~~

首轮写报告遇到 NumPy 矩阵秩的 JSON 类型错误，改为存储标准 Python 整数后在两个新目录重跑。两遍六张成片、三张对照和一份报告，十个文件逐一哈希一致。诊断设计矩阵与原渲染器的浮点结果最大差为 4.77×10⁻⁷。另读取 PNG，独立复算全部 30 项留块指标和退化限制。完整报告 SHA-256 为 `3b1fa2c71aeaa1f2deac9f9072de78f8a9a735b861b9970c38690086850f68c9`。
