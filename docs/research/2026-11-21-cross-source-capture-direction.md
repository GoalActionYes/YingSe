# 实拍换灯的常见偏移方向未能跨来源迁移

此前的单张参考判别规则把数字调色也当成实拍偏色。本轮跳过类别判断，直接测量 13 号使用实拍换灯参考后在目标成片中形成的偏移。在线性 RGB 中，用换灯输出减去原参考输出，为每个目标状态计算变化向量的二阶矩。将各状态归一化后等权拟合一条主方向，再在完整留出的另一来源上测试。MIT 参考对应两张目标，Beyond RGB 的两个参考对应四个目标状态。预测时只读取目标原图和普通 13 号成片，不读取成对参考、标签或目标正确成片。[事前协议](../../experimental/adaptive-reference-20261113/cross-source-capture-direction-protocol.md)

定向修正只移除成片相对目标原图在该方向上的分量。等量减弱对照把整个编辑量按一个比例缩小，令裁切前线性 RGB 改变量的均方根与定向修正相同。13 号原输出也保留。三条路线沿用 80% 强度、相同区域与原保护及风格门槛。[完整分数、方向与输出哈希](2026-11-21-cross-source-capture-direction.json)

## 成片结果

| 目标状态 | 13 号整图换灯差 | 定向修正 | 等量减弱 |
| --- | ---: | ---: | ---: |
| MIT 客厅 | 11.24 | 10.04 | 9.92 |
| Gros 人像 | 13.28 | 12.17 | 10.61 |
| WHITE_green 灯 1 | 10.80 | 10.70 | 8.73 |
| WHITE_green 灯 8 | 15.38 | 14.78 | 13.80 |
| BLUE_red 灯 1 | 10.74 | 10.48 | 8.94 |
| BLUE_red 灯 8 | 11.17 | 11.28 | 7.71 |

以上为 RGB8 MAE，衡量实拍换灯参考相对原参考引起的整图变化。两个来源的第一主方向各解释 98.96% 和 98.02% 的训练变化能量，却相差约 53°。在全部六个留出目标上，定向修正的整图换灯差均高于等量减弱。数字风格单项通过数为原 13 号 7／33、定向修正 0／33、等量减弱 4／33。三种结果都没有完整通过任一构图。六张同图控制逐像素保持，六次纯曝光参考中有四次定向修正比原成片多改了超过 1 RGB8 的明暗。

## 决策与复现

结束跨来源共用一条偏移方向的修复。来源内的高解释比例不足以支持跨场景使用。现有样本没有证实应加强整体保护或继续搜同一方向的力度。后续需要可验证的局部照明依据，或足以让跨来源模型体现有意编辑的真实配对。保留这轮成片与控制，软件未接入，独立画质、人工自然度和视频仍待完成。

使用[原始换灯对照](2026-11-21-mid-matched-wall-cast.md)、[Beyond RGB 对照](2026-11-20-beyond-rgb-matched-cast.md)和[前轮曝光控制](2026-11-21-two-image-chroma-quantile-gain.md)。从仓库根目录运行，选择新的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$sha = (Get-FileHash outputs/adaptive-reference-20261113/two-image-chroma-quantile-gain-v1/results.json).Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_cross_source_capture_direction.py outputs/adaptive-reference-20261113/cross-source-capture-direction-repro $sha
~~~

两次各生成 48 张成片、6 张对照和 1 份报告，55 个文件的 SHA-256 逐一一致。另用合成颜色验证，单独移除一个已知方向时，垂直方向的编辑可以保留。本机成片位于 outputs/adaptive-reference-20261113/cross-source-capture-direction-v1。
