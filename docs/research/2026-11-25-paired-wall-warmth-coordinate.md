# 给定参考编辑与作用范围，局部暖色可以跨场景重放

从实际参考前后图估出的一个局部颜色偏移，可以用到另一场景的浅色表面。MIT 客厅和 Beyond RGB 白绿场各试两档参考，RGB8 和 D50 CIELAB 两种估算的八张原尺寸结果均通过现有区域门槛。此次从参考估算颜色幅度，参考和目标的作用范围来自离线编辑配置。普通参考照片的编辑辨认和作用范围估计仍待验证。[事前协议](../../experimental/adaptive-reference-20261113/paired-wall-warmth-coordinate-protocol.md) · [MIT 实际出图](../../outputs/adaptive-reference-20261113/paired-wall-warmth-coordinate-run-a/mit/comparison.png) · [Beyond RGB 实际出图](../../outputs/adaptive-reference-20261113/paired-wall-warmth-coordinate-run-a/beyond_rgb/comparison.png)

## 为什么做这次对照

两版现有软件对普通参考都漏掉浅墙暖色。前一轮用已知目标开态证明了原生空间网格可表达这两处局部暖色。本轮把计算暖色所需的目标开态拿掉。每个参考灯档只用该档的真实前后图估算偏移，向同家族另一场景的目标原图重放。这能判断在给出正确的编辑位置之后，是否还需要知道目标的真实编辑结果才能把幅度做对。[普通双图失效](2026-11-24-beyond-rgb-warmth-workbench.md) · [原生网格表达见证](2026-11-24-matched-wall-warmth-grid.md)

同一份 RawTherapee 配置处理两张中性灰和一张浅暖平场。参考和目标用各自原有位置。只从 128 RGB8 平场提取归一化空间权重，两臂共享该权重。每个参考灯档从原登记核心、未近饱和且权重至少 0.8 的像素求稳健偏移。参考有效覆盖为 95.08%–96.63%。D50 CIELAB 按已登记的 Lab 编辑概念选取，使用同机已核验的 Colour Science 实现。两臂以相同 80% 强度保存目标成片。[Colour Science 0.4.7 转换](https://colour.readthedocs.io/en/v0.4.7/generated/colour.sRGB_to_XYZ.html)

## 成片结果

下表各指标按参考两档原顺序列出。暖色投影 100% 为复现目标已知编辑的幅度，过线区间 75%–125%。响应和开关差的既有误差上限为 3 RGB8。

| 目标 | 偏移坐标 | 墙面暖色投影 | 墙面响应误差（RGB8） | 两档参考输出间墙面差 |
| --- | --- | ---: | ---: | ---: |
| MIT 客厅 | RGB8 | 101.79%／102.79% | 1.43／1.88 | 0.64 |
| MIT 客厅 | D50 CIELAB | 106.41%／111.87% | 1.32／1.61 | 0.30 |
| Beyond RGB 白绿场 | RGB8 | 93.81%／106.67% | 1.25／0.89 | 0.67 |
| Beyond RGB 白绿场 | D50 CIELAB | 104.98%／104.16% | 0.94／0.61 | 0.36 |

所有成片的登记保护区及一网格宽外侧过渡带与各自预期图逐像素相同，无新增通道端点。这个结果依赖已给定的区域。关闭编辑时两灯输出都直接取目标原图，零误差来自恒等控制。D50 CIELAB 在两个目标上均减少两档成片差和响应误差，完整通过数和 RGB8 相同。MIT 上的幅度偏强，仍在原定区间内。[逐区、逐档和成片身份](2026-11-25-paired-wall-warmth-coordinate-evidence.json)

## EGM 审查与验证

独立审查提出三个区分条件。先用目标自身开关检验同类重放能否表达目标编辑。这个控制在全部四种目标与坐标组合上通过，最坏墙面响应误差 1.34 RGB8。再核验灰片投影会不会掩盖色偏，所有位置的两灰权重差 P95 最高 0.01090，浅暖片与 128 灰的权重差 P95 最高 0.00433，未裁切权重中的垂直投影残差 P95 最高 0.256 RGB8。三种平场中没有权重裁切。最后以准备、出图、评分顺序控制输入。主重放只由源前后图、目标原图和归一化权重推算，八张主输出全部保存并登记哈希后才读目标开态。第二次运行用 Python 文件访问审计拒绝提前读取目标开态。

两遍分别耗时 26.65 和 26.17 秒，共 58 份 PNG 或 JSON 逐字节一致。另从落盘帧按原分区复算 28 项记录，包括暖色投影、响应误差、换灯差、保护带与端点。两遍均使用本机 CPU、冻结素材和现有环境。实验未训练模型，未修改工作台。

## 对下一轮的影响

两处证据现在分别覆盖了「能渲染」和「给定参考编辑后能把幅度转到新场景」。下一项先测试从普通参考估计局部编辑时可以取得的具体线索，同时暂时固定这套已知范围的离线回放，直接以两组已知开关作对照。只在信号能预测应迁移的有意暖色、并拦住真实灯光误传时，才把它与自动编辑范围一起送进正常两图开发比较。这里仅有一种冻结构造的局部暖色、两处静物目标及各自同一采集家族的参考。真人肤色、其他风格与完整产品验收仍按原范围。

## 复现

沿用两组冻结素材、已固定的 RawTherapee 5.13 和 Colour Science 0.4.7。两个完整运行使用不同的新输出目录。可附 `--replay-only` 停在八张主图和回执写盘处，仅验证无目标开态的步骤。

~~~powershell
if (Test-Path 'V:/') { throw 'V: is in use' }
subst.exe V: (Get-Location).Path
try {
    $base = 'V:/outputs/adaptive-reference-20261113'
    $env:PYTHONPATH = "V:/;V:/.tools/tf-convert-env/Lib/site-packages;$base/scipy-capacity-20261122/wheel;$base/colour-tools-20261122/colour_science-0.4.7-py3-none-any.whl"
    $env:OPENBLAS_NUM_THREADS = '2'
    & V:/.tools/adaptive-env/Scripts/python.exe V:/experimental/adaptive-reference-20261113/probe_paired_wall_warmth_coordinate.py $base "$base/paired-wall-warmth-coordinate-new"
    if ($LASTEXITCODE -ne 0) { throw "Warmth replay exited $LASTEXITCODE" }
} finally {
    subst.exe V: /D
}
~~~
