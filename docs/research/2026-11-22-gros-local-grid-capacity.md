# 现有 13 号网格能够表达这批局部肤色编辑

两图出片的失败可能出在参考意图推断，也可能出在渲染器的表达能力。旧 Film B 实验只覆盖全图预设。本轮给定三组人物在冷、暖两档的局部编辑答案，比较整图共享的八层颜色矩阵和 13 号现有 16 × 16 × 8 空间网格。两臂使用同一冻结前后处理、80% 强度和原生渲染。拟合后，每个像素只取未用该像素所在方块估计系数的成片结果。[事前协议](../../experimental/adaptive-reference-20261113/gros-local-grid-capacity-protocol.md) · [旧全图预设表达验证](2026-11-21-gros-grid-expression.md)

| 留出块检查 | 同颜色在整图共享矩阵 | 保留 16 × 16 空间位置 |
| --- | ---: | ---: |
| 整图、双颊、双侧白幕和区域外全部通过 | 2／6 | 6／6 |
| 双颊各自通过 | 6／12 | 12／12 |
| 双颊最坏编辑误差 | 6.252 RGB8 | 0.376 RGB8 |
| 双白幕最坏偏色 | 0.295 RGB8 | 0.081 RGB8 |
| 编辑范围外最坏平均变化 | 0.674 RGB8 | 0.112 RGB8 |

以第 3 包暖灯为例，左颊误差由 **6.252** 降到 **0.288**，编辑投影由 **0.653** 恢复到 **1.000**。白幕仍低于原 3 RGB8 上限。原图、指定效果、共用颜色版本和原生空间版本都已保存。全局拟合表现差并不能否定所有全局颜色函数。普通 CUBE 对相同 RGB 的输入只能给出同一个输出；这次空间版本依赖像素所在位置，不能用一个普通 CUBE 完整保留这种位置关系。[第 3 包暖灯成片](../../outputs/adaptive-reference-20261113/gros-local-grid-capacity-v1/3-warm-comparison.png) · [六个样本的证据](2026-11-22-gros-local-grid-capacity-evidence.json)

这组实际成片为原生渲染器提供了可达见证。已知编辑目标时，当前渲染器可以在白幕几乎不变的同时达到所需的脸部颜色。本次拟合从同一目标读取了正确成片，只在同图上留出空间块。普通两图自动定位和风格推断仍未取得相同结果。后续优先查参考端的编辑位置与幅度。13 号当前主要把参考编码为全图颜色统计，已有控制也显示相同像素直方图可能掩盖编辑位置。现有结果支持优先改参考条件推断，普通实时渲染、独立人评和视频仍须按原要求验收。[参考布局控制](2026-11-19-reference-layout-collision.md) · [实际两图基线](2026-11-22-gros-local-editor-channel.md)

## 工具与复现

用固定 SciPy 1.18.1 LSMR 求解。仅下载官方 cp312 win_amd64 wheel，共 36,658,278 字节，SHA-256 为 `5e4d44984abc0020154ea81b247adeddcc3ac5527b975ff798bd1ba0adc513c2`，解包到本机忽略的研究目录。现有 Python 和产品依赖没有安装变更。12 个空间拟合折的三个通道均达到预设 240 步上限，结果作为已有解能做到的效果。不能据此声称数学最优。已知随机网格和实际各折的稀疏设计与原生浮点渲染差均低于 `1e-5`，零强度逐像素不变。[SciPy 发行文件](https://pypi.org/project/scipy/1.18.1/#files) · [许可证](https://github.com/scipy/scipy/blob/v1.18.1/LICENSE.txt) · [LSMR 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html)

两遍 30 张 PNG 和报告逐件同哈希。另从输出图复算 72 项分区读数，并核对棋盘拟合像素数。既有容量审计单测 5 项通过。报告 SHA-256 为 `252711ece1da355cf50335ad4062f4b52294b0cc7edad38772023cb7e0b2a12e`。本机 CUDA 只执行冻结模型的色彩引导与渲染，线性求解在 CPU。无权重训练或产品发布。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$launch = 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/scipy-capacity-20261122/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")'
& .tools/adaptive-env/Scripts/python.exe -c $launch experimental/adaptive-reference-20261113/probe_gros_local_grid_capacity.py $base "$base/gros-local-grid-capacity-new"
~~~
