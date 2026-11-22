# 开源色差工具复核真人成片

原来的风格响应分数衡量打开风格相对于关闭风格改变了多少。用户看到的是最终颜色。这里复用 GitHub 上的 Colour Science 0.4.7，从冻结 PNG 补算 CIEDE2000 色差。关闭态对目标原图，打开态对预期成片。原 RGB8 误差和风格评分一并保留。库采用 BSD 3-Clause，本机研究脚本直接读取经核对的发行包，产品运行环境无需安装该库。[开源库与许可](https://github.com/colour-science/colour/blob/v0.4.7/LICENSE) · [事前协议](../../experimental/adaptive-reference-20261113/gros-endpoint-colour-protocol.md)

## 自动两图对照

三位已用真人六个互换参考方向，每方向冷暖两档。用旧普通连续回归和局部比值回归的实际成片，逐项比较整图、双颊和两侧白幕。

| ΔE00 检查 | 降低 | 升高 | 持平 |
| --- | ---: | ---: | ---: |
| 关闭态中位数 | 17 | 43 | 0 |
| 关闭态第 95 百分位数 | 25 | 35 | 0 |
| 打开态中位数 | 33 | 27 | 0 |
| 打开态第 95 百分位数 | 37 | 23 | 0 |

60 项中有 18 项的原风格响应误差下降，打开态 ΔE00 中位数却升高。计数只表示变化方向，整图和局部有重叠，同一人物也出现在多个方向。

以 01 ← 02 的暖灯参考为例，右脸颊的风格投影由 0.25 到 1.13，打开态色差中位数由 1.10 降到 0.86。**同张成片的左白幕**风格响应误差也从 8.46 降到 3.00 RGB8，但关闭态色差中位数由 6.96 升到 10.17，打开态由 10.52 升到 12.05。右脸颊的收益和左白幕的退化同时成立。原完整门槛仍为 0／6，既有停止决定照用。[本机自动两图对照](../../outputs/adaptive-reference-20261113/local-log-ratio-replay-v1/portrait/01-from-02-comparison.png)

## 离线可达性对照

允许关态在原预算内移动，仍可让原联合评分由 2／6 到 3／6。新增通过的 01 ← 02 冷灯整图打开态，两种度量给出的方向不同。RGB8 平均误差从 2.90 升到 3.10，ΔE00 中位数从 2.89 降到 2.51，第 95 百分位数从 3.99 降到 3.79。右白幕也呈同样的方向分歧。暖灯整图则出现风格响应误差下降而打开态 ΔE00 中位数上升。套用原预算后，共有 10 项出现这种分离。原 RGB8 结果、通过见证和所有反例继续保留。区域与作者预期仅供离线检查，结果适用于这批已用开发成片。[原关态预算证据](2026-11-21-gros-off-budget.md) · [本机离线对照](../../outputs/adaptive-reference-20261113/gros-off-budget-v1/01-from-02-comparison.png)

下一项有实质不同的新候选时，先检查同一成片上的肤色、浅色区域和风格响应，并用已保存的对照辨认收益来自哪里。色差补测只帮助定位反例，独立素材、真实自然度评价及完整视频测试仍用于产品验收。

## 复现与验证

使用仓库原研究 Python 3.12.14、NumPy 2.5.4、Pillow 13.0.0。仅在复测命令的 Python 模块路径加载 Colour Science 0.4.7 wheel，大小 9,065,613 字节，SHA-256 为 `fe2383d86e507fc11a4b5206ff31a7399031e0dce527e22ec06eeb491d45d9a7`。所调用的 API 在本机无需 SciPy 或 Matplotlib，缺少这两个可选库时会提示。[固定发行文件](https://pypi.org/project/colour-science/0.4.7/#files) · [色差 API](https://colour.readthedocs.io/en/v0.4.7/generated/colour.difference.delta_E_CIE2000.html)

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$wheel = Join-Path $base 'colour-tools-20261122/colour_science-0.4.7-py3-none-any.whl'
$env:PYTHONPATH = (Resolve-Path $wheel).Path
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_gros_endpoint_colour.py $base "$base/portrait-endpoints-new.json"
~~~

两遍输出逐字节相同。每遍核对五份原报告、区域文件、108 次 PNG 读取的文件哈希，重新计算 480 项 RGB8 端点误差并与原报告匹配。复算 Sharma 等人提供的 34 对色差样例，最大绝对误差为 0.00004950，符合公开数据四位小数的舍入误差。留存四臂逐项结果，报告 SHA-256 为 `b71a5bdb99033ac012c726bf24495d68e977c6b9aa414e41b35051a09eaf2d32`。[标准样例](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/) · [本机完整数字](../../outputs/adaptive-reference-20261113/colour-tools-20261122/portrait-endpoints.json)
