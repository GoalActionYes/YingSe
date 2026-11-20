# 理想参考还原消除了换灯差，风格仍只有一组通过

上一轮带数字保留监督的参考校正在尼康白灯案例上出现 48.52% 裁切。这轮锁定原来六组目标和参考，先把固定岭回归展开为对训练标签的等价加权，再离线给出正确的参考来源。实拍绿灯参考替换为同机位白灯实拍，有意染色和白灯参考完整保留。理想对照只回答原 13 号在参考完全校正后能否响应正确风格。真实前图及身份只用于这项离线诊断。[事前协议](../../experimental/adaptive-reference-20261113/chromamill-oracle-reference-protocol.md) · [原候选与退化](2026-11-20-chromamill-matched-supervision.md)

## 白灯退化的归因

18 份参考的等价系数均能复现上一轮限幅后增益。其中 7 份至少有一通道预测超出所在训练折的标签范围。Scene3_a 尼康白灯的训练标签负权重绝对值之和为 3.001。预测的红增益 0.948 低于训练范围下限 0.961，属于外推。预测蓝增益为 4.855，虽触发 研究脚本的 4 倍上限，却仍低于该训练折真实绿灯校正的上界 5.868。白灯本来应是 1 倍。这个反例同时包含通道外推和错误地给白灯应用强校正，单独把增益限制在训练范围内不能直接排除后者。[预测、范围及原图身份](2026-11-20-chromamill-oracle-reference-evidence.json)

## 准确校正之后的成片

| 留出场景与相机 | 真实换灯变化 | 有意染色投影 | 有意染色误差（RGB8） |
| --- | ---: | ---: | ---: |
| Scene3_a 尼康 | 0 | 1.482 | 6.566 |
| Scene3_a 手机 | 0 | 1.729 | 8.716 |
| Scene9_b 尼康 | 0 | 0.794 | 3.293 |
| Scene9_b 手机 | 0 | 0.866 | 2.922 |
| Scene10_a 尼康 | 0 | 1.017 | 3.937 |
| Scene10_a 手机 | 0 | 0.926 | 5.076 |

换灯结果和白灯基准逐像素一致。数字染色仍按原门槛评分，整图只有 Scene9_b 手机同时满足方向和误差，六组共 1／6。灰阶区域为 11／30。多数失败包含幅度不准及色彩空间响应误差，不能仅归为参考校正不够。结论是这套准确参考替换在当前目标渲染器上的实测结果，仅覆盖固定参考路线，不能解释其他可能的目标渲染模型。

停止仅用参考前处理解决这批串色与风格冲突。后续若继续，需要直接改善目标侧的风格响应，并保留准确参考替换作为测量基准。产品仍只接收通常的两张 RGB 图。此次未改软件，也未运行独立白墙、肤色、自然评价或视频验收。

## 复现

使用已有自适应环境及[上一轮成片](2026-11-20-chromamill-matched-supervision.md#复现)。两次 18 张复跑成片、6 张拼图及报告逐字节一致。原图和成片留本机。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_chromamill_oracle_reference.py "$base/beyond-rgb-matched-cast-prepared-v1" "$base/chromamill-camera-pair-first-v1" "$base/beyond-rgb-matched-cast-first-v1" "$base/chromamill-matched-supervision-first-v1" "$base/chromamill-oracle-reference-replay"
~~~
