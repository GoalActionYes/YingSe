# 同一个已知编辑在目标原图和 13 号基底上给出不同结果

前一轮使用正确参考仍只有一组通过有意染色门槛。本轮从每组相机的参考白灯和数字染色 RGB 估计同一个线性乘法操作。每通道从交错采样块拟合，再在留出块检查误差。拟合不读取真实编辑参数、目标 RGB 或目标真值。固定 80% 强度，只在两个目标基底之间切换：原图，或者 13 号使用白灯参考后的成片。原 13 号数字参考成片留作对照。[事前协议](../../experimental/adaptive-reference-20261113/chromamill-target-base-protocol.md) · [理想参考替换](2026-11-20-chromamill-oracle-reference.md)

## 两种工作基底的结果

六对参考的留出块 MAE 不超过 0.025 RGB8，P95 均为零。以下表格同时保留编辑前后差的误差与最终成片相对目标直接编辑的误差。二者回答不同问题。

| 开发参考 | 原图上编辑响应误差 | 13 号基底上编辑响应误差 | 13 号基底结果对预期成片的误差 |
| --- | ---: | ---: | ---: |
| Scene3_a 尼康 | 0.000 | 3.058 | 23.131 |
| Scene3_a 手机 | 0.002 | 4.123 | 38.031 |
| Scene9_b 尼康 | 0.025 | 1.960 | 12.434 |
| Scene9_b 手机 | 0.002 | 2.292 | 11.942 |
| Scene10_a 尼康 | 0.003 | 1.409 | 16.168 |
| Scene10_a 手机 | 0.000 | 1.447 | 15.525 |

单位为 RGB8 MAE。把估计出的编辑直接应用于原图，整图响应 6／6、灰阶 30／30 通过，最终成片误差与响应误差相同，最高 0.026。用同一个编辑处理 13 号白灯基底，增量整图 4／6、灰阶 21／30 通过，但最终成片仍差 11.942–38.031 RGB8。两组 Scene3_a 在基底上的编辑响应都过强，投影分别为 1.317 和 1.451。实验中任何成片的裁切比例都不超过 0.013%，无法用裁切解释这些差异。[逐项结果、真实参数隔离及成片哈希](2026-11-20-chromamill-target-base-evidence.json)

此对照说明，准确应用同一编辑也会继承偏离原目标的底色，甚至在部分结果中改变编辑幅度。后续目标侧模型必须同时核对「白灯参考时目标底色漂移」与「开关编辑的实际增量」，不能将增量达标替代最终观感。此处只能估计明确的全局线性染色，需要普通两图额外缺少的参考前图。最高 0.026 RGB8 的结果属于已知参考对的离线上限，不代表目标画质已升级。显示位置、一般作者局部风格、真人、白墙和视频另行验证。

## 复现

先准备[双相机素材](2026-11-20-chromamill-camera-pairs.md)及[上一轮三个流程的成片](2026-11-20-chromamill-matched-supervision.md)。从仓库根目录运行。两遍 12 张输出、6 张拼图和报告逐字节一致，媒体留本机。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_chromamill_target_base.py "$base/beyond-rgb-matched-cast-prepared-v1" "$base/chromamill-camera-pair-first-v1" "$base/beyond-rgb-matched-cast-first-v1" "$base/chromamill-matched-supervision-first-v1" "$base/chromamill-target-base-replay"
~~~
