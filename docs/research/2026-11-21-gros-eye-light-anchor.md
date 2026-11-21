# 人像眼周亮点校正减轻了部分肤色偏差，仍未保住风格

真人换灯对照中，白幕和脸部受到的光不同。已有按白幕或材质校正的结果无法同时防止脸部偏色和保留应迁移的风格。眼部附近可能记录照明在眼球上的反光，因此本轮用原来登记的眼睛中心，在固定窗口里取未裁切的亮点，估计当前目标和参考的局部色度差。只在参考图脸部邻域校正这项差额，其余区域保持原码值。随后冻结 13 号原权重、输入尺寸与 80% 强度，渲染六个跨人物方向。亮点可能来自眼白、皮肤或反光，取样规则没有识别真实角膜反射。[事前协议](../../experimental/adaptive-reference-20261113/gros-eye-light-anchor-protocol.md) · [原始人像对照](2026-11-21-gros-film-b-cross.md) · [角膜反光研究](https://www.sciencedirect.com/science/article/abs/pii/S0925231208001112)

| 目标 ← 参考 | 暖灯关预设整图误差，原 13 号 → 眼周校正 | 暖灯关预设双颊误差，原 13 号 → 眼周校正 | 暖灯整图风格投影，原 13 号 → 眼周校正 |
| --- | ---: | ---: | ---: |
| 01 ← 02 | 5.72 → 5.40 | 6.79／6.25 → 4.56／3.89 | 0.735 → 0.719 |
| 01 ← 03 | 8.68 → 8.39 | 15.70／16.27 → 15.68／16.31 | 0.509 → 0.514 |
| 02 ← 01 | 10.87 → 10.92 | 2.99／5.36 → 2.38／4.87 | 0.592 → 0.592 |
| 02 ← 03 | 8.55 → 8.48 | 15.19／21.11 → 15.22／21.06 | 0.634 → 0.639 |
| 03 ← 01 | 12.50 → 12.64 | 7.68／4.92 → 7.22／4.36 | 0.595 → 0.556 |
| 03 ← 02 | 8.62 → 8.81 | 4.15／2.84 → 4.11／2.65 | 0.731 → 0.723 |

误差用 RGB8 MAE。01 ← 02 的暖灯双颊都改善，但仍超过 3，右颊的风格投影只由 2.501 降至 2.497。03 ← 02 的暖灯右颊原本满足风格要求，投影从 0.786 降至 0.741，低于 0.75。六组的整图、双颊、双白幕联合结果均未通过。单项风格由 4／60 降至 3／60。12 个眼部取样均有效，局部参考变换没有产生通道上溢。原片冷暖灯会改变所有三个取样者的眼周颜色，Film B 同样会改变眼周读数，故该采样无法独立区分现场灯光与有效编辑。[逐区指标与成片身份](2026-11-21-gros-eye-light-anchor-evidence.json)

三个同图控制均失败，整图误差分别为 6.172、7.403、6.390 RGB8。此时亮点差为零，校正原样返回输入，因此观察到的是冻结 13 号原入口的同图表现。依照事前门槛，结束固定眼周亮点校正，保留全部反例。手工眼位属于诊断输入，这些成片没有验证自动定位、独立画质、人工自然度或视频。考虑其他光照线索时，应先证明能在同一主体上区分实际换灯与编辑变化，再接入实时渲染。

后续从 13 号正常工作台复核了这三个同图控制。预览和 PNG 已有同图保护，手动亮度仍正常生效。上一段偏差只适用于直接调用空间模型。六个跨人物方向的 24 张真实导出 PNG 与原研究成片一致。[入口差异与结果](2026-11-21-gros-workbench-identity.md)

## 复现

用[冻结 Film B 对照](2026-11-21-gros-film-b-cross.md)中的环境和归档。选择未占用的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_eye_light_anchor.py outputs/adaptive-reference-20261113/gros-eye-light-anchor-repro
~~~

两遍各 24 张跨人物成片、3 张同图、6 张比较板及 1 份报告，34 个文件逐一通过 SHA-256 一致性检查。另直接从 PNG 和登记矩形复算 30 个范围的开关、风格投影和换灯误差，核对同图差及旧分数。本机成片在 `outputs/adaptive-reference-20261113/gros-eye-light-anchor-v1`。
