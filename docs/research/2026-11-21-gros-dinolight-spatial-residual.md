# 只用局部修光保留了参考风格，偏色仍然存在

上轮 DINOLight 把真人参考的换灯影响和 Film B 同时削弱。为检验按空间拆分能否保住两项收益，本轮从每张参考的 DINOLight 校正量中，减去一个只依赖原像素 RGB 的全图二次拟合。把余下的局部校正加回参考，再让冻结 13 号以 80% 强度渲染。全图函数逐张拟合，只读当前参考及它自己的恢复图。目标真值、区域和 Film B 标签仅用于对照评分。[事前协议](../../experimental/adaptive-reference-20261113/gros-dinolight-spatial-residual-protocol.md) · [完整修光结果](2026-11-21-gros-dinolight-reference.md)

| 检查 | 原参考 | 完整修光 | 仅局部校正 |
| --- | ---: | ---: | ---: |
| CL3AN 一号环境光误差 | 107.30 | 21.43 | 101.97 |
| CL3AN 二号环境光误差 | 76.66 | 21.88 | 72.39 |
| 真人参考 Film B 开关差，冷／暖 | 12.20／12.42 | 2.98／2.84 | 12.39／12.47 |
| 13 号整图开态换灯差 | 4.26 | 0.53 | 4.10 |
| 13 号整图关闭态误差，冷／暖 | 3.60／5.72 | 7.23／7.02 | 3.61／5.69 |
| 13 号整图风格投影，冷／暖 | 0.445／0.732 | −0.032／0.002 | 0.489／0.719 |
| 五区联合通过 | 0／5 | 0／5 | 0／5 |

除投影以外使用 RGB8 MAE。只用局部校正，两场强色灯的误差仅下降约 5%，低于预先要求的 20%，且分别有 32.0% 和 23.5% 的通道分量裁切。真人参考各态裁切约 0.60%–2.15%。真实换灯干扰重新出现，暖灯右颊过调投影从原 2.507 降到 2.072，仍高于 1.25，关闭态误差却从 6.26 升至 6.77。暖灯右白幕风格项新过线，开态换灯差也从 4.08 升至 4.77。整图、双颊和两处白幕没有一处同时通过所有条件。[逐区结果和成片哈希](2026-11-21-gros-dinolight-spatial-residual-evidence.json)

同一全图 RGB 分量中可以同时有拍摄灯光和创意色调。仅凭预测校正的空间形态，这两类变化无法在本组对照中分开。按原协议结束固定二次分离，也保留完整 DINOLight 修光的正对照。后续需验证带有明确拍摄与编辑来源的额外证据，才考虑重新使用此照明模型。此次为一组已用人像和两组已用强色灯，软件、正式独立验收与视频均未进入新的验收阶段。

## 复现

使用[同一 DINOLight 和参考图档案](2026-11-21-gros-dinolight-reference.md)，本轮不需要重新运行 DINOLight 或读取其权重。恒等及纯二次变化控制均逐像素保持原图。两次生成的 6 张参考或场景、4 张目标成片、2 张对照板和 1 份报告逐文件一致。另从实际 PNG 复算两场照明、真人参考开关差及五区所有条件。本机输出为 `outputs/adaptive-reference-20261113/gros-dinolight-spatial-residual-v1`。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_dinolight_spatial_residual.py outputs/adaptive-reference-20261113/gros-dinolight-spatial-residual-repro
~~~
