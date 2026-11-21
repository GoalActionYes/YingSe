# VeraRetouch 统一参考通道后，两图偏色仍未解决

作者的参考 CLI 由 OpenCV 读参考后直接送给 Pillow，参考顺序因而为 BGR。目标明确转 RGB。该提交的另一推理数据入口也将输入转为 RGB，但公开代码未确认参考分支的训练约定。上一轮保留原行为，本轮只将两张参考条件同时改为 RGB，使用相同冻结权重、原尺寸目标、512 参考编码与 80% 强度，判断通道顺序是否足以解释失败。[事前协议](../../experimental/adaptive-reference-20261113/gros-vera-reference-rgb-protocol.md) · [上轮 BGR 成片](2026-11-21-gros-vera-reference-pair.md) · [作者参考入口](https://github.com/OpenVeraTeam/VeraRetouch/blob/fefe6c4408f00beae07e132fc404cae0efee0985/infer_ref_retouch.py) · [其他数据入口](https://github.com/OpenVeraTeam/VeraRetouch/blob/fefe6c4408f00beae07e132fc404cae0efee0985/data/infer_dataset.py)

| 01 ← 02 成片 | 两图 BGR → RGB | 已知参考前图 BGR → RGB |
| --- | ---: | ---: |
| 整图关闭态，冷／暖 | 23.76／30.28 → 22.92／27.86 | 1.94／1.50 → 1.59／1.23 |
| 整图风格误差，冷／暖 | 5.07／5.62 → 5.78／6.40 | 3.67／2.98 → 3.05／3.10 |
| 整图开态换灯差 | 4.04 → 3.33 | 2.20 → 0.92 |
| 五区联合通过 | 0／5 → 0／5 | 0／5 → 2／5 |

所有误差为 RGB8 MAE。裸模型同图误差由 2.978 降到 1.427，仍非逐像素恒等。已知真实参考前图的两处白幕通过，暖灯双颊的原反向响应转正，但右颊冷／暖风格投影为 2.095／1.373，均超过 1.25。普通两图中，右颊误差有所下降，关态仍达约 25 RGB8，暖灯风格投影由 1.655 升至 3.168。保留完整反例，结束固定配置两种参考通道对照。[逐区成绩与成片哈希](2026-11-21-gros-vera-reference-rgb-evidence.json)

通道顺序对肤色、白幕和同图颜色确有影响。它无法解释普通两图的大部分关态偏色。三图的白幕收益仍需要真实参考原片，只用于离线证据。仅验证一组真人和一种全图编辑，软件、独立评价及视频流程均未改动。

## 复现

使用[上一轮已核验权重、源码、环境及 BGR 成片](2026-11-21-gros-vera-reference-pair.md)。两遍各生成 9 张成片、2 张对照板、1 份报告，逐文件 SHA-256 一致。对两套 RGB 输入的 10 个范围及同图误差，从 PNG 独立复算。成片在本机 `outputs/adaptive-reference-20261113/gros-vera-reference-rgb-v1`。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_vera_reference_rgb.py outputs/adaptive-reference-20261113/gros-vera-reference-rgb-repro
~~~
