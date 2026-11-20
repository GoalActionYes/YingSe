# 分材质冷暖校正减轻了部分肤色串色，整图仍未过线

Gros 人像的主体灯和白幕灯分开。为检验白幕不能代表的脸部变化是否值得使用，本轮从同一参考人物的冷暖两张未编辑实拍，分别读取双颊、左白幕和右白幕的线性对数 RGB 中位数。统一方案用三组增益的中位数校正参考整图，分材质方案以三个区域中心为锚进行高斯插值。双方都把同一拍摄校正作用于暖灯原片和暖灯 Film B，随后以 13 号原入口、80% 强度渲染另一位人物。目标标注和正确成片仅用于评分。参考原片对和参考材质标注属于离线输入。[事前协议](../../experimental/adaptive-reference-20261113/gros-material-capture-field-protocol.md)

原暖灯参考的人脸三通道中位值相对冷灯的差在 0.836–0.993（线性对数 RGB 向量范数），统一校正后为 0.717–0.893，分材质校正后为 0.136–0.252。分材质的两处白幕残差也普遍小于原灯，但统一方案在被用作中位增益的白幕上误差为零。分材质暖灯 Film B 参考的色域裁切最高为 6.80% 像素，未编辑暖灯最高为 0.67%。

## 输出

| 目标 ← 参考 | 13 号整图换灯差，关／开 | 统一校正 | 分材质校正 |
| --- | ---: | ---: | ---: |
| 01 ← 02 | 2.91／4.28 | 1.65／1.39 | 7.36／6.83 |
| 01 ← 03 | 4.50／4.58 | 1.82／1.28 | 5.45／5.68 |
| 02 ← 01 | 5.56／4.61 | 0.90／1.51 | 5.11／6.23 |
| 02 ← 03 | 4.17／3.62 | 1.91／2.21 | 6.18／6.16 |
| 03 ← 01 | 5.53／5.04 | 0.85／1.28 | 5.76／6.68 |
| 03 ← 02 | 3.25／3.42 | 1.91／1.99 | 8.51／8.83 |

单位为 RGB8 MAE。01 ← 02 的右脸颊在 Film B 开态换灯差由 8.005 降至 2.098，暖灯下风格误差由 8.081 降到 2.793。相同条件的整图换灯差却由 4.283 增至 6.827。统一校正使六组的整图换灯差都低于 3，但风格通过数从原 13 号的 4／60 降为 1／60。分材质为 5／60，仍损失了 03 ← 02 原本通过的暖灯右脸颊。三者完整通过方向均为 0／6。[逐区结果与成片哈希](2026-11-21-gros-material-capture-field.json)

## 决定与复现

保留按材质估计拍摄变化的证据，结束这套把局部增益插值到整图的做法。局部脸颊的收益没有覆盖整体失真与裁切。下一次若检验材质证据，应先限制校正作用域或单独隔离预设应用前后的处理顺序，再以相同的白幕和整图门槛验证。当前处理依赖已知参考前图和人工参考区域，尚不提供普通两图候选。

用[Gros Film B 开关归档](2026-11-21-gros-film-b-cross.md)，从仓库根目录选择新的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-film-b-cross-v2/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_material_capture_field.py "$base/gros-material-capture-field-repro" $sha
~~~

两遍均生成 12 张校正参考、24 张目标成片、6 张对照和 1 份数值报告，43 个文件逐一复核 SHA-256 一致。另核对零增益恒等及单锚点等同整图增益。完整成片保留在本机 outputs/adaptive-reference-20261113/gros-material-capture-field-v1 及同级复跑目录。
