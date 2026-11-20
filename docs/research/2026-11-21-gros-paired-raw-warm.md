# 移除暖灯参考预校正后，整图重放完整通过 4／6

上轮用限域校正的暖灯参考开关图拟合整图颜色表，12 项整图风格均过，却有四处暖灯右脸颊过调。本轮仅移除暖灯参考预校正，直接从同一拍摄的暖灯原图和 Film B 开态拟合。沿用三次拟合、80% 强度及目标原图关闭态。冷灯成片逐项复用归档。目标正确成片只有在输出后进入评分。[事前协议](../../experimental/adaptive-reference-20261113/gros-paired-raw-warm-protocol.md) · [预校正对照](2026-11-21-gros-paired-global-local.md)

三份暖灯参考颜色表均通过原留块资格。留块 MAE 为 0.256–0.281 RGB8，P95 均为 1，最差块均值不超过 1.959。两臂保持 12／12 项整图风格通过。

| 目标 ← 参考 | 暖灯右脸颊投影，预校正 → 原图 | 去预校正后完整通过 |
| --- | ---: | --- |
| 01 ← 02 | 1.162 → 0.938 | 是 |
| 01 ← 03 | 1.219 → 1.069 | 是 |
| 02 ← 01 | 1.446 → 1.294 | 否 |
| 02 ← 03 | 1.366 → 1.200 | 是 |
| 03 ← 01 | 1.441 → 1.353 | 否 |
| 03 ← 02 | 1.305 → 1.028 | 是 |

完整通过 2／6 提高到 4／6，60 项风格由 56 增至 58，没有丢掉上一轮整图方案的通过项目。相较更早已通过的局部方案，03 ← 01 暖灯右脸颊仍掉项。六组整图换灯差以原暖灯图拟合后均下降，为 0.367–1.525 RGB8。两处余下失败的暖灯右脸颊 MAE 仍低于 3，失败原因是风格投影高于 1.25。[逐区分数、拟合与文件哈希](2026-11-21-gros-paired-raw-warm.json)

## 决策与复现

同拍摄开关配对已经把实拍灯位固定住，额外预校正反而增加了本轮的过调。两处剩余反例都用了人物 01 的暖灯参考。后续应先区分其颜色样本覆盖不足与三次函数拟合偏差，再决定是否值得改变模型。该结果仍依赖参考编辑前后图，未验证普通两图、独立画质、人工自然度或视频。

沿用[上轮整图对照及其上游](2026-11-21-gros-paired-global-local.md)，在仓库根目录运行，选择尚未使用的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-paired-global-local-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_paired_raw_warm.py "$base/gros-paired-raw-warm-repro" $sha
~~~

两轮各保存 6 张暖灯成片、6 张对照、1 份报告，13 个文件逐一核对 SHA-256 一致。另逐项核对冷灯评分与上轮相同。完整成片留在本机 outputs/adaptive-reference-20261113/gros-paired-raw-warm-v1 及同级复跑目录。
