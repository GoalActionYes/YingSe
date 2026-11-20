# 限制分材质校正范围后，整图稳定但脸颊收益丢失

上轮从同一参考人物的冷暖实拍中估计脸颊、左右白幕的颜色增益，把三种增益插值到整张参考后，一处目标脸颊到达门槛，整图和参考色域却出现退化。本轮保持原增益和渲染器，只改变作用域。原登记脸颊和白幕框内使用原分材质校正，以每侧半个框宽或半个框高作余弦过渡，其余像素回到统一校正。用 13 号在同一 80% 强度生成新成片。冷暖参考原图与登记框仍为离线信息，目标预期只用于评分。[事前协议](../../experimental/adaptive-reference-20261113/gros-bounded-material-field-protocol.md) · [上轮对照](2026-11-21-gros-material-capture-field.md)

三位参考人物的非零作用域覆盖 7.81%–8.25%。范围外的六张校正参考均与上轮统一校正逐像素一致，参考三个测量区的色度及亮度合成残差与整图分材质校正一致。暖灯原片和 Film B 开态均无超色域裁切，上轮整图分材质开态最高为 6.80%。

| 目标 ← 参考 | 13 号整图换灯差，关／开 | 上轮全图分材质 | 局部限域 |
| --- | ---: | ---: | ---: |
| 01 ← 02 | 2.91／4.28 | 7.36／6.83 | 1.61／1.40 |
| 01 ← 03 | 4.50／4.58 | 5.45／5.68 | 1.82／1.28 |
| 02 ← 01 | 5.56／4.61 | 5.11／6.23 | 0.86／1.50 |
| 02 ← 03 | 4.17／3.62 | 6.18／6.16 | 1.93／2.24 |
| 03 ← 01 | 5.53／5.04 | 5.76／6.68 | 0.77／1.36 |
| 03 ← 02 | 3.25／3.42 | 8.51／8.83 | 1.88／2.02 |

单位为 RGB8 MAE。六组整图换灯差均低于 3。以 02 为参考、01 为目标时，暖灯 Film B 的右脸颊误差却由全图分材质的 2.793 增至 3.747，换灯差由 2.098 回升至 3.032，均超原门槛。两灯合计 60 项区域风格检查，原 13 号通过 4 项、统一校正 1 项、全图分材质 5 项、局部限域 1 项。四臂的完整方向均为 0／6。[结果摘要与全量报告哈希](2026-11-21-gros-bounded-material-field.json)

## 结论与复现

在参考脸颊和白幕处保持同样的校正，目标端仍未保持上轮局部收益。结果支持结束这套固定增益及作用域路线。后续若要利用材质颜色，应先验证渲染器是否真正把参考局部信息与目标部位关联起来。稳定版未改，普通两图候选、独立画质、人工自然度和视频验收仍待完成。

首次脚本误把像素坐标用作分母，产生除零警告与错误羽化范围。修正为以登记框宽高作常量分母，独立检查框内全权重、框外对称半权重、边界零权重和非法框拒绝。旧输出留作失败记录。仅采用修正后运行。使用上轮归档及其 [Film B 上游](2026-11-21-gros-film-b-cross.md)，选择尚未使用的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-material-capture-field-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_bounded_material_field.py "$base/gros-bounded-material-field-repro" $sha
~~~

修正后两轮各生成 6 张校正参考、12 张目标成片、6 张对照及 1 份报告，25 个文件逐一核对 SHA-256 一致。本机完整产物位于 outputs/adaptive-reference-20261113/gros-bounded-material-field-v3 和同级 recheck3 目录。
