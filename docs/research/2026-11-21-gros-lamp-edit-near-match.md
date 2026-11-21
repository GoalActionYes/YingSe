# 同快门真人的实拍暖灯与数字暖调仍有可见差异

旧对照把冷灯图数字染暖，脸颊能与暖灯实拍接近，但白幕的颜色变化明显不同。本轮用同快门的 02 冷暖真人图离线拟合一个全图二次 RGB 函数，测试能否让同一个数字调整同时接近整图、双颊和白幕。双颊按眼位做局部配准，白幕保持原坐标。16 像素块棋盘图只选一半拟合，另一半衡量。整图代理仍包含少量头部动作，保留相对于未经配准原始暖图的结果。[事前协议](../../experimental/adaptive-reference-20261113/gros-lamp-edit-near-match-protocol.md) · [旧肤色匹配对照](2026-11-20-gros-portrait-light-transfer.md)

| 02 参考范围 | 共同有效覆盖 | 数字暖调与对齐实拍暖灯的留块误差 | 与未经对齐的暖灯差 |
| --- | ---: | ---: | ---: |
| 整图 | 88.32% | 3.78 | 3.78 |
| 左颊 | 100% | 4.09 | 4.11 |
| 右颊 | 100% | 5.80 | 4.61 |
| 左白幕 | 100% | 3.21 | 3.21 |
| 右白幕 | 100% | 2.40 | 2.40 |

误差使用 RGB8 MAE。数字调整使参考整图的原冷暖差由 7.51 降到 3.78，五区只通过右白幕一项，未得到事前定义的近似输入对照。右颊按眼位配准后的分数反而更差，双眼定位不能保证脸颊像素完全对齐。本次更可靠的剩余差来自整图和没有做位移的左白幕，不能将每个剩余像素都归因于照明或材质。[完整度量及图像哈希](2026-11-21-gros-lamp-edit-near-match-evidence.json)

将冷灯、实际暖灯和数字暖调分别作为给 13 号的单张参考，固定目标 01。对照约定真实换灯保持目标原色，数字暖调按同样的函数在目标重放。两个期望结果整图相差 10.71 RGB8，实际 13 号两份目标成片相差 3.89。但参考自身已经未达近似门槛，两项相距不能证明对产品不可辨。数字暖调的整图风格投影为 0.225，双颊为 0.339／0.380，两侧白幕为 1.068／1.084。真实暖灯成片对原色的双颊偏差为 6.79／6.25，两侧白幕也均超过 3。保留两类失败，停止用这个固定全图匹配去逼近实拍换灯。该结果只覆盖一组同快门真人和人工构造编辑，未更改软件。

## 复现

沿用[冻存真人参考、13 号和运行环境](2026-11-21-gros-film-b-cross.md)，选择新的本机输出目录。两次各两张参考记录、四张目标渲染、一个对照板和一份报告，8 个文件逐项哈希相同。另用保存的系数及 PNG 独立复算留块误差、目标五区风格投影和偏色。本机产物保存在 `outputs/adaptive-reference-20261113/gros-lamp-edit-near-match-v1`。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_lamp_edit_near_match.py outputs/adaptive-reference-20261113/gros-lamp-edit-near-match-repro
~~~
