# 移动空间网格加重真人偏色，原位置已有保护作用

上一轮数字暖调在目标双颊的风格幅度低，在白幕接近预期。本轮固定三张冷、暖、数字暖调参考和同一目标，先生成原有 13 号网格，再沿宽度循环移动半幅。系数值和亮度分层都保持，最终渲染的目标像素、颜色引导、强度和五区评分均相同。这样直接检查系数放置对局部表现的影响。[事前协议](../../experimental/adaptive-reference-20261113/gros-shift-rendered-grid-protocol.md) · [上一轮公开数值](2026-11-21-gros-lamp-edit-near-match.md)

| 目标区域 | 实拍暖灯偏差，原 → 移位 | 数字暖调投影，原 → 移位 | 数字暖调误差，原 → 移位 |
| --- | ---: | ---: | ---: |
| 整图 | 5.72 → 10.22 | 0.225 → 0.179 | 7.85 → 9.09 |
| 左颊 | 6.79 → 19.08 | 0.339 → 0.183 | 14.67 → 17.72 |
| 右颊 | 6.25 → 16.84 | 0.380 → 0.165 | 15.84 → 21.22 |
| 左白幕 | 3.96 → 4.28 | 1.068 → 1.141 | 1.36 → 1.87 |
| 右白幕 | 3.83 → 4.05 | 1.084 → 1.101 | 1.63 → 1.92 |

误差使用 RGB8 MAE。两颊数字投影变化均超过 0.10，风格误差分别增加 3.05 和 5.38 RGB8。暖灯对脸颊串色也明显增大。两侧白幕的数字风格仍过其分项门槛，但任一区都未同时通过抗偏色与有效风格。对这一固定干预，位置确实重要，原网格比该移位同时更能限制脸部串色、保留应有的暖调。没有证据将当前风格不足归因于仅仅放错位置，也不认为网格已理解人脸语义。[数值与成片身份](2026-11-21-gros-shift-rendered-grid-evidence.json)

结束这一预定位置干预，不为同一个方向选新的移动距离。它支持继续保留现有空间位置分配，改进应优先改善从参考估计的局部颜色变化。这里只比较一组人工暖调及配套实拍换灯，未改软件或独立验收门槛。

## 复现

沿用[同快门对照和已冻结目标](2026-11-21-gros-lamp-edit-near-match.md)，从仓库根目录选择未使用的输出目录。网格循环两次恢复原值，逐项排序后完全一致。两遍各生成 3 张基线、3 张移位成片、1 张对照板和 1 份报告，8 个文件逐件一致。原位基线的 3 张成片和五区指标复现上轮，另从 PNG 独立核对两臂 10 个区域。本机成片在 `outputs/adaptive-reference-20261113/gros-shift-rendered-grid-v1`。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_shift_rendered_grid.py outputs/adaptive-reference-20261113/gros-shift-rendered-grid-repro
~~~
