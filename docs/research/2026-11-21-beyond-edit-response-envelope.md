# 静物偏色对照中的编辑混合上限

固定目标亮度分配两套编辑量时，静物灰卡有明显的迁移不足。为了区分比例选错与两套渲染器的调整范围不足，这次在相同成片上允许每个像素独立选择比例。使用 Beyond RGB 两个静物构图，每个构图保留此前的两个拍摄状态，全部为已用开发素材。

稳定版 1.3 和研究版 13 号各自在 80% 强度下生成数字编辑参考和原参考的两张成片。编辑增量取这两张成片之差。两种增量从同一目标原图出发，逐像素使用一个 RGB 通道共享、介于 0 和 1 的混合比例。离线用已知正确成片选择绝对误差最小的比例。原参考与实拍换灯条件沿用受控基线，直接保留目标原图。该设置只衡量目标侧已有编辑量的上限。[事前协议](../../experimental/adaptive-reference-20261113/beyond-edit-response-envelope-protocol.md)

## 哪些区域任何比例都过不了

沿用 RGB8 平均绝对误差上限 3、风格方向投影 0.75 至 1.25。先求裁切后、量化前的最低误差，再减去 8 位取整每通道最多可能带来的 0.5 误差改善。风格投影区间也计入取整余量。只有在误差下界仍超标，或整个投影区间都在门槛外时，才判定现有两套增量的任何比例都过不了。

| 拍摄状态 | 离线最佳成片通过 | 可证明没有合格比例 | 具有代表性的结果 |
| --- | ---: | ---: | --- |
| WHITE_green 灯 1 | 0／6 | 6／6 | 整图误差下界 4.434，风格投影上界 0.700 |
| WHITE_green 灯 8 | 6／6 | 0／6 | 整图误差 2.417，风格投影 1.123 |
| BLUE_red 灯 1 | 1／6 | 4／6 | 编号 19 色卡误差下界 10.186，投影上界 0.544 |
| BLUE_red 灯 8 | 4／6 | 1／6 | 编号 19 色卡误差下界 4.298 |

24 项检查中有 11 项取得不可达证明，均超出误差门槛，其中 8 项的投影上界也低于 0.75。BLUE_red 灯 1 整图和灯 8 编号 20 色卡的最小误差成片仍失败，但取整后的保守界无法排除其他比例。这两项保留为未证实。[完整分数与哈希](2026-11-21-beyond-edit-response-envelope.json)

## 对下一轮的影响

结束仅混合这两套现有编辑量的完整修复路线。WHITE_green 灯 8 表明两套结果在部分条件下确实互补。另一些条件下，即使位置选得完全正确，渲染出来的有效调色也太弱或离预设颜色太远。要验证新的目标侧算法，先看它能否补足这些缺失的颜色变化，再检验仅凭产品实际输入能否得到同样的调整。此次结果不覆盖真实两图自动识别、人工自然度或视频。

## 复现

沿用 [Beyond RGB 原对照](2026-11-20-beyond-rgb-matched-cast.md) 的成片和区域文件，选择一个新输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_beyond_edit_response_envelope.py outputs/adaptive-reference-20261113/beyond-edit-response-envelope-repro
~~~

两轮各保存 4 张成片、4 张对照和 1 份数值报告，9 份文件的 SHA-256 逐一一致。投影区间另用 32 个模拟 RGB 像素、4097 个等步长权重和 4096 组独立逐像素混合验证。本机输出位于 outputs/adaptive-reference-20261113/beyond-edit-response-envelope-v1 和同级 beyond-edit-response-envelope-recheck。
