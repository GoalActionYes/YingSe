# 保持现有色度方向，调整力度可恢复多数偏差

上轮在只染色的受控案例中恢复了目标原亮度。两套渲染器各多通过 5 项，但白幕和浅色卡的迁移仍不足。这轮将颜色方向和力度分开，希望确定渲染端是否需要产生新的颜色方向。

沿用六个开发状态、四个原始构图及 33 个既有区域。以目标原图为起点，从上轮的锁亮度成片取线性 RGB 变化，再去掉取整留下的微量亮度。每个像素只能沿原模型的颜色变化方向调整，力度可以超过原幅度，最大值由合法色域限制。用已知成片求线性 RGB 三通道绝对误差最小的力度，然后编码到 RGB8。另做保持亮度、允许改变颜色方向的参照。两种做法都读取正确成片且沿用理想的无编辑关态，只用于离线检验可达性。[事前协议](../../experimental/adaptive-reference-20261113/target-chroma-direction-protocol.md)

## 实际成片

| 目标状态 | 稳定版原力度 | 稳定版自由力度 | 13 号原力度 | 13 号自由力度 | 方向也可变 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MIT 客厅 | 2／4 | 4／4 | 1／4 | 4／4 | 4／4 |
| Gros 人像 | 5／5 | 5／5 | 2／5 | 5／5 | 5／5 |
| WHITE_green 灯 1 | 2／6 | 6／6 | 0／6 | 6／6 | 6／6 |
| WHITE_green 灯 8 | 1／6 | 6／6 | 3／6 | 6／6 | 6／6 |
| BLUE_red 灯 1 | 1／6 | 5／6 | 1／6 | 6／6 | 6／6 |
| BLUE_red 灯 8 | 2／6 | 6／6 | 5／6 | 6／6 | 6／6 |
| 合计 | 13／33 | 32／33 | 12／33 | 33／33 | 33／33 |

沿用 RGB8 平均绝对误差 ≤3、风格投影 0.75 至 1.25。13 号白幕左右误差由 5.325、5.660 降至 1.111、1.020。WHITE_green 灯 1 编号 19 色卡由 6.570 降至 0.539。稳定版唯一未通过的是 BLUE_red 灯 1 编号 19 色卡，11.469 降至 4.428，误差保守下界为 0.854。这个下界没有超过 3，无法证明稳定版该方向上的其他力度一定失败。该色卡用 13 号原方向调力度后通过，误差为 2.329。[完整分数、下界和哈希](2026-11-21-target-chroma-direction.json)

## 决策

有限混合范围所排除的反例，在允许按现有方向调节更大幅度后，部分可以恢复。这批测试的 13 号结果给出一个可行成片，说明无需改变当前方向也能满足这些指定区域的门槛。该结论只适用于已用开发照片和当前声明为只染色的操作。逐像素幅度可随内容变化，其中用到了正确成片。下一项应验证能否从实际输入估计足够的力度，以及同一估计遇到实拍换灯、同图或普通明暗编辑会不会误改。无需先训练一个新的颜色方向模型。普通两图自动画质、独立自然度和视频验收仍待完成。

## 复现

沿用[原亮度约束对照](2026-11-21-target-linear-luma-edit.md)及其上游文件，从仓库根目录选择尚未占用的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_target_chroma_direction.py outputs/adaptive-reference-20261113/target-chroma-direction-repro
~~~

两轮各生成 18 张成片、6 张对照和 1 份数值报告，25 个文件逐一核对 SHA-256 一致。另用 64 组模拟 RGB 颜色方向、每组 4097 个力度校验最小误差和固定亮度，包括零变化和色域边缘。本机输出在 outputs/adaptive-reference-20261113/target-chroma-direction-v1 和同级 target-chroma-direction-recheck。
