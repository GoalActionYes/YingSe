# 固定线性通道比例未能重放真实局部暖调

《Cuckoo》的脸颊和胸口先前用固定色差重放时偏强，统一减弱又会损害原片。这轮保留区域、强度和原亮度，改用红、蓝相对绿色的线性通道比例。比例从真人作者开关图估计，按冻结的棋盘分成两折。跨图用两折平均值，未读取目标成片来定参数。登记的目标区域仅模拟已知位置，作者蒙版用于离线估计。[事前协议](../../experimental/adaptive-reference-20261113/playraw-relative-channel-edit-protocol.md) · [此前统一减弱反例](2026-11-20-playraw-cross-editor-gain.md)

| 区域 | 固定色差 MAE | 通道比例 MAE | 通道比例响应投影 |
| --- | ---: | ---: | ---: |
| 《Cuckoo》脸颊 | 3.094 | 3.621 | 0.834 |
| 《Cuckoo》胸口 | 4.502 | 7.365 | 0.633 |
| 供样原片高权重区，留出折 0 | 1.582 | 3.860 | 1.056 |
| 供样原片高权重区，留出折 1 | 1.623 | 3.905 | 1.055 |

MAE 单位为 RGB8，门槛为 3。投影 1 表示实际变化在期望颜色方向上的幅度相等，范围为 0.75–1.25。两份旧编辑器缓存的登记区分数一致。胸口投影跌到 0.633，误差增加，目标两处都没有发生色域压缩，不能把退化归因于目标通道裁切。供样两折的羽化区误差有小幅下降（约 0.55 → 0.44），高权重区却超线。四张候选在区域外逐像素保持，Y′ 最大量化差不超过 0.478 RGB8。[本机成片对照](../../outputs/adaptive-reference-20261113/playraw-relative-channel-edit-v1/target-0-comparison.png) · [逐项数值和哈希](2026-11-20-playraw-relative-channel-edit-evidence.json)

结束固定线性通道比例。这两张真实照片仍属开发素材，而且借用了编辑范围。后续的强度证据需要来自更具代表性的真实同款操作，例如核验额外编辑记录能否在不同原片上提供比像素近似更可靠的响应。普通两图的肤色和近中性色改善、独立自然评价、完整视频验证均未通过产品验收。稳定版和手动调色未修改。

两次各四张成片、两张对照图和结果逐字节一致。保留父实验编辑器关态的复现限制。本轮不重新导出编辑器，只读取冻结缓存。报告 SHA-256 为 `3414e23cc8f93200933a39bf5b7b5d09fdcf33809e02dfa3a576877ea14ebc5b`。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_playraw_relative_channel_edit.py "$base/playraw-mixed-light-cross-editor-v2" "$base/playraw-relative-channel-edit-replay"
```
