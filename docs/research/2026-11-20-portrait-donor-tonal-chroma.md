# 用另一张照片的明暗分段迁移色度，仍有一边不足、一边过重

上一轮从目标自己的答案拟合偏移，五组人像都达到了局部色度误差门槛。这轮改用另一张照片估计偏移，检查是否能保留那个收益。《Cuckoo》和混合光人像用同一个 RawTherapee 色彩平衡开关，拍摄、用光和显影底色各自不同。每个方向只从供样照片的开关前后图取色度差，生成两张目标成片后才读该方向的目标期望图评分。两臂都从目标原图开始并锁定亮度。固定臂整图用一个偏移，分段臂按供样原图的三个明暗中心插值。[事前协议](../../experimental/adaptive-reference-20261113/portrait-donor-tonal-chroma-protocol.md)

| 目标及区域 | 固定误差 | 分段误差 | 分段投影 |
| --- | ---: | ---: | ---: |
| 《Cuckoo》整图 | 3.703 | 3.440 | 0.217 |
| 《Cuckoo》近中性色 | 1.767 | 1.483 | 0.455 |
| 《Cuckoo》脸颊 | 2.029 | 1.747 | 0.460 |
| 《Cuckoo》浅色树皮 | 3.196 | 2.911 | 0.324 |
| 《Cuckoo》绿色背景 | 9.509 | 9.221 | 0.150 |
| 混合光人像整图 | 3.519 | 1.609 | 1.208 |
| 混合光人像近中性色 | 2.698 | 0.746 | 1.317 |

误差单位为 RGB8。投影 1 表示沿预期改色方向的幅度相当，事前门槛为 0.75–1.25。四个必评项中只有混合光人像整图过线。该方向虽然总体更接近，近中性色仍过强。《Cuckoo》的三个登记区域均有改善，脸颊和浅树皮仍明显改色不足，绿色背景误差最高。两臂最大量化亮度漂移为 0.462 码值，最大色域回缩比例 4.92%。

从混合光人像拟合得到的亮度中心为 13.239、17.445、37.940，应用到《Cuckoo》时，95.35% 的像素落在中心范围外。反向的亮度中心为 71.791、113.799、151.463，98.32% 的目标像素在中心范围外。因此两个方向的大多数像素直接使用端点色度偏移。停止用供样的绝对亮度中心做跨场景插值。若检查相对明暗，先核对供样和目标的同类区域在各自画面中的亮度排序及所需改色方向是否一致。亮度排序是否具有可比性仍需证据。

这两张照片已用于开发。现有 13 号只收到目标与一张参考，这次两臂多获知供样的编辑前图，效果应在固定与分段之间比较。此次无产品改动，也未验证普通两图输入能自动获取这些偏移。[数值与哈希](2026-11-20-portrait-donor-tonal-chroma-evidence.json) · [《Cuckoo》四图对照](../../outputs/adaptive-reference-20261113/portrait-donor-tonal-chroma-v1/cuckoo-comparison.png) · [混合光人像四图对照](../../outputs/adaptive-reference-20261113/portrait-donor-tonal-chroma-v1/mixed-comparison.png)

## 复现

在仓库根目录运行下列命令。使用现有研究环境，末尾补入同版 Python 的本地 OpenCV 所在目录。脚本逐一检查来源哈希并复算原有评分。两次四张成片、两份并排图与结果 JSON 逐字节一致。

```powershell
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_portrait_donor_tonal_chroma.py outputs/adaptive-reference-20261113 outputs/adaptive-reference-20261113/rawtherapee-portrait-cross-v1 outputs/adaptive-reference-20261113/portrait-donor-tonal-chroma-replay
```
