# 模拟 JPEG 改善拟合校验，未新增有效重放

上一轮把后级细节比较范围扩大到 8×8，只恢复了一个双端 JPEG 正例。本轮固定三次色表拟合与后级细节判断，只比较三种验证预测：缩图上直接应用三次色表、在原尺寸应用后再缩图、在原尺寸应用后模拟同样的 JPEG 编码再缩图。后两者使用同样的照片尺寸与缩图方式，可以单独度量模拟编码的影响。沿用各条件 47 个重拍方向和 14 个同原片对，八个无损已成功正例固定计数。[事前协议](../../experimental/adaptive-reference-20261113/aligned-cubic-jpeg-forward-protocol.md)

| 条件 | 固定正例通过前置拟合，旧／原尺寸／模拟 JPEG | 重拍通过前置拟合，同顺序 | 最终有效正例，同顺序 |
| --- | ---: | ---: | ---: |
| 只压缩参考 | 2／2／4 | 0／0／3 | 0／0／0 |
| 两端都压缩 | 3／3／3 | 4／4／2 | 1／1／1 |

只压缩参考时，Scene1_a 白灯 213 的原尺寸验证误差经 JPEG 从 1.640 降到 1.214 RGB8，蓝灯 213 从 1.959 降到约 1.500。两者都通过前置拟合，后级细节相关性分别为 0.569、0.618，低于 0.65。三组前置放行的重拍也都被后级拦下，最终回到原 13 号。两端都压缩时，八个固定正例中，编码前后的原尺寸验证 MAE 全部上升，最终仍只有上轮同一组白灯例过线。各条件及各臂均无新增最终重拍误触。[只压缩参考的新增前置正例](../../outputs/adaptive-reference-20261113/aligned-cubic-jpeg-forward-v1/reference_only/CHROMA-Scene1_a-blue-213-comparison.png) · [通过前置后回退的重拍](../../outputs/adaptive-reference-20261113/aligned-cubic-jpeg-forward-v1/reference_only/CHROMA-Scene1_a-white-241-to-anchor-comparison.png)

编码路径会影响前置误差，也会改变正负例进入后级的机会。在这组材料中，较小的拟合误差未增加最终正确重放数。两种编码都低于六例门槛，结束固定 JPEG 模拟校验及单独放大细节尺度。保留原无损开发结果。此试验仅稳健性检查，未使用实际相机或社交平台的未知 JPEG 设置，也未给出跨场景偏色修复或产品接入结果。[逐对数值、路线及哈希](2026-11-20-aligned-cubic-jpeg-forward-evidence.json)

两遍各 16 张实际候选输出、16 张对照图和报告逐字节一致。候选输出中包含按原规则回退的 13 号成片。全部旧 JPEG 前置决策及固定正例得分通过名单均复现。使用已有研究环境，在仓库根目录运行，输出目录须未占用。

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '4'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_aligned_cubic_jpeg_forward.py outputs/adaptive-reference-20261113/aligned-cubic-jpeg-forward-replay
```
