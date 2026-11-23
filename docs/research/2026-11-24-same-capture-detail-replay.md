# 共同细节核验恢复同原片的局部调色

将同一张照片的原图和局部调色版本作为两个输入时，直接比较已对齐像素可以恢复调色。此前的几何方法已在另一位作者的单项色度编辑中可行，但固定机位换光也可能有相同几何结构。本轮比较正常 13 号、整数几何加 RGB 重放，以及增加共同细节核验的同款 RGB 重放。两个重放臂从目标和参考 RGB 自动选择。只在重叠处混合参考 RGB，强度沿用 80%，评分框、作者蒙版和效果答案均只用于离线计分。[事前协议](../../experimental/adaptive-reference-20261113/same-capture-detail-replay-protocol.md) · [上次几何方法](2026-11-20-playraw-mixed-light-geometric-replay.md) · [原基线的真实提亮成片](2026-11-24-playraw-flash-room-workbench.md)

| 作者和参考范围 | 原 13 号有意区域误差 | 核验后误差 | 开态支持格数 |
| --- | ---: | ---: | ---: |
| Martin Werner，完整参考 | 11.14 | 0.00 | 16／16 |
| Martin Werner，裁切参考 | 9.46 | 0.00 | 12／16 |
| Carlos，完整参考 | 12.79 | 0.00 | 16／16 |
| Carlos，裁切参考 | 10.68 | 0.00 | 15／16 |

误差为作者编辑区对 80% 预期成片的 RGB8 MAE。完整和裁切的四组均过线，关闭态及所有不重叠像素逐像素保持。Martin 的完整参考里，右颊由 13 号的 −1.29 恢复到预期 +10.18 个亮度码值，墙面与杯身的预选区域逐像素保持。Carlos 的高权重区域保住了真实的局部改色，零权重区的色度变化 P95 为 0。单次双端 JPEG 85、4:2:0 编码的对照也通过本组门槛，保留参考编码与编辑差。未经验证的缩放、不同压缩链和范围很大的编辑仍可能需要回退。[提亮前后对照](../../outputs/adaptive-reference-20261113/same-capture-detail-replay-v2/flash-full-comparison.png) · [局部改色对照](../../outputs/adaptive-reference-20261113/same-capture-detail-replay-v2/skin-crop-comparison.png)

12 次 Gros 同机位换灯、2 次 CHROMA 固定机位重拍、2 次白球服跨内容输入均拒绝直接重放，返回的 PNG 与本轮实际工作台基线逐字节相同。15 次在几何阶段已经被拦下。CHROMA 场景 1 通过整数几何检查，重叠区梯度相关达 0.9951。171,828 个有效相邻像素组中没有精确共享的组，判定应回退。仅按几何直接重放会把这一组浅色区域对原图的误差从 4.19 增至 10.52 RGB8，额外核验后回到 4.19。[此前同机位误触](2026-11-20-aligned-cubic-fine-residual.md) · [误触的成片](../../outputs/adaptive-reference-20261113/same-capture-detail-replay-v2/chart-Scene1_a-geometry.png) · [回退成片](../../outputs/adaptive-reference-20261113/same-capture-detail-replay-v2/chart-Scene1_a-guarded.png) · [数值与哈希](2026-11-24-same-capture-detail-replay-evidence.json)

本轮支持保留自动核验与局部重放作为隔离研究能力。四组正例来自两张原片，已进入开发。对于不同拍摄内容，核验没有提供作者该在什么物体上调多少的信息，仍沿用原 13 号输出，故不能计入跨场景画质通过。此处的真实换灯回退保住 13 号基线，基线 4.19 RGB8 的偏差本身仍超过 3 码值线。软件尚未接入本方法；跨场景色彩、独立人评、视频、工程回归和正式交付继续按原条件验证。重放依赖像素位置及重叠范围，单个普通 CUBE 无法完整表达。[开发通过和已知边界](2026-11-19-frozen-candidate-development-crosscheck.md)

## 复现与复算

首次启动在出片前因脚本固定的旧报告哈希漏抄两位而停止。原报告与上一阶段独立证据一致。将脚本的收据改成已核验的实际值，两遍用 56.01 秒和 56.70 秒完成。各遍 79 个指定 PNG（24 张正例成片、4 张图板、51 张负例成片）及报告逐项同哈希。独立以整数 80% 混合复算八组开关成片、裁切区外保持和原 RGB 预期，并核对重拍回退、两处保护区域和 JPEG 结果。[出图脚本](../../experimental/adaptive-reference-20261113/probe_same_capture_detail_replay.py)

~~~powershell
$env:PYTHONPATH = @((Get-Location).Path, (Resolve-Path -LiteralPath '.tools/tf-convert-env/Lib/site-packages').Path) -join [IO.Path]::PathSeparator
$env:OPENBLAS_NUM_THREADS = '2'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_same_capture_detail_replay.py 'outputs/adaptive-reference-20261113/same-capture-detail-replay-new'
~~~
