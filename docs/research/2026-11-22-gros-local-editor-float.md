# 浮点导出保留了暖光颜色差异，固定压缩损失编辑幅度

沿用六张实拍、原作者操作和原羽化区域。先用 darktable 4.0.1 导出 32 位浮点线性 Rec.709 和 Rec.2020。以第 2 包暖灯为例，此前 8 位成片新变零的 31,247 个蓝通道像素中，两种浮点导出转换到线性 Rec.709 后都有 **30,928 个负值**。Rec.709 数值第 5／95 百分位为 −0.00251／−0.00027，确认 8 位导出把有区别的值变成了同一个 0。两种输出的关闭态都转回原 RGB8，逐像素一致。[上一轮 8 位截断](2026-11-22-gros-local-editor-cross.md) · [事前协议](../../experimental/adaptive-reference-20261113/gros-local-editor-float-protocol.md) · [darktable 4.0.1 EXR 与色度标签源码](https://github.com/darktable-org/darktable/blob/release-4.0.1/src/imageio/format/exr.cc)

## 六张成片

用线性 Rec.709 的开关差再合成。直接截断作对照。另一处理保留编辑结果的线性亮度及相对中性轴的方向，对接近色域边界的色度用同一连续曲线压缩。低于可用色度 80% 的像素不做该压缩。区域外和白幕仍逐像素保留，关闭态也保持。

| 暖灯组 | 原 8 位新增端点 | 浮点直接截断 | 浮点色度压缩 | 压缩后双颊编辑投影 |
| --- | ---: | ---: | ---: | --- |
| 第 2 包 | 4.468% | 4.756% | 0.00029% | 0.472／0.417 |
| 第 3 包 | 1.942% | 2.020% | 0.00000% | 0.477／0.398 |
| 第 4 包 | 4.310% | 4.488% | 0.00071% | 0.519／0.417 |

投影 1 表示沿旧编辑变化方向的幅度一致，最低要求为 0.75。三组暖灯均未保住所需编辑幅度。第 3 包冷灯另外有一个像素的线性亮度达到 1.0200，超出 SDR 能保留的亮度。该像素在诊断成片用截断显示，整张仍判失败。其它冷灯双颊投影超过 0.75。三组均未冷暖联合通过，停止将这条曲线用于对照，无跨人物推理、训练或产品改动。[第 2 包暖灯成片](../../outputs/adaptive-reference-20261113/gros-local-editor-float-v2/2/warm/comparison.png) · [逐图证据](2026-11-22-gros-local-editor-float-evidence.json)

另作边界诊断。固定编辑后的线性亮度及颜色方向，只把超出 8 位码值 1–254 的像素拉到最近可用边界。三个暖灯组双颊投影都可达 0.847–0.990，但旧版蓝通道变零的像素全部变成同一个码值 1。调整端点统计可以掩盖细节仍被压平的问题，因此保留此读数作为诊断。后续判断真实画质需要同时验证调色幅度和通道细节；该固定强编辑不作为最终监督答案。

## 验证与复现

两遍各 18 张 PNG、6 张对照逐件同哈希。两份报告只在一张暖灯的最小亮度浮点值相差约 `7.96e-10`，所有成片、评分和准入结论一致。独立从落盘 PNG 复算新增端点。另复跑旧 8 位路径，报告与上一轮哈希完全一致。初次预检的 CLI 色彩意图缩写无效，正式预检沿用原 8 位实验的默认设置。首次汇总碰到亮度越界时中断；正式版保留其余成片并把该张标记为不合格，准入条件未变。

使用本机研究 Python 和已有 OpenCV。沿用上轮空闲 ASCII 盘符映射，从仓库根运行以下命令，新输出保留本机。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$env:OPENCV_IO_ENABLE_OPENEXR = '1'
$base = 'R:/outputs/adaptive-reference-20261113'
$launch = 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")'
& .tools/adaptive-env/Scripts/python.exe -c $launch experimental/adaptive-reference-20261113/probe_gros_local_editor_float.py $base "$base/gros-local-editor-float-new"
~~~
