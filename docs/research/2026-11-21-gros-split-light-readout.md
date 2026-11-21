# 人脸照明分量能跟随换灯，原片重建仍有失败

上轮用 SPLiT 表面色直接替换参考肤色时，部分区域少受换灯影响，另一些区域丢了 Film B。此次用作者的第二级模型单独估计漫反射照明和镜面反光，检查它们是否能把现场灯光和后期调色区分开。输入沿用三位真人、512 方形裁切和离线近似脸区。读取真实灯态及编辑状态只用于对照测量。[事前协议](../../experimental/adaptive-reference-20261113/gros-split-light-readout-protocol.md) · [上一轮成片](2026-11-21-gros-split-face-albedo.md) · [SPLiT 作者实现](https://github.com/costrice/split/tree/b2b559831bb65e54b820fd1d6dd7c0b3a99e8b65)

| 人物 | 四张脸部重建误差范围（RGB8） | 原图最大编辑／最小换灯 | 漫反射照明 | 镜面反光 | 通过完整门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| 01 | 5.18–6.34 | 0.705 | 0.324 | 0.376 | 两分量均未通过 |
| 02 | 2.89–4.51 | 0.377 | 0.292 | 0.315 | 两分量均通过 |
| 03 | 2.68–4.38 | 0.261 | 0.194 | 0.269 | 两分量均通过 |

比值越小，说明该分量读到的有意编辑相对于实拍换灯越少。两种预测分量的实拍换灯方向与原图色比方向余弦为 0.906–0.971，换灯向量长度为 0.540–0.802。两类分量在三人均符合方向、长度和编辑比门槛，但第一位的四张重建均超过 5 RGB8，事前要求三位全部通过，故两类分量都未获准作为后续调色输入。第二、第三位的原图色比也满足该比值阈值，模型优势主要体现在减小相对编辑干扰，不能将比值达标直接视作新增照明识别能力。第一位的照明方向读数仍保留，重建误差的原因尚未分清。[逐项分数及证据](2026-11-21-gros-split-light-readout-evidence.json)

结束当前裁切和掩膜下把这两种分量直接用于修光。保留两种分量的预览及全部反例，供后续核实重建误差的来源。此次未渲染新的目标调色成片，未采用作者第三层球面灯光估计，也未接入软件。已有三位开发真人与手工眼位不足以说明独立自然度和视频效果。

## 复现

沿用[第一阶段源文件、权重和人像环境](2026-11-21-gros-split-face-albedo.md)，新增同一作者 ZIP 的 `model_files/intrinsic_decom_f_DS.pth`。字节区间为 360858472–503087060，返回 HTTP 206 并核对 Content-Range。根据 ZIP 目录解压成员，尺寸 148,874,702，CRC32 为 `58a11af4`，SHA-256 见证据。两级各 93 项严格匹配，通过仅张量方式读取。新增权重和分解预览保留本机。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_split_light_readout.py outputs/adaptive-reference-20261113/gros-split-light-readout-repro
~~~

两遍的 12 张分解对照板和 1 份报告逐一通过 SHA-256 复核。另由保存的线性 RGB 中位数独立复算三人各分量的色比、向量余弦和资格判定。第一次的结果在本机 `outputs/adaptive-reference-20261113/gros-split-light-readout-v1`。
