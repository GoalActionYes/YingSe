# AceTone 色表编解码器在已知 Film B 映射下通过六张真人对照

AceTone 的公开方案用目标图和参考图预测 64 个码本索引，再解码为全图颜色表。本轮仅使用作者的编解码器，直接输入正确的身份表和已知 Film B 色表，先判断固定编码是否会丢失实际需要的风格。这样可以在下载完整模型前检查其渲染表示的基本可用性。[作者实现](https://github.com/martian422/AceTone/tree/916393b3f26bdf89c3d939cc5f2a9a3c115ccbc5) · [事前协议](../../experimental/adaptive-reference-20261113/acetone-vq-film-b-protocol.md)

两臂使用同一张 32³ 色表、三线性采样、80% 强度和 RGB8 量化。三位实拍人物均保留冷、暖灯照片，分别量整图、两侧脸颊与两侧白幕。未经压缩的对照逐像素保持关闭态，最大区域风格响应误差 0.220 RGB8，满足对照准入条件。作者标准编码后，六张照片 30 个整图及区域检查全部通过。关闭态最大区域误差为 1.250 RGB8，开关响应误差最高 2.828 RGB8，响应投影 0.783–1.093。有意风格开启成片相对已知成片的最大绝对误差为 3.024 RGB8。该值衡量开启成片整体误差，与扣除关闭态偏移后的风格响应误差不同。[数值摘要及成片哈希](2026-11-21-acetone-vq-film-b-evidence.json)

这项结果支持继续检查公开预览权重的两图预测。正确色表来自离线已知变换，不能判断完整模型能否预测它，也不评估参考换灯稳定性。全局颜色表仍不能给目标中两处相同 RGB 分配不同颜色。无需复制整个研究环境或替换现有软件。

模型卡将预览权重标为 Apache-2.0。核对的固定版本为 `9416b259dec1bf7bb01ba2dab38cf6470cfa98f0`，接口无需账号，约 8.15 GB 发布目录采用 BF16。配置为作者自定义 AceToneVLM，现有 RTX 4060 Laptop 总显存约 8 GiB，工作时空闲约 6.4–6.9 GiB。作者脚本使用 FlashAttention 2，当前视觉环境没有该依赖，也未装 bitsandbytes。下一步先用不含模型权重的配置和小尺寸参数检查作者架构能否在现有 PyTorch 上执行并保留自定义码本头，再决定整模型下载或量化。已下载的只是 21,383,912 字节的编解码器权重及少量源码和配置。[公开 BF16 权重与许可](https://huggingface.co/Vivre/AceTone-3B-PST-Preview/tree/9416b259dec1bf7bb01ba2dab38cf6470cfa98f0) · [作者评价入口](https://github.com/martian422/AceTone/blob/916393b3f26bdf89c3d939cc5f2a9a3c115ccbc5/eval/predict_lut_ddp.py)

## 复现

在本机 `outputs/adaptive-reference-20261113/acetone-vq-film-b-assets` 保存指定版本的 `model/vq.py` 与 `model/acetone-vqvae-d64.pt`，文件哈希由脚本核对。沿用 [Film B 交叉对照](2026-11-21-gros-film-b-cross.md)的原图和显影环境。从仓库根目录运行以下命令，新输出目录须不存在。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$base = 'outputs/adaptive-reference-20261113'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_acetone_vq_film_b.py "$base/acetone-vq-film-b-assets" "$base/acetone-vq-film-b-repro"
~~~

两次各生成 24 张成片和一份报告，全部 25 个文件逐件 SHA-256 一致。报告 SHA-256 为 `50eb9d746c153781d32850e6fd407634f806ec93b00adc6b3158a5f1a9ce5322`。实际成片位于本机 `acetone-vq-film-b-v1` 和同级复跑目录。
