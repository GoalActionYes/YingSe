# 滤镜反演差未能在真人冷暖灯下可靠识别编辑

上轮发现从单张参考猜原片会误改未编辑照片。本轮将 CIFR 恢复前后的有符号差限定为识别输入，检验它能否给固定的 Film A／B 已知款式提供信息。基线还是使用直接可见 RGB，候选读取反演差。两臂都用 256 方形、33 项相同的亮度和反色统计、四来源的 72 个训练状态、54 项同款拍摄差分，以及同样的固定读出与 0.65 触发门槛。三位 Gros 真人从整个拟合过程留出。Film A／B 只用作离线已知款式的渲染和监督。[事前协议](../../experimental/adaptive-reference-20261113/gros-cifr-residual-readout-protocol.md) · [已有四来源训练](2026-11-19-capture-invariant-look.md) · [上轮原片估计](2026-11-21-gros-cifr-edit-replay.md)

| 留出真人输入 | 直接参考识别正确 | CIFR 反演差识别正确 |
| --- | ---: | ---: |
| 未编辑 | 6／6 | 6／6 |
| Film A | 0／6 | 0／6 |
| Film B | 0／6 | 0／6 |
| 六个方向的完整 Film B 成片 | 0／6 | 0／6 |

两臂全部回退到未编辑，48 张 PNG 均保持目标原片。六个方向的每项 Film B 投影为零。全部测试分数最大分别为 0.494 和 0.423。补充只看最高分名次时，直接参考的未编辑、Film A 和 Film B 正确数分别为 0、2、4，反演差为 0、4、3。这个名次检查只读已有数值，不重新跑成片或放宽门槛。两臂的未编辑照片没有一张能凭最高分认出，支持不再靠调整触发阈值推进这套识别。[逐参考分数、成片和验证](2026-11-21-gros-cifr-residual-readout-evidence.json)

原样回退让未编辑输入通过保存检查，所有有意风格也被漏掉。按预设结束此固定反演差和线性读出。仅测试三位已用真人和两个已知预设，不能据此排除其他反演监督或开放风格方法。没有软件行为变更，也没有进行独立评价或视频测试。

## 复现

沿用[CIFR 固定权重及视觉运行环境](2026-11-21-gros-cifr-edit-replay.md)，还需[既有四来源](2026-11-19-capture-invariant-look.md)和[固定人像参考](2026-11-21-gros-film-b-cross.md)。指定新的本机输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_cifr_residual_readout.py outputs/adaptive-reference-20261113/gros-cifr-residual-readout-repro
~~~

本机成片和原始报告位于 `outputs/adaptive-reference-20261113/gros-cifr-residual-readout-v1`。两次运行的 48 张成片与 1 份报告逐件 SHA-256 一致。另从落盘文件核对全部原片恒等，从保存的分数独立复算 36 项触发判定及最高分名次。模型、参考图和成片仅留在本机。
