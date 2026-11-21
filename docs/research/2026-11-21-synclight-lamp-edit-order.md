# 已知实拍分灯差可还原参考，13 号的迁移仍不足

这是一个尽量有利的照明校正对照。SyncLight 环境光与单灯分量均来自同机位 RAW。冷灯由对单灯分量施加已固定的通道系数形成。复用原环境光倍数、曝光和白平衡，标准灯与冷灯参考逐像素复现此前的成片。选择两种灯光的照片，再分别施加同一 Film B。原始分量让我们知道每个像素的冷灯差，先校正再调色的标准灯成片是对照。实测比较在已调色参考上扣除相同物理差的两种方式，随后把四类参考分别输入原 13 号。[事前协议](../../experimental/adaptive-reference-20261113/synclight-lamp-edit-order-protocol.md) · [分灯原片及条件](2026-11-18-synclight-ambient-balance.md)

| 房间 | 参考整图误差，未校／码值校正／线性校正 | 参考强受光区误差，未校／码值校正／线性校正 | 目标开风格换灯差，未校／码值校正／线性校正 | 标准灯参考的目标风格投影 |
| --- | ---: | ---: | ---: | ---: |
| 走廊 scene35 | 4.321／0.718／1.026 | 6.490／1.154／1.550 | 3.132／0.665／0.704 | 0.487 |
| 床头 scene25 | 3.415／0.594／0.750 | 7.753／0.970／1.578 | 2.077／0.231／0.378 | 0.475 |

误差单位为 RGB8。参考误差相对正确的标准灯 Film B 成片计算，使用两种灯和两种风格共同不过暗、不过曝的像素。有效整图覆盖在两个参考上分别为 86.87% 和 94.63%，床头参考的弱受光有效覆盖为 4.30%，未达到 5% 资格，保留描述值。空风格控制中，码值扣光在两张参考上逐像素恢复，线性扣光有效整图误差为 0.166 和 0.163。两种方法都减少了参考偏色，且没有把目标换灯差留在原本水平。[分区评分及文件哈希](2026-11-21-synclight-lamp-edit-order-evidence.json)

即使直接把正确的标准灯参考交给 13 号，两个目标整图也分别只保留约 49% 和 47% 的既定 Film B 响应。对应的空风格输出还偏离目标原片 10.34 和 4.53 RGB8。按既有误差和响应门槛，全部最终输出仍没有完整通过。该结果支持在本对照中把照明校正与参考编辑估计区别看待。只改善上游照明不足以获得完整画质。可以保留分灯校正作为查原因的正对照，下一步应先验证标准灯下的风格推断，再复核换灯后是否同时改善。两房间同场不同视角，未涵盖肤色或跨场景自然评价。原始分灯差没有作为产品输入功能。

## 复现

沿用本机[分灯原片与解码目录](2026-11-18-synclight-real-pixel.md)、[原重建成片](2026-11-18-synclight-ambient-balance.md)和[Film B 库](2026-11-21-gros-film-b-cross.md)。从仓库根目录运行，指定未占用的输出目录。内部评分将标准灯对照放在 `cold` 键，将被比较的参考放在 `warm` 键，以复用原评分函数。表中物理状态按本实验重新命名。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_synclight_lamp_edit_order.py "$base/synclight-real-pixel-input/decoder" "$base/synclight-real-pixel-input/raw" "$base/synclight-lamp-edit-order-repro"
~~~

两次各 16 张参考、16 张目标成片和一份报告，全部文件 SHA-256 一致。完整报告哈希为 `2f9e6cda822c3811e087548a636f98133523e1bc3c280df5fbf9df70e2bbc7a5`。原片、权重和成片保留本机。
