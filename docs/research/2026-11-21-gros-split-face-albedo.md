# 人脸表面色估计降低参考换灯差，成片仍未保住全部风格

DINOLight 的整图照明恢复会连同有效风格一起改变。这轮选择在人脸和光照数据上训练的 SPLiT 第一阶段，检验它估计的漫反射表面色是否提供不同的线索。从已登记眼位裁切人脸，用固定椭圆近似皮肤范围，在保留输入明暗的前提下，仅替换该区域的色比。三位真人各有冷暖灯和 Film B 开关四张参考。第一组跨人物成片固定用 02 作参考、01 作目标，继续使用 13 号与 80% 强度。人工眼位只用于离线验证，脸颊和白幕只参与评分。[事前协议](../../experimental/adaptive-reference-20261113/gros-split-face-albedo-protocol.md) · [上轮整图照明边界](2026-11-21-gros-dinolight-spatial-residual.md) · [SPLiT 作者实现](https://github.com/costrice/split/tree/b2b559831bb65e54b820fd1d6dd7c0b3a99e8b65)

按双颊、三位人物和开关态统计，12 项参考换灯差中 10 项降低不少于 20%。同理，对双颊及冷暖灯分别计数，12 项参考 Film B 开关差中仅 7 项保留至少原幅度的 75%。三人的右颊同时满足两类检查，左颊保留了主要反例。02 冷灯左颊的开关差由 5.72 降至 2.34 RGB8，暖灯由 4.72 降至 2.41。换灯差用两张参考脸颊的逐通道中位数比较，以降低头部移动影响。预设开关差用同张照片的前后像素计算。[完整结果及文件哈希](2026-11-21-gros-split-face-albedo-evidence.json)

| 01 ← 02 成片检查 | 13 号原成片 | SPLiT 表面色参考 |
| --- | ---: | ---: |
| 整图关态误差，冷／暖 | 3.60／5.72 | 3.66／5.44 |
| 整图风格投影，冷／暖 | 0.447／0.735 | 0.440／0.722 |
| 整图开态换灯差 | 4.28 | 4.05 |
| 右颊关态误差，冷／暖 | 4.36／6.25 | 5.64／4.03 |
| 右颊风格投影，冷／暖 | 0.900／2.501 | 0.911／2.495 |
| 五区联合通过 | 0／5 | 0／5 |

除投影以外，单位均为 RGB8 MAE。暖灯右颊的底色误差减轻，冷灯对应区域反而退化，暖灯风格过调基本不变。参考白幕保持原色，目标两处白幕仍会因整图参考变化而响应。固定方案未满足事前门槛，结束直接以人脸表面色替换参考肤色。只验证了第一阶段及手工眼位加椭圆的适配，可能受光照、姿态、遮罩和模型泛化共同制约，尚不能归为单一原因，也没有验证作者完整的光照估计流程。软件和独立验收素材均未纳入改动。

## 复现

模型来自[作者公开归档](https://drive.google.com/file/d/1dAIH7VFnlxWIXn3NXl_L5O76m8g70jTr/view)，总长 776,488,973 字节。只取末尾目录及成员 `model_files/intrinsic_decom_f_AN.pth`，成员压缩段及文件头为字节 223593701–360858471。解压后 148,864,334 字节，CRC32 为 `328f7536`，核对完全匹配。采用 PyTorch 仅张量模式严格加载 93 项。另取固定提交的 `models/intrinsic_networks.py`、`models/blurpool.py` 与 Apache-2.0 许可。作者权重限非商业使用。源码、权重和媒体在本机保留。

将解压出来的权重和 `models` 目录按原名放入 `outputs/adaptive-reference-20261113/split-intake-v1`，用[已有真人成片和环境](2026-11-21-gros-film-b-cross.md)运行。两遍的 12 张脸部校正参考、4 张目标成片、2 张比较板和 1 份报告逐件 SHA-256 一致。另从实际 PNG 复核裁切范围外逐像素不变、参考双颊开关与换灯指标以及五区目标分数。结果在本机 `outputs/adaptive-reference-20261113/gros-split-face-albedo-v1`。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_split_face_albedo.py outputs/adaptive-reference-20261113/gros-split-face-albedo-repro
~~~
