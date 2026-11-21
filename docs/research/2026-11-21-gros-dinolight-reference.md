# DINOLight 能修正实拍强色灯，也削弱了真人参考的有效风格

此前的白平衡与眼部取样只能提供局部照明的间接线索。DINOLight 用色灯实拍与环境光对照训练，并以冻结的 DINOv2 中间层提供空间信息。本轮先检验它能否实际恢复照明，再把处理后的单张真人参考交给原 13 号，检查是否会同时消掉应迁移的 Film B。目标图保持原尺寸，两臂都使用 672×448 的参考与 80% 强度，差别只有是否预处理参考。所有图来自已经用过的开发材料。[事前协议](../../experimental/adaptive-reference-20261113/gros-dinolight-reference-protocol.md) · [DINOLight 作者实现](https://github.com/OBAKSA/Lighting-Restoration/tree/2d705f46cd887eaee702dbfee85c9ed926bc0af0) · [DINOv2 作者实现](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8)

| 检查 | 原图或直接缩放参考 | DINOLight |
| --- | ---: | ---: |
| CL3AN 一号色灯到环境光整图误差 | 107.30 | 21.43 |
| CL3AN 二号色灯到环境光整图误差 | 76.66 | 21.88 |
| 真人参考 Film B 开关差，冷／暖 | 12.20／12.42 | 2.98／2.84 |
| 13 号目标整图开态换灯差 | 4.26 | 0.53 |
| 13 号目标整图关闭态误差，冷／暖 | 3.60／5.72 | 7.23／7.02 |
| 13 号目标整图风格投影，冷／暖 | 0.445／0.732 | −0.032／0.002 |
| 五区联合通过 | 0／5 | 0／5 |

除投影以外，数值单位为 RGB8 MAE。两个强色灯场景误差各下降约 80% 和 71%，通过事前的 20% 正对照条件。场景来自作者公开测试目录，用于确认本地适配可实际修光，尚不能据此证明作者训练素材与本批照片完全隔离。真人参考的 Film B 差异缩小约四分之三。目标右颊关闭态误差从 4.32／6.26 升至 13.10／13.01，开关差在两种灯下还与期望方向反向。目标风格投影接近零，决定按事前要求结束将整张 DINOLight 恢复图直接作为参考。[完整评分与文件身份](2026-11-21-gros-dinolight-reference-evidence.json)

此次分开验证了照明恢复和有效调色。能够恢复实拍色灯，并不能保证同一变换会保留创意调色。后续若利用这类模型，需要先验证它从原图移除的内容能否用于区分真实光和后期调色。当前只运行了一组真人方向，未修改软件，也未启动独立画质、自然度或视频验收。

## 复现与资源

模型均以 PyTorch 仅张量模式读取。CL3AN 恢复与 DINOv2 主干严格加载 795 和 176 项，分别为 38,351,150 与 346,393,545 字节。源码及权重哈希见证据，DINOLight 源文件另与 Git blob 身份核对。上游源码和权重按作者公开推理说明在本机使用，均未随本项目上传。Meta 源码许可随原归档留存。DINOLight 仓库未明确给出再分发许可，因此也未接入发行包。[DINOLight 公开权重](https://drive.google.com/file/d/1XZg_qKbo3QUAIASx6LTKpwmft2GiTJ3F/view) · [Meta 官方模型](https://github.com/facebookresearch/dinov2#pretrained-models-via-pytorch-hub)

两次正式运行各产出 2 张色灯恢复、4 张真人参考恢复、8 张目标成片、2 张对照板和 1 份报告，17 个文件的 SHA-256 逐一一致。另从 PNG 复查两组实拍误差、两组开关参考差及 10 个分区。推理峰值张量显存为 1,103,826,432 字节，不含驱动与缓存池。第一次启动补了现有 SciPy 路径，整图宽度不符合作者内部分块后改用双 448 方块，前两次均没有产出有效成片。正式成片保存在本机 `outputs/adaptive-reference-20261113/gros-dinolight-reference-v2`。

从仓库根目录运行。权重和源码位于同一实验父目录下的 `dinolight-intake-v1`，文件名为 `dinolight.pth`、`dinov2.pth`、`dinov2.zip` 和 `model_DINOLight_amp.py`，Meta ZIP 需在该处解压到原提交命名目录。已有 CL3AN 和 Gros 路径按[实拍记录](2026-11-21-cl3an-pointwise-bound.md)和[冻结人像归档](2026-11-21-gros-film-b-cross.md)准备。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$env:XFORMERS_DISABLED = '1'
& .tools/vision-advisor-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/ModFlows-audit/vendor"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_dinolight_reference.py outputs/adaptive-reference-20261113/gros-dinolight-reference-repro
~~~
