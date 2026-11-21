# VeraRetouch 的已知参考前图减轻底色漂移，脸颊风格仍有反向

CIFR 试验表明，同组真人的 Film B 可以在已知参考原片时用固定色表重放。本轮换成按编辑前后关系训练的 VeraRetouch，用同一权重比较作者的两图入口和带真实参考原片的诊断入口。两图把目标作为参考前图使用，诊断臂改用参考在同一灯下的真实原片。目标、参考后图、全通道条件、权重、编码尺寸和 80% 强度都固定。[事前协议](../../experimental/adaptive-reference-20261113/gros-vera-reference-pair-protocol.md) · [既有固定色表正对照](2026-11-21-gros-cifr-edit-replay.md) · [作者推理代码](https://github.com/OpenVeraTeam/VeraRetouch/blob/fefe6c4408f00beae07e132fc404cae0efee0985/infer_ref_retouch.py)

| 01 ← 02 成片 | 普通两图 | 已知参考前图 |
| --- | ---: | ---: |
| 整图关闭态误差，冷／暖 | 23.76／30.28 | 1.94／1.50 |
| 整图风格投影，冷／暖 | 0.618／0.509 | 0.662／0.771 |
| 整图风格误差，冷／暖 | 5.07／5.62 | 3.67／2.98 |
| 整图开态换灯差 | 4.04 | 2.20 |
| 五区联合通过 | 0／5 | 0／5 |

误差采用 RGB8 MAE。普通两图的未调色暖参考使目标左脸颊偏离原片 49.76 RGB8，两处白幕约 22.54–22.88。有真实参考前图时，同区左颊误差降至 4.36，两处白幕均约 0.91。暖灯整图风格误差随之达标，但左、右脸颊的风格投影为 −0.564 和 −0.441，与期望方向相反。同图控制的整图误差为 2.978，仍有颜色变化。保留所有成片，结束固定配置扩大筛查。[五区结果及文件哈希](2026-11-21-gros-vera-reference-pair-evidence.json)

对这一组，参考前图能显著缓解两图输入带来的底色偏差。已有固定三次回放用同组真实前图能通过五区，说明带前图时的脸颊反向也是当前 VeraRetouch 渲染路径的具体限制。它的条件解码器对整图逐像素使用同一映射，这份结果仅覆盖已知全图编辑，局部空间调色也没有在此配置获得证据。当前软件和独立验收不受影响。

## 复现

使用[作者发布固定版](https://huggingface.co/Gyh68/VeraRetouch.Encoder_Renderer/tree/9b50d4c8241a3d811f4cccddf6024a12498c7a97)的 182,507,490 字节权重。只张量读取并严格加载 182 项。沿用作者目标 RGB、参考 BGR 的输入编码，参考在 Pillow 中双线性缩为 512×512，目标保持原尺寸。使用作者分块选项 262144，之后采用项目已有 80% 混合。隔离运行时按语法树加载这一配置实际使用的未修改类和配置函数，跳过其他可选后端的导入，配置容器用标准属性对象。源码哈希和路径见证据，作者权重仅限学术研究，不上传源码副本、权重或媒体。[作者许可证](https://github.com/OpenVeraTeam/VeraRetouch/blob/fefe6c4408f00beae07e132fc404cae0efee0985/LICENSE)

把证据所列同名源文件及 `encoder_renderer.pth` 放在本机 `outputs/adaptive-reference-20261113/vera-intake-v1`，复用[已有 Gros 和环境](2026-11-21-gros-film-b-cross.md)。两次的 8 张跨图、1 张同图、2 张对照板及 1 份报告逐项哈希一致。另从 PNG 独立复算 10 个范围的旧色误差、开关响应和换灯差。同图对照也独立复算。本机成片在 `outputs/adaptive-reference-20261113/gros-vera-reference-pair-v1`。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_vera_reference_pair.py outputs/adaptive-reference-20261113/gros-vera-reference-pair-repro
~~~
