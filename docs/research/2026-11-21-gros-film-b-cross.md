# 真人冷暖灯与 Film B 预设的交叉成片

为避免用相近的简单染色反复拟合，这轮引入已归档的厂商非线性预设。用 Gros 三位人物各自的冷灯、暖灯照片，在每张图上分别关闭和启用 ARRI Film B。两种灯光来自实际拍摄，Film B 是在已呈现的 RGB 图上借 SDR 逆变换施加的可复现编辑。这一设置没有把输入当成相机原始 LogC，也不代表摄影者为这些肖像所做的调色。[事前协议](../../experimental/adaptive-reference-20261113/gros-film-b-cross-protocol.md) · [原片和灯位](2026-11-20-gros-portrait-light-intake.md) · [官方预设归档](2026-11-19-arri-two-looks.md)

六张输入的无预设往返转换逐像素保持。开关预设的整图差为 11.929–12.439 RGB8，转换后无超色域裁切。每位人物的冷灯原片作目标，其余两人的冷／暖灯及开／关预设分别作单张参考。稳定版 1.3 和研究版 13 号均用 80% 强度。直接对目标应用同一预设得到离线期望。原有的整图、双颊和两处白幕均参与计分。模型只读取普通输入，没有读取预设、灯位标签或目标评分图。

## 成片

| 目标 ← 参考 | 稳定版整图风格投影，冷／暖 | 13 号整图风格投影，冷／暖 | 13 号整图换灯差，预设关／开 |
| --- | ---: | ---: | ---: |
| 01 ← 02 | 0.268／−0.190 | 0.447／0.735 | 2.912／4.283 |
| 01 ← 03 | 0.611／0.265 | 0.441／0.509 | 4.500／4.580 |
| 02 ← 01 | 0.911／0.686 | 0.673／0.592 | 5.560／4.612 |
| 02 ← 03 | −0.126／0.005 | 0.523／0.634 | 4.175／3.623 |
| 03 ← 01 | 0.877／0.874 | 0.507／0.595 | 5.533／5.045 |
| 03 ← 02 | 0.919／0.972 | 0.664／0.731 | 3.249／3.418 |

换灯差单位为 RGB8 MAE。风格投影门槛为 0.75 至 1.25，误差上限 3，换灯差和关闭预设时偏离目标原图的上限也为 3。稳定版、13 号分别完整通过 0／6。两套渲染器均没有一个区域同时通过整套条件。

以 02 为参考、01 为目标时，13 号右脸颊的预设响应投影随参考换灯从 0.90 变为 2.50，风格误差由 1.53 增至 8.08。左白幕却从迁移不足变为达到风格门槛，投影 0.47 到 0.78。这说明同一参考的拍摄灯变化可以在同一张成片上同时放大肤色过调、改善白幕迁移。24 张预设关闭态成片与此前同输入的冷／暖原结果逐字节一致。[完整分数、输入资格和文件哈希](2026-11-21-gros-film-b-cross.json)

## 用途与复现

这组开关可用于淘汰只会连灯色一起放大、或者把风格一起压掉的方案。三个人像已用于开发，应按原始人物组管理，预设变体也随原人物归组。该批没有新增训练、产品接入、正式独立评价或视频验收。

运行前已有两个输入依赖和对应的数值身份。首次启动缺少 OpenColorIO 导入目录，在任何成片生成前停止。复跑加入现有本机 OpenColorIO 2.5.2 目录，沿用同一协议和代码。使用[人像基础对照](2026-11-20-gros-portrait-light-transfer.md)和已归档的 Film B .cube，从仓库根目录选择尚未使用的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_film_b_cross.py outputs/adaptive-reference-20261113/gros-film-b-cross-repro
~~~

两次各生成 48 张成片、12 张参考状态、6 张目标期望、6 张对照及 1 份报告，73 个文件逐一核对 SHA-256 一致。媒体留本机 outputs/adaptive-reference-20261113/gros-film-b-cross-v2 与同级复跑目录。
