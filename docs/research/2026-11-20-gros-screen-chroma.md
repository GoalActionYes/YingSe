# 已知白幕减轻真人串色，仍漏掉只暖脸的编辑

上一轮在同快门参考下，13 号仍把现场换灯带来的颜色变化传到目标脸颊。本轮将目标和参考中两处已登记的白幕用作同类材料，估计两者的色度差，再以固定 80% 作用于目标原图。运行只读当前两图与材料标注。真实换灯、整图数字暖调沿用原对照，并新增同一数字暖调只作用于脸部的合成参考。其双颊权重中位数均为 1，白幕与原参考逐像素一致。[事前协议](../../experimental/adaptive-reference-20261113/gros-screen-chroma-protocol.md) · [原真人基线](2026-11-20-gros-portrait-light-transfer.md)

| 同快门 02 作参考 | 原 13 号 | 用白幕估计全图色度 |
| --- | ---: | ---: |
| 01 双颊的暖／冷实拍参考输出差 | 9.33–10.58 | 1.50–2.01 |
| 01 整图数字暖调响应误差 | 7.03 | 0.11 |
| 03 双颊的暖／冷实拍参考输出差 | 5.74–6.94 | 1.34–1.62 |
| 03 整图数字暖调响应误差 | 7.50 | 0.39 |

误差为 RGB8 MAE。六方向的候选双颊换灯变化在 1.34–2.75，均低于 3；整图有意暖调的实际投影为 0.985–1.018，误差为 0.04–0.44。把两处白幕及基底同时计入后，全局联合只有 2／6，原 13 号为 0／6。01 从 02 的左白幕为 3.00009，仍按预先的 3 码值门槛记失败。另有三方向的白幕误差达到 3.33–3.67。六个方向只暖脸时，白幕估计的成片都与对应关闭态逐像素一致，双颊风格投影为零。13 号能传一部分暖调，双颊投影 0.160–0.365，也未通过。两臂的总联合均为 0／6。[01 从 02 完整对照](../../outputs/adaptive-reference-20261113/gros-screen-chroma-v1/01-from-02/comparison.png) · [03 从 02 完整对照](../../outputs/adaptive-reference-20261113/gros-screen-chroma-v1/03-from-02/comparison.png)

同一白幕提供了估计全图色度的实测收益，仍无法说明脸部局部颜色是灯光还是编辑。结束用白幕作为整套编辑估计的做法，保留它作为全局颜色的离线基准。若继续分辨局部改色，须找独立于白幕的证据，并用真实作者的脸部和手部开关检验，不能关闭局部支路来换取较少串色。此次用人工登记的同类白幕；白墙身份、自动定位、真实作者局部编辑、人评和视频尚未在本轮验证。[完整数值、区域与成片哈希](2026-11-20-gros-screen-chroma-evidence.json)

两遍各 30 张软件输出、3 张局部参考、6 张期望局部图和6张对照图及报告逐字节一致。三组同图保持通过，并复算原 13 号全部既有分区结果。用已有研究环境运行，输出目录须未占用。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_screen_chroma.py "$base/gros-portrait-light-transfer-v1" "$base/gros-screen-chroma-replay"
```
