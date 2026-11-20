# 真人换灯会传到目标肤色，两个现有版本都未通过联合门槛

使用三名参与者的暖灯／冷灯实拍。把冷灯参考按实际换灯的两颊色度差做数字染色，并保持线性亮度；因此每个人有冷灯原图、暖灯实拍和肤色变化匹配的数字染色三种参考。对目标保持不变时，暖灯实拍用于测不希望传递的拍摄差；数字染色用于测有意要传的色彩。版本只收到两张普通 RGB 图。脸颊、白幕与双眼位置在原图上登记，仅用于配对资格和测量。[事前协议](../../experimental/adaptive-reference-20261113/gros-portrait-light-transfer-protocol.md) · [采集来源与曝光](2026-11-20-gros-portrait-light-intake.md)

三人均通过配对门槛。暖冷帧的眼中心位移最多 7.3 个工作像素。数字染色与实拍在脸颊对数色比上的剩余差为 0.00012–0.00343，低于 0.025 限制。02 的快门在两档相同。以其左侧白幕为例，真实换灯的两个对数色比变化为 +0.047、+0.035，匹配数字染色的变化为 +0.748、−0.503。两种参考的肤色变化接近，白幕的响应明显不同。两组有约三分之一档快门差，三组均为自动白平衡；素材只隔离到真实拍摄状态与数字染色的区别。[区域登记](../../experimental/adaptive-reference-20261113/gros-portrait-light-regions.json) · [区域及成片数据](2026-11-20-gros-portrait-light-transfer-evidence.json)

| 同快门 02 作参考 | 1.3 稳定版 | 13 号空间研究版 |
| --- | ---: | ---: |
| 01 目标双颊，暖灯参考相对冷灯参考的变化 | 3.52–4.75 | 9.33–10.58 |
| 01 目标两处白幕，同上 | 6.78–6.79 | 2.49–2.64 |
| 03 目标双颊，同上 | 2.39–2.84 | 5.74–6.94 |
| 03 目标两处白幕，同上 | 5.94–6.34 | 2.41–2.76 |

表中数值为 RGB8 平均绝对差。13 号在这两个目标的脸颊上，按 Rec.709 明暗 Y′ 分离后，R−Y′、B−Y′ 的平均绝对差为 7.89–12.29，Y′ 自身变化为 0.60–3.06。用两种参考成片的逐像素 RGB 差即可复算，说明该例主要变化在颜色。13 号有时能保住有意改色，例如 03 从 02 传数字染色时，两颊投影约 0.96、0.98，响应误差为 2.07、2.53 RGB8；同一目标实际换灯仍造成 5.74、6.94 RGB8 的变化。两个版本联合各 0／6。同图三组由两版的正常图片流程逐像素保持。[01 从 02 的成片](../../outputs/adaptive-reference-20261113/gros-portrait-light-transfer-v1/01-from-02/comparison.png) · [03 从 02 的成片](../../outputs/adaptive-reference-20261113/gros-portrait-light-transfer-v1/03-from-02/comparison.png)

这批素材提供了此前静物没有的真人对照。接下来先用这套成片检查，能否利用人脸与白幕受到不同照明的关系降低串色，并且保住有意染色。此思路须同时在既有真实局部编辑对照上检查，避免把用户想要的脸部暖调当成拍摄偏色。白幕仍只是中性材料，不能替代独立白墙；三人属于开发样本，无人类自然度或视频结论，本轮未改产品。

两遍各 36 张软件输出、6 张期望、9 张准备参考、6 张并排图及报告逐字节一致。使用现有研究环境，从项目根目录运行，输出目录须未占用。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_portrait_light_transfer.py "$base/gros-portrait-light-input-v3" "$base/gros-portrait-light-transfer-replay"
```
