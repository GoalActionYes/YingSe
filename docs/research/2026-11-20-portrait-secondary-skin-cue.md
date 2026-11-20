# 脸颈色差可识别只暖脸合成图，仍误判有意全局调色

参考图里的人脸变暖，可能是现场灯光，也可能是有意调色。两种情况会让参考的人脸看起来相似，但跨场景迁移时，前者可能把现场橙光带到目标人物，后者则可能是用户要保留的风格。本轮沿用三位人物的四态素材，比较只看脸颊和增加颈部参照的结果。同一人物的四张图完全留出，用另外两人拟合分类器。[事前协议、框与验收条件](../../experimental/adaptive-reference-20261113/portrait-secondary-skin-cue-protocol.md)

| 逐人留出结果 | 脸颊色比 | 脸颊减颈部色比 |
| --- | ---: | ---: |
| 暖灯实拍识别为拍摄偏色 | 3／3 | 3／3 |
| 冷灯原片被误判为拍摄偏色 | 0／3 | 2／3 |
| 有意整图暖调被误判为拍摄偏色 | 3／3 | 2／3 |
| 只暖脸合成图被误判为拍摄偏色 | 3／3 | 0／3 |
| 对照误报合计 | 6／9 | 4／9 |

所有颈部框均达到像素数门槛，三张只暖脸图的颈部也都和冷灯原片逐像素一致。只改脸时，脸颊和颈部的色差明显增大。整图有意调暖时，两处皮肤一起变色，色差仍接近冷灯原片。留出的 01 和 03 因此各产生两次误报。没有参考图调色前的图像时，固定色差也会受人物、颈部遮挡、阴影和区域选择影响。三人的手工框只能说明这批素材的表现，尚未验证自动定位或跨场景画质。

真实作者案例中，同一编辑开关也改变了儿童的颈部。框内 351 个像素全部有非零蒙版权重，最高为 212（8 位）。关闭与打开编辑时颈部的像素不相同，因此真实案例的脸颈对照记为缺测。用同帧成人皮肤代替会引入另一人物的肤色与光照差异。[详细数值、判定及输入哈希](2026-11-20-portrait-secondary-skin-cue-evidence.json)

结束固定脸颈色差直接决定肤色保护的路线。三张合成正向结果可用于解释局部选择的价值。真实编辑已经表明，用户也可能有意同时调整脸和颈部。下一轮应优先寻找能直接反映调整意图的独立信息，例如参考图配套原片、可靠的编辑记录或显式区域选择，再验证其可用范围。本轮没有生成新的调色成片、更新稳定软件或接触独立验收集。两次运行的报告 SHA-256 均为 `bf8ed690a74be0f9b4c2a5855ed69026b3af52631838b6bbebf1efa90a726498`。

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_portrait_secondary_skin_cue.py "$base/gros-portrait-light-transfer-v1" "$base/gros-screen-chroma-v1" "$base/portrait-secondary-skin-cue-replay"
```
