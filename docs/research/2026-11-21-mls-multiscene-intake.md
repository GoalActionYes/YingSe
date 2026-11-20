# MLS 蓝纸原图通过资格检查，两组绿色塑料棒配准分数偏低

从作者的 16 位线性照片 ZIP 仅分段读取 04、05、06、09 的基线灯和加 25% 蓝灯照片，另取原尺寸掩膜 ZIP。数据读取预算上限 48 MiB，照片请求上界约 35.06 MiB，掩膜 123,233 字节。复用 02、03 的照片用于蓝纸检验。12 张照片通过文件身份及尺寸检查。图像采用相机 RGB，固定增益仅用于测量。[事前协议](../../experimental/adaptive-reference-20261113/mls-multiscene-intake-protocol.md) · [作者发布及许可](https://github.com/Visillect/mls-dataset#license--citation)

| 场景 | 材质 | 双灯边缘相关 | 掩膜交并比 | 共同有效内部像素 | 资格 |
| --- | --- | ---: | ---: | ---: | --- |
| 02 | 蓝纸棱柱 | 0.558 | 0.991 | 58,270 | 通过 |
| 03 | 蓝纸棱柱 | 0.547 | 0.991 | 59,982 | 通过 |
| 04 | 绿色塑料棒 | 0.571 | 0.986 | 25,815 | 通过 |
| 05 | 蓝纸棱柱 | 0.575 | 0.992 | 45,453 | 通过 |
| 06 | 绿色塑料棒 | 0.204 | 0.987 | 18,570 | 未通过 |
| 09 | 绿色塑料棒 | 0.349 | 0.984 | 9,180 | 未通过 |

六个场景的原尺寸与预览掩膜交并比均超过 0.98，材质内部共同有效像素占比为 98.8%–100%。06 和 09 未达到既定的 0.5 边缘相关门槛。掩膜一致说明标注在两种分辨率间一致，无法替代实际换灯图之间的配准验证。全图明暗边缘相关也可能受光照方向影响。本轮保留两例失败，下一轮先检验稳定结构边缘是否发生几何位移，再决定两组能否支持已知材质的换灯对照。没有训练模型或新增产品成片。[数值与文件哈希](2026-11-21-mls-multiscene-intake-evidence.json)

## 复核

本机资料位于 `outputs/adaptive-reference-20261113`。若重新准备照片，导入既有 `fetch_mls_linear_pilot`，对新目录调用 `run(out, ("04","05","06","09"))`，将该次进程中的 `BUDGET` 设为 48 MiB 扣减原尺寸掩膜 ZIP 字节数。掩膜来自作者同一 FTP 目录的 `masks_16bit_color.zip`，完整读取前核查小于 1 MiB，保存到新增照片目录，核对报告内的 SHA-256。旧原图批次仍使用已有的 `mls-linear-pilot-inputs`。从仓库根目录运行检查，指定未占用的输出文件。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$base = 'outputs/adaptive-reference-20261113'
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/audit_mls_multiscene_intake.py "$base/mls-linear-pilot-inputs" "$base/mls-multiscene-intake-v1" "$base/mls-material-identity" "$base/mls-multiscene-intake-v1/results-repro.json"
~~~

两次运行的 JSON 字节一致，SHA-256 为 `3f6bc660401cac94f1fe1dd13acc3d91ac75f52c74a8158fabe049af087afe9f`。
