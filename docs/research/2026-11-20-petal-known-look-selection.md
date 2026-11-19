# 冻结 PETAL 风格编码认对了一些参考，成片联合检查仍未通过

此前固定纹理和已知材质都未稳定辨认普通参考的编辑。PETAL 编码器曾在旧训练配对中测得跨内容的风格对应。本轮保持编码器和渲染器冻结，只用它比较单张参考成片与从目标原图生成的三张预览，分别对应原样、Film A、Film B。目标效果真值只用于离线评价。[事前协议](../../experimental/adaptive-reference-20261113/petal-known-look-selection-protocol.md) · [编码器历史证据](2026-11-15-petal-encoder.md)

| 来源 | 识别正确，原样／JPEG 85 | 关闭态保持 | 整图及近中性色通过视图 | 同款最坏近中性色差（RGB8） |
| --- | ---: | ---: | ---: | ---: |
| DAST 人像曝光 | 24/45／24/45 | 12/15 | 0/15 | 8.20 |
| MLS 实拍换灯 | 6/12／6/12 | 2/4 | 0/4 | 8.91 |
| Middlebury 相机色卡 | 2/6／2/6 | 0/2 | 0/2 | 0.00 |
| ARRI 路径实拍 | 5/9／5/9 | 2/3 | 0/3 | 0.00 |

两种编码的选择逐项相同，各为 37/72。旧纹理方法原样／JPEG 为 30/72、28/72，但保持了全部关闭态。本轮每种编码只认对 1/24 个 Film A 和 20/24 个 Film B，另将 8/24 个关闭态误判为 Film B。实体色卡两台相机的六个输入均选择 Film B，同款换相机差为零却无法保留真实风格差异。四类来源均未通过，两种编码都没有任何视图通过整图与近中性色联合检查。[上一轮纹理对照](2026-11-20-target-patch-inverse-look.md) · [本机蓝灯假触发成片](../../outputs/adaptive-reference-20261113/petal-known-look-selection-v1/original/mls/02-blue-off.png)

结束用固定 PETAL 余弦距离直接选择这三种操作。编码器读数在本批压缩下稳定，误选也随之保留。已知操作时颜色渲染可以作为正对照，后续仍需能区分原本带色的画面与有意改色的独立信息或监督。该结果来自封闭三选一和已消费的开发素材，无法外推为开放参考的质量，也没有修改软件或模型。

## 复现

在既有 CUDA 视觉研究环境中将仓库根目录、OpenColorIO 2.5.2 和 SciPy 1.16.3 加入 `PYTHONPATH`，设置 `OPENBLAS_NUM_THREADS=4`，运行 `python experimental/adaptive-reference-20261113/probe_petal_known_look_selection.py <新的输出目录>`。两遍各 144 张成片及报告逐件一致，复算了全部选择并逐像素核对成片。原 PETAL 编码器三项测试通过，运行前后的模型状态哈希相同。报告 SHA256 为 `de52de84e26964e033ffc3d6c0d52b300587bdf294033758dc1625f38c016bc4`。[脚本](../../experimental/adaptive-reference-20261113/probe_petal_known_look_selection.py)
