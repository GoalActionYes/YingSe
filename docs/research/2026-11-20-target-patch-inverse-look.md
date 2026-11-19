# 目标图的相似纹理不足以判断参考用了哪款调色

已知编辑和足够色样能修复白衣及投影幕测试区。这轮回到普通输入，只用参考成片与目标原图判断是否重放 Film A、Film B 或保持原样。先从亮度纹理寻找互为最近的小块，再比较撤销两种候选风格前后，对应块的色差是否减小。原参考图、目标预期成片和编辑标签均不参与判断。[事前协议](../../experimental/adaptive-reference-20261113/target-patch-inverse-look-protocol.md) · [已知操作的局部正对照](2026-11-20-paired-neighbor-uniform-support.md)

| 参考来源 | 原图，判断正确 | JPEG 85，判断正确 | 原图／JPEG，整图和近中性色都通过的视图 |
| --- | ---: | ---: | ---: |
| DAST 人像曝光 | 21/45 | 19/45 | 3/15／2/15 |
| MLS 实拍换灯 | 4/12 | 4/12 | 0/4／0/4 |
| Middlebury 相机色卡 | 2/6 | 2/6 | 0/2／0/2 |
| ARRI 路径实拍 | 3/9 | 3/9 | 0/3／0/3 |

两种编码的 24 个关闭态全部保持原样。只在人像曝光来源触发调色，原图触发 7 次，压缩后触发 5 次。该来源原图 23/45 态不足 16 个匹配，JPEG 后是 25/45 态。同款换曝光后近中性色最差差值分别为 6.28 和 8.20 RGB8，均超过 3。其余三来源没有一次触发，两种编码均为 0/4 来源通过。结果说明，即便匹配到纹理相似的小块，也还缺乏证据确认两张照片里这些小块的材质与受光可以作颜色基准。[原图人像成片](../../outputs/adaptive-reference-20261113/target-patch-inverse-look-v1/original/dast) · [JPEG 对照](../../outputs/adaptive-reference-20261113/target-patch-inverse-look-v1/jpeg85/dast)

结束用固定亮度小块对应来选这两款调色。若再做目标引导，新增证据应能支持同材质或同拍摄关系，单靠放宽块匹配门槛没有依据。这里是有限的三种解释，且使用的是已经消费的开发内容，普通两图自动识别与独立画质验收仍待完成。

## 复现

沿用视觉研究环境，将仓库根目录、OpenColorIO 2.5.2 和 SciPy 1.16.3 加入 `PYTHONPATH`，设置 `OPENBLAS_NUM_THREADS=4`。运行 `python experimental/adaptive-reference-20261113/probe_target_patch_inverse_look.py <新的输出目录>`。两次运行的 144 张成片及报告逐件一致。按保存的读数独立复算全部决策，并逐像素复核每张成片与所选操作一致。报告 SHA256 为 `55e653b0f17a131896b02c54adc02d800b52d1696380fb16636118684af36520`。[脚本](../../experimental/adaptive-reference-20261113/probe_target_patch_inverse_look.py)
