# 参考成片的内嵌编辑字段无法区分这批风格

配对前后图能补足部分方向，普通两图只提供参考成片。本轮检查原始发布 JPEG，核对标准／扩展 XMP、EXIF 与 ICC，再读取规定范围内的 Camera Raw 色彩字段。未把拍摄字段、软件名或色域标记解释为滤镜身份。合成带标签 JPEG 通过读取正控；40 个实拍文件均按旧清单校验。[事前协议](../../experimental/adaptive-reference-20261113/reference-embedded-edit-protocol.md) · [Adobe Camera Raw XMP 字段](https://developer.adobe.com/xmp/docs/xmp-namespaces/crs/)

| 来源 | 有标准 XMP 的文件 | 含与原图不同的色彩字段的编辑结果 |
| --- | ---: | ---: |
| FACD 八组 | 12/32 | 3/24 |
| PSBench 四个编辑前后对 | 1/8 | 0/4 |

FACD 三个有字段差异的结果都属于 711468。三个滤镜的已登记色彩字段和值完全相同。将对应 JPEG 解码并缩放到 256 × 256 后，三个两两图像的 RGB8 平均绝对差仍为 24.19、42.91、53.14。说明这份相同记录不能区分这三份颜色不同的输出。PSBench 唯一有标准 XMP 的成片不含登记范围内的颜色字段。40 张无扩展 XMP，已找到的标准包均能解析。停止用这些导出文件内的字段作为通用风格输入；仍可用独立的编辑配对做监督和诊断。[此前拍摄白平衡核查](2026-11-19-capture-white-metadata.md)

两遍报告逐字节一致，并重新读取全部 40 个原始文件核对记录。报告 SHA256 为 `1f125c0c262e5cf65597641917ddb33beca8a267f58c2a5cfec1ff3c2cd98d36`。复现用已有视觉环境，把仓库根目录加入 `PYTHONPATH` 后运行脚本并传入新输出目录。公开记录只保存字段类别、计数与结论。现有软件没有改动。[复现脚本](../../experimental/adaptive-reference-20261113/probe_reference_embedded_edit.py)
