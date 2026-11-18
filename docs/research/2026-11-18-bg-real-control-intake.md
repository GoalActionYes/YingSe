# 核对到 87 组有匹配流程的人像换背景对照

BG-REAL 的 F1 把主体放到真实供体背景上，C1 用相同合成流程但沿用原背景。这个配对能比单纯修改已有背景色更直接地测试物体和场景替换是否影响目标人物。它仍是合成负对照，不能作为有意局部调色的正例。见[事前协议](../../experimental/adaptive-reference-20261113/bg-real-control-intake-protocol.md)和[作者数据卡](https://huggingface.co/datasets/anush-123/bg-real)。

固定发布版本 `aaba6cd1ced4ce3c846be9641fed092adb4919d1`。只读发布清单、署名表及主体／供体信息，共 14,571,242 字节。主清单的 LFS SHA-256 与 Git 指针均匹配，另三个元数据 Git blob 匹配。首读把 LFS 的 Git 指针当成原始文件做校验，在组配前停止；按发布 API 的 LFS 标识修正后通过。直连超时，改走本机网络代理后成功。见[固定发布](https://huggingface.co/datasets/anush-123/bg-real/tree/aaba6cd1ced4ce3c846be9641fed092adb4919d1/v2)和[身份摘要](2026-11-18-bg-real-control-intake-evidence.json)。

作者的 `test_id` 中，99 个不同的人类主体均有一对未做鲁棒性复压的 C1 与 F1。两臂的尺寸、编码、JPEG 质量、合成流程和主体掩膜路径匹配，且原始照片与供体可查 CC BY 2.0 的署名记录。按主体 ID 固定去掉 12 组重复供体后保留 87 对，供体类别包括客厅、公园、办公室、街景等。两次本机统计哈希一致，另从原 CSV 独立重算主体、供体数量、配对图片哈希及授权记录。

目前核对到的是发布身份和配对协议，尚未验证实际图像中主体内部的像素保持、边缘差、跨组图像独立性或调色输出。下一步可先小规模抽取 C1、F1 和共有主体掩膜，量化编码与合成后残留差异。满足条件再用作未消费开发负例，原有有效局部冷色和真实风格正例仍须同时保留。课程版未改动。

复现时用固定发布下 `v2/manifest_v2.csv` 与 `v2/meta/` 的三个对应 CSV，在现有环境运行[核查脚本](../../experimental/adaptive-reference-20261113/audit_bg_real_controls.py)，传入本机素材目录和新 JSON 输出。全量清单与作者署名保留本机；使用素材时须保留相应署名。
