# 跨场景同款调色来源的访问核查

## 核查范围

针对映色在参考背景换色时带偏人物和墙面的问题，核查两条公开线索是否能提供来自不同场景、带同一调色标识的配对。固定检查 INRetouch RTD 的基准目录、参考关系清单及作者代码仓库，以及 Adaptive Photo Retouching 6-Style 的发布结构与小型索引访问。只读取公开 API、源文档和响应头，元数据预算 1 MiB。不登录、不接受共享账号信息条件、不下载图像、模型、APK 或归档。

若确认可取得参考关系和同款输出，再考虑对少量照片做身份和局部编辑证据核查。若文件请求需要额外授权，或同款标签不能说明主体／背景编辑规则，则登记具体缺口并停止本轮。

## 结果

- RTD 固定版本 `3e100e1fa896d9ed023cd1545890400edd67f949` 的确包含 `Benchmark/references_file.txt`，清单 2,880 字节。未登录的固定版本请求返回 HTTP 401 和 `GatedRepo`。数据卡标注 CC BY-NC-SA 4.0，说明同一 Lightroom 预设应用于多个 RAW 场景，并主动排除专用人像预设。论文描述预设可含局部及全局操作，尚不能判断任何特定配对有意保护肤色或中性色。见 [数据卡](https://huggingface.co/datasets/omaralezaby/Retouch_Transfer_Dataset)及[作者论文](https://openaccess.thecvf.com/content/WACV2026/papers/Elezabi_INRetouch_Context_Aware_Implicit_Neural_Representation_for_Photography_Retouching_WACV_2026_paper.pdf)。
- Adaptive Photo Retouching 6-Style 固定版本 `a3d4bb6bbf4116587181028c9e00c67593e55668` 可列出约 8.34 MB 的索引和 4.51 GB 归档。索引请求也返回 HTTP 401 和 `GatedRepo`。卡片的许可字段为 `other`，六种结果来自模型逐图生成调整参数。没有核实明确的局部编辑记录或拍摄组。见 [发布清单](https://huggingface.co/datasets/Phitran21/adaptive-photo-retouching-6style/tree/main)。
- INRetouch 作者的公开代码仓库另有小样。固定提交 `cbf0db19487222c21d63116c320357764438360d` 的代码树有两对示例前后图、两个不同文件名的目标原图及同名调色结果。作者的多示例配置明确共同使用两对示例调色。此为公开研究样例，独立于受限数据集的访问途径。下阶段已经按单独协议核验八张公开 JPEG 并运行四个配对。见 [作者配置](https://github.com/omarAlezaby/InRetouch/blob/cbf0db19487222c21d63116c320357764438360d/options/train/InRetouch_Optimize_Multiple.yml)与[实图小样](2026-11-18-inretouch-public-pilot.md)。

首次 RTD 完整元数据请求触及 200,000 字节单请求限额。改为只请求版本、访问属性及数据卡字段后成功。GitHub 分支树响应把提交 SHA 列为响应 SHA，随后按提交记录中的树对象 SHA 读取，逐路径及 blob 核对一致。元数据与响应头在 `outputs/adaptive-reference-20261113/cross-scene-source-access-v1`。固定树、配置及数据 API 返回文件的 SHA256 见[实图小样证据](2026-11-18-inretouch-public-pilot-evidence.json)。
