# 作者编辑资料的公开取件核验

原生空间网格在两个已有答案的暖色案例中通过了表达检查，普通参考里仍需可核验的编辑线索。现有 PPR10K 四张原片都有三位专家成片，既有四折的固定全局映射试验显示，专家身份不足以复现同一调色。FiveK 的旧相邻对比度操作也已经核查过，公开最终 TIFF 无法对应所需的中间历史。本轮核对新发布的 LUTBench 入口，并对 EditChromin 补查数据托管层和第二组文件身份。预算约 10 分钟及少量公开文本，不取照片或权重，不运行上游代码。[双例网格表达](2026-11-24-matched-wall-warmth-grid.md) · [PPR10K 跨照片试验](2026-11-18-ppr-expert-transfer.md) · [FiveK 历史核验](2026-11-19-fivek-contrast-recovery.md) · [EditChromin 旧取件记录](2026-11-18-local-retouch-source-audit.md)

## 取件事实

| 来源 | 实际核验 | 用途判定 |
| --- | --- | --- |
| LUTBench | 论文用 91 个全局 LUT 处理 5,000 张统一显影的 FiveK 原片。作者指向的 GitHub 仓库可访问，GitHub 内容接口返回仓库为空，发布列表为零。 | 可借用固定输入转换与按原片分组的方法。当前入口未取得 LUT、文件级清单或可复现渲染包。可复现的全局色彩监督是一个用途，需要按空间位置改变同色像素的暖色仍缺对应真值。 |
| EditChromin | 固定提交 `8da72ef495b2c0c24f93c5ef92ff26f5881ad846` 的树有七个文本文件。仓库托管索引本次列了一个训练 split，大小接口可数行数为 0，Parquet 列表为空。固定提交的 `train_metadata.json` 与 `dataset_loader.rs` 逐字节相同，`schema.json` 与 `loadDataset.js` 逐字节相同。两个 `.json` 均无法解析。 | 当前公开提交未取得可用编辑行、原片及成片的关联或操作记录。数据卡声明需要与实际发布和使用许可一并核查。 |

两条都止于发布准入。小型技术文档与文件哈希留存，用来避免下轮从零重复查找。结论限于所核对的公开发布页和固定提交。[LUTBench 论文与作者仓库](https://arxiv.org/html/2611.23113v1) · [LUTBench 发布 API](https://api.github.com/repos/LouckXu/LUTBench/releases) · [EditChromin 固定发布](https://huggingface.co/datasets/Trouter-Library/EditChromin/tree/8da72ef495b2c0c24f93c5ef92ff26f5881ad846)

MirrorPPR 的公开说明覆盖五官、脸型和身材等几何操作，要求参考的前后两张图和目标。其专业子集通过修图流程在 PPR10K 原片上构建。本轮只核对作者说明，未发现符合当前浅墙或肤色对照的暖色开关和作用范围。[MirrorPPR 项目页](https://sjtu-deng-lab.github.io/MirrorPPR/) · [MirrorPPR 论文](https://arxiv.org/abs/2606.29308)

## 复核与用途

EGM 复核对照了旧记录。本轮对 EditChromin 增加的是第二组文件重复核验和托管索引读数。LUTBench 的全局颜色对照和前述两处网格表达都不能作为普通两图识别暖色的证据。

以 Hugging Face 仓库元数据的 `sha` 为输入，请求固定版本的 `train_metadata.json`、`dataset_loader.rs`、`schema.json`、`loadDataset.js`，逐字节比较并检验两个 JSON 文件的可解析性。核对 GitHub 的仓库、根内容及 releases 三个端点，Hugging Face 的仓库树与 datasets-server 的 size、splits、parquet。首次取回的响应体合计 32,122 字节。又向固定提交查询了七个树成员，从磁盘计算四个已取文件的 Git blob 标识，两项均与首次记录一致。文件作为未执行的数据样本，未运行发布方加载器。[HTTP 状态、文件长度及哈希](2026-11-24-authored-edit-release-evidence.json)

两个来源都没有进入像素阶段，现有软件仍缺合格的跨场景画质候选。用作本轮跨场景配对真值的暖色资料，应有可核对的原始场景标识、开关成片和同款编辑作用范围。作者身份、指令、术语或灰卡线索可以提供监督。开发比较应按拍摄来源和内容留出，实际推理只给目标原片和参考成片，同时比较局部暖色响应、换灯干扰和保护区误差，之后再做正式独立验收。
