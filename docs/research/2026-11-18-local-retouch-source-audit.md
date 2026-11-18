# 带局部编辑意图的公开素材核查

## 核查范围

判断公开来源是否有可以核对的跨场景同款调色，以及人物或中性色的局部编辑依据。现有候选在白球服和墙面偏色上未达标。已登记的 RTD 预设迁移和 PPR 专家组不足以同时给出两类证据。本轮不训练，不下载图像、模型或大型归档。仅核查作者页面、许可和发布文件结构，新增 API 响应总量限制为 1 MiB。

可进入实物小样核查的来源需要同时能取得原图、调色图、可追溯的局部编辑意图、跨内容共用的调色标识及素材许可。只有单图编辑对或缺少调色分组的来源归入其他研究用途，不用于本轮替代独立画质评估。

## 发布核验

**ControlArt-Bench 有实际配对与编辑意图。** 固定版本 `2494044edee52a1c5dcd5ded6509b4a7216de354` 的原图、蒙版、指令和目标各有 200 项，四类文件名主干集合一致。原图有 196 个不同的 LFS SHA256，其中 4 个原图各出现两次。文件名前缀也得到 196 组，组内原图哈希一致。可按该分组避免同图泄漏。作者将其定义为显式指令和用户蒙版引导的修图评估，其配对里的“参考”是目标修图结果。发布树没有额外的跨内容调色分组表，数据卡许可一节为空，API 也未标出许可。保留为有明确编辑意图的数据线索，暂不读取图像。见 [固定发布树](https://huggingface.co/datasets/Dragoniss/ControlArt-Bench/tree/2494044edee52a1c5dcd5ded6509b4a7216de354)和[作者项目](https://dragoniscv.github.io/projects/EyeControl/)。

**EditChromin 的配对目前未从所核版本取得。** 固定版本 `8da72ef495b2c0c24f93c5ef92ff26f5881ad846` 的数据卡宣称有 60,000 对，实际发布树只有 7 个文件，Python 源码从本地 `images/<split>` 读取配对，而该目录未发布。直接读取并按 Git blob 复核发现，`train_metadata.json` 与 Rust 加载器源码的对象一致，无法解析为 JSON。数据卡有 Apache 2.0 标记，这一标记不能填补缺失的数据文件。见 [发布数据卡](https://huggingface.co/datasets/Trouter-Library/EditChromin)。

**SPIRE 有可关注的独立审美来源。** 论文报告 650 对手机原图与专业修图，包含全局及局部编辑。所查作者论文未提供可核验下载入口、素材许可或跨内容同款调色分组。RefRetouch 则明确用调色前后参考对提取风格，主要实验仍用 FiveK／PPR 来源，项目页将源码标为待发布。两者都不构成可立即使用的本轮样本。见 [SPIRE 论文](https://arxiv.org/html/2609.24768v1#S4.SS2)和 [RefRetouch 项目](https://tememuruts.github.io/RefRetouch-Project-Page/)。

两套发布 API 响应加两个只读文本文件共 307,763 字节。PowerShell 与 Python 分别统计原图哈希，均得出 4 个重复组。Python 另核对四类文件的配对文件名及 EditChromin 加载清单身份。[小型证据](2026-11-18-local-retouch-source-audit-evidence.json)保存请求结果哈希，无原图、修图、蒙版或指令内容。复核可从 Hugging Face 的 `/api/datasets/<repo>` 确定版本，再读取 `/api/datasets/<repo>/tree/<revision>?recursive=true`。本机响应位于 `outputs/adaptive-reference-20261113/local-retouch-source-metadata-v1`。

本轮仍未取得合格的跨场景同款局部调色配对。若来源后续提供可核验的分组和许可，可以先做小样身份、局部调色差分和拍摄组核查。当前未消费独立样张，未训练或升级软件。

## 2026-11-19 核查 RefRetouch 的独立配对来源

RefRetouch 的 VCIRB 有 25 个留出组，每组使用一个 Lightroom 预设。作者先按语义和色彩分布把照片聚为相似组，并在挑选预设时排除涉及蒙版的操作。若后续数据可获得，可用于相似内容上的全局颜色与明暗迁移对照。当前白球服与白墙问题还需要不同场景的受控比较，以及分区编辑依据。论文另用 PPR10K 的 21 个专家 A 人像组，与本项目已核查的人像数据同源，后续仍须核对拍摄分组和已消费记录。[作者论文第 6、7 节](https://arxiv.org/html/2602.17044v1#S6)

作者仓库提交 `886252c835f10a86200617016d37bcba224522b1` 对应的递归树未截断，仅有 README。项目页和源码仓库均未提供可核验的 VCIRB 配对下载、分组清单或素材许可。当前只登记论文中的数据构成，未准入新照片或运行模型。[作者仓库固定版本](https://github.com/tememuruts/RefRetouch/tree/886252c835f10a86200617016d37bcba224522b1) · [项目页](https://tememuruts.github.io/RefRetouch-Project-Page/)
