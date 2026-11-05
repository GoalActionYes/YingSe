# 映色版本索引

2026年11月5日开始纳入Git。此前没有可恢复的原始Git历史，以下是按现有文件导入的阶段快照，提交时间为导入时间；阶段开发日期以研发日志为准。

| 标签 | 版本内容 | 状态 |
| --- | --- | --- |
| snapshot/preset-prototype | prototype目录的预设风格与权重搜索 | 早期比较基线 |
| snapshot/reference-color-stable | reference_color_app及测试、桌面源码 | 已有参考调色稳定源码 |
| snapshot/skin-protection | experimental/skin-protection-20261019 | 独立可选肤色保护实验 |
| research/regional-match | work/region-match-probe | 未证明整体收益，不接入 |
| research/content-balance-v1 | experimental/content-balance-20261105 | 研究候选，重复试验有反例 |
| snapshot/research-journal | 研发复盘与策略文档 | 历史记录和后续计划 |

标签对应累计源码快照，不代表每个阶段都已经通过完整产品验收。可用git show 标签:文件路径查看旧文件。需要实际切换时先保存修改，或使用git worktree add建立另一个检出目录，避免覆盖正在使用的素材和服务。

## 协作约定

main保存已整理源码；新功能从main创建feature/功能名分支。稳定交付后添加新的有说明标签；失败研究保留结果，不能冒称稳定发布。既有标签不挪动、不强推覆盖。

仓库保存源码、测试、依赖版本和简要证据，不保存私人媒体、项目数据、剪映DLL、逆向工具、便携Python、模型本体或历史ZIP。完整体验包仍留在原讨论目录。只有源码克隆时需要另行准备依赖、合法模型和测试资料；不能当作已包含便携运行时的体验包。

## GitHub远程

已按用户提供的地址配置origin：https://github.com/GoalActionYes/YingSe.git。本机通过Git Credential Manager官方浏览器流程认证，密码或令牌不写入源码。后续使用git push origin main和git push origin --tags同步；不强推，不重写既有标签。实际上传核验见讨论/12-Git版本管理/版本管理核验.json。

离线备份在讨论/12-Git版本管理/映色源码历史.bundle，使用git clone 映色源码历史.bundle 映色源码恢复可恢复历史和标签。该文件不是GitHub上传凭据。
