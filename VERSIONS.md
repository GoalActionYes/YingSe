# 映色版本索引

最新阶段：feature/aligned-reference-20261105，有条件的同构图颜色拟合，77项基础与97项肤色测试、图像/视频/导出验证；不同内容仍回退原算法，部分压缩和缩放被拒绝。

最新修复：fix/exact-identity-20261105，精确解码同图返回恒等映射，基础与肤色版均已打包；普通参考路径不变。相近图漂移和全面明暗校准仍未修复。发布证据在docs/research/2026-11-05-identity-fix。

2026年11月5日开始纳入Git。此前没有可恢复的原始Git历史，以下是按现有文件导入的阶段快照，提交时间为导入时间；阶段开发日期以研发日志为准。

| 标签 | 版本内容 | 状态 |
| --- | --- | --- |
| snapshot/preset-prototype | prototype目录的预设风格与权重搜索 | 早期比较基线 |
| snapshot/reference-color-stable | reference_color_app及测试、桌面源码 | 已有参考调色稳定源码 |
| snapshot/skin-protection | experimental/skin-protection-20261019 | 独立可选肤色保护实验 |
| research/regional-match | work/region-match-probe | 未证明整体收益，不接入 |
| research/content-balance-v1 | experimental/content-balance-20261105 | 研究候选，重复试验有反例 |
| research/content-balance-v2 | 全像素加权统计与12项测试 | 重复对照改善，独立自然素材和视频待验收 |
| research/reference-composition-v3 | 参考贡献适配、稳定矩计算与18项测试 | 新增三组受控占比通过；自然风格质量及非人物误识别待验证，不接入默认App |
| research/visible-face-v4 | 可见面部适用条件与25项测试 | 限定样例误触发回退通过；覆盖率与自然风格质量待验证，不接入默认App |
| research/reference-skin-residual-v1 | 参考肤色局部色度协调、33项测试与短视频原型 | 六组合成控制误差下降35%至49%，30fps90帧通过；自然质量及产品接入未验收 |
| snapshot/research-journal | 研发复盘与策略文档 | 历史记录和后续计划 |

标签对应累计源码快照，不代表每个阶段都已经通过完整产品验收。可用git show 标签:文件路径查看旧文件。需要实际切换时先保存修改，或使用git worktree add建立另一个检出目录，避免覆盖正在使用的素材和服务。

## 协作约定

main保存已整理源码；新功能从main创建codex/功能名分支。稳定交付后添加新的有说明标签；失败研究保留结果，不能冒称稳定发布。既有标签不挪动、不强推覆盖。

仓库保存源码、测试、依赖版本和简要证据，不保存私人媒体、项目数据、剪映DLL、逆向工具、便携Python、模型本体或历史ZIP。完整体验包仍留在原讨论目录。只有源码克隆时需要另行准备依赖、合法模型和测试资料；不能当作已包含便携运行时的体验包。

## GitHub远程

已按用户提供的地址配置origin：https://github.com/GoalActionYes/YingSe.git。本机通过Git Credential Manager官方浏览器流程认证，密码或令牌不写入源码。后续使用git push origin main和git push origin --tags同步；不强推，不重写既有标签。实际上传核验见讨论/12-Git版本管理/版本管理核验.json。

离线备份在讨论/12-Git版本管理/映色源码历史.bundle，使用git clone 映色源码历史.bundle 映色源码恢复可恢复历史和标签。该文件不是GitHub上传凭据。
