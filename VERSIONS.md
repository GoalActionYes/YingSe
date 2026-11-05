# 版本索引

## 当前推荐

开发从 `main` 开始。当前功能里程碑为 `feature/manual-controls-20261106`，主开发版新增基础／HSL／曲线手动调色。[操作说明](docs/MANUAL_COLOR_CONTROLS.md)。肤色实验副本暂未同步这些控件。

| 要做什么 | 选择 |
| --- | --- |
| 修改主程序 | `main` 的 `reference_color_app/` |
| 复现当前功能代码 | 检出 `feature/manual-controls-20261106`；按源码启动指南准备依赖 |
| 开发可选肤色保护 | `main` 的 `experimental/skin-protection-20261019/`，独立副本 |
| 比较早期八预设 | `snapshot/preset-prototype` 或当前 `prototype/` |
| 研究未上线算法 | `research/*` 与研究目录，先读最新审计 |

当前同画面拟合是在精确同图保护之后增加的路径；旧同图修复记录中的“相近图未修复”描述的是该历史阶段，不能代替当前状态。[改动记录](CHANGELOG.md) · [功能边界](docs/PROJECT_STATUS.md)

## 里程碑与历史标签

下表按阅读顺序排列。研究状态描述对应阶段；后续审计可能推翻早期乐观结论。

| 标签 | 内容 | 定位 |
| --- | --- | --- |
| `feature/manual-controls-20261106` | 主版基础 11 项、八色 HSL、RGB 曲线及组合导出 | 当前用户控件版本；自主实现 |
| `feature/aligned-reference-20261105` | 条件同画面 RGB 拟合、LUT 与真实导出验证 | 匹配算法基线，有限条件通过 |
| `fix/exact-identity-20261105` | 解码同图恒等保护 | 已被当前功能基线包含 |
| `snapshot/reference-color-stable` | 基础参考调色、桌面源码与测试 | 历史基础快照，不含后续修复 |
| `snapshot/skin-protection` | 默认关闭的目标肤色保护 | 历史实验快照，后续源码在 main 持续修复 |
| `snapshot/preset-prototype` | 八预设风格与权重搜索 | 早期比较基线 |
| `research/regional-match` | 分区迁移探针 | 未证明整体收益，不接入 |
| `research/content-balance-v1` | 初始内容占比适配 | 有重复试验反例 |
| `research/content-balance-v2` | 全像素加权统计 | 阶段数值检查，不代表产品验收 |
| `research/reference-composition-v3` | 参考贡献与稳定矩 | 研究候选，未默认接入 |
| `research/visible-face-v4` | 可见面部条件与回退 | 研究候选，未默认接入 |
| `research/reference-skin-residual-v1` | 学习参考肤色的局部协调、视频探针 | 后续审计收益有限，未默认接入 |
| `snapshot/research-journal` | 研发复盘和策略 | 历史资料 |

2026 年 11 月 5 日开始补入 Git。此前无可恢复的原始 Git 历史，早期阶段标签是按现有文件导入的累计快照，提交时间是导入时间。不要据此宣称这些日期前已经有版本管理。

## 查看旧版本与新改动

GitHub 仓库顶部的分支／标签选择器可浏览对应源码；[提交历史](https://github.com/GoalActionYes/YingSe/commits/main/) 查看每次差异。[当前里程碑以来的修改](https://github.com/GoalActionYes/YingSe/compare/feature/aligned-reference-20261105...main) 可以查看后续交接整理。

本机仅查看旧文件可执行 `git show 标签:文件路径`。需要运行旧版时，先保存当前修改，使用单独工作区，例如：

```powershell
git worktree add --detach ..\YingSe-baseline feature/aligned-reference-20261105
```

它创建独立源码目录；依赖、模型与测试媒体仍需另行准备。不要直接在正在运行服务的目录切换历史版本。团队提交和标签规则见 [协作指南](CONTRIBUTING.md)。

## 源码与安装包

Git 标签标识源码，不自动生成安装包。GitHub 目前没有便携 ZIP 发布；负责人本机包也不能用 Git clone 获得。完整体验包应注明源码标签、版本说明、依赖与 SHA-256，再独立分发。

负责人本机的 `讨论/12-Git版本管理/映色源码历史.bundle` 是离线历史备份，不随 GitHub 源码提供，也不证明已上传。远程最新内容以 GitHub 页面与提交 SHA 为准。
