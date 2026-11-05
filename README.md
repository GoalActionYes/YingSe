# 映色 YingSe

本地参考图调色原型：输入参考图和目标图片／视频，预览、调整强度并导出PNG、MP4或CUBE。不是剪映原生引擎，也没有完整复刻其算法。

## 接手先看这里

| 你想做什么 | 从哪里开始 |
| --- | --- |
| 第一次接手，跑起来 | [源码启动指南](docs/GETTING_STARTED.md) |
| 当前版本、历史区别 | [版本索引](VERSIONS.md) |
| 最近改了什么 | [改动记录](CHANGELOG.md) |
| 找代码、分任务、提PR | [协作指南](CONTRIBUTING.md) |
| 功能边界与待办 | [当前状态](docs/PROJECT_STATUS.md) |
| 查算法证据与失败记录 | [技术资料索引](docs/README.md) |

## 当前该用哪个版本

新开发从 main 开始。当前功能里程碑为 `feature/manual-controls-20261106`：在已有参考图匹配基础上，增加基础调色、HSL 和 RGB 曲线。[使用方法与界面](docs/MANUAL_COLOR_CONTROLS.md) · [验证记录](docs/research/2026-11-06-manual-controls/README.md)

先前匹配算法基线为 [feature/aligned-reference-20261105](https://github.com/GoalActionYes/YingSe/tree/feature/aligned-reference-20261105)，提交 cb79e4a。本批主要增加用户微调空间，不宣称参考风格匹配本身普遍改善。

| 版本 | 源码 | 建议 |
| --- | --- | --- |
| 主开发版 | [reference_color_app](reference_color_app) | 默认从这里改 |
| 肤色保护实验版 | [experimental/skin-protection-20261019](experimental/skin-protection-20261019) | 独立副本；保护默认关闭，本批手动控件未同步 |
| 算法研究 | [experimental/content-balance-20261105](experimental/content-balance-20261105) | 包含失败试验，不能整个当成已上线功能 |
| 八预设原型 | [prototype](prototype) | 历史比较，不是当前主程序 |

当前已实现：精确同图保持原样；确认画面对应且验证通过时学习全局颜色变换。不同场景参考仍可能受明暗和内容干扰，压缩、缩放也非全面支持。[查看边界](docs/PROJECT_STATUS.md)

## 源码与体验包不同

GitHub不包含EXE、便携Python、离线模型、测试媒体、个人项目数据或本机目录链接。目前没有在GitHub发布可下载的便携包；体验包向负责人索取对应里程碑的完整ZIP，源码按启动指南运行。

“打开映色.cmd”和“00-从这里开始”是负责人电脑的本机导航，克隆后不能据此启动EXE。源码开发不依赖这些路径，也不需要剪映。Python 3.12、NumPy、Pillow及FFmpeg的准备方法见启动指南。

本仓库自2026年11月5日补入阶段快照；早期日期不代表当时已有Git历史。研究成功、验证范围和正式接入分别记录，不能互相替代。
