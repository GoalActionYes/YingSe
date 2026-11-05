# 技术资料索引

接手运行先看 [启动指南](GETTING_STARTED.md)，功能边界看 [当前状态](PROJECT_STATUS.md)，版本选择看 [版本索引](../VERSIONS.md)。

最新用户功能：[手动调色指南](MANUAL_COLOR_CONTROLS.md) · [11 月 6 日验证记录](research/2026-11-06-manual-controls/README.md)。

## 阅读顺序

1. [最新同画面拟合](research/2026-11-05-aligned-reference/README.md)：已接入的条件路径、控制实验和拒绝条件。
2. [同图修复](research/2026-11-05-identity-fix/README.md)：原路径漂移、恒等保护与未采用方案。
3. [肤色研究审计](research/content-balance-20261105/skin-effect-audit/README.md)：扩展验证后的实际收益与反例。
4. [使用与算法说明](reference-color-usage.md)：基础处理链和操作资料，具体当前行为以源码和上述最新记录为准。

## 历史研究

| 资料 | 用途 |
| --- | --- |
| [基础适配对照](research/content-balance-20261105/baseline-adapter/README.md) | 对照研究与基础管线 |
| [加权统计 v2](research/content-balance-20261105/dense-v2/README.md) | 统计数值与受控实验 |
| [参考构图 v3](research/content-balance-20261105/reference-composition-v3/README.md) | 参考占比与稳定矩计算 |
| [可见面部 v4](research/content-balance-20261105/visible-face-v4/README.md) | 人物条件与误触发控制 |
| [参考肤色协调](research/content-balance-20261105/reference-skin-residual/README.md) | 早期合成与短视频探针；需结合后续审计 |
| [自然素材检查](research/content-balance-20261105/natural-check/README.md) | 自然素材条件与阶段结果 |
| [设计与计划](superpowers) | 各阶段需求和计划，不等于实现完成证明 |

研究记录保留当时的失败、数量与本机路径，不能把早期通过率当成当前泛化结论。部分媒体、GPU／原生冻结证据及运行时未入库；新克隆不能直接重跑全部历史脚本。[源码可执行检查范围](HANDOFF_VALIDATION.md) 单独记录。
