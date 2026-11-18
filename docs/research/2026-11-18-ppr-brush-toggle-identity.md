# MMArt 同图配置不能隔离 PPR10K 的笔刷效果

已知 PPR10K 样本 1645_1 的专家 A 配置有一项启用笔刷，MMArt 公开配置没有这一项。本轮核对两份配置是否只有笔刷不同。固定读取原始配置、专家 A 与上轮已取得的 MMArt XMP。解析最外层 Camera Raw 设置，排除已列明的局部操作及文件描述字段，统一数值与空白表示，保留曲线顺序。见[事前协议](../../experimental/adaptive-reference-20261113/ppr-brush-toggle-identity-protocol.md)及[上轮发布核对](2026-11-18-mmart-local-edit.md)。

专家 A 和 MMArt 配置有 113 个可直接比较的同名字段，其中 14 项不同。**色温、色偏、曝光、高光、对比度和饱和度均有差异**，处理版本、相机配置和主色调曲线一致。两份配置混有足以改变整图的调整。即使能取得成片，成片差也不能归为笔刷作用。停止这一同图开关来源。

两份本机报告字节一致。另用独立 XML 实现比较八项数值设置，结论一致。未读取照片或渲染成片，未使用独立保留集。见[可公开的字段差异和身份](2026-11-18-ppr-brush-toggle-identity-evidence.json)。

复现时在仓库根目录设置 `$env:PYTHONPATH = (Get-Location).Path`，用研究 Python 运行 `experimental/adaptive-reference-20261113/audit_ppr_brush_toggle_identity.py <本机固定MMArt配置> <新JSON>`，需要已有 PPR 两个 XMP 归档。原 XMP 不随 Git 发布。
