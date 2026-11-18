# FiveK 对比度操作已定位，公开档案缺少可核对的中间成片

按上一轮固定的跨文件匹配，重新定位两个版本的对比度历史，确认它们均为 **0 → 13**，前后完整状态只差对比度。两份设置均有 `Version=5.3`，和该状态对应的 `ProcessVersion` 未明确记录。不能从这个字段直接认定它用了 Adobe 哪一代处理模型。原文件、图像键、前后状态和摘要保存在本机重放包。[事前协议](../../experimental/adaptive-reference-20261113/fivek-contrast-recovery-protocol.md) · [Adobe 处理版本说明](https://helpx.adobe.com/lightroom-classic/desktop/process-and-develop-photos/develop-module-options.html)

| 检查 | 核实结果 |
| --- | --- |
| 原片与最终导出 | 两个公开 DNG 的容器头有效，每图另有五份官网最终 TIFF 可访问。目录副本名称未能直接对应官网 A–E 命名 |
| 目录当前状态 | 两份当前状态均等于各自历史末条，且都不同于本次要比的两个相邻状态。相对对比度提高后还分别有 4、8 个字段不同 |
| 归档预览 | 完整读到 479,232 字节的 `previews.db`。SQLite 自检通过，含 240 个图像索引项、233 个预览及 987 个分辨率条目。两个版本的图像键、全局标识和三个已知状态摘要均无精确匹配 |
| 本机复原 | 当前已核查的程序目录、开始菜单、命令与安装登记没有可用的 Lightroom／Camera Raw 渲染器。未验证旧管线，也未运行替代曲线 |

本轮只据范围响应读取目录页、完整预览索引、网页头段和 DNG 容器头，没有读取照片像素或预览文件体。总响应体不足 512 KiB。不能把最终 TIFF 或自定义对比曲线当作历史中间成片。将这对历史保留为明确操作证据，结束它作为当前可出图对照的准入。若取得兼容的渲染条件，可重放并验证四个状态；现阶段明暗迁移测试仍使用可复现实拍对照，肤色、白墙和独立画质验收未完成。[FiveK 发布说明](https://data.csail.mit.edu/graphics/fivek/) · [数值与哈希](2026-11-19-fivek-contrast-recovery-evidence.json)

用 `audit_fivek_contrast_recovery.py <本机页缓存> <新结果目录>` 复现。两遍的目录摘要、私有重放包与预览索引逐文件哈希一致。独立复核两份历史仅改变对比度、全部 233 个预览键的精确匹配结果及 12 个官网文件响应头。上传内容限脚本、协议和脱敏摘要。[脚本](../../experimental/adaptive-reference-20261113/audit_fivek_contrast_recovery.py)
