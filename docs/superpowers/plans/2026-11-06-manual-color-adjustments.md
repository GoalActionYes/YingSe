# Manual Color Adjustments Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. Steps use checkbox syntax.

**Goal:** 在参考图匹配后提供集中、可重置的基础／HSL／曲线微调，并保持图片视频导出一致。

**Architecture:** 独立 adjustments 模块校验版本化参数并合成单一 LUT，旧中性请求保持原路径。服务器、任务和项目保存同一参数快照；右侧新增三页签，主程序先交付，肤色副本不改。

**Tech Stack:** NumPy、Pillow、FFmpeg、原生 JavaScript／CSS，沿用现有依赖。

**Spec:** ../specs/2026-11-06-manual-color-adjustments-design.md

## Global Constraints

- 默认中性、原图对照独立于克隆开关；手动调色不启动模型。
- CUBE 只包含全局颜色变换，暂不加空间效果和无效控件。
- 视频维持既有时间戳和音轨；不称剪映像素等价。

## Review Focus

- 非法浮点、布尔和未知字段须拒绝。
- 快速输入及异步迟到预览不能回滚画面。
- 克隆关闭仍可手动调整；原始比较始终无调整。
- 老项目和中性参数保持兼容。
- 极端曲线／HSL 不出现 NaN；导出冻结完整设置。

## Tasks

- [x] 1. 新增 tests/reference_color/test_adjustments.py，先验证缺模块失败；实现 reference_color_app/adjustments.py 的 normalize、transform、compose；测试中性、方向、环绕、非法数据和 LUT 同源。
- [x] 2. 扩展 server.py、jobs.py 的可选 adjustments 请求，保存项目与导出快照；写 HTTP 和真实导出测试，验证旧请求继续通过。
- [x] 3. 新增 web/adjustments.js、调整 index.html／styles.css／app.js，三页签、分组开关、单项重置、曲线编辑及恢复；检查 JS 语法与浏览器操作。
- [x] 4. 在独立依赖环境运行源码检查、生成色卡和短视频验证，检查预览／PNG／CUBE、音视频与项目恢复，更新状态、操作指南和变更记录后统一提交上传。

## Execution ledger

- 用户已审阅方向并于 11 月 6 日明确要求自行制作和测试、次日交付。按该授权直接执行本计划，不再要求额外阶段确认；在当前仓库的 codex/manual-color-adjustments 分支工作，避免创建新入口加剧目录混乱。
- 使用 inline 执行。用户要求不为小变化上传，因此完成批次后一次推送。

- 完成记录：颜色模块先观察缺模块失败再实现；HTTP 微调请求先观察 400 后接入；独立审查发现恢复竞争与报告来源缺失，均增加失败回归并修复。浏览器实际操作发现页面拉长与滚动残留，已改正。合成三格式导出、便携 EXE 生命周期与两套依赖环境验证详见 research/2026-11-06-manual-controls。
