# 灰球筛出一组近等亮度浅色墙的实拍对照

上一轮用 14／24 号灯位，四处墙面均因亮度变化或裁切失去资格。本轮复用两处场景原坐标和四张照片，每处另取六个固定灯位。先看灰球有效覆盖、亮度和色度，选择符合门槛且色度变化最大的一对，再评估墙面。选灯不读取墙面表现。[事前协议](../../experimental/adaptive-reference-20261113/mid-gray-anchor-lights-protocol.md) · [原取片和反例](2026-11-21-mid-room-light-intake.md)

| 场景 | 灰球合格配对 | 选中灯位 | 墙面结果 |
| --- | ---: | --- | --- |
| 复印室 | 6／28 | 0／16 | 左墙合格，右墙失去资格 |
| 客厅 | 10／28 | 14／24 | 左墙缺测，右墙亮度变化超过限制 |

复印室左墙共同有效 12,413／12,420 像素，中位线性亮度比 0.908，色度差范数 0.324。灰球两维色度差为 −0.249／+0.237，左墙为 −0.236／+0.222，方向一致。右墙共同覆盖只有 32.14%，亮度比 0.485。客厅左墙无共同有效像素，右墙虽有 100% 覆盖，亮度比仍为 0.587。把这些不合格区域一并保留。灯位切换同时改变了受光几何，已通过的一块墙仅用于近等亮度的开发控制。[本机成对照片](../../outputs/adaptive-reference-20261113/mid-gray-anchor-selected-comparison.png) · [56 项筛选和文件身份](2026-11-21-mid-gray-anchor-lights-evidence.json)

新增 12 张 JPEG 共请求 3,305,062 字节，含两份 ZIP 尾部，保留 HTTP 206、ETag 和 CRC 检查。元数据和评分两遍一致，报告 SHA-256 为 `766c8e5df51edfc4c11d4d1b1e4cce2b20ea2fbadb226057b69ba7eeecf9408d`。下一步可在复印室用该实拍色度方向和匹配的人为染色构成两态对照。当前只有一个达标墙面，不能代表多场景、肤色或独立验收。软件及发布状态保持原样。

用已有视觉环境运行 [取片脚本](../../experimental/adaptive-reference-20261113/fetch_mid_room_pair.py)，在新输出目录后追加 `--directions 0 4 8 12 16 20`。再运行 [评分脚本](../../experimental/adaptive-reference-20261113/probe_mid_gray_anchor_lights.py)，参数依次为原 JPEG 目录、新 JPEG 目录、[上一轮取片记录](2026-11-21-mid-room-light-intake-evidence.json)和新的结果文件。
