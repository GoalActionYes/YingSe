# 三批后续开发素材已复核封存输入交叉身份

已有两套封存输入可以用于正式独立验证前的身份检查。原 BG-REAL 核查的对照名单和 PST50 旧报告完成后，项目又使用了多组实拍及编辑图。这轮追加核查八张 CHROMA-MILL 实拍 sRGB 原态、八张 PSBench 原始 JPEG、九张 COCO 屏幕人物照片。原报告及文件身份按哈希固定，没有读取封存目标真值、作者效果或视频。[事前协议](../../experimental/adaptive-reference-20261113/reserve-recent-source-crosscheck-protocol.md) · [BG-REAL 原资格检查](2026-11-19-bg-real-reserve-qualification.md) · [PST50 原指纹检查](2026-11-18-pst50-reserve-fingerprints.md)

| 封存输入 | 追加配对比较 | 最小指纹距离 | 精确或近重复预警 |
| --- | ---: | ---: | ---: |
| BG-REAL，24 张 | 600 | 18 | 0 |
| PST50，70 张 | 1,750 | 14 | 0 |

BG-REAL 核对已锁定文件后，只对输入生成已有的全图、水平翻转、中心裁切及翻转指纹。PST50 读取旧指纹及哈希。两者均使用 ≤6 位的预警阈值。25 张只覆盖这三批后续素材，跨镜头、强裁切、连拍及尚未经核对的其他开发来源仍可能漏检。原 10 个 BG-REAL 技术合格配对和 PST50 待核名单保留。PST50 仍缺逐组拍摄或镜头映射。此阶段不能解除封存或作为正式画质验收。

两次摘要逐字节一致。首次启动误抄了锁定资格报告的一个十六进制字符，核对文件身份后更正，未变更图像或比较范围。报告 SHA256 为 `87ce95d69c2b04bb9daca1f2ad1fe8a45197775e8e0331e5342fcc8ffeb60eed`。复现用已有视觉环境，将仓库根目录加入 `PYTHONPATH` 并运行脚本，传入新输出目录。输入、完整指纹和收据保留本机，现有软件没有改动。[复现脚本](../../experimental/adaptive-reference-20261113/probe_reserve_recent_source_crosscheck.py)
