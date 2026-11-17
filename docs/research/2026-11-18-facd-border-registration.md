# 同坐标去黑边后，FACD 七组通过结构复查

复用固定八组 32 张 JPEG，比较原图尺寸、共同内部裁切及去黑边后放大。通过条件保持全图相关至少 0.90、最佳整数位移为零、象限提升不超过 0.01。每组用三个滤镜统一裁切，保持原始像素，不按结果选取算法。见 [事前协议](../../experimental/adaptive-reference-20261113/facd-border-registration-protocol.md)和[原样本](2026-11-18-facd-range-pilot.md)。

| 处理 | 通过配对 | 三种滤镜均通过的原照片 |
| --- | ---: | ---: |
| 原图与公开成片 | 9/24 | 3/8 |
| 双方裁取同一内部区域 | 23/24 | 7/8 |
| 成片边框内部放大后再裁取 | 0/24 | 0/8 |

检测出的黑边宽度在 8–12 像素之间。同坐标裁切成功而重新缩放失败，支持黑边主要覆盖原画面边缘。未通过的是 488102 的 Amaro 婴儿近照。对齐分数从 0.305 提高到 0.741，仍低于阈值。去边后能看到 Amaro 暗角和强亮部变化，其中没区分其影响与剩余配准误差。保留该失败。七个通过整组达到原先至少六组的标准，可作同滤镜的受控开发对照。

两次运行的 56 份 PNG 和结果 JSON 哈希一致；另独立从源 JPEG 核对 32 份直接裁切逐像素一致。查看全部八组内部图片。婚纱及肤色在不同滤镜下依然有较强颜色变化，配准达标只确认配对可比较。它不证明滤镜具备保护肤色或中性色的语义，也未证明素材与历史库无重叠。见[逐组结果](2026-11-18-facd-border-registration-evidence.json)。

本机目录为 `outputs/adaptive-reference-20261113/facd-border-registration-v1`，含 `overlay-overview.jpg` 和两种裁切的 PNG。复现时在仓库根目录设置 `$env:PYTHONPATH = (Get-Location).Path`，使用研究 Python 运行 `experimental/adaptive-reference-20261113/probe_facd_border_registration.py <FACD 原图目录> <小样证据 JSON> <新结果目录>`。图片留本机，软件未更换。
