# 取得真人混合光下可独立开关的局部调色

此前用作者真值纠正保色位置后，现有改色预测仍无法恢复预期风格。本轮取得一份作者操作可单独开关的真人照片。原片作者 Joris Mak 标注 CC BY-NC-SA 4.0，编辑者 Carlos 说明用 RGB 色彩平衡处理脸与手，并另用蒙版曝光。公开发布的侧车为原编辑设置。原件、侧车、蒙版及成片均仅在本机作非商业研究。[原片与许可](https://discuss.pixls.us/t/handling-shade-direct-sunlight-with-skin-tones-filmic-sigmoid-preference/33540/1) · [编辑说明与 XMP](https://discuss.pixls.us/t/handling-shade-direct-sunlight-with-skin-tones-filmic-sigmoid-preference/33540/2)

原片 13,638,412 字节，XMP 为 22,912 字节，均与发布附件标识的 SHA1 一致。XMP 历史第 27、28 项是同一 RGB 色彩平衡实例，第 28 项为最终状态。混合参数的模式为 3，启用手绘蒙版。关态仅把第 28 项的启用位从 `1` 改成 `0`。两个配置只差一字节，曝光与原构图保持。选用编辑者注明的官方 Windows 4.0.1，112,740,008 字节安装包匹配官方 SHA256，用已有 7-Zip 解包至本机研究目录，没有安装。[事前协议](../../experimental/adaptive-reference-20261113/playraw-mixed-light-skin-protocol.md) · [混合模式源码](https://raw.githubusercontent.com/darktable-org/darktable/release-4.0.1/src/develop/blend.h) · [官方工具发布与哈希](https://www.darktable.org/2022/09/darktable-4.0.1-released/)

| 蒙版区域 | 像素数 | 开关色度差中位数（RGB8） | 色度差 P95（RGB8） |
| --- | ---: | ---: | ---: |
| 权重 ≥80% | 5,824 | 24.50 | 28.00 |
| 权重为零 | 651,445 | 0.00 | 0.00 |

符合预定准入。高权重像素中，去亮度后的红、蓝分量变化中位数分别为 +12.91 和 −36.59 RGB8。已得到改色方向、区域和区外保护约束，可用于下一轮局部风格开发。此处按作者实测蒙版评分，暂未将像素标为脸颊、手部或墙面，也不能将一个原片计作跨场景迁移、独立验收或自然度结果。[既有判断与改色失败](2026-11-20-tst-conditional-oracle-decision.md)

## 复现与身份

在仓库临时 ASCII 路径下，将研究视觉环境的 `PYTHONPATH` 指向仓库根，运行 `python experimental/adaptive-reference-20261113/probe_playraw_mixed_light_skin.py <原片、侧车、同版工具所在目录> <新输出目录>`。首遍实际生成两个 TIFF，蒙版页读取因只匹配英文页名而失败。根据 TIFF 标签加入中文页名及 UTF-8 处理后，在不同目录重新完成两遍，评分阈值保持不变。两遍区域读数一致，关态有 170 个通道值、开态有 54 个通道值、蒙版有 2 个像素值相差 1 码值。独立从 TIFF 用整数权重复算。四份主图均为 1024×702、sRGB8。首遍正式报告 SHA256 为 `b5717f8adb5920abfaa1f8be6e207875aa519b90b4f161a6b61d1b73418e82d9`，复跑为 `03458bec3f474486ebc2b74fa23380d2b843540a4b9b756df608d9ea2231c5c6`。产物位于本机 `outputs/adaptive-reference-20261113/playraw-mixed-light-skin-v1`，Git 只保存协议、脚本和数值。现有软件未修改。[复现脚本](../../experimental/adaptive-reference-20261113/probe_playraw_mixed_light_skin.py)
