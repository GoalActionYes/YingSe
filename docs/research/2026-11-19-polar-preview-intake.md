# POLAR 预览可定位人像换灯帧，均匀光对照仍待核实

前一轮的人像和鸟照无法固定同一局部编辑设置。现有蓝／白灯实拍对照则没有人物。本轮检查 POLAR 是否能补充实拍人像的光照变化，再用于交叉验证固定色彩编辑。只核对官方预览和作者编号，未进行调色或画质评分。[事前协议](../../experimental/adaptive-reference-20261113/polar-preview-intake-protocol.md) · [作者论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_POLAR_A_Portrait_OLAT_Dataset_and_Generative_Framework_for_Illumination-Aware_CVPR_2026_paper.pdf)

按预先固定的顺序检查 028、030。用公开版本、发布长度和 LFS 标识核对 HTTP 206 响应，每轮只读 ZIP 尾部和中央目录，合计 2,132,690 字节。每个包有四个表情目录、16 个机位，每个表情／机位恰有 157 张 JPEG。两次报告和完整目录逐字节一致。[固定预览版本](https://huggingface.co/datasets/Rex0191/POLAR/tree/77ec5422bc9003c202116575ca7f17a25dd85157)

| 核对项 | 两个预览包的共同结果 |
| --- | ---: |
| 每包 JPEG 数 | 10,048 |
| 每包表情／机位组合数 | 64 |
| 每组能按文件名匹配到公开光照表的帧 | 149 |
| 每组未列入该表的帧 | 8 |
| 独立的前景蒙版文件 | 0 |

光照表另按固定 Git blob 复核。未列入表的帧按每 21 个文件出现一次，尚无逐帧说明足以认定它们就是均匀光。关于论文中其余方向、预览帧身份和 JPEG 配准，本轮没有验证。公开的 `full` 目录只列出一个完整样本。数据卡显示 `openrail` 字段，作者处理说明将这个公开入口标为预览，使用范围仍待澄清。[固定光照表](https://github.com/Rex0191/POLARNet/blob/9580844bf52cce3ddfa19200c3deec3358e8be7c/OLAT_data_processing/light/light_157.txt) · [作者发布说明](https://github.com/Rex0191/POLARNet/blob/9580844bf52cce3ddfa19200c3deec3358e8be7c/OLAT_data_processing/README.md)

该来源已证实可以按范围定位两组人物拍摄。按事前规则暂停该预览的像素取样，保留已核对的光照帧、待核实的均匀光和使用范围。它只涉及拍摄光照，本身没有同款后期调色记录。现有产品和独立验收集没有改动。

复现用带 `requests` 的 Python 执行 `audit_polar_preview.py <新输出目录> [--proxy <本机代理>]`。以 `inventory.json` 中的表情／机位和 JPEG 文件名与固定光照表做集合对照。完整目录留本机。首次用不带 `requests` 的环境在导入处退出，改用现有视觉环境，两次均成功。[复查脚本](../../experimental/adaptive-reference-20261113/audit_polar_preview.py) · [数值与目录哈希](2026-11-19-polar-preview-intake-evidence.json)
