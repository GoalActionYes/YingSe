# 换拍摄灯会改变同一 Film A 的实际迁移

用 MLS 场景 02、03 互作目标和参考。每次保持目标及其拍摄灯，参考用另一场景的标准灯和弱蓝灯两张实拍。给两张参考分别开关同一个已知 Film A，固定 13 号 80% 强度。保存八张目标成片，用目标本身施加 Film A 的效果作为离线参照。[事前协议](../../experimental/adaptive-reference-20261113/mls-film-light-cross-protocol.md) · [实拍灯位与输入编码](2026-11-18-mls-linear-pilot.md)

| 目标 ← 参考 | 标准灯风格投影，整图／近中性 | 弱蓝灯风格投影，整图／近中性 | 同款换灯时的近中性 RGB8 差，关／开风格 |
| --- | ---: | ---: | ---: |
| 02 ← 03 | 0.241／0.335 | 1.344／0.560 | 4.18／9.16 |
| 03 ← 02 | 0.949／−0.084 | 1.083／0.260 | 8.62／9.62 |

投影 1 表示达到该区域的完整风格方向与幅度。03 用标准灯参考时整图投影接近 1，近中性色却反向；02 仅换参考灯就从偏弱变为整图过强。八个区域／灯位的风格响应均未同时满足原投影与 MAE 条件，两方向换灯差也都超过 3 个码值。换灯差度量 RGB，包含明暗和色度。该受控任务要求跟随数字编辑，目标灯保持；一般用户也可能有意请求迁移参考灯光。样本是两处实验室场景，未覆盖真实肤色或开放风格。

四份 MLS 输入和 Film A LUT 按旧哈希核对。沿用 SDR 逆变换后施加已知编辑，无风格往返在四张 RGB8 输入上逐像素保持。MLS 相机色域未校准，Film A 在此只定义一个可复现的数字操作。两次运行的四张准备图、八张成片、两张面板和报告逐字节一致。另从落盘图复算八项风格响应与四项换灯误差。现用自动结果未达到此受控任务门槛，保留开发对照，未修改软件或使用独立验收集。[02 号面板](../../outputs/adaptive-reference-20261113/mls-film-light-cross-v1/02-comparison.png) · [03 号面板](../../outputs/adaptive-reference-20261113/mls-film-light-cross-v1/03-comparison.png)

用既有研究环境，把仓库根目录、OpenCV、OpenColorIO 加入 `PYTHONPATH`，运行 `probe_mls_film_light_cross.py <本机研究输出根目录> <新输出目录>`。媒体留本机。[测量脚本](../../experimental/adaptive-reference-20261113/probe_mls_film_light_cross.py) · [数值与哈希](2026-11-19-mls-film-light-cross-evidence.json)
