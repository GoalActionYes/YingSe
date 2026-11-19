# 人物 Film B 对照补齐明暗与色度测量

保留旧 HSL／曲线对照及其门槛。在同四个已消费人物目标、八张参考原图上，使用固定 Film B 和既有 SDR 显示转换。四组目标的已知明暗变化为 4.04–8.63 RGB8，色度变化为 1.87–3.99 RGB8，均达到原 1 码值的测量下限。人物和近中性色覆盖也达标。四张目标原图的无风格显示往返逐像素相同。[事前协议](../../experimental/adaptive-reference-20261113/bg-real-film-b-qualification-protocol.md) · [原套件的测量限制](2026-11-19-paired-cubic-cross-content.md)

| 检查 | 13 号两图 | 扩展第 4 轮两图 | 离线参考前后图 |
| --- | ---: | ---: | ---: |
| 四组人物联合通过 | 0/4 | 0/4 | 1/4 |
| 白球服 C1／F1 明暗投影 | 0.247／0.364 | 0.423／0.481 | 0.940／0.949 |
| 白球服 C1／F1 色度投影 | 0.196／0.293 | 0.745／0.812 | 0.926／1.004 |
| 白球服开启后近中性色换背景差 | 5.05 RGB8 | 3.41 RGB8 | 0.77 RGB8 |
| 四组最坏保护区换背景差 | 30.74 RGB8 | 26.00 RGB8 | 2.25 RGB8 |

两图模型的 16 张关闭态逐像素复现此前对应归档。离线三次拟合只在白球服同时达到风格与保护要求。另两组在有意风格投影通过时仍有保护区换背景差超出 1.5 RGB8。9483–19432 的 F1 色度投影为 0.734，且保护区差最高 2.25。保留全部退化，两个普通两图候选和离线重放均未达到三组联合通过的要求。[白球服扩展成片](../../outputs/adaptive-reference-20261113/bg-real-film-b-qualification-v1/expanded-872-1268-comparison.png) · [白球服离线对照](../../outputs/adaptive-reference-20261113/bg-real-film-b-qualification-v1/paired-872-1268-comparison.png)

这是一套可同时评分明暗和色度的新增开发对照，照片来源已用于开发，不能作为独立验收。此前六个非换背景 Film B 方向的离线成功仍成立，这组实图说明它不能代替人物保护。未改产品，也未更改原门槛。

既有研究环境下，将仓库根目录与本机 OpenColorIO 加入 Python 模块路径，设 `OPENBLAS_NUM_THREADS=4`，运行 `python probe_bg_real_film_b_qualification.py <新输出目录>`。两遍的 12 张已知效果或参考、48 张对照成片、12 张对照图和报告逐件一致。从落盘图复算 48 项投影和 24 项近中性色差。[复现脚本](../../experimental/adaptive-reference-20261113/probe_bg_real_film_b_qualification.py) · [数值和哈希](2026-11-19-bg-real-film-b-qualification-evidence.json)
