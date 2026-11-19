# 参考背景灰化改善部分肤色，也加重其它场景的偏色

《Cuckoo》对照显示自动肤色权重能承载一项已知修色。本轮只测试普通目标与参考输入。候选用冻结肤色模型给参考生成软权重，把非皮肤颜色向同亮度灰色混合，再用同一 13 号和 80% 强度分析。只把新支路的色度按目标皮肤权重替换到基线，明暗与未选区域沿用基线。任一图没有可靠皮肤时返回基线，同图直接返回。已知成片与区域只用于离线评分。[事前协议](../../experimental/adaptive-reference-20261113/reference-skin-chroma-isolation-protocol.md) · [已知变化重放](2026-11-20-cuckoo-skin-support.md)

| 有意编辑打开时，换参考背景的区域色度差 | 13 号 | 候选 |
| --- | ---: | ---: |
| 白球服案例，前景手臂 | 3.17 | 1.16 |
| 白球服案例，后方脸颊 | 3.18 | 2.10 |
| 785–3553，脸部 | 3.01 | 6.75 |
| 9483–19432，前额 | 1.71 | 6.67 |
| 9483–19432，手背 | 2.59 | 7.07 |
| 8021–8690，投影幕中的脸 | 26.33 | 3.37 |

数值以 RGB8 色度码值计。雪地、浅墙和白球服的登记框与基线读数相同。投影幕内的人脸仍参与了肤色分支，因此它的换背景差下降不能视为屏幕误改已解决。白球服案例的有意色度投影由原背景／换背景 0.159／0.090 降为 0.142／0.075。模型保留了这组的部分皮肤稳定收益，同时进一步削弱有意色度迁移。[白球服本机成片](../../outputs/adaptive-reference-20261113/reference-skin-chroma-isolation-v1/background/872-1268/comparison.png) · [9483 实拍退化](../../outputs/adaptive-reference-20261113/reference-skin-chroma-isolation-v1/background/9483-19432/comparison.png)

四套检查都未过原门槛。换背景与有意编辑为 0/4，FACD 为 0/7，Film B 为 0/6，色卡为 1/7。背景历史协议的两组色度增量缺测保持原判定。实体色卡场景的右侧脸颊在关闭风格时，色度误差由 7.51 增至 15.12 RGB8。固定灰化参考背景无法同时提供稳定肤色和有效迁移，结束这条取色路线。后续须用能识别编辑意图的线索并检查实际人物、浅色材料和正常风格。[色卡与人脸退化](../../outputs/adaptive-reference-20261113/reference-skin-chroma-isolation-v1/chart/comparison.png)

两遍各 72 张成片、18 张对照图和报告，共 91 件逐文件同哈希。36 张应用肤色支路，24 张无可靠目标皮肤、12 张无可靠参考皮肤而回退。同图和无皮肤负控通过，复查全部落盘成片及零权重像素，基线明暗误差最大为 0.5 RGB8。报告 SHA256 为 `a2977240820be621b6c4acfddb1cb37b2798269cbead4b4bd8c07b5f48b2fe3f`。复现用现有视觉环境，设置 `PYTHONPATH`、`OPENBLAS_NUM_THREADS=2`、`OMP_NUM_THREADS=2`，执行 `probe_reference_skin_chroma_isolation.py <新输出目录> <旧肤色保护运行时的 site-packages>`。这批样例均为既有开发素材，软件版本、正式画质验收和视频验证未改变。[实验脚本](../../experimental/adaptive-reference-20261113/probe_reference_skin_chroma_isolation.py)
