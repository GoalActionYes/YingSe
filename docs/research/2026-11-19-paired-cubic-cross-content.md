# 给定参考前后图后，Film B 六个跨场景方向通过重放检查

参考前后图由既有素材记录确认属于同一原片。固定上一轮的三次颜色模型，只用配对参考拟合，保持普通 RGB8 渲染和 80% 强度。目标效果和人物区域只在离线评分时读取。两套共生成 14 张新的开启态，关闭态为目标原图。[事前协议](../../experimental/adaptive-reference-20261113/paired-cubic-cross-content-protocol.md) · [普通两图的冻结对照](2026-11-19-frozen-candidate-development-crosscheck.md)

| 检查 | 普通两图扩展第 4 轮 | 给定参考前后图 |
| --- | ---: | ---: |
| Film B 六方向联合 | 0/6 | 6/6 |
| 白球服两背景整图明暗投影 | 0.276／0.229 | 0.962／0.955 |
| 白球服两背景整图色度投影 | 0.176／0.118 | 0.748／0.802 |
| 白球服换背景时近中性色差，编辑开启 | 3.871 RGB8 | 0.366 RGB8 |
| 四组人物背景原联合 | 0/4 | 0/4 |

Film B 的整图投影为 0.802–1.201，近中性色为 0.847–0.964，范围内的最大近中性色响应误差为 2.55 RGB8。色卡作为参考时，拟合参考留出的 P95 为 5，原自动配准限制是 4。两次重放仍通过目标评分。由于配对身份在试验中已知，保留这两张离线成片，原认证门槛仍适用现有软件。[焊工本机对照](../../outputs/adaptive-reference-20261113/paired-cubic-cross-content-v1/film_b/sparks_4592-from-hdr_testimage-comparison.png)

白球服 C1 色度比 0.75 的原下限低 0.002，另一个场景的保护区换背景差为 1.81，高于单组 1.5 的上限。复核还确认此背景套件的两个目标已知色度增量小于原评测要求的 1 个码值，记为缺测。原规则将缺测的组判未通过，因此这组固定的四个目标最多只能有两组联合通过，无法达到既定的三组。保留原判定。这套素材仍可用于单独检查明暗、保护区域和明确可测的有意色度，后续整体开发判定需要预先冻结可实测对应维度的新增素材。[白球服本机对照](../../outputs/adaptive-reference-20261113/paired-cubic-cross-content-v1/background/872-1268-comparison.png)

结果支持把配对参考作为 Film B 的离线可达对照。它仍需要额外的参考原片，无法作为普通两图的候选。人物背景联合未通过，停止用这套固定拟合宣称跨场景修复。真实自然评价、视频及软件升级仍待完成。

在既有研究环境中将仓库根目录加入 Python 模块路径，设 `OPENBLAS_NUM_THREADS=4`，执行 `python probe_paired_cubic_cross_content.py <新输出目录>`。两遍 14 张原尺寸图、10 张对照图和报告逐件一致，并从落盘图独立复算两套主要指标。[复现脚本](../../experimental/adaptive-reference-20261113/probe_paired_cubic_cross_content.py) · [数值与哈希](2026-11-19-paired-cubic-cross-content-evidence.json)
