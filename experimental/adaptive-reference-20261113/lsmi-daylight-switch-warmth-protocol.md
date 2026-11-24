# 用 LSMI 的真实辅助灯替换复核局部暖色

白蓝和红白彩灯的变化过大。延续已验证的中性色偏问题，本轮只问共同保留灯 1 并替换一盏辅助灯，采用同一白平衡预设显影时，是否存在方向和幅度都接近既有局部暖色开关的样本。LSMI 原作者在三光源场景中实际拍摄了 12、13、123 三档。12 与 13 都保留灯 1，替换灯 2 或 3。两档与 123 的比较是加灯或减灯，单独记录。固定相机设置及三脚架来自作者的采集说明。光色和光量差是否足够小由实际成片检验，日光预设不提供逐档显影或环境光稳定的证明。[原拍摄，第 3.2 节和表 2](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Large_Scale_Multi-Illuminant_LSMI_Dataset_for_Developing_White_Balance_Algorithm_Under_Mixed_Illumination_ICCV_2021_paper.pdf) · [衍生显影，第 5 节](https://openaccess.thecvf.com/content/ICCV2025/papers/Serrano-Lozano_Revisiting_Image_Fusion_for_Multi-Illuminant_White-Balance_Correction_ICCV_2025_paper.pdf) · [彩灯边界](../../docs/research/2026-11-24-chromamill-warmth-match.md)

固定公开 Nikon ZIP 的原始日光档 `input/Place…_(12|13|123)_D_CS.png`。目录核验表明 955 场有这类输入，其中 39 场有完整三档。按数值编号升序，以 `i × (39−1) // 5` 固定六个布景，分别为 Place916、923、931、938、946、954。逐一核验 ZIP 区间、原始路径和 CRC。只取 18 个原始输入成员，不读作者白平衡目标、遮罩版本或其他显影档。预计 25–45 分钟，小样下载合计不超过 20,000,000 字节，目录另计 4 MiB。上游数据标 CC BY-NC 4.0，衍生版未单独发布图片许可。仅用作本机课程研究诊断，无训练、素材上传或商业交付。[原许可](https://github.com/DY112/LSMI-dataset#-dataset-license) · [数据发布](https://color.cvc.uab.es/revisitingmiwb/#download)

沿用原有 ICC 读取。无嵌入配置的 PNG 按衍生论文公开的 sRGB 解释，固定同一图像尺寸和方向处理。仅从每个场景 12 档选择一块自然出现的浅色材料和一个远处保护区，不用色卡作为用户会调暖的目标。场景缺材料或保护区时保存理由，不换同来源其他场景。固定测区再读取 13、123。输出固定 RawTherapee 暖色 1600、2800、力度 30 的开关，直接使用原共同支持、信号、余弦及幅度线。整图初筛沿用零位梯度相关至少 0.60、最佳整数位移 0 和改善不超过 0.02。拟准入方向须检查编辑区及其周围至少两处可辨认的固定地标，记录坐标、材料对应、遮挡和边缘污染。缺少足够信息时记录为局部对应未确认，不移动原评分框追分。原有暖色追加端点 2%、保护区不超过 1.5 RGB8 等素材线照旧。[原暖色准入](shared-local-warmth-protocol.md)

固定分别报告 12 个辅助灯替换有序方向和 24 个加减灯方向。保持固定白平衡也可能存在源 ACR 逐图处理，预览读数仍是完整可见 RGB。无命中即结束本批六组检测，不扩大测试名单追分。若某组替换方向命中且通过地标确认，保留测区外和保护区的真实变化，再在当前两版工作台做学术对照。产品训练和正式验收仍需要符合原许可和画质条件的来源。
