# 近场人像灯组能测换方向，缺少同位置换色的配对

核对 Quéau 等作者 near_ps 的固定发布。两份真人拍摄各有八张单灯 RAW PNG，另有环境、全灯和区域文件。作者明确允许将所附数据用于研究并要求引用。各单灯原件约 11.89–12.38 MB，可按文件取用。此次只读取目录和 918 字节的灯光标定，未下载照片。[作者说明与固定来源](https://github.com/yqueau/near_ps/tree/30881339ddac975c1faab3df5126b49034e22c13) · [事前规则](../../experimental/adaptive-reference-20261113/near-ps-portrait-intake-protocol.md)

八盏 LED 的 28 对标定 RGB 方向差为 0.223°–3.447°，位置距离为 105.35–490.58 毫米。颜色差最大的 4／7 号灯相距 266.38 毫米。距离最近的 7／8 号灯相距 105.35 毫米，色度角差 3.136°。没有一对同时满足找样片用的 ≥3° 色度差和 ≤5 毫米位置差。色度角按作者标定的线性 RGB 强度计算，可用来比较数据中的灯光配置；它不能代替感知色差或成片肤色得分。

这套素材适合后续检查光照方向、局部高光和阴影变化，无法用现有拍摄配对单独归因于灯色。保留两个原始拍摄家族，暂不消耗此批照片。另有 Meyer 等人的实验在同位置聚光灯上换红、绿、蓝滤片、固定相机设置，并有白色标定面。该设计更贴合本项目，但截至作者论文和现有公开入口核查，尚未定位到可用原图及使用范围。论文拼图不进入评分。[同位置滤片的作者实验，§4](https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2013/Meyer13-CCI.pdf)

标定文件的 Git blob 为 `9ec297ca5346e5702bdac79ead7e050a7604ff57`，SHA256 为 `fbfec9d8fd4cf3f0929ac8dbdebcef5ea700b684da5c25cdfaa66ece58b8670f`。树响应 SHA256 为 `994d7da25687c2a9765ec21c669318d4339feb64a97479239ff116697d23bb23`。两遍报告逐字节一致，并用向量叉积、点积和 `atan2` 独立复算 28 对。用已有视觉研究 Python 与 SciPy 路径 `.tools/DeepWB-audit/vendor` 执行 `audit_near_ps_lights.py outputs/adaptive-reference-20261113/near-ps-portrait-intake-v1`。这次仅缩小样片选择范围，肤色、白墙与有效风格的实际画质尚未验证。
