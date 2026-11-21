# 固定物体和机位时，照明变化仍会明显传到跨物体成片

自然光实拍同时改变了太阳光量、光色和阴影。本轮加入 OpenIllumination 的受控拍摄。固定三个物体、CA2 机位与 001／013 两种灯光模式，取六张去畸变图片和三个作者物体蒙版。原作者使用固定架上的相机和可控灯光。灯光编号只用作离线核查。实际 13 号只读目标图和当前一张参考成片。[作者数据与许可](https://huggingface.co/datasets/OpenIllumination/OpenIllumination/blob/006beced878d3fb7187ded8e20ee63d9c5020fcb/README.md) · [作者采集说明](https://papers.neurips.cc/paper_files/paper/2023/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html) · [事前协议](../../experimental/adaptive-reference-20261113/openillumination-fixed-camera-protocol.md)

九个文件共 1,927,813 字节，按固定发布核验 LFS SHA-256 或 Git blob。相机记录均为 0.2 秒、f/16、ISO 100，白平衡字段值为 1。JPEG 未附 ICC，使用既有 sRGB 假设，长边等比缩至 768。目视核对三个物体轮廓和支座位置，无可见移动。物体内部在 001 与 013 的亮度梯度相关为 0.391、0.762、0.356，反映变化很强，不能证明逐像素受光相同。013 画面里可直接看到多个开启的灯具，本轮结果涵盖物体反射光和入镜灯具的共同变化，暂不归因到其中一项。未取得标定的相机色谱或照明配方。

用每个物体的 001 照片作目标。其余两个物体分别用两档实拍照作为参考，每档单独关闭或打开同一 Film B。原目标施加相同预设得离线预期，所有成片使用 80% 强度。沿用原联合门槛，换灯差、原图偏差和风格响应误差 ≤3 RGB8，风格投影 0.75–1.25。

| 目标 ← 参考 | Film B 开态换灯差，整图／物体内 | 物体风格投影，001／013 |
| --- | ---: | ---: |
| 蛋形物体 ← 立方体 | 3.38／21.77 | 0.163／0.303 |
| 蛋形物体 ← 纸球 | 3.35／21.37 | 0.278／0.357 |
| 立方体 ← 蛋形物体 | 2.18／9.31 | 0.593／0.502 |
| 立方体 ← 纸球 | 2.56／9.05 | 0.503／0.667 |
| 纸球 ← 蛋形物体 | 6.31／49.86 | 0.212／0.292 |
| 纸球 ← 立方体 | 5.77／43.19 | 0.204／0.123 |

换灯差为 RGB8 MAE。六个方向均未通过，物体内 12 个分灯风格响应均未达到 0.75。尤其在两组立方体成片上，只看开启态的整图换灯差会显示小于 3，物体内部仍超过 9。纸球反例说明必须分区评分。该受控库可以进一步隔离「直接可见的灯具」与「同物体的反射颜色」两项，作为后续判断参考中的不相关区域是否在带偏物体的检验。三组均为已消费开发素材，无人体或肤色验收结论，现有软件功能和独立验收状态继续保持。[纸球本机对照](../../outputs/adaptive-reference-20261113/openillumination-fixed-cross-v1/obj_43_paper_ball-from-obj_02_egg-comparison.png) · [立方体对照](../../outputs/adaptive-reference-20261113/openillumination-fixed-cross-v1/obj_21_cube-from-obj_02_egg-comparison.png)

## 复现

按[发布清单](../../experimental/adaptive-reference-20261113/openillumination-release.json)取九个文件，保留目录。将清单以 CRLF 保存于输入根目录的 `release.json`，其 SHA-256 为 `99f94ae7fc8cb2c7662eafbc403eda57d61db9e1b557188aabff454fd0c64812`。沿用[现有研究 Python 和搜索路径](2026-11-21-gros-grid-expression.md)，执行 `probe_openillumination_fixed_camera.py prepare <输入目录> <新的显影目录>`。目视核对三个采集组后，执行 `probe_openillumination_fixed_camera.py run <显影目录> <新的出图目录> <inputs.json的SHA>`。

两个准备目录各 13 个产物、两次渲染各 31 个产物逐文件哈希一致。另独立由保存的 PNG 复算 24 项分灯风格响应及其成对换灯差。报告 SHA-256 为 `6b9f2c7af57709c4f04b236a3a798217777f17a85724121fd9bb32beefdf0d02`。相机照片、蒙版和成片保留本机。[验收数值与身份](2026-11-21-openillumination-fixed-camera-evidence.json)
