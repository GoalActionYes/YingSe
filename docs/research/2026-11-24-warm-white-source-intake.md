# SFU Solux 与 CML 浅色实拍的来源核验

需要一组可追溯的独立实拍，判断同一浅色物体受小幅暖光变化时，能否与已有固定局部暖色编辑形成近似对照。上一轮 Ribeira 真实日光主要改变亮度，并在测区产生高光裁切。这次先核对已记录光源的 SFU Solux，再核对带现场测光的 CML 实拍墙面。来源核验决定能否出四图。完成此项后，仍需实际暖色对照过线才讨论训练。[前一轮实拍结果](2026-11-24-ribeira-daylight.md) · [准入门槛](2026-11-24-spatial-condition-design.md)

## SFU 低反光档案缺少预选场景

作者用 11 种实物灯照同一布景，其中包含未加蓝滤片的 Solux 3500、4100 和 4700 K。作者总览使用一个光源、混排低反光与有介质反光的场景。先从总览登记 `plastic-1` 的右侧浅色容器，计划在 4100 K 原图确认无印刷和反光的本体可用面积。左侧物体只用于测同档编辑泄漏。按事前范围只下载低反光 8 位原片包，未读取其它场景的换灯像素。[SFU 总览及档案](https://www.cs.sfu.ca/~colour/data/colour_constancy_test_images/mondrian/index.html) · [作者论文](https://www2.cs.sfu.ca/~colour/publications/CRA-2002/CRA-2002.pdf) · [选择与停止条件](../../experimental/adaptive-reference-20261113/sfu-solux-warmth-protocol.md)

收到的档案为 73,011,656 字节，响应 ETag 为 `"45a11c8-3de69edf01ac0"`，SHA-256 为 `b503331dbb4299482959b49e825d6ac17f3cf47def1bb74a10246f26e599475b`。用系统 `tar` 与 Python `tarfile` 分别列目录。低反光档案有 22 个场景、223 张 TIFF、223 个 `.rgb` 和 223 个 `.illum`。61 张来自上述未加滤片的 Solux 档位，逐一具有同名的两类测量，19 个场景三档齐全。22 个场景中没有 `plastic-1`，任何文件名都没有 `plastic`。另一份有介质反光的档案说明提到塑料物体，未取件，也未将其作为低反光场景。既定候选在来源阶段停止，未换成色卡、纸片或其它物体。核对素材也发现发布包内谱文件扩展名为 `.illum`，作者概述使用的是 `.spect`。[低反光发布目录](https://www.cs.sfu.ca/~colour/data/colour_constancy_test_images/mondrian/index.html) · [介质反光档案说明](https://www.cs.sfu.ca/~colour/data/colour_constancy_test_images/specular/index.html)

EGM 第 5 阶段复核指出，源相机型号、光谱和标定资料足以定位相机域，显示域还需要与已线性化的 TIFF 兼容的固定转换。光源距场景会为控制裁切而调整，不能把变化完全归因于光谱。即使该容器通过测区和暖色门槛，仍需核查和现有浅墙的用途与材料可比性。本轮没有对相机码值做 sRGB 映射，所以 RGB8 暖色准入记为未评估。[相机标定说明](https://www2.cs.sfu.ca/~colour/data/camera_calibration/index.html) · [EGM 意见落实](../../experimental/adaptive-reference-20261113/sfu-solux-warmth-protocol.md)

## CML 已测现场照明，仍缺两张照片的曝光记录

从公开的灯具光束表选择同一 ETC fos/4 PL24 的无附件和标准柔光片两档。两档中心照度分别 2650 lx 和 2480 lx，比值 1.069，测得色温分别为 5219 K 和 5509 K。作者说明光束照同一面墙，用固定机位的 GH5、7 mm 镜头、5600 K 白平衡和标准照片风格，后处理为网页缩图。两档比较的是实际光束与附件共同变化，柔光片也可能改变照度分布。选对依据来自该表，未读成片颜色来挑选。[ETC 原始测量与照片](https://cinematography.net/2024-LMU-tests/etc_fos_4_pl24.html) · [拍摄方法，第 II 部分](https://www.cinematography.net/2024-LMU-tests/clTestSetup.html)

按明确链接取得两份全长 JPEG，仅检查文件结构与元数据。Pillow 两张均解码为 1280 × 720 RGB，两张均无 EXIF 和内嵌 ICC。独立扫描 JPEG 标记，两张都只有用于表示 JFIF 的 APP0 段。作者没有登记对应墙面两次拍摄的快门、光圈和 ISO，网页照片也无法核实。Part I 的人像和 DPX 则按灰卡逐片调到共同亮度，且跨灯档会切换白平衡，不能填补 Part II 所缺的原始曝光记录。因此停在来源核验，未打开照片比较墙面颜色，也未将测光仪读数当作同曝光证据。[原拍摄与显影说明](https://www.cinematography.net/2024-LMU-tests/clTestSetup.html)

| 作者文件 | 取件长度 | SHA-256 |
| --- | ---: | --- |
| `beam_patterns/96.jpg` | 46,003 | `39c803edcdf57355fb71b0ad028d071af602a1ebfe7b54cf8c83cc4abcd24fda` |
| `beam_patterns/98.jpg` | 47,727 | `949887bf9ff35d3b11b267056e845fba4a1c29547cab80c1f6bfc7b1af6f0910` |

## 重查与推进条件

从作者档案页取得 `mondrian_8_bit.tar.gz`，核对字节数、ETag 与 SHA-256，再运行 `tar -tzf mondrian_8_bit.tar.gz`，按第二级目录计场景，按后缀与 `solux-3500`、`solux-4100`、`solux-4700` 计帧。CML 两文件位于测量页的 `beam_patterns/` 相对路径，复查 `Image.getexif()`、`image.info.get('icc_profile')` 或 JPEG APP1/APP2；文件尺寸与 SHA-256 见表。媒体留在本机研究目录。

两条来源均未进入暖色出图，无产品变更或训练。当前缺的是同一浅色实体两档实拍及逐档相机记录。CML 的两档若取得相同曝光的原件或逐档拍摄记录，可从固定原始档接续。拿到可核查的补拍时，直接走已有首组预检，预计 30–60 分钟得到是否匹配的读数。[第一组补拍预检](2026-11-24-warm-capture-preflight.md)
