# 核对到两组可小批取得的实拍分光源素材

此前杯子与彩纸的换灯对照只能覆盖固定实验室场景。SyncLight 把真实房间的环境光和单灯响应分开拍摄，同一房间还有不同视角。可以用固定的灯色变化生成开发正例，检查软件能否只迁移该灯照到的颜色。作者公开发布标注 CC BY 4.0。本轮只核查发布索引与 TAR 成员头，没有执行重光照或调色。见[核查协议](../../experimental/adaptive-reference-20261113/synclight-real-intake-protocol.md)、[固定发布](https://huggingface.co/datasets/davidserra9/synclight/tree/8067588dc4ee901b805c3813d234a661973f8e71)和[作者生成说明](https://github.com/CVC-Color/synclight/tree/main/dataset)。

| 固定场景 | 可按相机键配对的组件 | 两场的拟取 RAW 体积 | 决策 |
| --- | --- | ---: | --- |
| scene15、scene16 | 每场相机键 1、2 均有环境光 0 和同编号可见灯 1，CR2 | 167,791,907 字节 | 超出 128 MiB 预算 |
| scene35、scene25 | 每场相机键 1、2 均有环境光 0 和同编号可见灯 1，DNG | 106,973,902 字节 | 可按固定范围准备后续小样 |

后两组是上游训练划分中含至少四份 RAW 的两个最小归档。本项目仅拟用于开发。同一房间下的视角与灯位变体必须归为一组，仍需从原片核实相机参数、对齐、黑电平、白平衡和灯位实物对应。使用真实采集的光响应合成彩色灯变体，该变体并非摄影师审美答案，也不为肤色贴上颜色标签。

逐项比对四个上游 test 归档及 35 个 train 归档的包长与 LFS SHA-256。四个固定归档以 HTTP 206 读取 512 字节成员头及尾记录，核对响应范围、版本和归档身份，再独立复算 48 个成员的偏移与拟取范围。合计读取归档头 26,672 字节，本机另存元数据 151,913 字节。未下载图像、灯图或权重，也没有安装 RAW 解码器。见[偏移核查脚本](../../experimental/adaptive-reference-20261113/audit_synclight_real_intake.py)及[成员与身份](2026-11-18-synclight-real-intake-evidence.json)。

另查了人像光场来源。POLAR 作者标注公开 Hugging Face 入口仅供预览，完整素材有申请途径。FaceOLAT 提供学术研究申请入口，需要提交资料。本轮只登记来源，未申请或读取人像。两者可作为未来肤色物理对照线索，当前室内资料优先用于浅墙等中性色。见[POLAR 访问说明](https://github.com/Rex0191/POLARNet/blob/main/OLAT_data_processing/README.md#polar-dataset-download)和[FaceOLAT 申请页](https://gvv-assets.mpi-inf.mpg.de/FaceOLAT/?page_id=12)。

复现时用本机已保存的 `outputs/adaptive-reference-20261113/synclight-real-intake` 元数据，依次以 `--split test`、`--split train` 运行脚本，各指定一个新 JSON 输出文件。进行像素实验前单独固定两个场景、相机配对和灯色操作，再核对解码器及原片。当前仍需新成片来判断映色是否能保住合理局部变化。
