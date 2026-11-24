# 室外实拍补齐与室内共用的局部提亮对照

MIT 室内已有真换灯与局部墙面提亮在同区域方向、幅度接近的对照。此前的户外版用了另一项压暗，不能复用同一种编辑。此次固定 WildRelight 的浅墙和柱旁铺地，仅补取各自第 1、2 时段的四张小尺寸对齐 HDR 照片。沿用每处先前固定的显影曝光，重放 MIT 同一份 Martin Werner 局部提亮设置。只变更三项选区几何，原拍摄者没有为这些图另行编辑。[事前协议](../../experimental/adaptive-reference-20261113/wildrelight-shared-tone-protocol.md) · [WildRelight 发布和许可](https://huggingface.co/datasets/Lez/wildrelight) · [原室内对照](2026-11-24-shared-authored-tone-admission.md)

| 场景 | 编辑开关逐档达标 | 有序实拍方向达标 | 四档编辑区亮度增加 |
| --- | ---: | ---: | ---: |
| 建筑浅墙 | 4／4 | 3／12 | 7.00–11.79 Y′ |
| 钢柱旁铺地 | 4／4 | 3／12 | 17.79–22.57 Y′ |

四张首次纳入本项目的时段与首时段，区域内水平亮度梯度相关为 0.782–0.891，全部超过 0.70。编辑关闭图与准备图逐像素一致。自己的另一段墙面、钢柱和远处保护范围，其 RGB8 差 P95 为零，均无新增端点截断。梯度相关帮助排除明显位置偏移，不能确认所有细节或色变化只来自单一光源。实验沿用来自配准 HDR 的显示转换。结论限定为该显影约定下的图像对照。[浅墙本机成片](../../outputs/adaptive-reference-20261113/wildrelight-shared-tone-v1/wall/comparison.png) · [铺地本机成片](../../outputs/adaptive-reference-20261113/wildrelight-shared-tone-v1/pillar/comparison.png)

铺地末时段至第 1 时段，换光使 RGB 中位增量为 23、20、23，末时段打开同款提亮的增量为 22、22、24，两个方向余弦 0.9984，幅度比 1.0291。浅墙末时段和第 2 时段在同一测区的匹配余弦为 0.9656、幅度比 0.9429。照明与编辑在登记范围内相似，景物其他部分仍可能有不同阴影或光色。以上可与 MIT 真实换灯构成两个采集来源的同款正、负开发对照。两套均已用于研究，室外两处也都没有人像。本轮没有自动推理或训练，暖色编辑、肤色、真实人评、独立画质和视频验收仍按原门槛。[六个匹配及逐档数值](2026-11-24-wildrelight-shared-tone-evidence.json)

## 复现与核验

原首末时段和元数据沿用[已核验本机归档](2026-11-21-wildrelight-wall-cross.md#复现)。新文件四张共 41,992,690 字节，以[固定发布清单](../../experimental/adaptive-reference-20261113/wildrelight-shared-tone-release.json)核对长度和 SHA-256，留在本机。使用旧 RawTherapee 5.13、研究 Python 和空闲 ASCII 映射盘，传入全新的输出目录。原首末显影图已逐像素复现；独立从两遍共 32 张编辑成片核对 ICC、哈希、区域梯度、匹配和保护读数。两次完整运行分别约 15.0 秒和 13.1 秒，不含下载及独立复核。报告哈希同为 `915122b02c31e69b79f070130b7b9ce11ebe268d81d7fd6d0d0062077e1a730b`。[复现脚本](../../experimental/adaptive-reference-20261113/probe_wildrelight_shared_tone.py)

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$env:OPENCV_IO_ENABLE_OPENEXR = '1'
$root = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_wildrelight_shared_tone.py $root $rawTherapeeCli "$root/wildrelight-shared-tone-input-v1" "$root/wildrelight-shared-tone-new"
~~~
