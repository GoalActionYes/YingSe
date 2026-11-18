# COCO 屏幕与人物同框照片准入

在本机已校验的 COCO 2017 验证集标注中，10 张照片满足授权元数据、尺寸和实例面积约定，其中 7 张人框与电视框有交叠。按事前规则复核全部 7 张，另取编号最早的两个无交叠控制。九份照片均通过来源、尺寸和文件身份核验，保留本机。见[事前协议](../../experimental/adaptive-reference-20261113/screen-occlusion-intake-protocol.md)和 [COCO 使用条款](https://cocodataset.org/dataset/termsofuse.htm)。

九张中没有确认到可登记直接皮肤的真实显示平面遮挡。三张清楚显示屏幕上的脸，可用作新的误触对照；一张屏幕内容过于模糊；一张商店照片边缘关系不清；其余四张是空间上分开的同框控制。9483 是已用样例，完整保留在本轮记录中。最高框重叠的 463618 里，实拍人物和电视在图像上挨近，人物没有挡住显示表面。图像类别与框不能提供前后关系。见[逐图裁决及三个屏幕脸部坐标](../../experimental/adaptive-reference-20261113/screen-occlusion-review.json)、[本机九图一览](../../outputs/adaptive-reference-20261113/screen-occlusion-intake/review.jpg)及[三个框的局部确认](../../outputs/adaptive-reference-20261113/screen-occlusion-intake/screen-face-controls.jpg)。

三张屏幕脸可加入后续开发误触检查。评价真人挡住屏幕时是否仍能正常调色，需要另找实际遮挡照片，当前控制不能替代。此次没有运行模型或改变软件。首次脚本把抽样上限误作最小数量，已按原协议保留全部七张重叠样本，两组筛选条件始终一致。九份图像文件都复核了字节数和 SHA-256，三份坐标也检查了边界。复现用已有视觉环境运行 `prepare_screen_occlusion.py <本机取件目录>`。见[来源清单及哈希](2026-11-19-screen-occlusion-intake-evidence.json)。
