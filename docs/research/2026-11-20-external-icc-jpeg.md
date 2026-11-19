# 第三方宽色域 JPEG 通过稳定工作台图片流程验证

从 WebKit 获取三张未经项目重新编码的 600×400 JPEG。Italy 和 Yellow Flower 嵌入不同字节的 Display P3 配置，Shoes 嵌入 Adobe RGB (1998)。每张图分别作目标和参考，比较原始 JPEG、从该 JPEG 按标签转换的 sRGB 基准、以及保留原始 RGB 数值但去掉标签的控制图。两张输入均走正式导入，应用相同 80% 强度、明度 12 和红色 HSL 饱和度 8 后进行预览和 PNG 导出。[事前协议](2026-11-20-external-icc-jpeg-protocol.md) · [原图发布页](https://webkit.org/blog-files/color-gamut/)

| JPEG | 当作目标，忽略／读取 ICC 的成片误差（RGB8 MAE） | 当作参考，忽略／读取 ICC 的成片误差（RGB8 MAE） |
| --- | ---: | ---: |
| Italy · Display P3 | 4.311／0 | 1.138／0 |
| Yellow Flower · Display P3 | 1.952／0 | 0.877／0 |
| Shoes · Adobe RGB (1998) | 6.574／0 | 3.075／0 |

带标签输入和 sRGB 基准的原图读取及成片逐像素一致，全部去标签控制达到事前定下的 0.5 RGB8 差异门槛，每臂预览和导出也逐像素一致。首次调用因验证脚本把 HSL 数据传成对象而被 HTTP 拒绝。脚本改用已有的一组三个数值接口后，从新的数据目录重新运行。没有修改应用代码、数据或门槛。稳定工作台的 93 项 Python 回归本轮重新通过。

结果验证了三份外部 JPEG 的图片 ICC 输入工程修复。基准与应用使用同一颜色管理库，主要核验文件配置是否贯穿导入、分析、预览和导出。未涵盖所有 ICC 格式、宽色域导出或开放跨场景自动风格。现有稳定版发布物与现有研究版不受本轮核查影响。体验和回退使用[候选启动说明](2026-11-20-image-icc-candidate.md)。

## 复现

将发布页的 `Italy-P3.jpg`、`YellowFlower-P3.jpg` 和 `Shoes-AdobeRGB.jpg` 保存在本机目录。在候选工作树和已有视觉环境中，设置仓库根目录为 `PYTHONPATH`，运行 `python scripts/probe_external_icc_images.py <三份原图目录> <新的输出目录>`。脚本核对原图文件哈希、图像尺寸、方向和 ICC 标识。两个不同数据目录中的 18 张 PNG 和数值报告逐件一致，已从 PNG 独立复算六项误差。报告 SHA256 为 `99b2c34b50c3d7f4a6299409306477e8e271f3614d4d598774b3dfbbf603fc40`。照片、导出和工作台数据不随 Git 上传。[复现脚本](../../scripts/probe_external_icc_images.py)
