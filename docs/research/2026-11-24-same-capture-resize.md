# 同原片缩小参考保留局部调色

使用 15 号同原片工作台，把原尺寸照片作为目标，把同一原片的缩小调色成片作为参考，原尺寸 PNG 可以保留参考里的局部提亮或改色。此次验证完整画幅的半尺寸和四分之三尺寸，来源分别用 OpenCV 面积降采样和 Pillow Lanczos。需要能在小图坐标上核对共同细节和梯度。变化经小图差值叠回目标，原尺寸细节继续来自目标。正常重拍和跨图回退至现有 13 号路径。[启动、依赖和回退](2026-11-24-same-capture-workbench.md) · [实验协议](../../experimental/adaptive-reference-20261113/same-capture-resize-protocol.md) · [EGM 审阅与处理](2026-11-24-same-capture-resize-egm.md)

## 实际出片

用两位作者既有的局部提亮和局部改色对照，在各自两种尺寸和两种缩放滤镜下运行，两版都通过真实上传、分析和 80% 强度的 PNG 导出。编辑区误差是与作者预期成片的 RGB8 平均绝对误差。

| 作者编辑 | 13 号的编辑区误差 | 15 号本轮误差 | 保留编辑幅度 |
| --- | ---: | ---: | ---: |
| Martin 的右颊局部提亮 | 11.12–11.15 | 0.44–0.58 | 99.23%–99.56% |
| Carlos 的局部肤色调整 | 12.78–12.82 | 0.32–0.47 | 98.65%–99.45% |

八组从 13 号全部未过线到本轮全部过线。八组关闭态逐像素保持原图，有意编辑和保护区域通过作者对应协议，端点检查也通过。两位作者的素材均已用于开发。[本机右颊对照](../../outputs/adaptive-reference-20261113/same-capture-resize-green-v1/flash-full-half-area-comparison.png) · [本机肤色对照](../../outputs/adaptive-reference-20261113/same-capture-resize-green-v1/skin-full-three-quarters-lanczos-comparison.png)

70 个负对照的 PNG 与 13 号逐字节一致。包括 16 个真实重拍或跨图配对分别使用四种缩放设置，右半幅人为移位分别使用四种缩放设置，以及两张缩小后保存为 JPEG80／4:2:0 的参考。在四幅半边移位图中，匹配滤镜仍检出 8 个共同细节区，梯度相关仅 0.470–0.478，低于 0.98 而被拦下。两张缩小 JPEG 均回退。其余缩放链、广泛结构改动、颜色配置与重新编码的组合超出本轮证据。

旧 15 号四组全幅和裁切及双端同尺寸 JPEG 对照重新出片，所有旧回归记录与改动前一致。16 个原路径负例、旧视频和含手动调整输出的哈希未变。新缩小路径另过强度为零、交互预览、亮度与色调微调、项目保存和重启恢复。旧项目携带原有效果配置重新分析，导出和升级前逐字节一致。旧工作台用例及状态测试覆盖普通 CUBE 的拒绝、撤销重做和预览调度。旧短视频是兼容回归，普通跨场景画质、真实人评、新视频稳定性和完整视频性能仍待验收。

## 复现

使用原研究环境，OpenCV 4.11.0、Pillow 12.1.1。无新模型、下载或训练。两遍新组合分别约 270 秒和 261 秒，旧用例复核约 169 秒。两遍新报告逐字节一致，独立核对每遍的 182 张 PNG 及其哈希，又从成片重新计算 16 个作者区域评分，并比较 70 对负例的 PNG 字节。改动前的首次回归导出两个失败方向后，本机 HTTP 轮询超时终止。完整基线来自正式同轮运行。报告、成片与原图保留本机，分支保存源码、复现说明和小型数值摘要。[数值与哈希](2026-11-24-same-capture-resize-evidence.json)

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$env:NO_PROXY = '127.0.0.1,localhost'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$python = './.tools/adaptive-env/Scripts/python.exe'
$code = 'experimental/adaptive-reference-20261113'
$root = 'outputs/adaptive-reference-20261113'
& $python "$code/verify_same_capture_resize.py" "$root/same-capture-resize-new"
& $python "$code/verify_same_capture_workbench.py" "$root/same-capture-legacy-new"
~~~
