# PSBench 提供可核对的局部编辑单例

为局部调色寻找有原图、作者结果和编辑说明的来源。固定 PSBench 发布版本 `929d0710eb7c86a988f8a69a74619052b60ac293`，读取任务 035–055、201–240 共 61 份 JSON，合计 26,033 字节。排除全图处理、几何、模糊及物体增删，找到六项可定位的颜色或明暗操作线索，其中有主体对比增强、背景去饱和和画面局部增强。所查范围未核实两组同时共享明确区域语义和同一调色意图。作者发布页标注数据集 MIT，媒体与任务全文保持本机。见 [固定发布](https://huggingface.co/datasets/zyn1216/PSBench/tree/929d0710eb7c86a988f8a69a74619052b60ac293)、[作者论文](https://proceedings.mlr.press/v306/zhang26x.html)和 [读取范围](../../experimental/adaptive-reference-20261113/psbench-local-color-source-protocol.md)。

固定检查任务 230 的灯塔前后图和评分规则。两张 JPEG 共 4,934,523 字节，逐张匹配作者发布 SHA256，尺寸均为 2475×3719。256 缩图的亮度梯度相关为 0.99768，整数位移峰值在零，符合已有结构检查。原尺寸近中性色代理上，单一 RGB 查表对实际编辑差的误差下界为 1.2471 码值，四象限独立查表为 1.2051，打散为 1.2392。独立按颜色组重算下界一致。这个量也可能包含 JPEG、颜色管理或细微几何差异，不能完全归因于语义局部调色。见 [数值与来源哈希](2026-11-18-psbench-local-color-evidence.json)。

本机图像对照中，背景天空去色，旗帜保留红色，塔身呈偏冷色。旗杆和屋顶边缘可见蓝色残留，作者结果可以说明哪里被编辑，画质仍有瑕疵。当前可用作带明确局部意图的诊断单例。跨场景同款参考、肤色与近中性色的完整画质验收仍需其他证据。

复现时用固定版本取得上述 61 份任务 JSON 的本机清单和任务 230 的发布树及四个文件。在仓库根目录设置 `$env:PYTHONPATH = (Get-Location).Path` 后运行 `python experimental/adaptive-reference-20261113/audit_psbench_local_color.py <本机输入目录> <新输出 JSON 路径>`。本机目录为 `outputs/adaptive-reference-20261113/psbench-local-source-v1`。本轮无训练、软件升级或独立素材使用。
