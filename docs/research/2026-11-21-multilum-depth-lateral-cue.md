# 单图深度梯度未能提供跨房间灯位线索

固定画面条带的左右亮度难以跨房间解释灯位，因为每间房的材质分布、遮挡和表面朝向各不相同。本轮固定已使用的两个实拍场景、十对镜像灯位和同一 Film B 处理，新增相对逆深度作为几何线索。用已有 Depth Anything V2 Large 原权重从每张图单独估计深度，原图与 Film B 成片各自推理。没有借用未调整图的几何，也没有将灯位标签或人工区域用于逐张计算。[事前协议](../../experimental/adaptive-reference-20261113/multilum-depth-lateral-cue-protocol.md) · [固定条带对照](2026-11-21-multilum-lateral-cue.md) · [本机模型和许可清单](2026-11-17-stylegallery-preparation.md)

固定取画面上方 70%，排除曝光超界及深度梯度最大的 5%。在 4×6 网格内各自去中心后，测量原图对数亮度与水平深度梯度的偏相关，控制垂直梯度。此处的梯度是估计的相对变化，不能解释为标定的表面法线。合成曲面上的方向反转检查通过。80 张实拍输入均有有效分数，参与测量的像素数为 56,184–115,763。

| 留出的场景 | 原左右条带，原图／Film B | 新深度梯度，原图／Film B |
| --- | ---: | ---: |
| main_experiment120 | 4／10 · 3／10 | 3／10 · 4／10 |
| kingston_dining10 | 5／10 · 5／10 | 1／10 · 1／10 |

两房间用于校准的符号相反。加入近似几何后，最好的一个条件只比原统计多一对，另一个房间两种状态均由 5 对降至 1 对。四个条件均未达到事前门槛。当前不能用这项相关值判断应从参考中去除哪些拍摄灯影响。停止这个固定深度梯度统计，不用调整分块或阈值来追分。现有样本没有真人，数据也没有可以核对模型逐像素法线精度的真值，因此不能区分失败来自估计误差、材料与照明混杂，还是房间受光本身不满足该统计假设。[全部配对结果与预览身份](2026-11-21-multilum-depth-lateral-cue-evidence.json)

## 复现

沿用前轮[两房间归档](2026-11-18-multilum-pilot.md)、[40 张 Film B 成片](2026-11-21-multilum-lateral-cue.md)和已有 Depth Anything V2 目录。权重按原 CC BY-NC 4.0 用于本机课程研究，未分发。使用本机 OpenCV 与原 GPU float32 推理。从仓库根目录执行，输出路径须未占用。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$base = 'outputs/adaptive-reference-20261113'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_multilum_depth_lateral_cue.py "$base/multilum-pilot-v1" "$base/multilum-lateral-cue-v1" .tools/StyleGallery-audit "$base/multilum-depth-lateral-cue-repro"
~~~

两次各 80 张 16 位相对深度预览、1 份报告，81 个文件的 SHA-256 均一致。预览已做分位缩放，仅供查看。实际分数在缩放和量化前计算。完整报告 SHA-256 为 `31febea9901166e2ff8a45ddbadc0e97bc3490a3f722fcbe2500ce6a9863baaa`。原程序和正式验收集均未改变。
