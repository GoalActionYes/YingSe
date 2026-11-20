# 明暗与色度边缘同向仍无法区分实拍灯色和数字编辑

整体颜色分位不能判断参考颜色是否该迁移。本轮检验局部边缘有无额外证据。把每张参考缩至长边 256，分别在 4 像素和 16 像素的差分跨度计算对数明暗及两个对数色比的梯度。只在未裁切且梯度达到最低强度的像素上统计，再在强明暗边缘和强色度边缘各取方向一致度的中位数。方向一致度取绝对值，因此同向和反向均计作对齐，垂直记为零。它衡量的是边缘是否共线。共线可以来自照明，也可以来自材质或物体交界。[事前协议](../../experimental/adaptive-reference-20261113/capture-edge-alignment-protocol.md)

一次只留出一个素材来源。在其余三个来源拟合此前固定的最近中心判别，再从留出来源的每张参考单独计算结果。原片与数字编辑均应保留正常调色，实拍换灯可触发受控保护。数据均为已有开发对照。[逐图描述、判别距离和哈希](2026-11-21-capture-edge-alignment.json)

| 留出来源 | 实拍换灯识别 | 数字编辑误触 | 原片误触 | 缺测 |
| --- | ---: | ---: | ---: | ---: |
| CHROMA-MILL | 5／6 | 5／6 | 6／6 | 0 |
| Beyond RGB | 2／2 | 2／2 | 2／2 | 0 |
| Gros | 1／3 | 3／3 | 3／3 | 0 |
| MIT | 1／1 | 1／1 | 1／1 | 0 |
| 合计 | 9／12 | 11／12 | 12／12 | 0 |

## 实际画面与决定

MIT 的原参考、实拍换灯和同向数字编辑全被判作换灯。接到此前相同的诊断保护分支后，客厅和人像的六份结果逐像素保持原图，换灯串色为零，数字风格投影也在九个区域全为零。人像两颊数字风格误差为 5.214、6.720 RGB8，白幕为 11.322、11.333 RGB8。两张目标的完整门槛均未通过。

结束这项边缘同向描述。已有负例中，真实灯色和数字调色共用的边缘关系足以让保护分支误触。后续需要此前输入缺少且可交叉核验的证据，才值得重新开发触发规则。六张失败输出及两张对照均保留。产品软件未接入，独立画质、真实自然评价和视频仍待完成。

## 复现

使用[四来源固定复现环境](2026-11-21-capture-tile-tone-color.md)及其中的本机输入，从仓库根目录选择一个新输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_capture_edge_alignment.py outputs/adaptive-reference-20261113/capture-edge-alignment-repro
~~~

两轮各生成 6 张输出、2 张对照和 1 份报告，9 个 SHA-256 逐一相同。两个尺度额外使用同向、垂直梯度验证方向读数，并检查全无有效像素时返回缺测。本机产物在 outputs/adaptive-reference-20261113/capture-edge-alignment-v1 及同级复跑目录。
