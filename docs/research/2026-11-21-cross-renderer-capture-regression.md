# 两套渲染器一起拟合未能跨场景保住参考风格

固定去偏方向在另一来源失效后，本轮检验两套算法的不同反应是否含有独立线索。固定同一个目标与参考，取稳定版 1.3 和研究版 13 号的两套既有 80% 成片。两种方法都拟合由成片 RGB8 增量到目标正确增量的线性映射。一种只看 13 号，另一种同时看两套输出，可以生成两套增量凸混合范围以外的颜色。正则、样本权重以及训练标签事前固定。参考原片和实际换灯状态用零变化作训练标签，匹配数字编辑用受控成片作训练标签。[事前协议](../../experimental/adaptive-reference-20261113/cross-renderer-capture-regression-protocol.md)

分别用 Beyond RGB 拟合、MIT／Gros 目标检验，以及反向留出 Beyond RGB。留出推理只用目标原图和两套各自由同一张参考生成的普通成片。目标正确成片和评估区域只在留出输出后计分。这批素材均已用于开发，每个静物构图的两种拍摄状态作为同一组。

## 成片

| 目标 | 原 13 号换灯差 | 单输出拟合 | 双输出拟合 | 双输出风格误差 |
| --- | ---: | ---: | ---: | ---: |
| MIT 客厅 | 11.24 | 12.23 | 1.57 | 3.99 |
| Gros 人像 | 13.28 | 12.69 | 3.60 | 6.77 |
| WHITE_green 灯 1 | 10.80 | 5.47 | 9.31 | 9.76 |
| WHITE_green 灯 8 | 15.38 | 9.79 | 14.87 | 7.94 |
| BLUE_red 灯 1 | 10.74 | 7.27 | 12.21 | 9.43 |
| BLUE_red 灯 8 | 11.17 | 6.71 | 12.77 | 7.87 |

单位为 RGB8 MAE。换灯差是同一目标换参考后的整图变化，风格误差是数字编辑相对原参考的变化与预期编辑之差。客厅双输出的换灯差低于 3，但以原参考生成的结果仍比目标原图偏 3.61，风格误差也从 1.70 增至 3.99。人像以原参考生成的结果改善到 2.20，但实拍换灯差仍超门槛。四个静物状态中，双输出换灯差均高于同折单输出。33 项中，原 13 号数字风格通过 7 项，两个拟合臂各通过 0 项，三者完整构图均为 0／4。[拟合系数、逐区域分数与文件哈希](2026-11-21-cross-renderer-capture-regression.json)

纯曝光参考控制中，双输出六次均使成片明暗偏离相应 13 号结果超过 1 RGB8，范围 1.86–7.37。两套成片均等于目标时，零增量在拟合层逐像素保留原图。该项只验证拟合层，未另跑两个工作台的同图入口。额外合成正控制验证，当第二套输出明确含有区分实拍和编辑的信号时，双输出确实能分开两种变化。

## 决策与复现

结束用这批输出和固定线性映射做跨来源修复。客厅证明两套结果有时提供互补信息，但它未保住有意调色，换到静物还出现更大串色。现有条件涉及三个不同的参考构图，无法据此判断更广的编辑监督能否泛化。下一轮应优先找带真实编辑开关、且有不同拍摄条件的跨场景素材，避免在同一批对照上更换正则或扩大线性模型。产品功能未改动。

沿用 [MIT](2026-11-21-mid-matched-wall-cast.md)、[Beyond RGB](2026-11-20-beyond-rgb-matched-cast.md) 和[曝光控制](2026-11-21-two-image-chroma-quantile-gain.md)保存的输入。在仓库根目录运行，选择新的输出目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$sha = (Get-FileHash outputs/adaptive-reference-20261113/two-image-chroma-quantile-gain-v1/results.json).Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_cross_renderer_capture_regression.py outputs/adaptive-reference-20261113/cross-renderer-capture-regression-repro $sha
~~~

两轮各保存 48 张拟合输出、12 张曝光输入、6 张对照图和 1 份报告，67 个文件的 SHA-256 逐一一致。媒体留在本机 outputs/adaptive-reference-20261113/cross-renderer-capture-regression-v1 及复跑目录。
