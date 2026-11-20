# AceTone 真实 BF16 两图首组未通过格式与画质筛查

从作者公开固定版本取得 8,131,668,008 字节的两个 BF16 分片，分片均与发布 SHA-256 一致。加载 825 个张量，实际张量名称、尺寸与类型均匹配。显存常驻权重约 4.99 GiB，18 个文本解码层的 216 个张量按需从原 safetensors 迁到 GPU，未量化。PyTorch 峰值已分配约 5.33 GiB，峰值保留约 5.70 GiB。模型加载约 5.97 秒，该值从文件哈希校验之后计起。四个参考各生成约 119–150 秒，只计生成调用及 CUDA 同步，未包括图片读取、tokenizer、CUBE 渲染和视频。精度未改变，小尺寸模型的磁盘卸载与全 GPU 三 token 结果一致，两次复算一致。[事前协议](../../experimental/adaptive-reference-20261113/acetone-bf16-first-portrait-protocol.md) · [固定发布](https://huggingface.co/Vivre/AceTone-3B-PST-Preview/tree/9416b259dec1bf7bb01ba2dab38cf6470cfa98f0)

运行使用现有 Qwen2.5-VL 图像处理和发布 tokenizer，两个输入的视觉网格各为 1×36×54，作者提示词沿用，固定贪心生成和 SDPA。先固定人物 01 冷灯图作目标，人物 02 的冷／暖灯各有 Film B 开关图。四条件各输出 73 个 token，其中 66 个是色表码，均无 <EoT>。原严格门槛通过 0／4，保留原始 token，在首次失败后登记作者容错规则，另把每组前 64 个码用同一 VQ 权重解码出图。这四张成片均来自真实两图生成，人工区域和正确目标只参与离线评分。[容错诊断记录](../../experimental/adaptive-reference-20261113/acetone-permissive-parse-note.md) · [作者评价脚本](https://github.com/martian422/AceTone/blob/916393b3f26bdf89c3d939cc5f2a9a3c115ccbc5/eval/predict_lut_ddp.py)

| 评分区域 | 关风格原图误差，冷／暖 | 开关风格响应误差，冷／暖 | 风格投影，冷／暖 | 开风格换灯差 |
| --- | ---: | ---: | ---: | ---: |
| 整图 | 3.44／9.04 | 7.82／4.23 | 0.235／0.627 | 10.03 |
| 左脸颊 | 4.78／3.78 | 4.98／5.91 | 0.714／−0.174 | 13.74 |
| 右脸颊 | 5.38／2.19 | 6.95／5.21 | −0.089／0.395 | 17.47 |
| 左白幕 | 2.97／11.21 | 8.98／3.62 | 0.187／0.711 | 13.61 |
| 右白幕 | 2.99／11.22 | 9.06／3.53 | 0.179／0.720 | 13.76 |

除无量纲投影外，单位均为 RGB8。五个区域联合通过 0／5。即使用作者容错得到图像，普通两图生成的颜色表仍将参考换灯带入目标，也未保住足够的 Film B 风格。上一阶段给定正确色表的编解码结果仍成立。此次保留实际工作流作为研究对照，结束对这一个公开预览权重、贪心配置的扩大筛查。一次跨人物方向无法评价其他权重、解码策略或数据。没有修改产品，没有取得独立画质或视频成绩。[前轮已知表对照](2026-11-21-acetone-vq-film-b.md) · [数字与成片哈希](2026-11-21-acetone-bf16-first-portrait-evidence.json)

## 运行与复核

在上一阶段的本机模型配置目录补齐两份经 SHA-256 核验的 safetensors。编解码器沿用上一阶段资产。用现有视觉和 OpenCV 环境，在仓库根目录运行。新输出目录须不存在。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:USE_TF = '0'
$base = 'outputs/adaptive-reference-20261113'
$runner = 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")'
& .tools/vision-advisor-env/Scripts/python.exe -c $runner experimental/adaptive-reference-20261113/probe_acetone_bf16_first_portrait.py "$base/acetone-architecture-assets" "$base/acetone-vq-film-b-assets" "$base/acetone-bf16-first-portrait-repro"
& .tools/vision-advisor-env/Scripts/python.exe -c $runner experimental/adaptive-reference-20261113/probe_acetone_permissive_parse.py "$base/acetone-bf16-first-portrait-repro" "$base/acetone-vq-film-b-assets" "$base/acetone-bf16-permissive-repro"
~~~

完整 BF16 推理跑了一次，输出保存在本机 `acetone-bf16-first-portrait-v1`。基于同一批原始 token 的容错解码运行两次，四张成片和报告逐文件 SHA-256 一致，位于 `acetone-bf16-permissive-v1` 和同级复算目录。小模型排查依次解决了混合设备范围、safetensors GPU 设备编号与卸载参数的 BF16 类型。两个已验证的模型文件保留本机。
