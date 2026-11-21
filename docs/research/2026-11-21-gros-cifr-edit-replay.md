# 从参考估计调色前图后，回放仍会把错误颜色迁到目标

已知参考的编辑前后图时，Film B 可以跨人物重放。普通两图没有参考原片，本轮用专门做滤镜反演的 CIFR 估计这张原片。从估计原片与当前参考拟合现有三次颜色映射，使用 80% 强度应用到另一人物。离线正对照使用同一套拟合和渲染，只把估计原片换成真实未调色图。两臂都用 256 方形参考。未调色真值、人工脸颊与白幕只进入评分和正对照。[事前协议](../../experimental/adaptive-reference-20261113/gros-cifr-edit-replay-protocol.md) · [既有已知编辑正对照](2026-11-21-gros-paired-two-light-palette.md) · [CIFR 作者实现](https://github.com/birdortyedi/cifr-pytorch/tree/6b0a9b28cb2a724db869c02f376f123d05137e46)

| 01 ← 02 成片 | CIFR 估计原片 | 已知真实原片 |
| --- | ---: | ---: |
| 整图关闭误差，冷／暖 | 6.84／3.70 | 0／0 |
| 整图风格误差，冷／暖 | 6.37／4.81 | 1.18／0.96 |
| 整图风格投影，冷／暖 | 0.434／0.578 | 0.905／0.911 |
| 整图开态换灯差 | 6.22 | 1.14 |
| 五区联合通过 | 0／5 | 5／5 |

除投影以外为 RGB8 MAE。有 Film B 时，参考到真实原片的整图误差从 12.17／12.40 降到 10.81／6.87。参考本来未调色时，CIFR 也改动了 6.94／4.77。该变动被后续正向映射当作编辑，冷灯关闭态的目标两处白幕各偏离原图约 7.62。使用真实参考原片时，关闭态逐像素保持，暖灯右颊风格投影从估计方案的 1.681 变为 0.944。全部局部反例包括部分暖灯白幕风格单项过线，仍不满足换灯或关态要求。[逐区域结果](2026-11-21-gros-cifr-edit-replay-evidence.json)

按原门槛停止这套固定 CIFR 恢复与三次回放。对这一组，真实前图与估计前图之间的对照支持优先解决参考编辑估计，并说明固定重放能表达这次 Film B。该对照仅验证明确的全图预设，仍需新的数据判断场景依赖编辑和开放风格。未改软件、独立集或视频流程。

## 复现

CIFR 使用[作者演示相同的权重](https://huggingface.co/birdortyedi/cifr/tree/e343520df71b8ee199ba6c10f69c1ce076f876f4)，文件 91,520,207 字节。VGG16 取[标明来源为 torchvision 的已固定权重](https://huggingface.co/timm/vgg16.tv_in1k/tree/b8d8aa2dd860af9233c8c67385a8097fd6c35d3f)，只下载 safetensors 字节 0–16383 的索引和 2810–58861561 的卷积特征，后者 58,858,752 字节。两个范围均核验 HTTP 206 和范围字段。按作者评测方式在 [-1,1] 输入，再将输出裁到 [0,1]。严格核对 107 个 CIFR 和 26 个 VGG 张量，CIFR 采用仅张量加载。源码和权重保存在本机 `outputs/adaptive-reference-20261113/cifr-intake-v1`，文件与哈希见证据，未随研究上传。

两次输出 4 张估计原片、8 张目标成片、2 张对照板及 1 份报告，15 个文件逐一核验 SHA-256 一致。独立从 PNG 复算 4 张参考误差和两臂 10 个目标区域。第一次成片在本机 `outputs/adaptive-reference-20261113/gros-cifr-edit-replay-v1`。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_cifr_edit_replay.py outputs/adaptive-reference-20261113/gros-cifr-edit-replay-repro
~~~
