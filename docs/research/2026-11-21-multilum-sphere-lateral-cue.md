# 已知灰球几何仍未使镜像灯位判断达标

前两轮测到同一单图亮暗指标在两个房间不能稳定判断灯位。本轮用同次拍摄的灰球裁切，将近似已知几何和单图深度放在同一张照片上比较。以裁切中心和半径构造正交投影半球，和 Depth Anything V2 Large 所估计的相对深度分别进入同一偏相关计算。两臂使用一致的灰球内部范围、有效曝光和原始灯位配对。原图与 Film B 各测一次。合成半球的相反侧光方向能被正确区分。[事前协议](../../experimental/adaptive-reference-20261113/multilum-sphere-lateral-cue-protocol.md) · [上一轮室内深度结果](2026-11-21-multilum-depth-lateral-cue.md)

| 灰球来源 | 解析球面，原图／Film B | 估计深度，原图／Film B |
| --- | ---: | ---: |
| main_experiment120 | 5／10 · 5／10 | 5／10 · 4／10 |
| kingston_dining10 | 6／10 · 6／10 | 6／10 · 6／10 |

两种几何都未达到预定的 8 对。80 个实拍状态中，两臂各有 2 个因可用像素或分块不足而缺测，均保留。每个房间近似几何与估计值最多只差一对。至少在这组已知形状的表面上，补足几何不足以让固定亮暗统计成为有效的跨场景线索。

作者 SDK 标注十对对照中七对同时含一个可直接看见闪光的方向和一个只见间接闪光的方向。按这项事前登记的分组，从解析球面的结果看，两场景的七组混合对都只通过 4／7，三组同可见性对分别通过 1／3 和 2／3。这里的混配可能影响结果，样本规模不足以单独将失败归因于它。存档角度定义的是闪光方向，灰球收到的房间反射并不必然呈镜像。后续如果回到照明证据，应先获得能够核对实际入射光的测量或同一位置的明确光色干预；结束这套固定梯度与灯位排序方案。灰球对照不证明人物皮肤及普通两图风格效果。[作者 SDK 的方向分组](https://github.com/lmurmann/multi_illumination) · [每对结果及文件哈希](2026-11-21-multilum-sphere-lateral-cue-evidence.json)

## 复现

沿用[原两室内 ZIP](2026-11-18-multilum-pilot.md)、[原 Film B 资产](2026-11-21-gros-film-b-cross.md)和[深度模型准备](2026-11-21-multilum-depth-lateral-cue.md)，从仓库根目录运行。输出目录须未占用。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$base = 'outputs/adaptive-reference-20261113'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_multilum_sphere_lateral_cue.py "$base/multilum-pilot-v1" .tools/StyleGallery-audit "$base/multilum-sphere-lateral-cue-repro"
~~~

两次各 40 张 Film B 灰球成片、80 张相对深度预览及报告，121 个文件的 SHA-256 一致。完整报告 SHA-256 为 `7182fcea178c3fb63224d6b16482b28b4c57b63240f13d8d08a01c68c816ee93`。成片、原始资产和权重留本机，产品未修改。
