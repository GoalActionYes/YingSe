# 固定左右明暗差未能跨房间判断灯位

AceTone 的实拍成片显示，参考拍摄灯会影响有意风格的迁移。下一项假设是，真实灯位改变会留下可区分的左右明暗关系，而同一全图调色大致保留这个关系。固定两个已用室内实拍场景，按作者 `meta.json` 选择 10 对仰角相同、方位角相反的灯位。每个房间只用原片来校准单个符号，验证在另一个房间。每张图片只读取自身上方 70% 的有效亮度，不读取配对像素、不读取参考探针。只比较固定左右条带与整图亮度两种分数。[事前协议](../../experimental/adaptive-reference-20261113/multilum-lateral-cue-protocol.md) · [MIT 数据和灯位说明](https://projects.csail.mit.edu/illumination/)

| 留出的场景 | 左右差正确，原图／Film B | 整图亮度正确，原图／Film B |
| --- | ---: | ---: |
| main_experiment120 | 4／10 · 3／10 | 3／10 · 3／10 |
| kingston_dining10 | 5／10 · 5／10 | 5／10 · 4／10 |

两房间均未达到每状态至少 8／10，左右条带也未比整图对照多通过两对。固定灰阶镜像验证了左右差的符号随镜像反转，整图量不变。实拍室内场景的材质、遮挡与相对受光位置并不对称，不能从这项统计可靠地识别灯的位置。结束固定左右条带作为自动照明证据的方向。该处只检验同房间镜像灯位的相对排序，不能作为单张新图的方向分类或人物肤色结果，也不能排除使用实际几何约束的空间线索。[各方向结果与哈希](2026-11-21-multilum-lateral-cue-evidence.json)

## 人像取样执行情况

预先登记原计划为三名已用人物，固定姿态、暖灯设定和两个侧灯位置，各取一张。旧目录 1,576 张照片可定位全部六个成员，但此次无论直连还是经现有代理，Mendeley 的公开文件入口都返回 395 字节错误 JSON，未返回同版本的 ZIP 范围。取样在下载任何照片之前停止。没有放宽档案身份、换照片、申请权限或把论文配图当原图。原计划保留，待官方范围读取恢复后可按原顺序验证。[原计划](../../experimental/adaptive-reference-20261113/gros-side-light-intake-protocol.md) · [固定数据版与许可](https://data.mendeley.com/datasets/bpy24trj6s/1)

## 复现

沿用[室内原档案](2026-11-18-multilum-pilot.md)和[Film B 显影](2026-11-21-gros-film-b-cross.md)。从仓库根目录运行，指定未占用的输出路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_multilum_lateral_cue.py "$base/multilum-pilot-v1" docs/research/2026-11-18-multilum-pilot-evidence.json "$base/multilum-lateral-cue-repro"
~~~

两次各 40 张 Film B 成片和一份报告，全部文件 SHA-256 一致。完整报告为 `8d9713c9f56a7c5cac2d8db76e5833ea302912d3ef7ba7aef33ac8c7d67941bb`。成片留本机 `multilum-lateral-cue-v1` 及同级复算目录。
