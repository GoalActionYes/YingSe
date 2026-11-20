# 整图参考开关重放达到 2／6，剩余问题集中在暖灯右脸颊

上轮证明，恢复目标底色后只修改脸颊和白幕仍无法达到整图门槛。本轮复用已有三次颜色拟合，分别从原冷灯参考和限域校正的暖灯参考中学习 Film B 开关变化。将两张参考得到的颜色表分别用于另一位人物的目标原图，保持 80% 强度。另做合成臂，范围外使用这张整图结果，范围内按旧羽化权重保留上一轮的局部结果。关闭态仍为目标原图。目标正确成片只用于评分，仍需参考编辑前后图。预设身份和目标答案均不参与拟合。[事前协议](../../experimental/adaptive-reference-20261113/gros-paired-global-local-protocol.md) · [已用三次拟合](2026-11-19-paired-cubic-cross-content.md)

参考配对的 6 个留块拟合全部达到已有资格，误差为 0.204–0.493 RGB8，P95 为 1–3，最差 8×8 块均值不超过 2.709。位置合成逐像素验证了四处核心保持原局部图、范围外保持整图结果。

| 方案 | 区域风格通过 | 整图风格通过 | 整个方向通过 | 原通过区域掉项 |
| --- | ---: | ---: | ---: | ---: |
| 重放原局部编辑 | 34／60 | 0／12 | 0／6 | 0 |
| 整图配对重放 | 56／60 | 12／12 | 2／6 | 3 |
| 合成整图和原局部 | 46／60 | 12／12 | 0／6 | 0 |

01 ← 02 和 01 ← 03 的整图重放完整通过。其他四个方向均仅在暖灯参考的右脸颊未达完整联合。四个脸颊的风格 MAE 为 1.363–1.989 RGB8，投影为 1.305–1.446，超出 1.25 上限，且其中三个原已过线。整图配对的 12 项整图风格误差为 0.222–2.082 RGB8，六组整图参考换灯差均低于 3。保留原局部的合成虽无掉项，也保留了旧局部失败。[逐区分数、拟合资格与哈希](2026-11-21-gros-paired-global-local.json)

## 决策与复现

已经找到可覆盖原局部范围之外的编辑表示。整图直接重放仍违反原有局部退化限制，当前不接入软件。剩余问题有可定位的对照。暖灯参考先经过拍摄校正，校正可能与 Film B 的非线性颜色操作相互影响。下一轮可保持参考配对和拟合方法，单独隔离是否使用这层拍摄校正，查看能否保住 12 项整图与原脸颊。这个结果只覆盖已确认的参考开关对，普通两图、独立画质、真实自然度和视频仍待完成。

沿用[原底色重放](2026-11-21-gros-edit-base-coverage.md)及其上游，从仓库根目录运行，选择新的输出路径。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-edit-base-coverage-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_paired_global_local.py "$base/gros-paired-global-local-repro" $sha
~~~

两轮各保存 24 张成片、6 张对照及 1 份报告，31 个文件逐一核对 SHA-256 一致。另核对符号混合、两个端点及羽化中点。完整图像和结果保留在本机 outputs/adaptive-reference-20261113/gros-paired-global-local-v1 和同级复跑目录。
