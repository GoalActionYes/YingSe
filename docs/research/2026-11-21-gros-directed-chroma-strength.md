# 校准方向后补足力度，五项冷灯肤色达标

上轮改变部位编辑方向后，几处脸颊的误差已低于 3 RGB8，但风格投影说明迁移仍偏弱。本轮从这些方向已校准的成片出发，仅改变幅度。保持原部位的编辑中位向量方向，将附近参考色的开关差中位向量的长度乘 80% 作为预期幅度。倍率限 0 至 4，近色覆盖不足 75% 或编辑信号小于 1 RGB8 时沿用原结果。羽化范围外逐像素不变。目标正确成片仅在输出后参与原有评分。[事前协议](../../experimental/adaptive-reference-20261113/gros-directed-chroma-strength-protocol.md) · [上轮方向](2026-11-21-gros-target-color-direction.md)

| 新增通过的冷灯脸颊 | RGB8 风格误差，原 → 新 | 风格投影，原 → 新 |
| --- | ---: | ---: |
| 01 ← 03 右颊 | 2.867 → 0.849 | 0.400 → 0.989 |
| 02 ← 01 右颊 | 2.451 → 1.557 | 0.331 → 0.886 |
| 02 ← 03 左颊 | 2.216 → 1.877 | 0.652 → 0.758 |
| 02 ← 03 右颊 | 2.094 → 1.531 | 0.406 → 0.811 |
| 03 ← 01 右颊 | 1.273 → 1.351 | 0.699 → 0.931 |

60 项区域风格通过从 29 项增至 34 项。原通过项目全部保住，且无超过 1 RGB8 的误差增幅。新增项都属于冷灯脸颊，原暖灯脸颊及此前恢复的白幕仍通过。六个方向的整图换灯差均低于 3 RGB8，无超色域裁切。[逐区倍率、分数和文件哈希](2026-11-21-gros-directed-chroma-strength.json)

## 整体限制

原评分仍要求关闭态保色、开关风格与两态换灯差共同通过。所有 12 个整图关闭态到原目标的误差为 3.12–9.89 RGB8，整图风格响应误差为 3.25–6.23 RGB8，六个跨人物方向完整通过仍是 0／6。本轮没有更改关闭态和局部范围之外的像素，因此新增局部收益不会消除整体偏差。下一步按原门槛分离「关闭态底色」与「有效风格只覆盖小区域」的影响，再决定是否值得继续估计普通两图中的编辑量。当前仍依赖参考编辑前后图和人工部位，独立画质、人工自然度与视频均待候选。

## 复现

沿用[上轮方向成片与既有环境](2026-11-21-gros-target-color-direction.md)，从仓库根目录运行，使用新目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$sha = (Get-FileHash "$base/gros-target-color-direction-v1/results.json").Hash.ToLowerInvariant()
& .tools/adaptive-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_directed_chroma_strength.py "$base/gros-directed-chroma-strength-repro" $sha
~~~

两轮各生成 12 张成片、6 张对照和 1 份报告，19 个文件逐一核对 SHA-256 一致。另检查已知幅度、倍率上限、无信号回退和方向保持。完整报告和成片位于本机 outputs/adaptive-reference-20261113/gros-directed-chroma-strength-v1 及同级复跑目录。
