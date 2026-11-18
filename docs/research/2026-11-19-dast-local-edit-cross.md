# 实拍曝光变化下，13 号未能准确迁移脸部暖调

在五组既有实拍曝光序列上，固定另一人的目标原图。每组用标称曝光、轻欠曝、轻过曝三种参考，每种参考分别关闭、开启同一个只作用于脸部的暖色操作。13 号用现有权重和 80% 强度生成 30 张图片。曝光变化来自实拍，局部编辑由程序生成。脸部定位用于制作对照和离线评分。软件推理只接收目标原图和参考成片。[运行前协议](../../experimental/adaptive-reference-20261113/dast-local-edit-cross-protocol.md)

| 检查项 | 预设条件 | 实测 |
| --- | --- | --- |
| 目标脸部收到的暖调 | 预期方向上 75%–125% | 23.2%–51.4%，0/15 达标 |
| 目标脸部的 RGB 增量误差 | ≤3 码值 | 3.98–6.01，0/15 达标 |
| 脸部羽化范围外的 RGB 变化 | ≤1.5 码值 | 0.80–3.20，12/15 超限 |
| 曝光单独导致的脸部色度变化 | 对数色度范数 <0.03 | 0.132–0.349，20/20 超限 |

联合通过 0/5 组，预设要求至少 3/5。第 3 组轻欠曝时，脸部暖调仍只有预计的 51%，脸外改变量达到 3.20 码值，为本轮最大。第一组开启局部暖调后，过曝参考相对标称参考造成的肤色色度差由约 0.29 增为 0.35。这里的暖调百分比衡量指定方向的色度增量，画质、自然度和全部颜色的误差另有指标。

这个对照把两种期望放在同一组图上。参考只换曝光时，目标肤色应稳定。参考脸部有明确的暖调时，目标脸部应能继承，脸外应少受影响。结果支持同时保留两项要求，统一减弱响应会进一步削弱本来就不足的暖调。本轮形成开发对照，软件版本保持 13 号，后续依然需要独立自然图像和视频验收。[完整数值与输出哈希](2026-11-19-dast-local-edit-cross-evidence.json)

两遍准备及出图的 55 个 PNG 与两份报告逐一匹配。15 个关闭状态逐字节匹配上一轮曝光成片。独立读取 PNG 重算全部目标脸部投影、RGB 增量误差、脸外变化和 20 项曝光色度，结果一致。便携稳定版与课程软件入口继续见[体验与课程交付](../../00-从这里开始/0-体验与课程交付.md)。

## 本机复现

使用既有冻结目录 `outputs/adaptive-reference-20261113/dast-exposure-prepared-v1` 和 `dast-exposure-response-v1`。依赖和素材准备沿用[第一次曝光对照](2026-11-19-dast-exposure-response.md)。目标和参考照片、定位结果、操作后参考和生成图均留在本机。当前结果位于同级 `dast-local-edit-cross-v1`，对应的编辑输入位于 `dast-local-edit-prepared-v1`。

```powershell
$root = 'outputs/adaptive-reference-20261113'
$script = 'experimental/adaptive-reference-20261113/probe_dast_local_edit_cross.py'
$runner = 'import sys,runpy; from pathlib import Path; script=Path(sys.argv[1]).resolve(); sys.path.insert(0,str(script.parent)); sys.argv=sys.argv[1:]; runpy.run_path(str(script),run_name="__main__")'
& 'experimental/skin-protection-20261019/runtime/python.exe' -c $runner $script prepare "$root/dast-exposure-prepared-v1" "$root/dast-local-edit-prepared-repro"
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '1'
& '.tools/adaptive-env/Scripts/python.exe' $script render "$root/dast-exposure-prepared-v1" "$root/dast-local-edit-prepared-repro" "$root/dast-exposure-response-v1" "$root/dast-local-edit-cross-repro"
```
