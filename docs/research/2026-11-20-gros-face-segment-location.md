# 面部分割找到了漏检人物，也把白幕判成面部

上一轮明确给出调色意图和参考编辑前后图，整图暖调与保色均达到开发门槛，局部暖调四次回退均涉及同一位人物漏检。本轮只在局部模式替换定位来源。读取已有肤色模型的面部类别，把有效连通块转换为脸框，原色比估计、羽化权重、强度、其他三态成片和评分规则均沿用。[事前协议](../../experimental/adaptive-reference-20261113/gros-face-segment-location-protocol.md) · [上轮成片](2026-11-20-gros-declared-edit.md)

| 六方向的比较项目 | 上轮人脸检测 | 本轮面部类别 |
| --- | ---: | ---: |
| 局部暖调与原保色／风格联合通过 | 2／6 | 2／6 |
| 局部回退 | 4／6 | 4／6 |
| 无人控制检测块数 | 0 | 0 |
| 两脸控制检测块数 | 2 | 4 |

02 原图的真实脸框为 [291, 180, 188, 246]，画面右侧白幕另被框为 [603, 110, 84, 188]，坐标为横向起点、纵向起点、宽、高。出现多个块后，四个相关方向均回退，最差双颊误差仍为 18.33 RGB8。两脸控制也回退，但检测得到四块，未通过预先指定的块数检查，因此不能据此声称它正确区分了两张脸。原先通过的两方向继续通过，其中一个方向误差从约 0.01／0.02 升至 0.35／0.25 RGB8。[本机漏检与误检同例](../../outputs/adaptive-reference-20261113/gros-face-segment-location-v1/01-from-02/comparison.png) · [本机原通过例](../../outputs/adaptive-reference-20261113/gros-face-segment-location-v1/01-from-03/comparison.png)

结束把粗定位与固定色差直接组合的这两轮验证。白幕误检说明区域资格仍待解决；既有真实编辑对照又表明，即使有正确区域，固定跨照片色差仍会超线。若验证用户给定范围，还需在真实作者照片上检查颜色幅度，不能把这批合成暖调当作成片验收。稳定软件、普通两图、白墙、自然评价和视频验收均无变更。[真实跨照片幅度反例](2026-11-20-playraw-mixed-light-cross-editor.md)

两次各六张成片、六张比较图及报告逐件一致。报告 SHA-256 为 `b20fae81da9894b46d5389989c5b95dfe1556c71c7a1c7558687d49e9774e5d2`。使用现有肤色研究运行时与权重，无需安装依赖。[逐项数据及成片哈希](2026-11-20-gros-face-segment-location-evidence.json)

```powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_face_segment_location.py "$base/gros-portrait-light-transfer-v1" "$base/gros-screen-chroma-v1" "$base/gros-declared-edit-v1" "$base/gros-face-segment-location-replay" experimental/skin-protection-20261019/runtime/Lib/site-packages
```
