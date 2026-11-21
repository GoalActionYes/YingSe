# 13 号实际入口保住同图原色，跨人物偏色仍存在

上一轮眼周试验直接调用空间模型，三个真人同图控制出现了 6.172–7.403 RGB8 的偏差。13 号工作台在模型外还有同图保护和经校验的对齐拟合。为确认裸模型成片是否代表实际操作，本轮用同批三位人物，通过 13 号正常 HTTP 工作台上传、分析、预览和下载 PNG。权重、图片尺寸与 80% 强度固定，正式回归保留素材未参与。[事前协议](../../experimental/adaptive-reference-20261113/gros-workbench-identity-protocol.md) · [上一轮原始记录](2026-11-21-gros-eye-light-anchor.md)

| 检查 | 实际结果 |
| --- | --- |
| 参考是另存的同一照片，文件字节不同 | 3／3 识别为解码像素完全相同 |
| 常规预览与连续预览基底 | 3／3 逐像素保持原图 |
| 无微调 PNG 下载 | 3／3 逐像素保持原图 |
| 手动亮度设为 12 后下载 PNG | 3／3 与只应用手动亮度的图像逐像素一致 |
| 六个跨人物方向的冷／暖灯及 Film B 关／开态 | 24／24 与冻结研究成片逐像素一致，均未走对齐拟合 |
| 局部偏色与参考风格联合评价 | 0／6 个完整方向，0／30 个整图、双颊和双白幕范围 |

同图控制的旧偏差适用于裸模型。软件现有的同图保护能够让用户保留原色，手动操作仍然独立生效。跨人物的 24 张实际 PNG 复算得到原来的失败结果，因此角色不同、照明变化时的偏色仍需在参考分析或渲染上解决。此后裸模型同图结果用于诊断，验收则走工作台预览和导出。此次只验证了图片从上传到 PNG 的路径，未开展新视频测试、自然评价或候选验收。[证据及成片哈希](2026-11-21-gros-workbench-identity-evidence.json)

## 复现

使用[冻结 Film B 对照](2026-11-21-gros-film-b-cross.md)的环境及输入。使用新的本机输出目录，脚本会创建独立临时工作台数据目录。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/verify_gros_workbench_identity.py outputs/adaptive-reference-20261113/gros-workbench-identity-repro
~~~

输出 30 张实际导出 PNG、3 张独立重编码输入和 1 份报告，存于本机 `outputs/adaptive-reference-20261113/gros-workbench-identity-v1`。两次运行的 34 个文件逐一核对 SHA-256 一致。另独立核对 24 张导出 PNG 的 SHA-256 与原研究成片一致，30 个分区的完整分数也一致。已有存储、导出和增强预览回归 12 项全部通过。Git 保留脚本、报告和小型证据摘要。
