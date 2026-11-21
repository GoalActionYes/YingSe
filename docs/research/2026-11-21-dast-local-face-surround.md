# 脸周颜色能补救一组暖调，跨人物仍易误判

13 号在五组真实曝光的开发对照中，传到另一位人物脸上的暖色偏弱，部分编辑还影响脸外。这轮只检查是否可以从单张参考判断脸部暖调是否开启。用 OpenCV 正脸框采样，分别在脸内取平均，或减去脸周的同方向平均值。每折排除参考和目标人物。两臂都沿用同一个离线已知的暖色操作、自动目标脸部定位和 80% 强度。目标答案、参考状态与人工评分区域未进入选色函数。实拍的是曝光，局部暖调由程序产生。[事前协议](../../experimental/adaptive-reference-20261113/dast-local-face-surround-protocol.md) · [原 13 号实际成片](2026-11-19-dast-local-edit-cross.md)

| 方法 | 未编辑参考保持原图 | 有意暖调正确开启 | 三档曝光完整通过 |
| --- | ---: | ---: | ---: |
| 只用脸内颜色 | 11／15 | 13／15 | 0／5 |
| 脸内减脸周颜色 | 9／15 | 9／15 | 1／5 |

第 3 组从标称到轻欠曝、轻过曝全部判对，目标脸部按预期变暖且脸外保持。第 4 组则只在轻欠曝时开启编辑，并同时把未编辑参考误认为有意暖调。五组的 30 张参考均取得足够的脸内和脸周中间调像素。供给已知开关时，五个自动定位正控与目标预期逐像素一致。这个正控使用与程序编辑生成阶段相同的目标定位器，只用来检查合成暖色的应用路径。参考取样用独立的几何人脸框。[本机通过组](../../outputs/adaptive-reference-20261113/dast-local-face-surround-v1/3-comparison.png) · [本机误判组](../../outputs/adaptive-reference-20261113/dast-local-face-surround-v1/4-comparison.png)

未达到预设的至少三组完整通过和全部未编辑参考保持要求，结束固定脸内、脸周暖色判断。实际脸周像素可能来自头发、衣物和背景，其颜色变化难以充当普遍的脸部调色基准。这轮只验证了一种局部暖调，保留失败供后续有确切编辑位置、且承受真实拍摄变化的方案比较。稳定版和 13 号仍用于体验，独立画质、人工评价和视频验收未用于这次开发判定。[逐案读数及成片身份](2026-11-21-dast-local-face-surround-evidence.json)

## 复现

用[实际曝光与暖调归档](2026-11-19-dast-local-edit-cross.md)及已有肤色研究运行环境。直接启动时本机嵌入式 Python 未将脚本目录加入搜索路径，在读取数据前停止。沿用原 DAST 的脚本启动方式后，两个新目录分别完成 60 张两图结果、五张正控、五张图板和一份报告，71 个文件逐一一致。另读取 PNG 核对全部 60 个自动状态、五张正控和两臂的完整方向计数。图板上排按目标、预期、正控排列，中排为只用脸内的三档开态，下排为脸内减脸周的三档开态。完整报告 SHA-256 为 `0336196a76eb7d8e2e4ac89ef99283db5e59dc04288fbc805bb2d82e98eedf6a`。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '1'
$runner = 'import sys,runpy; from pathlib import Path; script=Path(sys.argv[1]).resolve(); sys.path.insert(0,str(script.parent)); sys.argv=sys.argv[1:]; runpy.run_path(str(script),run_name="__main__")'
$base = 'outputs/adaptive-reference-20261113'
& experimental/skin-protection-20261019/runtime/python.exe -W 'error::RuntimeWarning' -c $runner experimental/adaptive-reference-20261113/probe_dast_local_face_surround.py $base .tools/tf-convert-env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml "$base/dast-local-face-surround-repro"
~~~
