# 比例增益与颜色偏移无法区分实拍换灯和局部编辑

上轮的普通两图结果中，参考只换灯就使目标白幕偏色，有意开启暖调后脸颊响应又不足。本轮用 OpenCV 查找脸部，分别拟合脸与亮中性环境的 RGB 分位。对每组比较只乘通道增益、乘增益后再加通道偏移两种解释。前者用来近似拍摄变化，后者只有在脸部效果明确、环境亦支持局部处理时才写入目标脸。推理只读取两张现有 RGB 图及同一输入对的冻结 13 号回退输出，脸部检测器保持既有哈希。评分真值与区域在试验出片结束后读取。[事前协议](../../experimental/adaptive-reference-20261113/gros-affine-intent-protocol.md) · [OpenCV 定位文档](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html) · [既有实际导出](2026-11-22-gros-local-editor-channel.md)

## 六方向与负例

| 检查 | 结果 |
| --- | --- |
| 24 张跨人物候选与 13 号逐像素相同 | 24／24 |
| 缺唯一人脸或取样不足，回退 | 14／24 |
| 环境增益或脸部拟合未获支持，回退 | 10／24 |
| 整图、双颊、双白幕联合通过 | 0／6 |
| 同图保持与无人回退 | 两个控制均通过 |
| 已知全图简单编辑保留原通路 | 2／3 |

13 号原有偏色与局部响应不足全部留在本次实际成片中。第 3 包的已知统一编辑没有通过现有同画面对齐，试验把它判成只含拍摄增益，输出逐像素回到原图。该参考与原图的 RGB8 MAE 是 20.18。试验输出比 13 号实际下载的 PNG 额外相差 5.99 RGB8。保留这组退化，停止把增益解释直接当作保持原图的开关。[跨人物实图](../../outputs/adaptive-reference-20261113/gros-affine-intent-v2/3-from-4-comparison.png) · [第 3 包原路径 PNG](../../outputs/adaptive-reference-20261113/gros-affine-intent-v2/control-3-base.png) · [第 3 包试验 PNG](../../outputs/adaptive-reference-20261113/gros-affine-intent-v2/control-3-trial.png)

第 2 包四张参考只有暖灯关态识别出唯一人脸。第 3、4 包八张参考均识别出脸，候选仍缺环境或颜色拟合支持。在离线复核中，把同一人的准确双颊框直接给模型，最少内容差异的拍摄与编辑对照仍不能分开。第 2、3 包冷灯下有意编辑的仅增益误差分别为 2.74 和 2.70 RGB8，均会被当作只需增益。第 4 包仅换灯和冷灯编辑都达到偏移触发条件，纯增益改成允许偏移后误差分别由 6.44 降至 2.22、由 3.48 降至 1.70。三组暖灯的编辑开关均不满足数值拟合资格。区域定位确有缺项，但给定准确区域仍可发生错误归因。该离线复查不计入候选成绩。[逐例路由、对照及哈希](2026-11-22-gros-affine-intent-evidence.json)

当前证据不支持按这类 RGB 分位拟合决定用户意图。结束这套拟合与回退规则，不调整人脸阈值或增益门槛。后续样本应提供与颜色分布之外可验证的线索，再判断两图能否分别限制中性色误移和肤色风格漏移。无产品版升级、正式独立素材、人评或视频测试。

## 验证与复现

第一次只用第 2 包检查全图编辑，并要求先通过现有同画面对齐。进一步检查发现第 2、3 包因颜色矩阵条件不佳被现有路径拒绝。保留第 2 包，同时对三组都使用相同全图仿射、通过真实工作台导出原 13 号对照，再检查候选回退。没有改变候选判断或分数门槛。两遍报告、24 张候选 PNG、6 张比较图及 8 张控制输出共 39 件逐件哈希一致。另从所有落盘 PNG 核对 24 张等于冻结 13 号及负例失效。报告 SHA-256 为 `461858fe907abf9e8016d9abb1c488387b3aadc4d70ea1b812de44b2b0da4ffa`。

仓库根目录运行。[脚本](../../experimental/adaptive-reference-20261113/probe_gros_affine_intent.py)

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
$launch = 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")'
& .tools/adaptive-env/Scripts/python.exe -c $launch experimental/adaptive-reference-20261113/probe_gros_affine_intent.py $base "$base/gros-affine-intent-new"
~~~
