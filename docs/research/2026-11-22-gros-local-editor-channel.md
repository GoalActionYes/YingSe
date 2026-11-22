# 三组局部调色取得可用冷暖对照，两版迁移仍未达标

前两轮在颜色余量和风格幅度之间存在冲突。旧 8 位导出会让暖灯脸部一大片蓝通道相同，保持线性亮度的色度压缩则让原编辑幅度降到约四成至五成。这里继续使用同一批冻结的 32 位浮点文件、原片和作者参数。在带符号的 sRGB 编码之后，逐通道使用同一个单调曲线平滑靠近边界的数值，并锚定到原像素。保留原区域外和关态。允许相对原编辑亮度或色相有小幅变化，用原有照明和编辑门槛复查出片。[事前协议](../../experimental/adaptive-reference-20261113/gros-local-editor-channel-protocol.md) · [上轮边界证据](2026-11-22-gros-local-editor-float.md) · [W3C 带符号 sRGB 转换示例](https://www.w3.org/TR/css-color-4/#color-conversion-code)

| 暖灯组 | 旧蓝通道归零像素的新版码值类别 | 蓝通道 P5–P95 | 暖灯双颊编辑投影 | 全图新增端点 |
| --- | ---: | --- | --- | ---: |
| 第 2 包 | 8 | 3–5 | 0.911／0.922 | 0.00014% |
| 第 3 包 | 7 | 3–6 | 0.946／0.961 | 0.13269% |
| 第 4 包 | 8 | 3–6 | 0.898／0.925 | 0.00071% |

三组冷暖同时通过，本地得到 12 张可复核的局部编辑开关。白幕和蒙版外保持原色。和浮点直接截断比较，三个暖灯样本的双颊亮度平均误差最高 0.324 RGB8，低于事前上限 3。数据反映实拍变光与一项有记录的固定局部操作。原作者没有逐张调这六张照片，幅度和细节通过也不证明自然度。[原版、截断、压缩及本轮成片](../../outputs/adaptive-reference-20261113/gros-local-editor-channel-v2/2/warm/comparison.png)

## 用真实软件入口导出

稳定 1.3 和 13 号各通过本机工作台上传、分析、导出和下载 24 张跨人物 PNG，共 48 张。使用同一个目标原图、另一人的冷暖灯及开关参考，强度 80%，无手动参数。每次推理只读两张正常图。答案及区域只用于离线评分。另从两版各下载三张同图对照，均逐像素保持。13 号的 24 份跨人物映射均未走同画面对齐拟合。

| 旧门槛检查 | 稳定 1.3 | 13 号 |
| --- | ---: | ---: |
| 同图逐像素保持 | 3／3 | 3／3 |
| 整图、双颊、双白幕联合通过 | 0／6 | 0／6 |
| 目标双颊最强编辑响应投影 | 0.277 | 0.419 |
| 只换参考灯光，白幕最大差 | 8.79 RGB8 | 7.49 RGB8 |

第 3 包作目标、第 4 包作参考时，参考关闭编辑，只在冷暖原片之间切换，13 号目标左白幕变动 7.49 RGB8，超过上限 3。第 4 包作目标、第 3 包暖灯原片作参考时，参考也关闭编辑，但 13 号输出的左白幕与原目标相差 28.69 RGB8。前一对照隔离了真实换灯与其它固定差异，后一数值也包括参考内容的差别。已知局部编辑在参考白幕不动，输出仍出现较大基础白幕偏色，脸颊对编辑开关又响应不足。后续候选需分别降低关闭态的白幕误移色，并提高两颊针对有意编辑的响应；这些反例用于同时核对两项变化。若只减小整体强度，已经偏弱的编辑响应会进一步衰减。[仅换参考灯光的对照](../../outputs/adaptive-reference-20261113/gros-local-editor-channel-v2/3-from-4-comparison.png) · [明显白幕失真的对照](../../outputs/adaptive-reference-20261113/gros-local-editor-channel-v2/4-from-3-comparison.png)

## 复现及边界

首次同图检查误用 13 号裸模型，已有工作台的同图保护没有参与。改用两个实际 HTTP 工作台重新运行，光照、素材及准入条件保持。两遍每遍 85 个输出文件逐件哈希一致，含 48 张跨人物 PNG、6 张同图 PNG、预期输出和对照图。另从下载成片独立核对了 120 项分区、灯位及模型读数，重算导入素材的端点和蓝通道类别。报告 SHA-256 为 `0198b52a85999a5f04a7939742712dec26c1e29d31cdca96db32d08425beb0c7`。[哈希与数值摘要](2026-11-22-gros-local-editor-channel-evidence.json)

仓库根目录使用既有研究 Python 和 OpenCV，命令读取上一轮落盘文件。EXR 文件、原片、成片和工作台数据留本机。此次未训练、未发布或进行独立画质、人评和视频验收。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
$env:OPENCV_IO_ENABLE_OPENEXR = '1'
$base = 'outputs/adaptive-reference-20261113'
$launch = 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")'
& .tools/adaptive-env/Scripts/python.exe -c $launch experimental/adaptive-reference-20261113/probe_gros_local_editor_channel.py $base "$base/gros-local-editor-channel-new"
~~~
