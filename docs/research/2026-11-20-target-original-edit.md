# 从普通参考估计编辑并直接作用原图，仍出现漏调

此前从参考编辑前后图提取同款染色，只作用于目标原图可通过六组。普通两图没有编辑前参考。本轮从单张参考 RGB 提取现有冻结外观，用训练折内已有编辑的非负加权估计染色幅度，再直接作用于目标原图。实拍绿光和白光只在训练中标为零编辑，有意染色为待迁移编辑。从训练折确定支持边界，超出时回到原目标。不同于前一轮三通道线性回归，预测不超出训练标签的逐通道范围。[事前协议](../../experimental/adaptive-reference-20261113/target-original-edit-protocol.md) · [已知编辑的目标原图对照](2026-11-20-chromamill-target-base.md)

## 成片

| 留出组 | 输入在支持内 | 白灯偏离原目标 | 换灯相对白灯变化 | 数字编辑投影 | 整图联合 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CHROMA-MILL Scene3_a，尼康、手机 | 0／6 | 0 | 0 | 0 | 0／2 |
| CHROMA-MILL Scene9_b，尼康、手机 | 6／6 | 3.883–3.919 | 0.347–0.445 | 0.039–0.040 | 0／2 |
| CHROMA-MILL Scene10_a，尼康、手机 | 6／6 | 4.204–4.258 | 0.412–0.575 | 约 0.038 | 0／2 |
| Beyond RGB，两种参考场景的四个目标 | 0／12 | 0 | 0 | 0 | 0／4 |

距离单位为 RGB8 MAE。照片来自既有开发对照，CHROMA-MILL 每次留出同一布景的两台相机，Beyond RGB 不进入拟合。18 份回退成片与目标原图逐像素相同，但有意染色也未迁移。12 份进入处理的参考没有超出训练增益范围，换灯造成的目标差低于 0.6；与此同时，四份白灯参考都给目标原色带来超过 3.8 的额外变化，有意染色的响应仅约预期的 4%。CHROMA-MILL 30 项、Beyond RGB 20 项灰阶局部测量联合通过均为零。[逐条件与成片哈希](2026-11-20-target-original-edit-evidence.json)

## 决定

直接从目标原图开始不会继承 13 号基底，但错误估出的编辑仍会把干净参考变成不应存在的调色。该固定特征的有界加权避免了大增益外推，没有支持「保原色同时迁移编辑」的成片。结束基于同一数据和冻结特征的插值、阈值或带宽试验。保留已知前后参考可以应用同款编辑的离线上限。普通两图仍需要新的可验证编辑线索或更可靠的学习方法。此候选限定全局染色，软件默认、局部场景、真人肤色、独立白墙和视频的完整门槛仍单独保留。

## 复现

使用已有自适应环境和冻结外观检查点，准备[CHROMA-MILL 成片](2026-11-20-chromamill-matched-supervision.md)及[Beyond RGB 成片](2026-11-20-beyond-rgb-matched-cast.md)。从仓库根目录运行，输出目录须未占用。两次 30 张成片、10 张拼图与报告一致，图像及权重留本机。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_target_original_edit.py "$base/beyond-rgb-matched-cast-prepared-v1" "$base/chromamill-camera-pair-first-v1" "$base/beyond-rgb-matched-cast-first-v1" "$base/chromamill-matched-supervision-first-v1" "$base/target-original-edit-replay"
~~~
