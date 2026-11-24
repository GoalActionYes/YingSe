# 两个室内场景得到与有意提亮接近的实拍换灯对照

用 Gros 人像轮相同的 RawTherapee 作者参数，把局部点分别放到 MIT Multi-Illumination 两处室内场景已登记的左墙。在同一个房间的八档真实灯位上保持点位和参数相同，只开关局部工具。目的是在同一块墙上取得可见变化相近、成因明确的两种参考，供自动两图调色到另一场景时测试。编辑由项目重放，拍摄人与原设置作者没有为这两间房重新调色。[事前协议](../../experimental/adaptive-reference-20261113/shared-authored-tone-admission-protocol.md) · [实拍与选区](2026-11-21-mid-gray-anchor-lights.md) · [原局部提亮](2026-11-24-gros-author-tone-switch.md)

| 素材 | 单图开关达标 | 有序换灯方向通过 |
| --- | ---: | ---: |
| 复印室左墙 | 7／8 | 4／56 |
| 客厅左墙 | 7／8 | 8／56 |
| Gros 两位右颊 | 6／6，复用有记录的原结果 | 0／12 |

复印室从灯位 4 换到 20 时，左墙 RGB 中位增量为 19、15、16。在原灯位打开局部提亮时，增量为 17、17、17，方向余弦 0.9948，幅度比 1.015。客厅从 16 换到 4 也通过，换灯增量为 25、21、16，编辑增量为 23、21、21。右墙、灰球及远处区域进入保护检查。两房间各有一张 24 号灯位的左墙有效像素不足，复印室 76.89%，客厅为 0%。所有灯位及方向均保留。[复印室对照](../../outputs/adaptive-reference-20261113/shared-authored-tone-admission-v1/14n_copyroom1/comparison.png) · [客厅对照](../../outputs/adaptive-reference-20261113/shared-authored-tone-admission-v1/everett_living2/comparison.png)

两位右颊中到高档的方向余弦为 0.9975 和 0.9899，可见方向都相近。但编辑／换灯幅度比分别为 1.576 和 1.336，超过 1.25 上限。按原门槛，眼下只有室内来源能提供这个匹配对照。两房间同属 MIT 采集，12 个方向含共用照片，不能计为 12 个独立场景。人像与室内均是已用素材，未准入训练或独立验收。本次只运行编辑器，室内成片可进入两个实际工作台的下一项开发对照。[全部通过方向及失败统计](2026-11-24-shared-authored-tone-admission-evidence.json)

## 复核与复现

使用已有 RawTherapee 5.13 和研究 Python。冻结复印室、客厅各八个灯位及源登记报告，复用 Gros 的六组已出开关图。在项目根为便携工具临时映射空闲 ASCII 盘符，运行脚本并传入新目录。[本机来源准备方式](2026-11-21-mid-gray-anchor-lights.md) · [复现脚本](../../experimental/adaptive-reference-20261113/probe_shared_authored_tone_admission.py)

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$root = 'outputs/adaptive-reference-20261113'
& ./.tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_shared_authored_tone_admission.py $root $rawTherapeeCli "$root/shared-authored-tone-admission-new"
~~~

两遍实际运行 38.6 秒和 28.8 秒。每遍 32 张新开关图、两份对照图的像素或文件身份独立核对。从落盘成片复算所有新室内方向、原人像方向、有效像素和保护区，并核对侧车只改变三项几何字段或一个开关。两遍完整报告逐字节相同，SHA-256 为 `51dfe0f5e9e13517b6f43b3e0c11678ef5d8b970c47940e4d386cd8be8e79aef`。图片、原设置、运行时及完整评分留本机。
