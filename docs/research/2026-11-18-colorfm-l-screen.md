# ColorFM-L 改善铁塔，人物与背景联合门槛未过

ColorFM-L 的源码有语义交叉注意力，但先为两张输入生成一组共享的颜色核，再对各像素的 RGB 变换。原生小图验证同一目标内不同位置的相同 RGB 输出一致。ColorFM-O 可以用语义信息拟合颜色流，拟合后也使用一个全局函数，最终求解只读 RGB 和时间。它们可以用全图语义选映射，渲染时无法按位置给同一个输入 RGB 分配不同输出。见[作者论文](https://arxiv.org/abs/2607.07119)、[ColorFM-L 固定源码](https://github.com/cszn/ColorFM/blob/64c84c228cb505425c18546b886d3c80906f5e27/models/colorfm_l.py)和[ColorFM-O 求解器](https://github.com/cszn/ColorFM/blob/64c84c228cb505425c18546b886d3c80906f5e27/solvers/colorfm_o_solver.py)。

## 源码限制和本批照片的误差是两项证据

对局部冷色开关的四张人物目标，允许每个输入 RGB 任意查表，以人物近中性色和背景核心的原有误差门槛归一化，求两项误差和的量化下界。各图下界为 0.0913、0.0123、0.0061 和 0，均低于可排除同时通过的 2。白球服的 3,908 个近中性人物像素中，只有 95 个与背景核心精确同色。两张无人控制的作者局部版与基础版逐像素相同。这些照片里精确同色冲突很少，仅凭下界不能判定实际全局模型无法通过整组门槛。查表使用已知作者结果，是离线表达测量。见[能力测量协议](../../experimental/adaptive-reference-20261113/global-local-capacity-protocol.md)。

## 62 张实际成片的结果

使用固定的公开 `ColorFM_L.pth`，123,249,227 字节。源码由 15 份 Git blob 核对，模型严格加载全部权重。现有环境缺少 Lightning 和 xFormers，独立启动器把单次 Lightning 日志调用改为标准库告警，保留作者的 PyTorch 注意力回退及所有数值路径，未安装依赖。沿用 80% 强度，原尺寸，模型只读取目标和参考，评分证据在出图后使用。原生同图合成样片误差为 6.875 码值，现有同图早退规则令实验入口逐像素保持。见[权重发布与许可](https://huggingface.co/heyh97791/ColorFM)及[实测协议](../../experimental/adaptive-reference-20261113/colorfm-l-screen-protocol.md)。

| 检查 | 结果 |
| --- | ---: |
| 人物背景抗干扰与整体着色联合 | 0/8 |
| 近中性色平均干扰变化 | 增加 39.72% |
| 近中性色单项最大干扰增幅 | +16.35 码值 |
| 对比度与去饱和响应 | 0/12 |
| 人物局部对照／整图冷色／无人负例 | 0/4、3/6、2/2 |
| 实拍两方向 | 铁塔通过，人像失败 |

铁塔整图误差从 13 号的 15.03 降至 10.58，低于原图的 11.39。近中性色误差由 15.91 降至 7.78。台阶人像近中性色误差从 16.77 增至 17.74，实际成片中台阶仍偏青。白球服的近中性色局部误差 5.47 → 5.07，未达到 1.5 上限。背景品红对照 `785-3553` 近中性色干扰从 3.49 增至 19.84。保留铁塔收益与全部退化样例，结束该固定全局模型筛查。

两遍实验报告、62 张 PNG 和两份下界报告匹配，独立复算两幅实拍的 12 项误差。见[小型证据](2026-11-18-colorfm-l-evidence.json)。SAGE-Color 的渲染路径有区域条件，但其原入口依赖受限访问的 SD3.5 Medium 和多个额外模型，此轮未安装或运行。SemiNFT 作者页仍未提供可执行代码入口。这两项不能用作者展示图补成产品证据。见[SAGE-Color 依赖说明](https://github.com/chenxib/sage-color/blob/main/model/README.md)和[SemiNFT 作者页](https://melanyyang.github.io/SemiNFT/)。

复现时在既有研究环境设置仓库根目录为 `PYTHONPATH`，分别运行[下界脚本](../../experimental/adaptive-reference-20261113/audit_global_local_capacity.py)和[两图脚本](../../experimental/adaptive-reference-20261113/probe_colorfm_l.py)。前者传入新输出目录，后者传入作者源码目录及新输出目录。权重许可限非商业研究，源码依赖副本、图像、配对图和权重仅留本机。稳定版与 13 号未升级，未读取独立保留样本。
