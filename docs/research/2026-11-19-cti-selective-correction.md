# 灯色开关保护了普通参考，却把有意蓝调也当成偏色

固定 CTI-Retinexformer 的白／蓝最近点判定。只有判为蓝灯才应用上轮的成对全图条件校正，其它参考逐像素保持。本轮没有使用作者的增强输出。实拍部分复用同一权重及输入的原 13 号和校正成片。补充八张有意加蓝参考，取同场景未加风格白／蓝实拍的线性通道均值比，作为对白灯原片的数字编辑，在 Film A 关闭、开启两种状态沿用。它是一组可核实的程序编辑，拍摄条件本身保持白灯。[事前协议](../../experimental/adaptive-reference-20261113/cti-selective-correction-protocol.md) · [固定灯色估计](2026-11-19-cti-reference-screen.md)

四组未加 Film A 的数字蓝调参考均符合输入资格。与实际蓝灯的线性通道均值最大差为 0.00561，所有数字参考与实拍蓝图的平均差为 5.83–10.90 RGB8。开 Film A 后有一组裁切比例达 7.67%，保持在结果中。均值对齐用于匹配色偏强度，两类像素图仍有差异。

| 对照 | 结果 |
| --- | --- |
| 六张普通实拍及 Film B | 全部回退，参考逐像素保持 |
| 八张有意加蓝参考 | 全部误触发，八张目标均出现严重退化 |
| 实际蓝灯相对白灯 | 近中性色变化 2.15–7.92 RGB8，八项均减半，三项达到 ≤3 |
| 实际蓝灯下的有意 Film A | 四项近中性色方向投影为 0.559–0.744，均低于 0.75 |

最明显的有意蓝调反例是 Scene1_a、B_241、Film A 开启。目标近中性色与编程效果的绝对误差由 5.84 升至 39.25 RGB8，增加 33.41。该控制能检查是否抹掉有意蓝调，不能替代真实作者风格和自然评价。这一固定开关未通过，结束用二类白点判定直接触发旧校正。[原流程成片](../../outputs/adaptive-reference-20261113/cti-selective-correction-v1/Scene1_a-241-on-original.png) · [误触发成片](../../outputs/adaptive-reference-20261113/cti-selective-correction-v1/Scene1_a-241-on-gated.png)

两遍各 25 份文件逐件匹配，含八张数字参考、十六张目标成片和报告。另独立复算线性均值及参考图、数字风格误差、16 个实拍成片身份和八项近中性色换灯差。复现沿用[灯色估计的本机环境与权重](2026-11-19-cti-reference-screen.md)，将仓库根目录、OpenCV 和 einops wheel 加入 `PYTHONPATH`，运行 `probe_cti_selective_correction.py <研究输出根目录> <本机作者资产目录> <新输出目录>`。无新训练、媒体上传、产品更换或独立验收。[脚本](../../experimental/adaptive-reference-20261113/probe_cti_selective_correction.py) · [数值与哈希](2026-11-19-cti-selective-correction-evidence.json)
