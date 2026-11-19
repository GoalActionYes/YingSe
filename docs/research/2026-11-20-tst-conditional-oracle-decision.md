# 给出正确保色决定后，学到的改色增量仍未达标

将是否改色和改多少分开训练后，第 23 组第三参考下应改色的像素全部被判断为保色。本轮固定之前学到的系数、亮度、像素限幅及渲染，只在五组明确保／改切换区用真实作者记录纠正判断。两份诊断一份使用 13 号原增量，一份使用原编辑预测头的增量。区外仍保留原判断和成片。[事前协议](../../experimental/adaptive-reference-20261113/tst-conditional-oracle-decision-protocol.md) · [原自动判断](2026-11-20-tst-conditional-chroma.md)

| 开发组 | 自动判断及学到的增量，换款投影 | 正确判断及 13 号增量，换款投影 | 正确判断及学到的增量，换款投影 |
| --- | ---: | ---: | ---: |
| 2 | 0.422 | 0.289 | 0.131 |
| 3 | 0.519 | 0.518 | 0.365 |
| 8 | 0.349 | 0.876 | 0.410 |
| 10 | −0.012 | −0.024 | −0.190 |
| 23 | 0.063 | 0.550 | 0.559 |

正确判断把两臂的保色误差都控制到 0.393–0.510 RGB8。应用学到的增量时，第 23 组换款幅度恢复到预期的 55.9%，相较原联合候选进步 49.6 个百分点，但低于 75% 门槛。第 10 组方向反向。应用 13 号增量时，第 8 组虽过风格投影门槛，同款换参考差为 4.523 RGB8，也未通过。两臂联合均为 0/5。[第 23 组本机离线成片](../../outputs/adaptive-reference-20261113/tst-conditional-oracle-decision-v1/23-2-conditional.png) · [第 10 组本机反例](../../outputs/adaptive-reference-20261113/tst-conditional-oracle-decision-v1/10-2-conditional.png)

在已知正确的局部决定下，当前编辑预测也无法形成可靠的换款响应。收尾只优化此判断器的路线，后续如继续需要同时取得更可靠的编辑颜色信息。作者区域只用于离线诊断，现有软件和独立验收未变。

## 复现

在已有视觉环境将仓库根目录加入 `PYTHONPATH`，设置 `OPENBLAS_NUM_THREADS=2`、`OMP_NUM_THREADS=2`，运行 `python experimental/adaptive-reference-20261113/probe_tst_conditional_oracle_decision.py <原训练缓存> <新输出目录>`。每遍验证原 216 张成片逐像素一致。两遍各 30 张诊断成片及报告逐件同哈希。从落盘图复算局部风格投影和响应误差，并确认测量区外像素保持。报告 SHA256 为 `fc3e66fb264606849b2f8d44595db9b56a04a2ba276c5455c2241f7bf9cbf70f`。[脚本](../../experimental/adaptive-reference-20261113/probe_tst_conditional_oracle_decision.py)
