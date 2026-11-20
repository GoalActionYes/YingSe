# 平坦区域细尺度对照拦住两次换灯重拍，JPEG 后失效

旧三次拟合可以重放同一原片的调色，但把两张同机位换灯强度的重拍当成同原片编辑。本轮在普通两张 RGB 输入上增加一个判断：三次拟合后，与参考成片比较平坦区域的 2×2 像素亮度差分。两张图的平坦块都需达到事前亮度与波动范围，至少 64 个块，残差标准差不低于 0.25，相关系数至少 0.65 才保留三次拟合。判断也会受到取样、局部细节、噪声和编码影响，不能据此声称识别了相机传感器指纹。[事前协议](../../experimental/adaptive-reference-20261113/aligned-cubic-fine-residual-protocol.md)

| 原有开发对照 | 相关系数 | 实际成片 |
| --- | ---: | --- |
| 场景 1，白灯高档目标和低档参考 | 0.286 | 回到原 13 号，逐像素一致 |
| 场景 2，白灯低档目标和高档参考 | 0.251 | 回到原 13 号，逐像素一致 |
| 八组已通过的同原片 Film A | 0.943–0.971 | 八张原三次成片逐像素保留 |

场景 1 的原误触曾把近中性色相对原图的偏差由 4.19 抬高到 10.49 RGB8，回退后恢复到原 13 号的 4.19。场景 2 回退也恢复原 13 号结果；该例原误触的近中性色变化较小，回退是为了不将重拍误当成已知编辑。其余 45 个重拍方向仍沿用原决策。14 对正例先复现原决策，八组原三次重放全部保留；整图响应误差 0.344–1.071 RGB8，六项可测的近中性色仍过线，两项缺测保留。完全同图的原身份旁路也保持。达到本轮无损开发门槛。[场景 1 成片对照](../../outputs/adaptive-reference-20261113/aligned-cubic-fine-residual-v1/CHROMA-Scene1_a-white-241-to-anchor-comparison.png) · [场景 2 成片对照](../../outputs/adaptive-reference-20261113/aligned-cubic-fine-residual-v1/CHROMA-Scene2_a-white-241-from-anchor-comparison.png)

JPEG 85（4:2:0）往返后，两项重拍的相关性仍低于门槛。八个 Film A 正例中，五个在旧拟合阶段已经回退，三个进入细尺度比较但相关系数只到 0.126–0.414，最终全部无法重放。下一步用同一来源的真实编辑输出核对只压缩参考、不同导出尺寸及压缩前共同低频信息，先验证普通文件能保留什么证据。不能把当前无损开发门槛扩大为适用于常见 JPEG 或跨场景的成绩。本轮未改正常软件路径，未使用正式独立素材。[逐项结果、哈希与缺测](2026-11-20-aligned-cubic-fine-residual-evidence.json) · [原误触及偏差](2026-11-19-aligned-cubic-capture-stress.md)

两次各十张实际成片、两张并排图和报告逐字节一致。使用固定旧压力测试归档和既有研究环境，从仓库根目录运行，输出目录须未占用。

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '4'
& .tools/adaptive-env/Scripts/python.exe -c 'import runpy,sys; sys.path.insert(0, "experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1), run_name="__main__")' experimental/adaptive-reference-20261113/probe_aligned_cubic_fine_residual.py outputs/adaptive-reference-20261113/aligned-cubic-fine-residual-replay
```
