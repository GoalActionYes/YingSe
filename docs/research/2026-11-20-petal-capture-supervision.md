# 加入拍摄变化监督后，冻结 PETAL 仍未稳定辨认调色

固定 PETAL 余弦检索曾在四来源中认对 37/72 个状态，但把 8/24 个未编辑参考误判为 Film B。本轮固定 PETAL，只拟合两个小型岭回归判断器。两者使用同样的原样／Film A／Film B 标签，实验臂还使用同一场景换灯、曝光或相机的配对，要求同款读数保持一致。每次留出整个拍摄来源，JPEG 85 也仅用于测试。推理判断只用参考成片，选定操作后以 80% 渲染目标原图。[事前协议](../../experimental/adaptive-reference-20261113/petal-capture-supervision-protocol.md) · [原余弦检索](2026-11-20-petal-known-look-selection.md)

| 留出来源 | 无约束 → 成对约束，正确状态 | 无约束 → 成对约束，关闭态保持 | 无约束 → 成对约束，风格通过视图 |
| --- | ---: | ---: | ---: |
| DAST 人像曝光 | 23/45 → 20/45 | 14/15 → 13/15 | 1/15 → 0/15 |
| MLS 实拍换灯 | 6/12 → 5/12 | 2/4 → 2/4 | 0/4 → 0/4 |
| Middlebury 相机色卡 | 2/6 → 2/6 | 0/2 → 0/2 | 0/2 → 0/2 |
| ARRI 路径实拍 | 3/9 → 3/9 | 0/3 → 0/3 | 0/3 → 0/3 |

原样与 JPEG 的汇总读数相同。逐张核对发现，成对约束臂有一个原应选 Film A 的人像参考，压缩后从未调色改选 Film B，两次都错。成对约束将总正确数由 34 降至 30，关闭态误触由 8 增至 9。两臂的最坏近中性色换拍摄差均为 8.91 RGB8，超过 3。没有任一来源通过。四个拟合折分别使用 27–66 个原样参考和 9–54 个同款配对差，范围有限，无法把失败单独归因于编码器或监督数量。[人像曝光成片](../../outputs/adaptive-reference-20261113/petal-capture-supervision-v1/paired/original/dast) · [蓝灯成片](../../outputs/adaptive-reference-20261113/petal-capture-supervision-v1/paired/original/mls)

结束用固定 PETAL 特征对这三种已知操作做余弦检索和这套成对监督判断。两次对照均表明，编码器的制备和压缩稳定性没有消除原样误触与风格漏判。后续需要新的拍摄／编辑监督或能直接验证开放风格迁移的输入证据，继续在当前三态上调阈值没有依据。现有软件未改，旧开发照片不计入独立验收。

## 复现

在既有 CUDA 视觉研究环境中将仓库根目录、OpenColorIO 2.5.2 和 SciPy 1.16.3 加入 `PYTHONPATH`，设置 `OPENBLAS_NUM_THREADS=4`。运行 `python experimental/adaptive-reference-20261113/probe_petal_capture_supervision.py <新的输出目录>`。两遍各 288 张成片、一个特征缓存和一份报告均同哈希。用保存的 144 个向量独立建立增广最小二乘问题，复算 288 个选择及成片。报告 SHA256 为 `5f921d4d63545bff2e732052b10dd0aa061868b3dcb92d5beaa035c29144a578`。特征和成片留本机。[脚本](../../experimental/adaptive-reference-20261113/probe_petal_capture_supervision.py)
