# StyleGallery确定性：单步通过，整图在原预算内未完成

本轮没有新增画质结果。严格确定性条件解决了当前完整尺寸单步测试的梯度重复差异，但整图基准在600秒预算内未完成，未生成图片；没有启动后续检查点整图，旧逐像素桥接仍失败，旧两例不合并。

## 固定条件与单步结果

依据已安装版本的[PyTorch2.6确定性API](https://docs.pytorch.org/docs/2.6/generated/torch.use_deterministic_algorithms.html)和[复现说明](https://docs.pytorch.org/docs/2.6/notes/randomness.html)，在CUDA初始化前设置CUBLAS_WORKSPACE_CONFIG=:4096:8，启用严格确定性算法、关闭warn_only与cuDNN benchmark，设置cuDNN deterministic并保持TF32关闭。官方也说明，单一设置不能保证整条应用跨平台／版本复现，确定性计算可能更慢；本报告不外推这种保证。

按[运行前协议](../../experimental/adaptive-reference-20261113/stylegallery-deterministic-protocol.md)，复用相同22927缓存、空文本、排序构造的区域配对、64×64 latent、缩放1和原150步调度的首个t，只运行三个单步，不产生新候选图、不读取GT。

原执行A、原执行B、检查点C的损失均为3.990152359，三个梯度SHA256相同；A/B及A/C最大／平均梯度差均为0。增量CUDA峰值约1.160GB／1.161GB／0.398GB。会话35462正常退出0；[单步证据](2026-11-17-stylegallery-deterministic-evidence.json)保留全部设置、源身份与统计。它使用构造区域配对和固定缩放1，不代表历史匹配或完整动态GradScaler轨迹。

## 完整图实际执行与停止

随后按[整图对照协议](../../experimental/adaptive-reference-20261113/stylegallery-deterministic-image-protocol.md)，先在新目录运行同一22927的严格确定性、无检查点基准，仍保持原FP32准备、原512图、seed42、深度模型、150步和全部损失参数。没有运行已OOM的39532默认配置。

1130项主组件state逐值核验通过，加载13.504秒，准备／分区／匹配306.725秒，准备阶段CUDA峰值6920829952字节，约6.92GB。进入优化后明显变慢，终止时最后日志显示第10/150步。外层在600.047秒结束自有子进程，保存stop_reason=child_timeout，无原512图或256副本；不能把未完成案例记为0误差、画质失败或画质通过。

当前没有满足条件的完整确定性基准，因此不能运行或宣称原执行／检查点整图一致，更不能跳过基准直接合并旧结果。会话90879已退出，不轮询旧句柄、不增加步数或延长预算。

## 已核验与仍未知

audit_stylegallery_deterministic.py核对三个单步记录及哈希一致性、当前和冻结源身份、两个输入PNG及像素身份、1130项初态核验记录、超时范围和无有效输出，并确认旧桥接仍失败；[审计证据](2026-11-17-stylegallery-deterministic-audit-evidence.json)明确它没有独立重跑GPU梯度或整图。

单步诊断不到1秒与整图内的慢步骤同时改变了组件驻留、准备历史等条件；前者还显式清理过空闲CUDA缓存。当前尚无优化阶段allocated／reserved分项观测，不能直接认定慢因是缓存、碎片或某个算子，也不能把单步速度当完整图速度。

下一项只测同一FP32加载转FP16链中的实际分配／保留／驱动空闲显存，并比较释放空闲缓存前后同一单步损失与梯度。先证实执行数值不变及资源收益，才决定是否值得新的整图尝试；没有新证据，不重跑150步。不改变模型、照片、种子、强度、步数或画质门槛，不上传媒体／权重。稳定软件保持，整体目标仍未完成。
