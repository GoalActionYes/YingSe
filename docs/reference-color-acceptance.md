# 映色参考调色：交付与逐项验收索引

当前最新版：[10位SDR兼容版222448](../dist/reference-color-20261016-222448.zip)。用户批准扩展原8位输入边界，明确10→8位转换；真实G盘4K相机素材读取、调色与整段MP4导出通过，64项Python/13项界面测试通过。当前页面后台已更新并保留项目，见[修复记录](../outputs/video10-validation/修复记录.md)。下方214939及首次8位交付均为历史基线。

晚间面板改版交付：当前新版为 [reference-color-20261016-214939](../dist/reference-color-20261016-214939.zip)，深灰青色三栏界面、参考图自动分析、启用开关和数字强度，PNG/MP4/CUBE导出保留。Python62项、JS11项和便携副本实导出通过，独立审查发现的1项竞态已修复。算法文件未变，一致性未知仍保留。详见[本轮交付与调查记录](../outputs/reference-color-validation/clone-panel-20261016/交付与续查记录.md)。下方首次交付记录与数值证据保留为历史基线。

2026-10-16。依据用户批准的[设计](superpowers/specs/2026-10-15-reference-color-app-design.md)及[计划](superpowers/plans/2026-10-15-reference-color-app.md)。Tasks 1–9、最终广泛审阅及精度修复复审均已通过；本文件保留逐项证据与适用边界。不把有限样例扩写成完整原生源码恢复或逐像素一致。

## 交付入口

- [干净便携 ZIP](../dist/reference-color-20261016-203854.zip)、[SHA256](../dist/reference-color-20261016-203854.sha256)、[解压目录](../dist/reference-color-20261016-203854/)、[发布清单](../dist/reference-color-20261016-203854/package-manifest.json)。ZIP 62,836,998 字节，SHA256 `efc8d55e0204fc08a4fbaade4c06bf2bab1859ed39cb54843ae322821e77b075`。
- [使用说明](reference-color-usage.md)、[独立源码](../reference_color_app/)、[启动与构建源码](../desktop/reference_color/)、[合成素材清单](../outputs/jianying-color/test-kit/manifest.json)。完整研究素材在 test-kit；便携包仅带白名单演示子集。
- [运行验证报告](../outputs/reference-color-validation/验证报告.md)、[算法与证据分级](reference-color-algorithm.md)、[专项调查](../outputs/jianying-color/专项调查报告.md)。旧 prototype 与旧启动入口保留。

## 证据分级与原生/本软件对应

| 层次 | 原生证据与已知对应 | 本软件入口/保存 | 结论及限制 |
|---|---|---|---|
| 输入绑定 | 资源 source feature_type0→LUT input0，target feature_type1→input1；真实252/228字节缓存及暖参考缓存PNG支持UI方向 | pipeline.py：目标→mode0/63项，参考→mode1/57项 | 当前实现方向由资源、缓存参考身份与输出支持；缓存PNG不能单独证明完整UI绑定，宿主目标纹理仍未知 |
| 预处理与特征 | 默认680最长边、20步长、256亮度桶、两种采样、RGB/YUV、区域顺序及回退来自反编译；缓存63/57 little-endian float32 | features.py、contracts.py；诊断source_packet/reference_packet | 静态移植和真实缓存逐字段测量；视频解码、缩放、宿主颜色处理仍有残差，不承诺缓存逐字节兼容 |
| 矩阵与参数 | 逆矩阵、T/R/A/F/C、37 vec4、常量位模式、退化fresh-object初始化来自定点反编译 | matrices.py、parameters.py、curve_tables.json；诊断矩阵/parameter_buffer | 运算顺序和边界有独立数学真值/分支证据；未取得原生矩阵/参数运行转储。跳写F按已证明新对象零初值；未知且未消费的eigen scratch保留未知 |
| 亮度曲线与双LUT | 资源/明文GLSL、256三通道曲线、64³ identity atlas、内部.8/1两次绘制、512×1024布局及最终混合公式 | lut.py 两层64³RGB float32；Mapping.layers、LUT摘要 | 静态证据；提取shader在系统NVIDIA OpenGL对CPU测试最大1字节，不等于剪映GPU运行中间量 |
| 强度 | 用户确认原生导出UI50/100；明文最终shader分段混合 | UI0..100→内核0..1；原图/第一层smoothstep，再第一/第二层插值；cube采最终组合 | 暖色端到端输出支持当前绑定；完整UI强度扫描未采集。.cube已由独立FFmpeg实际ASCII reader验证，非仅自写公式互证 |
| 原生持久化 | PersistentColorMatchInfo三键target_feature_path/source_feature_path/target_image_path及getter/setter双向拷贝 | storage.py 自有JSON、NPZ、内容哈希、时间、预处理、算法版本、项目设置、完成任务记录 | 对应功能但非同格式；三路径序列化函数未见强度，不能推成全程序没有强度。原生草稿编码、完整失效/复用策略未恢复 |
| 视频状态 | 原生仅有限时点人工导出观察，无逐帧中间量 | media.py实际PTS，固定选定分析帧；jobs.py整段复用本次映射 | 固定映射、缓存键与原子保存是独立设计。稳定样例不证明剪映固定LUT或相同状态更新 |

原生证据入口：[布局](../outputs/jianying-color/deep/feature-layout-audit.md)、[真实缓存与预处理](../outputs/jianying-color/deep/原生特征缓存与预处理验证.md)、[区域顺序](../outputs/jianying-color/deep/region-order-audit.md)、[退化状态](../outputs/jianying-color/deep/degenerate-state-audit.md)、[矩阵](../outputs/jianying-color/deep/matrix-probe-port-report.md)、[参数规则](../outputs/jianying-color/deep/parameter-rules-audit.md)、[亮度表及输入LUT](../outputs/jianying-color/deep/亮度纹理与输入LUT恢复.md)、[双LUT](../outputs/jianying-color/deep/双层LUT与亮度表追踪.md)、[渲染审计](../outputs/jianying-color/deep/render-probe-audit.md)、[系统GPU](../outputs/jianying-color/deep/系统GPU与CPU着色器对照.md)、[持久化](../outputs/jianying-color/deep/persistence-report.md)、[视频解码差异定位](../outputs/jianying-color/deep/视频解码差异定位.md)。反编译伪代码不是原始C++源码。

“运行观察”包括用户真实剪映导出、真实缓存、独立程序实际PNG/MP4/cube、浏览器操作和外部路径便携运行；各自只能支持各自范围。“数学推导/合理推断”包括协方差transport解释及误差来源候选，不当作原生运行转储。“独立设计”包括HTTP服务、固定帧映射、项目格式、缓存键及发布流程。“未知”包括宿主纹理精确颜色管理、部分GPU舍入、对象复用、全部UI强度及自然人像泛化。

## 设计要求逐项核对

| 批准要求 | 具体证据、测试或交付文件 | 状态/边界 |
|---|---|---|
| 固定目标版本和输入身份、0/50/100绑定 | baseline.json、test_evidence.py、task-1-report.md、test-native-cache/evidence.json | 28项身份；版本11.6.0.14552；未把截图默认80当导出强度 |
| 本地、独立入口、无目标DLL/模型/服务与旧预设依赖 | reference_color_app全源码、desktop/reference_color/build.py显式白名单、task-8-review.md | PASS；包内实际路径与外部副本真实导出证明独立运行 |
| 特征采样、符号/方向、float32、退化与真实缓存 | test_features.py、features.json、task-2报告/修复审阅、上述特征审计 | PASS静态移植/误差披露；缓存非字节一致 |
| 矩阵规则与37vec4、未知不编造 | test_matrices.py、test_parameters.py、task-3-cache-measurement.json、task-3-report.md | PASS；原生运行矩阵/参数转储仍缺失 |
| 双表布局、通道轴、蓝切片、端点、UV分支、共享参数 | test_lut.py、lut.json、task-4-report.md、系统GPU报告 | PASS当前独立真值；不是剪映宿主GPU等价证明 |
| 对外组合cube而非单层 | test_lut.py/test_jobs.py、cube-oracle.json、verify_cube_reader.py | 实际ASCII cube经FFmpeg7.1 trilinear：8192随机RGB+1250蓝切片边界；四档强度，非零最大1字节，仅一份真实暖映射 |
| 图片原尺寸/方向；明确SDR和高位深拒绝 | media.py、test_pipeline.py真实16位RGB/RGBA PNG/TIFF回归、task-5报告/修复审阅 | PASS常规8位SDR；Log/完整ICC/HDR专业流程不支持或未验证 |
| 视频选帧与实际画面、输入摘要、非法/越界/坏文件 | test_pipeline.py、pipeline-run、ui/video-100-export.jpg、metrics.json实际PTS | PASS覆盖；复杂VFR/尾部精确选帧非全面保证 |
| 固定映射、PNG/MP4/cube、音轨/补边/编码说明 | jobs.py、test_jobs.py、runtime-smoke、acceptance-runtime、task-6-report.md | PASS；首可选音轨AAC、H264、奇数尺寸pad；MP4有损误差分列 |
| 内容/时间/算法/预处理缓存键；临时文件原子发布；坏缓存重算 | storage.py、test_jobs.py，task-6修复：真实CRC/zlib损坏NPZ | PASS；自己的可复现格式，不伪装成剪映草稿 |
| 项目输入身份、帧、强度、证据版、映射及导出设置恢复 | storage.py、test_http.py、test_ui.cjs、ui/浏览器验证记录.md | PASS；真实服务重启与快速Home→End后100/MP4/1s恢复 |
| 本机HTTP、上传限制/真实解码、ID/穿越/失效拒绝 | server.py、test_http.py、task-7-report.md/task-7-review.md | PASS；127.0.0.1，错误JSON，旧media/mapping不能误导出 |
| 同帧预览、强度/重置、换输入/选帧失效、格式清旧链接 | web/app.js、test_ui.cjs四项独立VM、ui/浏览器验证记录.md | PASS；两P2修复后独立审阅通过；0强度两图src相同 |
| 进度/取消/错误/完成文件，不能假成功 | test_jobs.py/test_http.py、ui/cancelled.jpg及真实浏览器取消、任务报告 | PASS；取消error均无结果入口；完整后重命名并公开 |
| 空态/长中文/小窗口/真实下载 | ui/empty.jpg、long-name.jpg、small-window.jpg、download-verification.json | PASS实际浏览器；720×850无横向溢出；MP4下载351707字节哈希等于任务结果 |
| 双击本机窗口、隔离依赖、中文空格路径、重启/端口/缺依赖 | task-8-report.md/task-8-review.md；外部副本data/verification.json | 可见窗口为180112实际exe+唯一Edge标题+用户明确确认正常；原生截图受策略阻止。最终203854外部副本真实HTTP导出与生命周期验证，未伪称该干净原件已打开 |
| 白名单、合成demo、授权、清单和哈希、不带个人资料 | package-manifest.json、THIRD-PARTY-NOTICES、task-8-review.md | 当前203854候选1234项全校验，ZIP1235项；原件无data。FFmpeg精确配置依赖/source provenance和GPL直达导航为更广分发前P3；本地作业不阻塞，不代表已完成公开分发安排 |
| 冷暖/低饱和/对比/阴影高光/恒等/内容比例/运动 | test_features.py/test_matrices.py/test_lut.py；21 PNG、3 MP4，metrics.json/comparison.png | 实际执行；另有degenerate-product.json实际16对40×40黑白灰红全管线，有限LUT/0精确；同图100变化MAE8.6388/max45明确披露，无同条件原生导出 |
| 原生同PTS差异、静止首尾、误差分位数与分层定位 | metrics.json四时点；endpoint-frames.json、首轮验证报告0.5/7.5静止端点；独立映射链首次对照.md | 0/1/3/7实际PTS匹配；补测各自真正首尾帧，尾PTS7.958333/7.966667不同，静止保持端点MAE0/50/100为2.0864/2.1316/2.1748，p95=7；不是全运动段逐帧验证 |
| 使用说明、源码/程序/测试素材/报告/证据索引 | 本文件交付入口、reference-color-usage.md、验证报告.md | 交付齐备；最终源码审阅与精度修复复审已闭合（见台账） |

任务报告/审阅位于[非Git台账目录](../.superpowers/sdd/2026-10-15-reference-color-app/)，测试位于[tests/reference_color](../tests/reference_color/)，运行结果统一位于[outputs/reference-color-validation](../outputs/reference-color-validation/)。原Task9记录全Python58项和实际app Node VM4项通过；最终精度修复后全Python60项通过，受影响整批编码及诊断已重做。

## 尚未关闭的原生一致性要求

没有恢复全部原始源码；没有取得宿主输入纹理、完整GPU链、原生矩阵参数转储及所有对象历史。剩余差异继续由分层证据定位，不能以界面可导出作为算法等价结论。七种参考测试属于本软件行为验证，只有暖色视频有剪映导出；自然人像/复杂照明/跨场景尚无广泛对照。更完整强度扫描、同图原生验证、全运动段逐帧与宿主中间量仍需后续证据，当前不使用“一致复现”或百分比精度。

Task7旧映色入口要求的针对性复查见[old-portable-entry.json](../outputs/reference-color-validation/old-portable-entry.json)：既有0.2.2便携入口实际 --check退出0，认证健康正常，palette cinematic-v2，FFmpeg可用；没有既有活动实例，测试自有实例通过认证HTTP安全停止200。旧--stop显示确认模态，首次wrapper30秒超时，未将其冒称为成功。旧根目录0.2.1入口保留但未启动，已有FFmpeg unavailable错误未修复；旧可见完整工作流未重测，不能把0.2.2启动证明扩写成旧根目录已修复。


2026-10-16 最终精度修复：视频在RGB24转换前读取FFmpeg像素格式的各分量位深；超过8位、浮点及无法确认精度的输入明确拒绝。真实10位BT709 SDR上传返回400；真实8位YUV/RGB/灰度控制正常。此为独立App输入策略，不是原生算法恢复结论。修复后Python完整60项通过；受影响受控批次、cube、16对纯色及首尾帧诊断已重做，8位数值结论不变。当前算法摘要 `3fe2b1eafd3e2498017b0172faf0dfed2ba5ae985b2d88b1689c6c4e7ff03405`。
新包清单与源码匹配、ZIP摘要及全新外部副本的包内运行时实际PNG/cube/MP4导出和生命周期验证见 [最终修复发布审计](../outputs/reference-color-validation/final-fix-release-audit.json)。旧180240及其ZIP保留；当前干净原件无data。

最终审查记录：[精度修复复审](../.superpowers/sdd/2026-10-15-reference-color-app/final-fix-review.md)。已按批准的独立实验软件范围交付，不将未知原生层次升级为一致性承诺。
