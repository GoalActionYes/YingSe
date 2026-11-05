# 独立参考图调色 App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现独立 Windows 本地 App，把一张参考图的色彩风格应用到目标图片或视频，并用剪映真实缓存和导出逐层验证复现程度。

**Architecture:** 独立 `reference_color_app` 包负责媒体、特征、矩阵、参数、LUT、任务和页面；算法不依赖旧预设混合模块。先通过诊断程序隔离数值差异，再将核实后的模块用于产品。原生缓存只作为验证真值，不能成为用户处理新素材的必要输入。

**Tech Stack:** Python、NumPy float32、Pillow、本地 HTTP 服务、HTML/CSS/JavaScript、现有 FFmpeg、Windows 启动封装；优先复用已安装运行时，不增加未验证的产品依赖。

**Spec:** `docs/superpowers/specs/2026-10-15-reference-color-app-design.md`（用户已认可）。

## Global Constraints

- 目标版本固定为本机剪映专业版 11.6.0.14552。
- 全部媒体在本机处理。
- 不使用剪映DLL、模型或未公开服务作为本软件运行依赖。
- 不把旧的八种预设混合作为参考图匹配算法。
- 图片输出保留原分辨率。
- 首版限定常规SDR素材；HDR/Log专业色彩管理、多轨时间线、语义分割、账号和云端功能不纳入本次范围。
- 当前不承诺逐像素或逐字节完全一致。
- 未恢复项必须在开发报告和版本说明列明；实验性替代实现需要单独标记，不混入“已核实复现”结果。
- 仅完整完成的导出可显示成功；便携包采用文件白名单。
- 本目录没有 Git 仓库，不初始化仓库或制造提交；各任务通过源文件、验证记录和文件摘要留痕。

## 实际起点与证据

已经有独立特征探针 `work/jianying-color/deep/recovered_feature_probe.py`，完整57/63浮点布局、矩阵与参数审计、原生明文 shader、真实252/228字节缓存及未调色/50/100导出。参考图特征探针能独立恢复六个阈值，但其余字段仍有误差；这不等于完整算法已复现。矩阵、参数诊断移植在进行中，产品尚未接入。

执行前读取：`outputs/jianying-color/deep/原生特征缓存与预处理验证.md`、`feature-layout-audit.md`、`parameter-rules-audit.md`、`parameter-rules-rawconstants.json`、`亮度纹理与输入LUT恢复.md`、`outputs/剪映人工验证/分析结果/首轮验证报告.md`。先检查文件是否存在；缺证据时返回调查，禁止补写猜测为真值。

## 文件边界与公共接口

新增目录 `reference_color_app/`：`contracts.py`（数据结构）、`features.py`、`matrices.py`、`parameters.py`、`lut.py`（独立算法）、`media.py`、`pipeline.py`、`storage.py`、`jobs.py`、`server.py`、`web/index.html`、`web/app.js`、`web/styles.css`。测试在 `tests/reference_color/`；构建在 `desktop/reference_color/`；应用数据在 `reference_color_app/data/`，验证结果在 `outputs/reference-color-validation/`。

共用数据约定如下，所有后续任务保持这些名字：

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FeaturePacket:
    mode: int                 # 0:63项，1:57项；UI语义由输入绑定验证确认
    values: np.ndarray        # little-endian float32
    diagnostics: dict

@dataclass(frozen=True)
class Mapping:
    layers: np.ndarray        # 两层64x64x64x3，float32，轴顺序R/G/B
    diagnostics: dict

# extract_features(rgb_u8, mode) -> FeaturePacket
# solve_matrices(source63, reference57) -> dict
# solve_parameters(source63, reference57, strength=1.0) -> (np.ndarray, dict)
# build_mapping(source: FeaturePacket, reference: FeaturePacket) -> Mapping
# apply_mapping(rgb_u8, mapping: Mapping, strength: float) -> np.ndarray
# analyze(target_path, reference_path, time_seconds=0.0) -> Mapping
```

`rgb_u8` 是 H×W×3 RGB uint8；强度范围0..1。参数为37×4 float32。诊断包含算法版本、输入摘要、已核实项和未核实项，不将 NaN 静默转成零。首版不添加风格预设、语义分割或训练模型。

### Task 1: 固定可复现基准及对照方向

**Files:** 新增 `tests/reference_color/test_evidence.py`、`outputs/reference-color-validation/baseline.json`；读取现有 `test-native-cache/evidence.json` 及人工验证素材，不改原件。

**Interfaces:** 输出 baseline.json，包含相对路径、SHA256、尺寸、时长、UI强度0/50/100、缓存模式、输入绑定证据及未知项。

- [x] 用独立检查固定特征字节长度及精度，不以探针自输出充当期望值：

```python
def test_native_packet_sizes():
    from pathlib import Path
    p = Path('outputs/jianying-color/deep/test-native-cache')
    assert len(next(p.glob('*source_feature*.match')).read_bytes()) == 252
    assert len(next(p.glob('*target_feature*.match')).read_bytes()) == 228
```

- [x] 执行 `python -m unittest discover -s tests/reference_color`（测试采用 unittest.TestCase 包装，不安装 pytest）；先确认缺少基准/字段时失败。
- [x] 计算文件摘要，记录视频分析1秒的解码条件；用缓存参考PNG与暖色参考图的已有误差核对方向，保留目标视频原生输入像素仍未知这一限制。
- [x] 重跑证据检查；基准只保留测试素材，不复制用户其他剪映工程。

### Task 2: 完成特征提取的独立移植

**Files:** 新增 `reference_color_app/contracts.py`、`features.py`、`tests/reference_color/test_features.py`；读 `recovered_feature_probe.py` 与特征审计。

**Interfaces:** 输入RGB uint8，输出 `extract_features(rgb_u8, mode) -> FeaturePacket`；mode0为63项，mode1为57项。

- [x] 编写真实缓存对照及退化输入检查；误差作为测量，不直接扩大阈值掩盖差异：

```python
packet = extract_features(reference_pixels, 1)
assert packet.values.dtype == np.dtype('<f4')
assert packet.values.shape == (57,)
np.testing.assert_array_equal(packet.values[42:48], [30,87,126,137,167,225])
```

- [x] 运行测试观察缺失模块失败；移植680最长边、RGB/YUV、量化、两种采样、分区、回退、参考筛选、均值/协方差/逆矩阵与饱和度。去掉诊断探针的全局 mode，避免并发串线。
- [x] 对纯黑、纯白、纯色、单像素、低饱和与正常图检查长度和有限值；按原生分支保留失败条件，拒绝空图或非法mode。
- [x] 对真实缓存记录逐字段误差；单独调查视频解码、缩放、方向和舍入。无法消除的误差写入诊断与验证报告，不能标成字节兼容。

### Task 3: 验证矩阵与经验规则

**Files:** 新增 `matrices.py`、`parameters.py`、`tests/reference_color/test_matrices.py`、`test_parameters.py`；审阅进行中的 `matrix_probe_port.py`、`parameter_probe_port.py`。

**Interfaces:** `solve_matrices(source63, reference57)->dict` 返回T/R/A/F/C、flags、max_excess、eigenvalues；`solve_parameters(...,strength=1.0)` 返回37×4数组和诊断。

- [x] 给解析特征分解、逆矩阵回退、近零特征、特征值比70、外区限幅和饱和度比写边界用例。独立矩阵真值示例：

```python
# 非退化对角协方差的 transport 应接近 diag(2, .5)，
# 误差需保留原生分母1e-6而非强求数学精确值。
np.testing.assert_allclose(A, np.diag([2.0, .5]), atol=5e-5)
assert parameters.shape == (37, 4)
assert parameters.dtype == np.float32
```

- [x] 先运行测试；然后审阅移植与反编译的运算次序、位运算符号翻转、矩阵方向和每个参数槽，保留IEEE除零后的原生夹紧规则。
- [x] 查清参考 negligible 分支跳过 F 写入时的初始化与shader消费方式；有证据后实现，不把单位阵假设直接变成产品逻辑。
- [x] 在真实缓存上输出矩阵、标志和37vec4。没有原生矩阵转储时明确“静态移植，运行中间量未对照”；不把数学单测通过称为原生一致。

### Task 4: 串通双层LUT与最终渲染

**Files:** 新增 `lut.py`、`tests/reference_color/test_lut.py`；诊断生成器放 `work/jianying-color/deep/render_probe.py`。

**Interfaces:** `build_mapping(source,reference)->Mapping`；`apply_mapping(rgb_u8,mapping,strength)->uint8 RGB`；`cube_text(mapping,strength,size=64)->str`。

- [x] 测试已知身份表、通道轴顺序及强度边界，期望图来自明确RGB值：

```python
x = np.array([[[0,127,255],[255,0,127]]], dtype=np.uint8)
np.testing.assert_array_equal(apply_mapping(x, mapping, 0.0), x)
# 身份表还须检查R/G/B角点与每个蓝色切片边界。
```

- [x] 运行测试确认缺失实现；按已恢复shader写独立数值实现，或先用系统OpenGL诊断执行提取shader以隔离CPU参数问题。记录驱动、精度、坐标与RGBA8舍入；诊断shader文件不自动进入便携包。
- [x] 恢复256项三通道亮度表、NEAREST采样、64³身份atlas、两次输出绘制及0.8/1.0内部强度；核对UV有/无分支和512×1024上下层坐标。
- [x] 按资源最终公式实现原图/第一层smoothstep与第一层/第二层插值；先独立核对UI50/100到该强度的映射。
- [x] 用真实特征缓存隔离“参数→渲染”误差，再用独立提取结果跑完整链；生成对照图与误差分布。外部.cube必须采样最终组合映射，并测量其插值误差，不直接导出某一层冒充预览结果。

### Task 5: 媒体输入和完整算法管线

**Files:** 新增 `media.py`、`pipeline.py`、`tests/reference_color/test_pipeline.py`。

**Interfaces:** `decode_image(path)->uint8 RGB`；`decode_frame(path,time_seconds)->(RGB,metadata)`；`analyze(target_path,reference_path,time_seconds=0.0)->Mapping`。

- [x] 测试图片EXIF方向、原尺寸、视频指定秒数和无音轨素材；输入坏文件、越界时间、HDR标记返回明确异常。
- [x] 从旧 `prototype/server.py` 提取并审阅FFmpeg调用模式，独立实现参数列表调用；禁止拼接用户路径为shell代码。保留实际采样时间、解码元数据和输入摘要。
- [x] 串联两路特征与LUT；视频分析预览显示实际帧，不能用缩略图替代分析输入。
- [x] 对同一文件重复运行比较诊断摘要，验证没有旧素材或全局mode污染。

### Task 6: 状态保存、缓存失效与导出任务

**Files:** 新增 `storage.py`、`jobs.py`、`tests/reference_color/test_jobs.py`。

**Interfaces:** `start_export(mapping_id,media_id,strength,kind)->job_id`；`get_job(job_id)->dict`；`cancel_job(job_id)->None`。状态为 queued/running/done/cancelled/error。

- [x] 编写状态竞争检查：取消后的任务不能转done；损坏缓存必须重算；输入内容、分析时间、算法版本或预处理变化会改变缓存键。
- [x] 采用临时文件加原子替换保存映射和项目JSON，保存输入身份/时间/强度/证据版本/输出设置；诊断包含feature、matrix、parameter、LUT摘要。
- [x] 实现PNG原尺寸输出、组合.cube、MP4视频。视频使用固定分析映射，保留首条可选音轨，编码AAC；奇数尺寸用pad并告知输出尺寸，记录SDR及编码影响。
- [x] FFmpeg非零退出记error；取消终止子进程并移除对临时输出的下载入口；成功后才重命名与公开结果。
- [x] 对导出PNG和预览逐像素核对；对MP4按编码误差单独报告，对.cube比较多个随机RGB与切片边界。

### Task 7: 本地接口与最小独立界面

**Files:** 新增 `server.py`、`web/index.html`、`web/app.js`、`web/styles.css`、`tests/reference_color/test_http.py`。

**Interfaces:** POST `/api/media` 上传；POST `/api/analyze` 返回mapping_id和分析帧；POST `/api/preview`；POST `/api/export`；GET `/api/jobs/<id>`；POST `/api/jobs/<id>/cancel`；GET `/api/diagnostics/<mapping_id>`。

- [x] 写HTTP集成测试，覆盖完整图片流程、非法ID、失效mapping、路径穿越与导出失败。服务只绑定127.0.0.1，限制上传大小并校验真实解码，错误以JSON返回。
- [x] 界面提供目标导入、参考图导入、视频选帧、执行、同帧前后对比、0..100强度、重置、PNG/MP4/.cube导出与进度/取消。新素材或选帧变化立即清除旧mapping_id。
- [x] 将按钮状态关联实际任务；分析失败或取消不显示成功。诊断放单独下载入口，主流程只显示必要SDR限制和错误。
- [x] 在浏览器用测试素材操作并保存截图，检查空状态、长文件名、小窗口和服务重启；确认旧映色入口仍可用。

### Task 8: 独立启动与便携包

**Files:** 新增 `desktop/reference_color/boot.py`、`Launcher.cs`、`build.py`、`verify.py`、`README.md`。

**Interfaces:** 启动器选择可用本机端口，启动独立服务并打开页面；包位于新的 `dist/reference-color-<timestamp>/`，不覆盖旧映色包。

- [x] 检查构建清单：仅应用源码、运行时、所需依赖、授权说明与合成演示素材；拒绝包含 `.match`、剪映DLL、工作区日志、个人上传、浏览器profile。
- [x] 复用 `desktop/portable_boot.py` 与 `BuildPortable.py` 的成熟结构，修正模块入口和包名；使用隐藏后台进程、可用端口探测及启动失败提示。
- [x] 双击启动，完成一次图片和视频导出，再关闭重启；检查空格/中文路径、端口被占用及FFmpeg缺失行为。
- [x] 输出文件清单与SHA256，确认从包内运行，不借用开发目录或剪映安装目录。

### Task 9: 验收与可说明的复现结论

**Files:** 新增 `outputs/reference-color-validation/验证报告.md`、`metrics.json`、`comparison.png`；更新使用说明和 `outputs/jianying-color/专项调查报告.md` 的交付链接。

**Interfaces:** 报告按源码/反编译实证、运行观察、合理推断、仍然未知分级。

- [x] 运行恒等、受控冷暖/饱和/对比/阴影高光、内容占比、强度0/50/100和运动视频用例；写明每项实际执行条件与未执行原因。
- [x] 和现有剪映三份导出比较同时间点与静态首尾帧；报告RGB MAE、分位数、最大误差及解码/编码限制。避免把视频不闪烁解释为剪映必然固定LUT。
- [x] 单独列出当前剩余原生差异；若全链偏差仍大，继续从中间量定位，不以“界面能导出”结束算法复现。
- [x] 检查全部设计要求覆盖到Task 1–9；交付独立源码、程序、说明、合成素材、验证报告和证据索引。只有验证结果支持时才使用“一致复现”描述。

## 执行顺序与停止条件

依赖链：1→2→3→4→5→6→7→8→9。Task 3的矩阵/参数审计可以分别调查，Task 4的渲染诊断可在缓存输入下先行；产品接入仍需依次评审。算法差异是需要继续调查的问题，不是换用其他滤镜的许可。需要用户补充运行样本时先提供具体素材、操作与文件名。

自审：书面设计的输入/输出、固定选帧、自主保存、强度、取消、SDR限制、白名单打包和分级报告均有对应任务；公共接口命名统一。原生未知项有具体调查动作与验证层，不要求实现人员凭空补齐；产品代码尚未按本计划开始。

2026-10-16 收尾：Tasks1–9及最终精度修复复审通过；最终交付为203854。计划制定时的自审/未开始描述作为历史记录保留，当前状态以交付审计和台账为准。
