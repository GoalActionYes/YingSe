# 从源码启动映色

面向 Windows 64 位。先运行主开发版；GitHub 源码不包含便携运行时、模型、演示素材或 EXE。使用自己的测试图片即可，不需要剪映。

## 第一次运行

安装 Git 和 Python 3.12 后，在 PowerShell 执行。路径有空格时使用引号。不需要激活虚拟环境。

```powershell
git clone https://github.com/GoalActionYes/YingSe.git
cd YingSe
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\prepare_ffmpeg.py
.\.venv\Scripts\python.exe -m reference_color_app.server --port 0 --data-dir .\dev-data
```

最后一条命令会打印 `http://127.0.0.1:端口`，手动在浏览器打开。保持终端运行；按 Ctrl+C 停止服务。下次只需进入仓库，重新执行最后一条命令。不要把本地服务直接暴露到公网。

`prepare_ffmpeg.py` 从已安装的 `imageio-ffmpeg` 包复制 FFmpeg 到程序读取的 `.deps` 目录；仅安装依赖还不够。脚本不读取系统 PATH 中的 FFmpeg。若设置了 `IMAGEIO_FFMPEG_EXE`，先在当前终端移除该覆盖，再执行准备命令。

## 五分钟体验

1. 导入目标图片或短视频，再选择参考图。
2. 等待分析完成，比较原图与效果；调整强度，关闭色彩克隆查看原色。
3. 视频先选分析帧；分别测试图片导出、短视频导出与 CUBE 导出。
4. 记录使用的提交、输入类型、强度和预期效果。素材不随源码上传。

建议先用同一张无损 PNG 同时作为参考和目标，验证不产生变化；再使用不同参考观察局限。内置演示素材未入库，新克隆不要依赖演示按钮。输入兼容范围见 [当前状态](PROJECT_STATUS.md)。

主版现有基础／HSL／曲线手动微调，默认不改变结果。按 [控件指南](MANUAL_COLOR_CONTROLS.md) 逐组尝试，再导出图片或视频；关闭色彩克隆不等于关闭手动调整。

## 修改后检查

```powershell
.\.venv\Scripts\python.exe scripts\run_portable_tests.py
```

这组检查只依赖源码和上述依赖，覆盖映射、参数、管线、HTTP 与导出任务。明确排除了需要原生／GPU 外部证据的 `test_evidence`、`test_features` 和 `test_lut.FrozenGPUChecks`。它不等于全部本机逆向验证，也不证明视觉质量普遍改善。验证记录见 [交接检查](HANDOFF_VALIDATION.md)。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 找不到 Python 3.12 | 先确认 `py -3.12 --version`；不要直接使用来源不明的便携 Python |
| 找不到 FFmpeg | 在当前仓库重新执行准备脚本，确认出现 `FFmpeg ready` |
| 页面打不开 | 检查终端是否仍运行，使用本次打印的地址；随机端口每次可能不同 |
| Git GUI 中中文异常 | 使用 UTF-8 编辑器打开原文件确认；不要直接把显示乱码另存覆盖原文件。HTML 用浏览器打开 |
| Git GUI 左侧看不到全部文件 | 这里通常显示的是变更列表。用资源管理器打开仓库根目录，按 README 找源码 |
| 克隆后没有 EXE／模型 | 这是预期行为。源码按本文运行；便携体验需要完整 ZIP |

## 肤色保护实验版

先完成主开发版启动。实验副本在 [experimental/skin-protection-20261019](../experimental/skin-protection-20261019)，其运行依赖与模型需要单独准备，参考该目录的说明和依赖锁定记录。主版本的最小依赖不等于完整的分割模型环境。

FFmpeg 可在仓库根目录执行 `python scripts/prepare_ffmpeg.py --skin` 准备；服务必须从实验目录启动，才能导入该副本的 `reference_color_app`。缺少离线模型时不要开启肤色保护。保护视频首版为 30 fps；CUBE 无法携带空间蒙版。不要把参考肤色协调研究当作已上线开关。
