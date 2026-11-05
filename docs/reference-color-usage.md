# 映色参考调色使用说明

下载并完整解压[Windows便携包](../dist/reference-color-20261016-222448.zip)到可写文件夹，双击“映色参考调色.exe”或“ReferenceColor.exe”。无需安装Python。请保留完整runtime和.deps目录，不从ZIP内直接运行。处理都在本机进行。旧包及其data保留，使用10位SDR支持请启动本链接中的新版入口。

1. 左侧导入目标照片或视频，右侧“调整 → 基础 → 色彩克隆”点击加号或缩略图选择参考图片；两份素材就绪后自动分析。也可先使用内置合成示例。
2. 视频在播放器底部选择分析时间，点击“应用此帧”，重新分析后核对显示的实际画面和时间。
3. 中间切换调色结果、原始画面或同帧对比。右侧滑块/数字输入调整0–100强度；关闭“色彩克隆”显示并导出原色，保留强度，再次启用恢复应用。“恢复原画”将强度归零。更换素材或参考图后自动重新分析，也可点击“重新分析”重试。
4. 顶部“导出”定位右侧导出区。图片可选择PNG或.cube；视频可选择MP4或.cube，导入视频后自动选择MP4。只有显示完成后才点击“下载完成文件”；可另存诊断记录。取消视频导出后不会发布未完成文件。视频整段采用本次选定帧的固定映射。

支持8位图片及8/10位SDR视频。10位视频进入分析和着色前转换到RGB8，左侧素材信息和导出说明明确提示，MP4输出为8位；这是一项兼容转换，不是剪映原生10位处理链的恢复结论。HDR/BT.2020、未知视频位深、12/16位视频和高位深图片暂不支持；Log及完整ICC管理未验证。图片保留解码方向和原尺寸。MP4使用H.264/AAC有损编码，保留首条可选音轨，奇数尺寸补边。.cube是按当前强度合成的64级单表，请在其他软件选择trilinear并检查其色彩管理设置。单个文件仍不超过512MB；更大素材需先另存短片段。

关闭窗口后后台仍保留；再次双击恢复本包服务和最近有效项目。所有导入副本、项目、导出、日志和窗口配置在本包data内。移动文件夹前，等待导出结束或取消，再运行 `ReferenceColor.exe --stop` 安全关闭后台；该操作不会结束其他应用。

启动失败查看data/launcher-error.txt、data/native-launcher-error.txt和data/server.log；缺少FFmpeg时会拒绝启动。请重新完整解压原包，不移动单个exe。发布文件摘要见[SHA256文件](../dist/reference-color-20261016-222448.sha256)和包内package-manifest.json。`ReferenceColor.exe --check`可隐藏启动并检查后台。

本版为独立实验实现，暖色视频有限样例已与剪映11.6.0.14552比较，尚不承诺逐像素一致。同一张图同时作为目标/参考，在最大强度也可能变化；内容比例会影响全局统计，不提供语义分割或人像保证。详见[验证报告](../outputs/reference-color-validation/验证报告.md)、[逐项验收与证据索引](reference-color-acceptance.md)、[算法说明](reference-color-algorithm.md)。旧映色原型与本版独立入口均保留。

开发与复查入口：[产品源码](../reference_color_app/)、[便携启动/构建/校验](../desktop/reference_color/)、[受控素材](../outputs/jianying-color/test-kit/manifest.json)、[测试](../tests/reference_color/)。从源码启动可使用已有Python依赖运行 `python -m reference_color_app.server`；开发接口与便携入口详情以源码及[启动说明](../desktop/reference_color/README.md)为准。


历史记录：2026-10-16早前精度修复曾按8位输入合同拒绝10位SDR，保留60项测试及对应数值证据，摘要为 `3fe2b1eafd3e2498017b0172faf0dfed2ba5ae985b2d88b1689c6c4e7ff03405`。之后用户明确批准新增10位SDR兼容导入，当前输入策略见上文，64项Python及13项界面测试通过。数学、曲线和LUT文件未变；媒体输入策略变更后的摘要为 `0024ea158e5b3d7dca14195c806b80beb9236be13b534470f2d04f5e14364514`。详见[修复记录](../outputs/video10-validation/修复记录.md)。
