# 同期 darktable 仍未恢复作者暖墙开关

为取得有意改变浅墙颜色的真实正例，用 Terry Pinfold 发布的模块设置重放同一张混光原片。作者增强的是受暖灯照射墙面的黄色，用手绘蒙版限制对桌面的影响，蒙版还包括部分肤色。上一轮的 Windows darktable 5.4.1 无法导出这张富士 RAW。此次隔离使用 5.0.0，其 Color Zones 模块版本为 5，混合参数版本为 13，与侧车一致。发布时间早于帖子。这些线索支持用来测兼容性，尚不能确定作者用的精确版本。[编辑者说明](https://discuss.pixls.us/t/playing-with-different-light-colors/48178/13) · [5.0.0 发布及校验值](https://www.darktable.org/2024/12/darktable-5.0.0-released/) · [上次成片与未通过项](2026-11-24-playraw-flash-room.md)

| 输入与方式 | 结果 |
| --- | --- |
| 富士 RAF，侧车只关 Color Zones，标准导出 | 访问异常，无 TIFF |
| 富士 RAF，无侧车，标准导出 | 访问异常，无 TIFF |
| 富士 RAF，同一关态侧车，缩小尺寸路径 | 访问异常，无 TIFF |
| 另一张奥林巴斯 ORF，无侧车，标准导出 | 退出 0，得到可解码且带 ICC 的 1024 长边 RGB TIFF |

两个富士详细日志都在 RAW 解码、传感器准备、白平衡和高光处理后，进入约 6240 × 4155 的去马赛克输入时停止，随后报 0xC0000005。缩小输出仍把原尺寸附近的数据交给去马赛克。现有对照把故障范围缩小到这张 RAF 的去马赛克路径，尚未确定具体故障指令、内存根因或实际受影响的其他相机。相同版本可导出另一张 RAW，富士无侧车也复现了异常，所以没有进入作者颜色模块的画质评分。按约定结束本地 Windows 版本尝试。恢复这项研究需可验证的原始前后导出，或有实质运行证据的另一渲染路径。[事前协议与故障对照](../../experimental/adaptive-reference-20261113/playraw-terry-wall-switch-protocol.md) · [日志摘要和文件哈希](2026-11-24-terry-wall-switch-evidence.json)

## 复现

Windows 官方包 127,802,285 字节，用现有 7-Zip 解包后约 524 MB，未安装。临时映射项目根到空闲 ASCII 盘符，使用已有 `darktable_off` 生成关闭侧车，其余原始字节不变。用以下参数执行 CLI，在新目录分别配置缓存与日志。在无侧车对照中省略侧车位置。在缩小路径使用 `--hq false`。两型 RAW 和侧车保留本机，日志亦不上传。复核覆盖全部实际运行的退出值和文件、单字节侧车差、有效 ORF 输出以及两份 RAF 日志的最后处理环节。未取得暖墙开关成片，未运行两个工作台或训练。[固定开关函数](../../experimental/adaptive-reference-20261113/probe_playraw_flash_room.py)

~~~powershell
& $cli $raw $disabledSidecar "$case/result.tif" --width 1024 --height 1024 --hq true --apply-custom-presets false --export_masks false --icc-type srgb --core --configdir "$case/config" --cachedir "$case/cache" --disable-opencl -t 2 -d pipe --conf write_sidecar_files=never --conf plugins/imageio/format/tiff/bpp=8
~~~
