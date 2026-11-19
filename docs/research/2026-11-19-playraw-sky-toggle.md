# 已冻结两张原片与开关状态，Windows 5.4.0 导出未完成

从两位原片作者的授权帖子取回 24,088,448 与 24,200,704 字节的 RAW，长度与 HTTP 响应一致。用上一轮固定的第 9 楼与第 7 楼 XMP 生成开关状态。A 仅切换历史第 16 项，B 仅切换第 28 项。独立逐字节核对两对文件，每对仅有一个 ASCII 启用位由 `0` 变 `1`，反向切换恢复原字节。扫描有效历史也确认只在开态出现指定的活动蒙版曝光。[事前协议](../../experimental/adaptive-reference-20261113/playraw-sky-toggle-protocol.md) · [素材与作者操作](2026-11-19-playraw-sky-intake.md)

| 文件 | SHA256 |
| --- | --- |
| `DSCF0419.RAF` | `9f0af3b458bfdf04cc8de3daa6d3b03c6d776e100c64983191aeff1bf164e4ca` |
| `P1005231.RW2` | `241b75a13e95fa1f3be53d9eef810658374e8375021ab3f3389bb407c90b2b7a` |
| `A-off.xmp` | `2c40ede1b4e183bd2c534d4b6783804ebf68a6ff7a81fb8be101475cd77d59fb` |
| `A-on.xmp` | `ae87d3b1626aa809356bbe715e2d2420defd9ecd3267b904de80a38e5882abf1` |
| `B-off.xmp` | `36f7e399a7c88a801a42072dca8bbf6e69becb00d60d6e51cc41b0f1b131141f` |
| `B-on.xmp` | `ed9da91b50e550b5faa838735a4dead00f00eec8f4742f261371cd2173a03d90` |

固定 Windows 5.4.0 包为 125,621,422 字节，SHA256 `655ce83046f27d20e1385bffa1e5471b702033083772f65db1fe1a07445c3eb8`，与官方发布一致。直连超时后通过项目已有代理取得完整文件。使用官方 7-Zip Extra 展开 7-Zip 自解压包，再用其中完整的 `7z.exe` 展开 NSIS 包。两个解包文件合计 3,418,044 字节，均匹配官方发布 API 哈希。编辑器只在本机忽略目录展开，没有运行安装程序。命令行版本输出为 5.4.0。[编辑器发布与校验值](https://github.com/darktable-org/darktable/releases/tag/release-5.4.0) · [解包工具发布](https://github.com/ip7z/7zip/releases/tag/26.06)

用临时 ASCII 路径、斜杠输出路径、隔离配置及缓存，在 CPU 上重复定位。未导入任何 XMP 的 B 可以导出并完整解码为 1024 × 769 RGB PNG，解码像素 SHA256 是 `d409032d84fba5c6e47c6479e5c9202bf859ccdcf1aa2dd482fbb4adfa4a9658`。A 在无 XMP 时也以 `0xC0000005` 退出。接回两份**未经修改的作者 XMP**，两张都在列出导出管线后以相同访问异常退出，没有 PNG。B 的日志已列出 `exposure.1`。以匹配文件名自动导入 B 的关闭状态、另试 TIFF／JPEG、关闭元数据或 OpenCL 均未得到可用的开关对照。这里能确认 Windows 版的实际失败及 B 的正控，无法将异常定位到某个 DLL、蒙版或元数据字段。

因此尚无可用于测量的暗天空、前景或色边对照。保留两张原片、逐字节开关和失败日志。这个输入构建步骤可在另一套经验证能读相同 XMP 的官方运行环境复查。先用两张无 XMP 的原片及未经修改的作者 XMP 验证导出，再按原协议出图。此时不把 B 的默认直出当作作者关闭态，不用模拟曝光代替。既有软件、模型和独立验收状态保持原状。

本机目录为 `outputs/adaptive-reference-20261113/playraw-sky-toggle-v1`，正控和作者 XMP 原样对照日志在 `identity-control`。用 `prepare_playraw_sky_toggle.py <上轮 XMP 目录> <输出目录>` 重建开关。用 `render_playraw_sky_toggle.ps1 -Cli <darktable-cli.exe> -InputDirectory <RAW 与开关 XMP 目录> -OutputDirectory <空目录> -State off` 复现此次 CLI 路径。保留 `on` 调用以供可用运行环境运行，根据协议先从关态登记区域。
