# Resolve 教程提供可控的参考调色练习

Blackmagic Design 的 Resolve 20 调色教材第 7 课把 05–07 号车库镜头放在同组。教程以 07 为主镜头，对照一张外部参考，在输出色彩转换之前加入共享冷调、红色管线修饰和暗角。06 又做了单独的阴影修正，以减少蓝色偏重并让人脸更清楚。它适合用两个状态拆开检验：同一组共享风格是否能迁移，以及肤色调色是否还需要按镜头修正。车库片段是同场景不同镜头，可作为开发对照。跨拍摄组和独立自然度需要另外的素材。[官方教材，第 7 课，313–319 页](https://documents.blackmagicdesign.com/UserManuals/DaVinci-Resolve-20-Colorist-Guide.pdf)

## 已核对的获取条件

- 官方 Part 3 链接的 HEAD 请求最终返回一份提示地址无效或过期的 HTML。按官网下载入口带分段头发送 GET 后，直连和代理均返回 `206`，精确范围为 `2389626073-2389691629/2389691630`。后者才是 ZIP 响应。读取一次尾部 65,557 字节，核对了强 ETag、范围和长度。尾部 SHA256 为 `4c8368e771d1434ba2125e277cc21a25ce6c89a6a8cff5e6923dfe4d9a7a40c3`。[官方练习包入口](https://www.blackmagicdesign.com/dvres/grading-with-resolve20-pt3)
- ZIP 中央目录确认有 `Project 03 - The Long Workday SCD.mov`、`Project 03 - The Long Workday Commercial COMPLETED.drp` 和 `Lesson 07 Timeline SETUP.drt`。两张 PNG 位于参考图目录，未见单列的车库成片帧。车库源 MOV 是压缩大小 2,149,650,946 字节的单个 Deflate 成员。读取中央目录只能定位成员，未验证能按帧高效取出 MOV。
- 本机常见安装路径、命令路径和 Windows 卸载登记均未检出 Resolve。本轮未安装软件，未下载 ZIP 成员，未验证工程能否在免费版中导出所需的无特效帧。练习包用于按教材操作，训练和再分发授权尚未核实。[官方培训页](https://www.blackmagicdesign.com/products/davinciresolve/training)

若以后能取得获准的少量帧，应在固定时间码保持输入与输出转换，先只切换共享创意节点。06 单镜头阴影修正可另做离线对照，07 的车灯特效需单独关闭，避免把光效差异混入色彩误差。当前仅记为有明确编辑步骤的来源线索，优先继续寻找可直接取得前后帧、且含不同拍摄组的资料。

复核时对上述 Part 3 短链接发送 `Range: bytes=-65557`、`Accept-Encoding: identity`，使用官方培训页作为来源页，允许常规重定向。在响应状态、尺寸和类型均匹配后读取尾部，用 Python `zipfile.ZipFile` 列出成员，勿解压。本机响应和清单位于 `outputs/adaptive-reference-20261113/resolve-course-source-v1`。[事前协议](../../experimental/adaptive-reference-20261113/resolve-course-source-protocol.md)
