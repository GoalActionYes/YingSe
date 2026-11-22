# 真人换灯的局部编辑重放出现暖灯颜色截断

三组新真人各有冷、暖两张实拍。用本机 darktable 4.0.1 重放此前存档的作者 RGB 色彩平衡参数。每张原件先依其颜色配置转换为 1024 × 683 sRGB8。外部固定羽化区域把编辑限制在人脸周围，同一人两种灯共享同一区域。六张图均逐像素关闭态保持，区域外及登记白幕保持原色。这里检验一个固定操作的开关，作者未替三人逐张编辑。编辑参数、区域和出图由研究端使用，未送进产品推理。[事前协议](../../experimental/adaptive-reference-20261113/gros-local-editor-cross-protocol.md) · [已有作者操作](2026-11-20-playraw-mixed-light-skin.md) · [官方工具发布](https://www.darktable.org/2022/09/darktable-4.0.1-released/)

| 实拍组 | 冷灯新增 RGB 端点像素 | 暖灯新增 RGB 端点像素 | 暖灯双颊色度变化中位数 | 冷暖均合格 |
| --- | ---: | ---: | ---: | --- |
| 第 2 包 | 0.044% | 4.468% | 23.00／28.00 RGB8 | 否 |
| 第 3 包 | 0.007% | 1.942% | 32.00／32.50 RGB8 | 是 |
| 第 4 包 | 0.101% | 4.310% | 23.00／30.50 RGB8 | 否 |

六张的双颊都超过预设 2 RGB8 的有效变化门槛，登记白幕均为 0。冷暖联合的额外端点比例须各不超过整图 2%。第 2、4 包暖光面部明显橙化，蓝通道新到 0 的像素分别为 31,247 和 30,146。两组这些像素在外部区域合成之前的编辑器导出中蓝通道已经是 0。这个现象把问题定位到编辑器生成的 SDR 结果，暂不能区分具体色彩平衡内部与输出编码环节。第 3 包暖灯接近整图上限，脸部也有局部截断点。整图资格不代表局部自然度通过。[本机开关与新增端点图](../../outputs/adaptive-reference-20261113/gros-local-editor-cross-v2/editor-cross-local.png) · [逐图数值与哈希](2026-11-22-gros-local-editor-cross-evidence.json)

按协议至少两组合格才进行跨人物互换，只通过一组，所以停止稳定 1.3 和 13 号的跨人物输出。保留 12 张开关成片作开发诊断，不把已截断的暖灯脸部当作联合监督答案。本轮新增的是关于操作和拍摄光交互的证据。下一种对照优先核验编辑是否有色域余量，以及开关能否保留局部色彩细节，再评价迁移算法。独立画质、人评、视频或候选接入均未运行。原件、作者设置和成片留本机。

## 验证与复现

第一遍在读取第一个编辑结果时，因引用了另一处参数签名不同的色度函数而退出。换回局部编辑脚本已使用的函数，固定阈值和区域后从新目录执行两遍。每遍从第一张输入 TIFF 写完到报告落盘约 19 秒，含十二次编辑器导出及本机计分。12 张最终 PNG 和报告逐字节一致，12 张编辑器中间 TIFF 的 RGB 像素也一致。另从落盘文件核验六份关闭态逐像素一致、六对侧车各只差一个字节，以及端点统计。适配坐标文件在 Windows 下的换行后，最终脚本再次得到相同报告。三份报告 SHA-256 均为 `3dd7043172870f6d79229cee012f39a0346f18b4c4dfc4e261dd6d1c4f83bb76`。

在仓库根目录使用已有研究环境。示例映射空闲 ASCII 盘符供 darktable 使用。已有映射时选另一个空闲盘符。

~~~powershell
if (Test-Path 'R:/') { throw 'R: is occupied' }
subst R: (Get-Location).Path
if ($LASTEXITCODE) { throw 'Cannot map a temporary editor drive' }
try {
    $env:PYTHONPATH = (Get-Location).Path
    $env:OPENBLAS_NUM_THREADS = '2'
    $base = 'R:/outputs/adaptive-reference-20261113'
    $launch = 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); runpy.run_path(sys.argv.pop(1),run_name="__main__")'
    & .tools/adaptive-env/Scripts/python.exe -c $launch experimental/adaptive-reference-20261113/probe_gros_local_editor_cross.py $base "$base/gros-local-editor-cross-new"
} finally {
    subst R: /D
}
~~~
