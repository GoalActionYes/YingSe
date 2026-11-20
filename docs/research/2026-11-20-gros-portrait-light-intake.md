# 取得三组同机位暖冷光真人原图

此前缺少许可与配对身份都明确的真人换灯小样。Gros 的固定第 1 版提供 CC BY 4.0 的原始 JPEG。作者记录了固定相机、分开的主体灯与背景灯，以及白色投影幕。暖灯为 3200 K，冷灯为 5500 K；两档在第 4 灯位的测光表读数为 280 和 391，且相机使用自动白平衡。这套资料能检验实际拍摄变化怎样传到肤色与浅色背景；不能单独归因于色温。白幕也单独登记，不当作实拍白墙。[作者采集说明与测光表](https://pmc.ncbi.nlm.nih.gov/articles/PMC6321972/) · [Mendeley 固定版与许可](https://data.mendeley.com/datasets/bpy24trj6s/1)

先读取 232,550 字节目录，找到 1,576 个 JPEG 成员。固定三名参与者，取无帽、无眼镜、正面姿势 `pos_2`、灯位 `angle_4`。每组有五张连续相机编号，依作者的文件顺序说明，首两张对应暖灯和冷灯。最终只取这六张，原图共 29,158,784 字节，含 ZIP 目录及成员头的成功读取量为 29,281,331 字节。直接传输超时后，按相同成员范围分段重试，不更换采样。原图逐件按 ZIP 目录复算 CRC32、长度、SHA256 和完整 RGB 解码。[取样协议](../../experimental/adaptive-reference-20261113/gros-portrait-light-intake-protocol.md) · [作者的命名与顺序说明](https://www.mdpi.com/2306-5729/4/1/26)

| 开发组 | 暖灯快门 | 冷灯快门 | 用途 |
| --- | ---: | ---: | --- |
| 01 | 1/100 秒 | 1/125 秒 | 换灯并伴随快门变化 |
| 02 | 1/125 秒 | 1/125 秒 | 优先检验同快门条件 |
| 03 | 1/125 秒 | 1/160 秒 | 换灯并伴随快门变化 |

六张均为 Canon EOS Rebel T2i、ISO 800、f/8、自动白平衡、5184×3456，EXIF 朝向相同。同快门组仍受光量和相机自动白平衡影响。下一阶段先核对姿态与白幕可用范围，然后登记实际肤色、白幕和数字匹配染色三种响应，用另外两个快门有变化的分组检查退化。三人均进入开发，按参与者绑定，不打开已封存的白墙。此轮只完成取样和技术核验，未进行调色或修改产品代码。[六张身份与技术字段](2026-11-20-gros-portrait-light-intake-evidence.json)

## 复现

脚本从 Mendeley 按范围读取相同 ETag 和长度的 ZIP，独立复查时只从已下载原图读取白名单技术字段。下面从已登记证据恢复六个成员名，传给脚本。网络直连稳定时可省略代理参数。输出路径须未占用。

```powershell
$evidence = Get-Content docs/research/2026-11-20-gros-portrait-light-intake-evidence.json -Raw | ConvertFrom-Json
$selection = [ordered]@{}
foreach ($photo in $evidence.photos.PSObject.Properties) {
    $selection[$photo.Name] = $photo.Value.member
}
$list = 'outputs/adaptive-reference-20261113/gros-portrait-selection-replay.json'
$selection | ConvertTo-Json | Set-Content -Encoding utf8NoBOM $list
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/audit_gros_portrait_light.py outputs/adaptive-reference-20261113/gros-portrait-intake-replay --selection $list --proxy http://127.0.0.1:7897
```
