# Gros 人像库新增三组实拍冷暖灯原件

原先只检查了第 1 包。用同一套限量读取工具复核作者公开的第 2、3、4 包后，确认还有 **8 名此前未用的参与者**，均有指定正面机位的五张连续照片。实际每包只取一人的前两张，共得到 **3 组冷暖灯实拍、6 张原件**。三包共 4,200 个 JPEG 文件。没有编辑项目或侧车，也没有同一档灯的有意调色前后图。论文注明这批照片没有预处理。联合监督需要的四格因此仍不齐备。[采集说明](https://pubmed.ncbi.nlm.nih.gov/30627602/) · [作者全文](https://par.nsf.gov/servlets/purl/10093806) · [第 2 包](https://data.mendeley.com/datasets/ycjd7mdsbs/1) · [第 3 包](https://data.mendeley.com/datasets/55wmmr8j3g/1) · [第 4 包](https://data.mendeley.com/datasets/zsv3bgp7t4/1)

| 分包 | 此前未用的参与者 | 选取组的暖／冷快门 | 光圈 | 用途 |
| --- | ---: | --- | --- | --- |
| 第 2 包 | 3 | 1/800 秒、1/800 秒 | f/2.8 | 同快门。脸部与白幕均有可见颜色变化，优先作只换拍摄灯的对照 |
| 第 3 包 | 3 | 1/50 秒、1/60 秒 | f/9 | 实拍变光伴随曝光变化。浅幕有局部亮斑，单独作为混合干扰检查 |
| 第 4 包 | 2 | 1/800 秒、1/800 秒 | f/2.8 | 同快门。目视可见眼睑和下巴姿势变化，留作区域级开发对照，逐像素比较前需配准 |

三组均为相同机型、ISO 800，同组内方向、尺寸、光圈与白平衡模式一致。六张 EXIF 白平衡模式均为手动，字段只反映模式，未证明具体增益恒定。作者测得两档灯的亮度也不同。色温、光量和可能的机内处理应作为实际拍摄干扰一起保留。六张图都作开发素材并按原参与者归组；同组其他机位和近重复不得进入最终独立验收。[EXIF 字段定义](https://www.cipa.jp/e/std/std-sec.html) · [本机冷暖灯对照图](../../outputs/adaptive-reference-20261113/gros-remaining-archives-v1/capture-pairs-local.png)

实际用途是给两图算法增加真实的「参考只换拍摄光」负例。上轮自动候选在人脸与白幕上有方向相反的变化，这批实拍可以检查未来候选是否仅在原三人上成立。做联合监督仍需取得同场景、两档灯均可独立开关的可追溯编辑。三份作者发布均按 CC BY 4.0 标注，原图和对照留在本机。本轮无训练、自动画质候选或产品发布。

## 复现

参照事前协议从 ZIP 服务按范围读取，并绑定发布地址、总长和 ETag。若正常直连，省略示例中的代理。只在新的本机目录重跑。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$script = 'experimental/adaptive-reference-20261113/audit_gros_remaining_archives.py'
& .tools/adaptive-env/Scripts/python.exe $script "$base/gros-portrait-catalog-v2/catalog.json" "$base/gros-remaining-archives-replay" --proxy 'http://127.0.0.1:7897'
~~~

目录范围通过 619,502 字节取得，登记六张原件共 29,479,179 字节。三个成功来源记录合计接受 29,995,911 字节范围数据。第 4 包初次读取冷灯图时遇到 TLS 连接中断，只重试第 4 包，两遍已落盘的暖灯原件 SHA-256 相等。总数只含三份通过核验的成功记录，连接中断前服务已传输的字节未计入。未下载完整 ZIP。

另读六张原件，逐张复算 SHA-256、ZIP CRC32、字节数和 JPEG 解码，检查六张唯一身份及八名新增参与者与旧目录无交集。旧入口另重取第 1 包目录，哈希与原目录完全一致。第 2、3 包原始报告位于本机 `gros-remaining-archives-v1`，第 4 包最终报告位于 `gros-part-four-recovery-20261122`，下列汇总保存三个成功来源的报告和原件哈希。[事前协议](../../experimental/adaptive-reference-20261113/gros-remaining-archives-protocol.md) · [汇总证据](2026-11-22-gros-remaining-archives-evidence.json)
