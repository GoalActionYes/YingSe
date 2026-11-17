# 既有训练照片与专家设置的身份关联

问题：现有扩展训练的1024个目标原文件是否保留可与PPR原始／专家A XMP精确关联的标识，从而避免下载整包照片或按编号猜配对？本轮只查已使用训练content原文件，不读GT、开发或独立图像，不训练。

绑定cohort-training-coverage-v2及其sRGB manifest，按全部1024个不同content路径排序，依据manifest原SHA从既有输入缓存定位原件；不下载。PNG／JPEG／TIFF都纳入，不因是否有元数据择样。每个原件全文件SHA须一致，Pillow仅取文件头／XMP／EXIF信息，不主动解码图像像素。PNG额外扫描所有块以避免遗漏IDAT之后的XMP文本，仅解压受限元数据文本，不解压像素数据。

仅读取白名单标识：Adobe xmpMM DocumentID、OriginalDocumentID、InstanceID，CameraRaw RawFileDigest与EXIF ImageUniqueID。值只在内存规范化并SHA256比较，不输出或保存原标识；不读取姓名、GPS、日期、设备序列号、版权或文件名字段。DocumentID／OriginalDocumentID可在同一document类别匹配；InstanceID及两种digest各自匹配，不跨类别凑数。排除空值、非十六进制UUID／摘要形式与全零标识。

PPR原始／专家A两侧建哈希到数字照片ID的集合。多个键命中时，取非空候选集合的交集；交集为空单列冲突，不能忽略冲突凑单例。报告0候选、1候选、多候选及冲突；多候选不能自动选择。唯一候选仍只是文档身份关联，不是已验证图像版本、渲染状态或像素配准，更不能直接把相机RAW设置赋给普通sRGB输入。

对唯一候选，另描述原图XMP六个主要数值字段是否齐全并与PPR source／expert A精确相等，缺失不当0。这只核对元数据状态，不把附带元数据相同等同于实际像素应用相同。

验收：全部1024原件身份与结构检查完成，全部已取得PPR XML安全解析，无未处理错误；成功是得出可信的关联覆盖率，而非强求存在匹配。匹配失败不放宽UUID规则或改用编号／拍摄时间／设备信息猜测。预算CPU180秒、内存300MB，无新增下载。保存逐训练路径的文件SHA、是否含标识、匹配数量和公开数字候选ID，不保存原标识或原XML。关联结果不改变数据划分或正式验收要求；仍须另查PPR组与已消费开发家族交集。
