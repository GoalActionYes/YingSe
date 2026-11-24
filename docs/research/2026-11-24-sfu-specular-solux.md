# SFU 三档 Solux 原片出图与局部配准

此前在 SFU 低反光档案里缺预选的 `plastic-1`。本次从作者另一组公开介质反光档案确认该场景，找到同一容器和 3500、4100、4700 K 三档未加滤片的实拍，三张均附相机白参考和实测谱。三档实际出图后，两组选定配对都在预先固定的地标处存在 1 像素横移。按原定条件停止，暖色方向与幅度未评估。只涉及一组浅色容器，跨来源暖墙证据仍待补齐。[事前协议、测区与停止线](../../experimental/adaptive-reference-20261113/sfu-specular-solux-protocol.md) · [此前的低反光档案](2026-11-24-warm-white-source-intake.md)

## 来源和共同显影

原片压缩包 31,613,681 字节，ETag 为 `"1e262f1-3de6a0fda5b00"`，SHA-256 为 `67d47f4eae5b4708d69e4ed496a9ddd8e572fc26a5d1f1962309a2e77dce2820`。只解出预选三档 TIFF 和对应 `.illum`、`.rgb`。三档均为 637 × 468、无内嵌 ICC 的相机 RGB8。作者已线性化并校正原像素，仪器白参考幅度可能有约 10% 的测量误差。灯距为避免裁切可能改变，同档相机设置、绝对白点和实际布光不能从 TIFF 自行恢复。样本及结果保留本机。[作者档案](https://www2.cs.sfu.ca/~colour/data/colour_constancy_test_images/specular/index.html) · [拍摄与预处理](https://www2.cs.sfu.ca/~colour/publications/CRA-2002/CRA-2002.pdf)

用公开灵敏度和既有 CIE 观察者计算 598 条标定辐亮度的相机 RGB 与 XYZ，同除以 Y 后拟合固定矩阵。三档共用首帧 4100 K 的曝光系数 47.68538956 和同一标准显示转换。实测参考白与传感器积分的同通道最大相对差为 1.52%。相机映射白参考与实测谱直接积分的绝对 xy 距离为 0.0133–0.0155，两对色度变化向量的相对误差分别为 4.09%、13.84%。EGM 指出标定光谱与传感器曲线拟合同源，而且绝对白点及近阈值匹配易受转换误差影响。保留近似显影和白参考读数，后续若使用贴近门槛的结果，需事前定义标定允许误差并取得独立校验。[原标定](https://www2.cs.sfu.ca/~colour/data/camera_calibration/index.html) · [实际矩阵、数值及哈希](2026-11-24-sfu-specular-solux-evidence.json)

## 固定地标结果

只看 4100 K 的锚图登记标签下方未印刷的容器表面、两个保护区及两块标签地标。显影后两对的最佳平移都为横向 1 像素，两个地标读数一致。零位相关及最优增益列在表中。旧门槛要求最优平移为零、零位相关至少 0.60，相关增益不超过 0.02。

| 固定灯档 | 两地标零位相关 | 两地标最优相关增益 | 判定 |
| --- | --- | --- | --- |
| 4100 → 3500 K | −0.1179、0.6467 | 1.0273、0.1090 | 未通过局部配准 |
| 4700 → 4100 K | 0.3905、0.7291 | 0.5312、0.0344 | 未通过局部配准 |

两次独立显影的 15 个文件逐字节一致，包括三张 PNG、三份源 TIFF、三份 XYZ、三份裁切前掩码和三份显影记录。分别从两遍落盘 PNG 重算两个方向与两个地标，共八项，结果一致。另用原 TIFF 的相机绿通道独立算横向差分相关，两个配对的两个地标都在横向 1 像素处相关最高，偏移在显影前即可观测。固定矩阵恢复、秩亏和无有效标定亮度三项小数组检查通过，既有暖色预检五项测试通过。实拍暖色四图没有启动，当次没有更新产品、训练或独立验收。当前建议仍以可核验的同机位浅墙补拍为先，避免在这组带几何变化的容器上反调配准或色彩门槛。[后续恢复条件](2026-11-24-cross-scene-quality-handoff.md)

## 本机复现

从作者发布页获取 `specular_8_bit.tar.gz` 和 `sensors.spect.gz`，后者为 1026 字节。另用 SFU 相机标定的 `spectra.txt.gz` 与现有 CIE 1931 表。脚本逐项核对源身份，下列输出目录应未占用。先生成 4100 K，固定测区后再读其它两档。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$dir = "$base/sfu-specular-solux-source-v1"
$inputs = @("$dir/specular_8_bit.tar.gz", "$dir/sensors.spect.gz", "$base/sfu-solux-source-v1/spectra.txt.gz", "$base/skin-spectra-input/CIE_xyz_1931_2deg.csv")
$script = 'experimental/adaptive-reference-20261113/render_sfu_specular_solux.py'
$python = '.tools/adaptive-env/Scripts/python.exe'
$anchor = "$base/sfu-specular-solux-4100-new"
$env:OPENBLAS_NUM_THREADS = '2'
& $python $script @inputs 4100 $anchor
if ($LASTEXITCODE -ne 0) { throw 'Anchor render failed' }
foreach ($light in @(3500, 4700)) {
    & $python $script @inputs $light "$base/sfu-specular-solux-$light-new" --anchor $anchor
    if ($LASTEXITCODE -ne 0) { throw 'Alternate light render failed' }
}
~~~

把两块地标按原坐标从 PNG 裁入已有的 `probe_inretouch_registration.py` 中 `gradient` 和 `shifts`，取完整地标块的第一个结果。所有公开源、成片及裁切前数组保留本机，Git 仅保存复现脚本、协议和数值。[本机 4100 K 图](../../outputs/adaptive-reference-20261113/sfu-specular-solux-4100-v1/srgb.png) · [本机 3500 K 图](../../outputs/adaptive-reference-20261113/sfu-specular-solux-3500-v1/srgb.png) · [本机 4700 K 图](../../outputs/adaptive-reference-20261113/sfu-specular-solux-4700-v1/srgb.png)
