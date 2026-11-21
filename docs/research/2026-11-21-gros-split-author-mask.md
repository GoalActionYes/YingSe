# 作者脸部分割复核了掩膜影响，暖灯重建仍超线

SPLiT 上轮能在三位真人中跟随换灯方向，但第一位人脸重建超线。原近似椭圆纳入额前头发和部分胡须，因此本轮只替换为作者发布的 DeepLab 人脸掩膜，保留原裁切、两级分解权重及门槛。为区分选区变化与重建改善，同时在两种掩膜的共同像素上评分，并单独测量旧掩膜被移除的区域。自动分割读取裁切图。裁切仍使用已登记的眼位，不能当成完整自动产品入口。[事前协议](../../experimental/adaptive-reference-20261113/gros-split-author-mask-protocol.md) · [原分解与门槛](2026-11-21-gros-split-light-readout.md) · [作者原预处理](https://github.com/costrice/split/blob/b2b559831bb65e54b820fd1d6dd7c0b3a99e8b65/1_preprocess.py)

| 人物 01 | 原椭圆全范围 → 作者掩膜全范围 | 交集内旧 → 新 | 旧图被剔除区 |
| --- | ---: | ---: | ---: |
| 冷灯，关 Film B | 5.18 → 4.62 | 4.15 → 4.14 | 8.06 |
| 冷灯，开 Film B | 6.34 → 5.27 | 4.68 → 4.59 | 10.55 |
| 暖灯，关 Film B | 5.75 → 5.56 | 5.29 → 5.22 | 7.25 |
| 暖灯，开 Film B | 6.15 → 5.61 | 5.28 → 5.17 | 8.82 |

单位为 RGB8 MAE。12 张照片的共同范围均覆盖旧掩膜至少 71.71%。第一位的交集误差仅降低约 0.01–0.11，暖灯两张仍高于 5。全范围变小主要来自原椭圆中的高误差区域被移除。对第二、第三位，作者掩膜下的四张重建仍都通过 5 的门槛。三人的两类照明分量依然通过方向、长度和相对编辑影响条件，完整资格仍各为 2／3。[12 张的分区误差和照明判分](2026-11-21-gros-split-author-mask-evidence.json)

更准确的脸区能减小原掩膜的测量偏差，第一位在共同范围内的重建误差仍然存在。按事前约定结束当前掩膜适配的直接修光，不调整分割阈值，不将其接入参考调色。这里验证了三位已用真人和固定眼位裁切，尚未辨明暖灯重建的剩余误差来源。

## 复现

复用[SPLiT 两阶段权重及已有环境](2026-11-21-gros-split-light-readout.md)。从同一公开 ZIP 仅补取 `model_files/deeplab_model.pth`，区间 42–215133484，按 HTTP 206 的 Content-Range 验证。解压后 232,279,081 字节，CRC32 为 `c3926ec0`。作者源码使用固定提交的 `models/deeplab.py`，按原脚本加载 332 个张量。权重和源码与原两级权重放在同一个本机 `split-intake-v1`，外部文件和人像均未上传。

~~~powershell
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '2'
& .tools/vision-advisor-env/Scripts/python.exe -W 'error::RuntimeWarning' -c 'import runpy,sys; sys.path.insert(0,"experimental/adaptive-reference-20261113"); sys.path.append(".tools/tf-convert-env/Lib/site-packages"); sys.path.append(".tools/DeepWB-audit/vendor"); sys.path.append("outputs/adaptive-reference-20261113/sparks-aces-ocio-2.5.2/wheel"); runpy.run_path(sys.argv.pop(1),run_name="__main__")' experimental/adaptive-reference-20261113/probe_gros_split_author_mask.py outputs/adaptive-reference-20261113/gros-split-author-mask-repro
~~~

两次运行均输出 12 张本机对照板和 1 份报告，文件哈希逐件一致。旧掩膜的重建及线性 RGB 中位数精确复现上一轮。独立用共同与剔除两区误差及像素比例复核 12 组分解，另重算三人各分量的资格判定。首次正式结果位于本机 `outputs/adaptive-reference-20261113/gros-split-author-mask-v1`。
