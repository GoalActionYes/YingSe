# 暖色两折 GPU 预检

两个已登记拍摄来源分别用小规模拟合检查暖色监督能否进入 13 号的参考条件头。四臂各跑两步，两遍均通过工程检查。这一步确认计算与数据通路可用，正式留出成片尚未生成。[两折素材及身份](2026-11-24-warmth-source-holdout.md) · [正式试跑规则](../../experimental/adaptive-reference-20261113/positional-warmth-trial-protocol.md)

- 复核 13 号当前权重的 SHA-256 为 `d8d8e6727a44100e6fedb11b605a17e33afc675de7448d4ebcf6d24d492ad4c9`。旧拟合记录中的 TST100K 分组清单哈希与本机清单一致。这个结果只核实本轮继承哪个主干以及它已知的旧来源。两折留出只约束本轮参考头拟合。
- 两臂读取各自拟合家族的相同场景与镜像序列。确认可训练参数全部在 65,664 参数的参考头里。初始训练批次的网格与关闭该参考头逐值一致，区域及暖色标签在缩小到 128 × 128 后仍有实际信号。
- 两遍池化臂都对参考块倒序保持逐值一致。位置臂的最大网格变化在 0.0000944–0.000103。两遍浮点数末位有微小差异，重复验证的是位置通路存在、池化不变和两臂损失有限。记录不用于判定画质。
- 两遍脚本内部总耗时约 9.09 秒和 6.11 秒，最大已分配显存为 79,980,032 字节，约 76.3 MiB。这个范围包括本短测的源核验、装载、两步更新和检查，不含完整成片导出。显卡为 RTX 4060 Laptop GPU。保留 4 小时的训练、出片及评分上限。两遍预检状态均未保存为候选权重。[实测回执](2026-11-24-positional-warmth-preflight-evidence.json)

EGM 复核要求将两个臂解释为带坐标的局部块和整体均值的整体比较。两折各自的四态顺序、实际评分门槛和外部人像检查已经写进正式协议。正式运行从固定 13 号权重和新初始化参考头重启。除比较墙面暖色和换灯误传，还检查旧的两个有意暖肤病例是否进一步变弱。预检没有读取正式封存图和新留出成片。

## 复现

在仓库根目录，沿用现有自适应研究环境和本机素材，输出使用新目录。

~~~powershell
$env:PYTHONPATH = '.;.tools/tf-convert-env/Lib/site-packages'
$env:OPENBLAS_NUM_THREADS = '2'
$base = 'outputs/adaptive-reference-20261113'
& '.tools/adaptive-env/Scripts/python.exe' 'experimental/adaptive-reference-20261113/run_positional_warmth_trial.py' $base "$base/positional-warmth-preflight-replay"
~~~
