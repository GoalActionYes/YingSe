# 训练条件监督：1024配对生成及全量审计完成

仅从cohort-multiref-v1的训练1024行生成条件参数目标，256目标各四种风格。没有读取开发64图的拟合参数，也没有使用新保留图片。原epoch6主干冻结，只为每个训练配对求128维条件；GT用于优化及快照选步。这是训练监督准备，不是正常推理效果。

固定128步、Adam lr0.03、±3边界及RGB/梯度损失，使用已核验的固定输入缓存；每批4例共享梯度裁剪系数。单图快照与参数在0及每16步保留。全部1024例选择第128步，247例所选条件触及边界，因此不能声称全局最优或收敛。未按误差删例或继续加步。

训练GT拟合平均RGB8误差13.26718→3.57134，1024例均改善超过0.1；因为直接使用答案，不能用这个降幅宣传产品画质。主干状态前后相同，通用预测器没有在此生成阶段学习。

任务正常退出0。全量CPU审计与独立审查核验3072个目标/参考/GT角色记录、2048张PNG的哈希和重新计算的GT误差、9216个有限且有界的条件快照、零初始化、所选参数及步数关联。训练1024索引唯一且与开发64索引不交叉，源文件/数据/权重/模型状态关联一致。中间快照误差沿用已保存日志，未逐个重新GPU渲染；所选与基线误差已独立重算。

本机结果在 `outputs/adaptive-reference-20261113/tst100k/condition-labels-train-v1`。[协议](../../experimental/adaptive-reference-20261113/condition-labels-protocol.md)、[生成程序](../../experimental/adaptive-reference-20261113/prepare_condition_labels.py)、[审计程序](../../experimental/adaptive-reference-20261113/audit_condition_labels.py)及[证据摘要](2026-11-15-condition-labels-evidence.json)保留过程。图像与参数缓存不随Git上传。

后续[学习对照](../../experimental/adaptive-reference-20261113/condition-supervision-training-protocol.md)使用相同PETAL输入与条件头，比较像素损失控制和额外0.1条件MSE监督。GT条件仅进入训练损失，不进入开发预测；只有不提供标准答案时的实际输出改善才能支持下一步。学习是否完成及其效果以该次结果为准，不能从本准备文档推断。现有软件保持第10阶段。
