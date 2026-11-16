# PhotoWCT2：本机运行前置检查

恢复目标后发现此前没有实际运行PhotoWCT2。它与已运行的WCT2、PCA蒸馏模型不是同一检查点；[作者说明](https://github.com/chiutaiyin/PhotoWCT2/tree/834a7701769de1cf0d7b6cee250f583f4cca1e8f)采用分块训练的编码／解码器与高频残差跳接。这提供一次有限筛查的依据，不证明更好的风格理解或画质，不能因为换模型名称重启旧头调参。

本轮仅检查作者relu模型运行前提：四份检查点合计28052932字节；逐文件核对Git树字节数、Git blob SHA1并记录SHA256。三份本机运行所需源码／CSV也逐项记录哈希。作者源码和权重仅保留本机`.tools/PhotoWCT2-audit`，未转载到Git。

使用已有TensorFlow2.20及tf_keras环境，无安装、新训练、受限底模、费用或图片上传。设置legacy Keras后构造原作者结构。首次严格加载拒绝解码器block0尚未创建的变量；先用64×64零特征建立该层，再执行`assert_consumed()`，编码器和解码器均通过。不使用`expect_partial()`掩盖缺权重，也没有修改作者网络或映射参数。

固定种子随机RGB64×64输入通过编码及带跳接重建，输出为1×64×64×3且全部有限。[证据](2026-11-17-photowct2-runtime-evidence.json)记录来源、权重、版本与加载范围。这只是运行前置检查，未执行风格协方差迁移、作者引导后处理、真实画质或速度比较，不能据此接入产品。

下一步先核验作者notebook与引导滤波依赖，再冻结旧开发64对上的固定relu／ZCA对照，不搜索模型分支、滤波参数或强度。若可运行，保留所有输出后才读取GT评分，并沿用原有完整门槛；若前置不成立则记录具体原因，不用替代后处理冒充作者默认结果。既有停止路线不恢复，稳定版和第10阶段保持，目标仍未完成。

本机复查命令：

```powershell
& '.tools/tf-convert-env/Scripts/python.exe' experimental/adaptive-reference-20261113/check_photowct2_runtime.py
```

该命令依赖上述本机固定源码和检查点，克隆本项目不自动包含它们。历史失败与成功范围如上，不把随机图重建当独立画质验收。
