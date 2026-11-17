# ModFlows默认作者流程开发筛查

## 唯一问题与新依据

既有参考直方图／局部修正和主色彩损失未解决跨场景偏色。作者ModFlows直接从图像预测颜色流函数，再把目标颜色经过目标流与参考逆流迁移，提供不同于旧条件头的信息利用方式。本轮只检验作者默认预训练流程是否值得继续开发，不假定它理解人物、保证肤色或能通过验收。旧路线不重训。

## 固定身份与条件

代码maria-larchenko/modflows固定commit ca5da1a2742fb6a63e03b7a1245662f47d278b24，9个所需文本文件共25025字节，逐文件Git blob及SHA256核验，MIT。模型MariaLarchenko/modflows_color_encoder固定revision f7b7077946ed02d70a410a58b780cdc3b3883635，默认B6_dim8195_iter700000，239763134字节，SHA256 3d2aa5e24502d098cb55e98a0abab919718834fc87405db29c27bd57d1e1958f。只读取weights_only完整state，严格加载，不执行作者CLI中清理目录的代码。

使用已有主环境Torch2.6.0 CUDA；缺失依赖按固定wheel及哈希放入.tools/ModFlows-audit/vendor，约9.6MB，不改现有环境，不安装作者旧Torch。兼容性必须实际运行确认，不能以成功下载称为运行成功。

直接调用固定作者run_inference函数和Encoder B6，hidden1024，8195输出，8步、strength1、crop=False、compress=False，不调强度或积分步数。作者enc_preprocess先把HWC数组reshape为3×528×528，随后EfficientNet再作预处理；本轮忠实保留这一发布行为，不擅自transpose修正。它是否为作者意图不能由本轮确认。作者输出转换为截断uint8，也原样保留。

## 数据、预算与判断

只用已消耗旧开发64对，canonical主环境PIL双线性RGB256，输入从既有prepared manifest核对，保存输入PNG及像素哈希。GT在全部预测完成后才访问用于离线评分。基线使用current13原始epoch6保存PNG，先验证哈希和重算完整评分与既有记录一致；不重跑基线训练。

预算64对加首例一次完全重复，无训练、无新照片、无其他权重版本。两次首例RGB必须一致；检查全部输出尺寸／范围、模型参数不变及所有哈希。记时包含作者每对打开已准备PNG、编码两图和颜色流输出，另列模型加载与输入准备；不声称完整软件／视频速度。

沿用既定完整开发门槛，不改RGB／响应／严重退化规则。默认模型先对current13比较，完整门槛未通过就停止默认配方，不试B0／另一检查点／步数／强度追分。即使开发门槛通过，也只获得继续独立验收的资格，不直接接入产品。全部反例和输出保留本机；Git只上传驱动、协议、源身份及数值摘要。
