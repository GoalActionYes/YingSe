# 三项显式局部记录：结构与非零参数资格核查

问题：479项暂余候选中仅有的三项局部结构，是空记录／关闭记录，还是含启用修正、非零参数与非空笔刷定义？固定全部244_4、1369_1、1645_1，不按画面效果或误差重选。只是决定是否值得取对应专家像素，不把字段本身当成实际渲染效果或软件能力。

绑定上轮local-record-availability、v3组级排除和缓存专家A XMP的SHA。只读Camera Raw命名空间内CorrectionActive、CorrectionAmount、Local开头数值调整、CorrectionMasks笔刷属性及Dabs非空条目数；不导出原XML、文档标识、姓名、设备、日期或笔刷坐标。完整三项均保留，包括缺失、不合法及关闭状态。

分开记录颜色／明暗参数（Exposure、Brightness、Saturation、Temperature、Tint、Contrast、Highlights、Shadows、Whites、Blacks、Toning）与细节参数（Sharpness、Clarity、Texture、Dehaze、Noise、Moire、Defringe）。这只是按字段用途的诊断分类；非零不等于已证明作者渲染器产生对应有效像素变化。未知字段或无法确认的启用语义记未知，不能自动算有效。

取得像素的必要条件：至少一项CorrectionActive布尔值为真且CorrectionAmount数值>0，至少一项非零颜色／明暗字段，且具有MaskValue>0、Flow>0、Radius>0、非空Dabs的笔刷。布尔值严格识别XML Schema的true／false／1／0，其他记未知。若三项均无此条件，不下载像素、不因局部字段存在而重启训练。若有，仅对固定符合项规划小范围原图／专家A／人物蒙版验证，仍需另定预算与比较；不把推断笔刷蒙版接入产品。

解析修正记录：初版误按大写True识别，实际三记录为小写true，因此初版只能记未知。其脚本、协议与结果完整保留在ppr-local-effect-structure-v1；v2修正字面值解析，不改变启用、非零与笔刷资格条件，也不将修正当成第二次空间实验。

本轮CPU60秒，无网络、GPU、图像像素读取或训练。参数值仅在内存用于布尔判定，公开只保留字段名、启用／非空计数及结论。后续若做空间拟合，沿用既有固定全局／人物／位置／打散控制，不改参数；作为同一空间监督假设的第二轮，仍无可重复信号则停止当前支路，不通过换阶数或模型名称继续。
