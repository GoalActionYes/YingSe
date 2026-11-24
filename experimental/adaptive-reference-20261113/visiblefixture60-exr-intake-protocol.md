# VisibleFixture-60 实拍开关灯的原件核验

两图参考调色在浅墙有意加暖和真正换光时仍会误伤肤色。现有合格匹配参考来自一个拍摄来源。本轮只核查 VisibleFixture-60 的发布能否提供真实有灯／无灯、同一相机且可共同显影的一对素材。论文描述实拍灯具开关，公开清单列出 EXR 和原 DNG。公开存储的 `Originals/` 清单为空，要判断线性 EXR 自身或作者文档能否提供同档曝光、相机色彩和处理依据。[作者基准](https://github.com/Vrroom/VisibleFixture-60/tree/59657838536bd9cbedd70df67c43c584ce41c13e) · [论文](https://openaccess.thecvf.com/content/CVPR2026/html/Chaturvedi_TokenLight_Precise_Lighting_Control_in_Images_using_Attribute_Tokens_CVPR_2026_paper.html) · [原任务门槛](../../docs/research/2026-11-24-spatial-condition-design.md)

按 GitHub 固定清单统计有方向的记录、去除反向重复的文件对、原始文件和 EXR 的存储覆盖。若缺 DNG，先用公开存储对象键与大小筛重复和缺件，再按有对应的唯一 EXR 文件名请求首 4096 字节。最多核验公开列出的全部 121 份 EXR 文件头，并校验 HTTP 区间、对象身份和完整属性表。失败样本保留为不可读，不推断其内容。检查相机、快门、光圈、ISO、输入白平衡、色彩原色与线性尺度证据。若文件没有这些属性，寻找作者的逐对记录及共同转换说明。解码为线性浮点数只证明码值表示方式，曝光比例和白平衡的相等性还需来源记录。[OpenEXR 标准字段](https://openexr.com/en/latest/StandardAttributes.html)

前置核验预算 15 分钟。EXR 请求上限 121 × 4096 字节，另外保留一套小型作者索引、存储列表与结果，不下载预测图、整图或权重。只有核实了相同曝光或可追溯的曝光校正、共同白平衡及色彩变换，且可用范围和许可清楚，才按首个有可核查浅墙的场景登记基准区域，再运行冻结暖色预检。若发布只有不带条件记录的显示导出或线性渲染，止于来源核验，暖色是否匹配仍未知。此来源任何已消费场景只用于开发，按原始拍摄归组。无需训练或修改产品。
