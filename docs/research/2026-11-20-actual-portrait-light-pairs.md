# 真人换灯对照的原片与使用范围核查

现有两张实拍人像能检验同一项有意调色跨场景是否仍有效。原片的拍摄光照各不相同，也没有同一人物在白光和有色光下的实拍配对。要区分灯光与后期，候选数据应能固定相机、主体姿态和原始场景，在只切换已知光源时提供对应的原图。白色标靶或同一浅色材料有助于检查局部偏色。原图身份与使用范围也需要能从发布者资料核验。[已有双人像编辑对照](2026-11-20-rawtherapee-portrait-cross.md)

| 来源 | 可提供的对照 | 原片与使用范围的核查结果 |
| --- | --- | --- |
| Meyer 等的换灯色采集 | 5 人、9 种灯光组合；同一正面聚光灯可加红、绿或蓝滤片，白色漫反射目标随拍。相机设置固定，关闭自动白平衡及其他后处理。 | 论文图 2 与采集说明明确。作者论文和公开检索未找到逐人原图、配对清单或小样使用条款。优先保留用于固定机位的换灯色诊断。[论文 §4](https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2013/Meyer13-CCI.pdf) |
| Oulu Physics-Based Face Database | 采集包含四种灯光和不同的相机白平衡标定。用同一标定的一组，可以检验实际换灯。另有皮肤反射率、灯光光谱和相机通道响应。 | 原论文记载光盘分发。原实验室页面和旧下载地址本轮未能打开，学校学位论文仓提供采集说明。未找到当前可直接取用且附许可的小样。[学校存档](https://oulurepo.oulu.fi/handle/10024/37592?show=full) · [原论文](https://citeseerx.ist.psu.edu/document?doi=d38efaa0b297955fb9b2b456177512f3a9ba1eb6&repid=rep1&type=pdf) |
| Katsimerou 等的 CI 数据 | 16 人、1,680 张 RGB 照片，记录七种室内灯光设置。原表同时记录颜色与强弱差异。 | 可用于复杂室内光照。若要单独归因颜色，仍需配对曝光和强度。所查出版记录与论文未给出原片发布及许可。[学校出版记录](https://research.tue.nl/en/publications/face-detection-in-intelligent-ambiences-with-colored-illumination/) · [采集论文](https://citeseerx.ist.psu.edu/document?doi=70bf1769d2d5737fc82de72c24adbb7882d2effd&repid=rep1&type=pdf) |
| FAID 闪光与环境光 | 论文约 12% 为人物类，分离两类照明，可作光源变化的辅助对照。 | 作者已停止托管，历史数据入口返回 404。人物相邻机位和重复拍摄需按同人处理。[作者项目](https://yaksoy.github.io/flashambient/) · [论文 §3](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yagiz_Aksoy_A_Dataset_of_ECCV_2018_paper.pdf) |
| DeepFlash 与 FNF Matting | 前者的同人同姿态采集用手机闪光与影棚灯，后者含闪光／无闪光视频和前景标注。 | DeepFlash 项目入口本轮超时，未核实小样权限。FNF 作者的仓库说明需要邮件申请数据。两套也同时改变光场和方向，可作闪光负例或区域配准检查。[DeepFlash 原论文](https://iris.unibas.it/retrieve/3b8937f3-d904-4138-9da7-f28c07a34861/DeepFlash%20Turning%20a%20flash%20selfie%20into%20a%20studio%20portrait.pdf) · [FNF 作者访问说明](https://github.com/xty435768/DeepFlashMatting#dataset-access) |

这次确认了两套采集设计直接对应缺口，尚未取得任何可用原片。已下载的 UMINHO-HSFD 是实测皮肤反射率，可继续作为按光谱生成的局部对照。其光照替换通过积分计算，与这次寻找的真实换灯重复拍摄对应不同证据。[已有光谱人像对照](2026-11-19-skin-spectral-portraits.md)

若合法可用的小样出现，先按相机、人物、姿态及白平衡分组，固定同一组内的无滤片与滤片条件，检查面部局部配准和标靶裁切。再从无滤片原图另做一份同方向有意染色，记录操作参数与强度。光照状态、标靶和编辑记录只给离线评分。对候选同时测试实际换灯参考的额外偏色，以及有意染色参考的变化幅度。开发使用过的来源仍按已消费素材管理。本轮仅核查公开资料。
