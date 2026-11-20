# 实拍林地和春季黄叶补回更多风格，整图仍未过线

上一轮十二个房间只覆盖统一显影 Cuckoo 剩余缺口的 1.11%。本轮据关闭态缺口偏黄绿的特点，事前固定三份作者公开的室外原片。共 81,476,544 字节，先核对附件长度和作者许可，再下载到本机研究目录。对目标照样只在出图后读取编辑答案。[事前协议](../../experimental/adaptive-reference-20261113/rawtherapee-foliage-coverage-protocol.md)

| 原片与作者 | 下载字节 | 单张覆盖的剩余像素 | 选择 |
| --- | ---: | ---: | --- |
| _6I_8176.CR3，Sherebyah Tisbi，署名－相同方式共享 | 21,149,270 | 6,309 | 第一，[原帖](https://discuss.pixls.us/t/green-yellowish-cast/43989) |
| E5Dii252_2021_12_04_1546.CR2，Guillermo Rozas，CC BY-SA 4.0 | 33,093,508 | 1,125 | 第二，[原帖](https://discuss.pixls.us/t/yellow-tree-in-spring/28481) |
| IMG_9562.CR2，Brian Innes，署名－相同方式共享 | 27,233,766 | 0 | 未入选，[原帖](https://discuss.pixls.us/t/purply-blue-green-bluebells-in-the-grass-play-raw/30738) |

两张入选原片联合覆盖 6,601／20,478 个剩余像素（32.23%），超过 15% 出图条件。沿用已知操作、统一显影、80% 强度和原亮度。整图色度响应投影由 0.67860 升至 0.74555，仍低于 0.75；响应误差由 0.84937 降至 0.73455 RGB8，改善 0.11482，小于本轮要求的 0.25。原合格四区仍通过，脸颊、浅树皮和草地分数逐值相同。保留实际图和剩余 13,877 像素未覆盖的事实，按协议停止再从同类素材补这张目标。不能把相近阈值当通过。[全部评分与哈希](2026-11-20-rawtherapee-foliage-coverage-evidence.json)

这套条件下，室外材料可以补充室内样本缺少的颜色。它的收益只验证在这张消费过的人像、共用显影以及已知编辑操作下。供体前后成片的使用条件无法直接搬到普通两图，下个阶段应验证能从目标图和参考成片自身得到的可重复信息。独立自然效果、白墙和视频仍未达标。

将原片按表保存本机 `rawtherapee-foliage-input-v1`，使用原 Python 环境和仓库根目录 `PYTHONPATH`，以与室内筛查相同的六个路径参数运行 `probe_rawtherapee_foliage_coverage.py`。两遍结果及所列出图哈希一致。原室内命令重新运行，结果与上一轮逐字节一致。室外结果保存在本机 `rawtherapee-foliage-coverage-v1`，原片和成片不随 Git 上传。
