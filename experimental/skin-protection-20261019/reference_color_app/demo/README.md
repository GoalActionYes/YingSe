# 受控输入示例

从项目 `outputs/jianying-color/test-kit` 复制的独立合成输入，供课程演示与界面检查；运行时仅依赖本目录固定文件。无个人照片、原生调色输出或旧版预设处理结果。应用仍通过正式 Store / pipeline 分析这些输入。

原始作者脚本为 `work/jianying-color/make_fixtures.py`。图片包含色块、灰阶和近似肤色色块，不含真实人物或语义真值。暖色参考为输入侧通道增益/红偏移合成，视频为受控运动输入。

| 文件 | 来源 | SHA256 |
| --- | --- | --- |
| source.png | test-kit/source.png | ea207d3efd2effff9a9bb2b2fd06fcaa6d64b707bc5ec85adcdc637c743fc6bb |
| reference.png | test-kit/reference-warm.png | d4059fdfa51be5d5b5a971827f7a3dc419bec1101b3192edf9409491102ab367 |
| source-motion.mp4 | test-kit/source-motion.mp4 | ed5095191a8c5f142e59f8fc5807ba3a35c8dd0eff05b45c57f65b7bf03ca759 |

示例不证明与剪映整体效果相同。界面中的“使用示例”与“视频示例”只选择输入，不选择算法或滤镜。
