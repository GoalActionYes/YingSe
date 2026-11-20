# 三个布景通过白／绿灯双相机像素核验

固定 Scene3_a、Scene9_b、Scene10_a，尼康和手机各自用 241 档白灯与绿灯照片。按上轮清单校验原图字节数和 SHA256，复用三张手机白灯，实际新增九张、139,294,691 字节。官方验证及测试区没有读取像素。按场景把两台相机照片绑定成同一家族，所有材料只用于本项目开发。[取件清单](2026-11-20-chromamill-camera-sources.md) · [事前协议](../../experimental/adaptive-reference-20261113/chromamill-cross-camera-pixel-protocol.md) · [作者数据与采集说明](https://color.cvc.uab.cat/chromamill/)

## 像素资格

发布 PNG 解码后均为 8 位三通道。尼康为 6036 × 4020，手机为 1557 × 1037。按项目既有方式将线性 RGB 缩到长边 512，再应用统一 sRGB 曲线。无逐张曝光或白平衡。用两台相机各自的实拍白／绿图全图线性通道均值比，给对应白灯图加一个匹配的数字染色。

| 场景 | 白／绿机位零位相关，尼康／手机 | 实拍整体色度变化，尼康／手机 | 数字染色裁切，尼康／手机 |
| --- | ---: | ---: | ---: |
| Scene3_a | 0.995／0.958 | 1.295／1.413 | 0.078%／1.096% |
| Scene9_b | 0.999／0.969 | 1.435／1.706 | 0.049%／3.502% |
| Scene10_a | 0.998／0.987 | 1.408／1.748 | 0.206%／3.194% |

机位比较均在同相机内部进行，六组最佳整数位移都是零。数字参考与实拍的线性 RGB 最大均值距离为 0.00031，均低于 0.02。色度变化是全图通道对数比的差，不能换算为人眼感知距离。三个布景的两台相机均通过，无需更换任一场景。这组资料可分别比较同布景换相机及同相机换布景。两台相机的取景仍有细小差异，不按跨相机像素一一对应使用。8 位发布格式不足以证明传感器噪声或高位深特性。[逐组来源、门槛与成片哈希](2026-11-20-chromamill-camera-pairs-evidence.json)

## 接续

可将这三个场景用于开发新的跨相机对照。检验时需按布景成组留出，两台相机的同布景份额不能分到拟合和验证两边。官方验证家族仍未使用。现有全部条件为绿色照明，不替代多个偏色方向、人像肤色或真实白墙。此次只验证实拍和匹配编辑能否作为监督，未运行调色或训练模型。

两次各 18 张准备图、3 张拼图与数值报告逐字节一致。已有本机 OpenCV 和自适应环境可复算。从仓库根目录运行，报告路径须未占用。图像留本机。

~~~powershell
$base = 'outputs/adaptive-reference-20261113'
$env:PYTHONPATH = (Get-Location).Path
$env:OPENBLAS_NUM_THREADS = '4'
& .tools/adaptive-env/Scripts/python.exe experimental/adaptive-reference-20261113/probe_chromamill_camera_pairs.py "$base/chromamill-camera-sources-v1/first.json" "$base/chromamill-correction-training-assets-v1" "$base/chromamill-camera-pair-input-v1" "$base/chromamill-camera-pair-replay" --opencv-site .tools/tf-convert-env/Lib/site-packages
~~~
