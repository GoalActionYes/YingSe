# 联合模型路线：权重访问与本机资源核查

本轮完成公开元数据、固定作者源码与硬件核验，没有下载权重、安装环境、运行模型或上传图片。结论仅是当前标准推理的前置条件未满足，不是画质失败，也不能据此断言量化／逐层卸载绝对无法运行。

## ICTone 的具体前提

Hugging Face 元数据给出的 Diffusers 目录权重如下；没有把根目录原始格式副本重复算入下载量，也没有把仅评测用的指标网络算作推理必需项。

| 项目 | 字节数 | 十进制 GB |
| --- | ---: | ---: |
| FLUX-Fill Transformer 三分片 | 23804920120 | 23.805 |
| 全部标准底模推理权重 | 33911019268 | 33.911 |
| ICTone LoRA | 1854698344 | 1.855 |
| 底模＋LoRA | 35765717612 | 35.766 |

这些是权重文件体积，不是实测显存峰值或完整下载体积；分词器、配置、环境和运行缓存另计。每份权重的文件名、LFS SHA256、字节数与仓库revision都保存在[证据摘要](2026-11-17-joint-model-resource-evidence.json)。依据为[底模元数据](https://huggingface.co/api/models/black-forest-labs/FLUX.1-Fill-dev?blobs=true)及[LoRA元数据](https://huggingface.co/api/models/ToneStyle/ICTone-Fill-LoRA?blobs=true)。

本机GPU为RTX4060 Laptop、8188MiB总显存，系统可见物理内存16858472448字节（约15.70GiB）；本次瞬时可用显存5705MiB、物理内存约2.59GB。瞬时空闲会变化，不当作固定硬件容量。磁盘空间不是本轮发现的主要缺口。

当前[作者推理脚本](https://github.com/dengyuhai/ICTone/blob/00a6b4a8e133057e3db76db03d46e663a3cb48a8/inference.py)与本机此前留档SHA一致，正常路径BF16加载后整条pipeline移入CUDA，另提供`enable_model_cpu_offload()`。源码没有配置量化或逐层卸载。仅从文件体积和加载方式推算，标准完整Transformer不具备本机8GB整组件驻留条件；没有通过故意OOM验证这一推算。

[Diffusers官方内存说明](https://huggingface.co/docs/diffusers/main/en/optimization/memory)区分整组件CPU卸载与更慢的逐层卸载，并指出超大模型仍可能无法以整组件方式放入单GPU。因此不能直接把作者现有CPU卸载开关解释为本机可运行保证。量化、逐层／磁盘卸载属于另一个需要验证的执行方案，不能只改开关便宣称效果和性能保持。

## 权重访问

固定FLUX-Fill revision的`transformer/config.json`匿名请求返回401；固定SD3.5 Medium revision同一路径也返回401。两份公开模型元数据均标为gated，本机Hugging Face登录凭据可用性检查为false。这证明当前已核验环境不能按所测方式访问配置，不证明用户本人没有申请权限。

若决定继续这些底模，需用户在本机完成模型页面的访问流程和登录；不要在聊天里发送Token。[FLUX页面](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)与[SD3.5页面](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)为实际访问入口。该前提来自服务访问结果，不是技能要求或自动审批拒绝。

SAGE公开final checkpoint为4838257536字节，但它还需要SD3.5及其他对应／结构模型；单个checkpoint体积不能当成整条链资源需求。本轮没有重算所有可选外部模型总量，也未验证其低显存运行，不能据此估算速度或硬件最低配置。

## 下一步与用户所需信息

已请求用户确认是否存在可用的其他电脑／学校工作站，未租用、付费或对外发送素材。需先明确资源配置及底模访问，再固定一轮小范围现有开发素材筛查的时间／资源预算；只有结果支持时才扩大评估。拥有更多资源不保证模型通过画质、结构、视频和性能要求。

在当前资源未改变时，不启动上述标准大模型推理，也不重跑已失败轻量方案。匿名人类诊断材料已经准备，真实反馈仍为0；可用反馈提出下一项不同假设，但不能用偏好结果推翻既有数值失败或冒充独立验收。

复现元数据及硬件核查：

```powershell
& '.tools/vision-advisor-env/Scripts/python.exe' experimental/adaptive-reference-20261113/audit_joint_model_resources.py --proxy http://127.0.0.1:7897 --run 新的核查目录名
```

脚本仅请求公开元数据、配置和固定作者源码，每响应最多2MB，不请求权重或图像；新目录拒绝覆盖。复测的时间、上游版本和空闲资源可能不同，旧结果保留。公开证据不含凭据或第三方完整源码。完整目标仍未达成。
