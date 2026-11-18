# 肤色测量与曝光实拍样例的配对核查

单图把一些物体本色或背景变化判成编辑。此轮寻找能验证物体本色保持的实拍标签。MIII 发表了换光前后的相对反射率、着色和镜面分量，但校方低分辨率 ZIP 及论文项目地址均重定向至校方主页，作者上传的备份文件请求返回 HTTP 403。没有取得 MIII 原始像素，原定的换光与反射率检查没有执行。另核查作者参与发布的公开演示，HTML 只引用两段合成演示视频，无法从中核定原始分量。见[事前范围](../../experimental/adaptive-reference-20261113/miii-reflectance-intake-protocol.md) · [MIII 校方发布](https://www.cg.informatik.uni-siegen.de/en/shida) · [作者备份](https://www.researchgate.net/publication/298415036_intrinsic_lowres) · [官方演示](https://light-field-appearance-intrinsic.mpi-inf.mpg.de/results/Appearance_Editing_GT_Intrinsic/index_appearance_edit_gt_intrinsic.html)

替代来源是 DAST 作者公开的曝光样例。它能检查同一张脸的相机 RGB 会怎样随曝光而变，同时有独立仪器记录说明脸部测量如何取得。只取固定提交的 8,043,401 字节示例 ZIP，按 Git blob、SHA-256 与 ZIP CRC 检查。作者说明全量数据需要申请，本轮只使用公开样例，媒体与个体测量留在本机。[固定发布](https://github.com/dasec/DAST-SkinTone-database/tree/839482cd09c487a1a0f9c342b0fd76626e04dc3e)

- 公开样例有 3 人、36 张裁切照片、15 条色度仪记录，其中脸颊和额头共 9 条。该仪器测量与照片在参与者级关联，可作为独立佐证。未取得照片上的精确测点坐标。
- 36 张照片均能与发布表一一关联。24 张文件名精确相同，12 张需去掉档案名末尾的数字后缀，按参与者和拍摄编号核对仍唯一。4 张没有完整相机和曝光信息，保留在原档，排除出曝光比较。
- 5 个同参与者、同相机组同时具有名义曝光与四种非名义曝光，覆盖全部 3 名参与者。样例实为 RGB JPEG，尺寸 796 或 797 × 1024，均无内嵌 ICC 与 EXIF。分组依据是作者随档表。全量说明中的 TIFF 格式不能代替样例头部实测。[聚合计数和档案摘要](2026-11-19-dast-exposure-intake-evidence.json)

这批输入可进入同脸同相机的曝光稳定性开发诊断。它能帮助确定目前把肤色偏移判成风格时，曝光与相机处理是否已经造成相近变化。样例来自受控拍摄且经过裁切，只能检验有充分配准的对应区域。当前没有解码样例像素、运行模型或发布个人记录。它没有跨场景调色意图、整幅白墙或视频证据，不能承担完整画质验收。

用要核验的固定 ZIP 运行 `audit_dast_example.py <样例 ZIP> <新 JSON>`。首次解析对仪器文件的空行处理有误，修正后两遍报告逐字节一致。PowerShell 独立复核全部 36 个关联键、24 项精确文件名关系与 5 组完整曝光关系。[核查脚本](../../experimental/adaptive-reference-20261113/audit_dast_example.py)
