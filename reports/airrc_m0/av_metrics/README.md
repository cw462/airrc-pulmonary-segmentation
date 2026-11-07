# AirRC M0 动静脉错分补充评价记录

本次状态为 **incomplete：缺少现有预测标签**。没有可报告的真实 AV 指标数值，没有启动训练或推理。

| 数据划分 | 固定病例数 | 可读 GT 几何 | 现有预测数 | 六项指标各自 n_valid | 缺失病例 |
|---|---:|---:|---:|---:|---:|
| 验证 | 38 | 38 | 0 | 0 | 38 |
| 测试 | 38 | 38 | 0 | 0 | 38 |

所有新增指标的 mean、sample_std、median 当前均为 null，不表示错分率为零。逐例比率为 nan，原始计数为空。由于预测缺失，76 例均未进行预测/GT 几何比较，不能声称已完成空间配对检查。

已查阅 prepare_airrc.py 的 LABEL_MAP=[0,3,0,1,2]，确认原始标签 3/4 为肺动脉/肺静脉，统一标签为 1/2；当前固定 GT 已是统一编码。验证、测试各实际读取一例 GT 的标签值为 [0,1,2,3]，并读取了全部 76 例 GT 的 shape、spacing、origin、direction 和 affine。固定队列与旧 178/38/38 划分一致。

已有 evaluate_airrc.score_case 只返回指标，没有调用影像写入导出预测。本机已知 full/evaluation 输出目录和服务器对应两个运行目录均未找到保存的 AirRC 预测影像。下列路径是补充脚本约定的输入位置，不是旧评估曾导出的文件位置。完整缺失病例和路径见 av_missing_files.csv。

~~~text
av_mcs = (A_to_V_voxels + V_to_A_voxels)
         / (GT_artery_voxels + GT_vein_voxels + pred_artery_voxels + pred_vein_voxels)
~~~

该口径没有额外系数 2。尚未核对 HiPaS 作者代码，不声称与其官方实现完全一致。内部保存 0–1，比率展示时乘 100。每例独立计算后按患者分别汇总验证/测试，sample_std 的 ddof=1；零分母不填 0。

已通过用户提供的最小合成测试以及 15 项单元测试，涵盖物理网格还原、原始标签映射、NaN、患者汇总、缺失预测、COPD 标注完整性和 PARSE22 排除。合成测试数值不作为患者实验结果。

原 per_case.csv 保留，复制后的 228 行原有字段和值全部保留。AV 指标是病例级，追加在每例的三行结构级记录中；汇总只使用 76 个独立患者记录。

- [追加列后的逐例表](per_case_with_av_metrics.csv)
- [分验证/测试汇总](av_metrics_summary.json)
- [原始错分计数](av_confusion_counts.csv)
- [全部缺失文件与预期路径](av_missing_files.csv)
- [异常和不可用原因](av_metrics_exceptions.csv)
- [几何检查记录](av_geometry_checks.json)
- [已有预测检索记录](av_prediction_inventory.json)

公开副本使用 <GT_ROOT>、<PREDICTION_ROOT> 等占位符表示运行目录。本机持久副本 av_missing_files.csv 保留完整绝对路径。协议、公式、标签映射和 A1/A2 复用命令见 [方法说明](../../../docs/av_metrics.md)。

已按用户指出的 LIDC-IDRI-0004/label 实际读取原始 GT，确认原始编码为 [0,1,2,3,4]，经已有映射与固定测试 GT 的 shape、affine 和全部标签体素一致。见 [GT 映射核对](av_gt_mapping_check.json)。
