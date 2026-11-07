# AirRC M0 动静脉错分补充评价

状态为 **complete**。用户明确允许补跑推理后，使用原第 400 epoch、100,000 步最佳权重，为固定 38 例验证和 38 例测试导出完整预测标签，再读取这些标签补算六项指标。

| 指标 | 验证 mean ± sample_std（%） | 验证 median（%） | n_valid | 测试 mean ± sample_std（%） | 测试 median（%） | n_valid |
|---|---:|---:|---:|---:|---:|---:|
| av_mcs | 0.233 ± 0.136 | 0.178 | 38 | 0.215 ± 0.107 | 0.188 | 38 |
| A→V | 0.437 ± 0.293 | 0.371 | 38 | 0.411 ± 0.269 | 0.311 | 38 |
| V→A | 0.513 ± 0.371 | 0.385 | 38 | 0.463 ± 0.272 | 0.378 | 38 |
| 合并血管 Dice | 96.891 ± 0.528 | 97.006 | 38 | 96.901 ± 0.515 | 96.971 | 38 |
| 合并血管 Precision | 97.152 ± 0.995 | 97.416 | 38 | 97.110 ± 0.663 | 97.177 | 38 |
| 合并血管 Recall | 96.642 ± 0.755 | 96.623 | 38 | 96.701 ± 0.949 | 96.832 | 38 |

上表为患者级统计，sample_std 的 ddof=1。所有比率内部保存 0–1，表格乘 100 显示百分数。验证和测试始终分开，不使用全队列体素池化比率作为主要结果。

~~~text
av_mcs = (A_to_V_voxels + V_to_A_voxels)
         / (GT_artery_voxels + GT_vein_voxels + pred_artery_voxels + pred_vein_voxels)
~~~

MCS 没有额外系数 2，分母不替换为 GT 血管量。尚未核对 HiPaS 作者代码，不声称与其官方实现完全一致。错分率需结合合并血管检出指标解释。

GT 沿用现有统一编码，原始 AirRC 的 LABEL_MAP=[0,3,0,1,2] 已从项目代码确认；用户指出的 LIDC-IDRI-0004 原始人工标注经映射后，与固定测试 GT 的 shape、affine 和全部标签体素相同。

两组各 38 例均检查了预测与 GT 的 shape、spacing、origin、direction 和 LPS affine。实际需物理最近邻重采样的病例为 0 例；其余已在 GT 物理网格。

异常、零分母或不可用指标总数为 0。NaN 不填 0；逐例不可用原因保存在异常表。全部预测已经保存为完整 0/1/2/3 标签。

重算的 228 项病例/类别 Dice 与原评分比较，最大绝对差为 0.0。原有 per_case.csv 以及 Dice、clDice、HD95、Precision、Recall 实现和分数均保留；228 行原字段和值完整保存在追加列后的副本中。AV 指标是病例级，在旧长表的三行结构记录中重复，统计仅使用 76 个独立患者。

指定最小合成测试及 15 项单元测试已通过；实际导出另外逐例检查空间及 Dice 一致性。A1/A2 复用同一 AV 脚本、固定 GT、空间流程和 cohort 文件。

- [逐例补充表](per_case_with_av_metrics.csv)
- [验证/测试统计汇总](av_metrics_summary.json)
- [原始错分计数](av_confusion_counts.csv)
- [缺失文件清单](av_missing_files.csv)
- [异常和不可用原因](av_metrics_exceptions.csv)
- [空间检查记录](av_geometry_checks.json)
- [预测导出记录和 Dice 比较](av_prediction_export_summary.json)
- [原始 GT 映射核对](av_gt_mapping_check.json)

公开副本使用运行目录占位符。完整绝对路径保留在本机持久副本。[公式、范围与复用命令](../../../docs/av_metrics.md)
