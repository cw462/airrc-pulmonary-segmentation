# HiPaS 公开250例：冻结 AirRC M0 外部评价

250例均已完成，动脉和静脉每项指标 n_valid=250。使用原100000步最佳权重，未训练、微调或依据HiPaS选模。无缺失、最终失败、未定义指标或排除病例。

这是公开250例子集的外部评价，不等同于作者完整1073例或官方独立测试198例。无气道GT，不报告气道准确率。

以下为患者级均值 ± 样本标准差（ddof=1）；百分比指标内部保存0–1。

| 指标 | 肺动脉：均值 ± SD | 肺动脉：中位数 | 肺静脉：均值 ± SD | 肺静脉：中位数 |
|---|---:|---:|---:|---:|
| Dice（%） | 81.16 ± 5.74 | 82.25 | 77.95 ± 5.86 | 78.75 |
| clDice（%） | 76.59 ± 8.81 | 77.51 | 78.14 ± 8.29 | 79.45 |
| HD95（mm） | 8.45 ± 3.79 | 7.70 | 9.11 ± 5.88 | 8.40 |
| Precision（%） | 81.79 ± 7.11 | 82.80 | 78.62 ± 10.60 | 80.52 |
| Recall（%） | 81.32 ± 8.44 | 83.95 | 79.32 ± 10.53 | 78.35 |
| IoU（%） | 68.61 ± 6.74 | 69.86 | 64.17 ± 6.48 | 64.95 |

## 动静脉错分与合并血管指标

| 指标（%） | 均值 ± 样本SD | 中位数 | n_valid |
|---|---:|---:|---:|
| av_mcs | 0.49 ± 0.57 | 0.43 | 250 |
| artery_to_vein_rate | 1.66 ± 1.26 | 1.51 | 250 |
| vein_to_artery_rate | 0.37 ± 1.02 | 0.25 | 250 |
| merged_vessel_dice | 80.46 ± 4.81 | 81.21 | 250 |
| merged_vessel_precision | 81.10 ± 8.03 | 82.73 | 250 |
| merged_vessel_recall | 81.06 ± 8.60 | 82.14 | 250 |

MCS = (A→V体素数 + V→A体素数) / (预测A/V总体素数 + GT A/V总体素数)，没有额外系数2。每例独立计算后按患者平均，不声称已与HiPaS作者代码完全一致。

## 数据、空间与异常

- 来源：[Zenodo v2 record14879605](https://zenodo.org/records/14879605)，保存原始CT、两套二值NPZ和metadata.xlsx；此目录仅公开代码、指标与病例编号。
- 实际CT为原始HU，未按归一化示例重复反变换。NPZ布局YXZ，逐例采用官方Resolution的XYZ间距。
- 原患者origin/direction未公开；CT和GT共享明确声明的零origin、单位direction canonical LPS基准网格，不能解释为恢复了患者原始世界坐标。导出预测与GT同网格；持久文件的250对NIfTI头也已核对。HD95使用公开物理间距，单位mm。
- 统一GT：0背景、1肺动脉、2肺静脉。原始两套GT有重叠，按[作者issue10答复](https://github.com/Arturia-Pendragon-Iris/HiPaS_AV_Segmentation/issues/10)归动脉；原始NPZ不改写，逐例重叠计数保存在原始计数CSV。模型可预测3气道，但无气道GT评价。
- 沿用evaluate_airrc.py原始Dice、clDice、HD95、Precision、Recall；AV公式沿用evaluation/av_metrics.py。
- 5例服务器暂存读取异常已恢复：108、148、151、155、199；均使用相同原始文件，未改变病例、标注、空间、参数或已有成功结果。详情见异常清单。没有排除或填0的病例。3次正常读取仍失败会停止，不跳过归档校验。
- 空掩膜口径保持原代码，实际500组结构均非空；所有指标均有250个有效患者。

## 文件

- [逐例结构指标，500行](per_case.csv)
- [完整统计JSON，含mean/sample_std/median/n_valid](summary.json)
- [逐例原始混淆计数，250行](av_confusion_counts.csv)
- [已恢复输入异常清单](evaluation_exceptions.csv)
- [固定协议、代码版本和空间说明](evaluation_protocol.json)
- [固定250例及间距](cohort.json)
- [完成清单](completion_manifest.json)

原始数据及预测保留在用户本机源目录；公开仓库不包含医学影像、模型权重或私有连接配置。

## 骨架与参考分支补充评价

已完成相同250例的动脉、静脉骨架及分支补算。[完整结果与固定协议](skeleton_branch/README.md)。clDSC即本报告clDice。
