# HiPaS250例：肺动脉、肺静脉骨架与GT参考分支评价

复用现有M0预测及固定统一GT，不训练、不重新推理。250例、500组病例/结构；独立标记为external，没有并入AirRC验证/测试。无气道GT，不评价气道。

骨架、分支图及旧clDice数值代码指纹与前次AirRC一致。scikit-image0.21.0 Lee骨架，26邻接、junction簇、全部连通分量、物理边长半权重，最短有效GT分支3mm。主指标BDR80，BDR90只作敏感性分析，空间容差0。

最终失败0，缺失0；异常记录0条，详见异常CSV。重算clDice（clDSC）与原表最大绝对差为0.0。没有覆盖原结果。

16项合成及输入范围测试已在本机和原服务器运行环境通过。全量前已检查001/002两例的动脉、静脉GT分支及覆盖图。

## 肺动脉

| 指标 | 患者级均值±样本SD | 中位数 | n_valid |
|---|---:|---:|---:|
| skeleton_recall（%） | 95.71 ± 6.15 | 96.62 | 250 |
| skeleton_precision（%） | 64.65 ± 11.34 | 64.99 | 250 |
| cldice_recomputed（%） | 76.59 ± 8.81 | 77.51 | 250 |
| branch_detection_rate_80（%） | 94.26 ± 6.70 | 95.41 | 250 |
| branch_detection_rate_90（%） | 92.62 ± 7.07 | 94.10 | 250 |
| branch_coverage_mean（%） | 95.98 ± 6.12 | 97.02 | 250 |
| gt_branch_count | 559.22 ± 166.38 | 549.00 | 250 |
| detected_branch_count_80 | 525.70 ± 157.07 | 517.00 | 250 |
| detected_branch_count_90 | 516.96 ± 156.21 | 509.00 | 250 |
| excluded_short_branch_count | 42.59 ± 26.90 | 37.00 | 250 |
| excluded_cycle_component_count | 0.00 ± 0.00 | 0.00 | 250 |

## 肺静脉

| 指标 | 患者级均值±样本SD | 中位数 | n_valid |
|---|---:|---:|---:|
| skeleton_recall（%） | 95.73 ± 6.21 | 97.00 | 250 |
| skeleton_precision（%） | 66.89 ± 11.27 | 67.69 | 250 |
| cldice_recomputed（%） | 78.14 ± 8.29 | 79.45 | 250 |
| branch_detection_rate_80（%） | 92.49 ± 7.96 | 94.58 | 250 |
| branch_detection_rate_90（%） | 90.78 ± 8.46 | 92.97 | 250 |
| branch_coverage_mean（%） | 94.67 ± 6.77 | 96.26 | 250 |
| gt_branch_count | 650.53 ± 220.13 | 616.00 | 250 |
| detected_branch_count_80 | 593.34 ± 175.52 | 577.50 | 250 |
| detected_branch_count_90 | 581.39 ± 169.22 | 571.50 | 250 |
| excluded_short_branch_count | 55.82 ± 34.06 | 49.50 | 250 |
| excluded_cycle_component_count | 0.01 ± 0.11 | 0.00 | 250 |

sample_std采用ddof=1，患者先独立计算再宏平均；不合并全体病例的分支作为主要比率。NaN不填0。内部比率0–1，表格显示百分数，长度单位mm。

分支是无根拓扑参考路径，不是解剖命名分支。覆盖型BDR不保证连续，不能称作完整连接率。3mm、80%和90%为本项目预设阈值，不是HiPaS官方参数。

延续原外部评价GT编码：0背景、1动脉、2静脉；原始NPZ重叠按作者规则归动脉。原origin/direction未公开，继续使用共享canonical LPS评测网格与官方逐例Resolution；预测经过既有物理对齐检查，不按数组尺寸resize。

全部GT分支明细共328249行。500份预测无关GT图缓存已持久保存在本机，供后续模型使用同一GT/病例/协议复用。

- [逐例指标](skeleton_branch_per_case.csv)
- [完整统计JSON](skeleton_branch_summary.json)
- [分支明细](branch_details.csv)
- [异常清单](skeleton_branch_exceptions.csv)
- [固定协议](evaluation_protocol.json)
- [固定病例](cohort.json)
- [缺失清单](missing_predictions.csv)
- [001动脉预览](previews/external_hipas_001_pulmonary_artery.png)
- [001静脉预览](previews/external_hipas_001_pulmonary_vein.png)

[方法与复用命令](../../../docs/hipas_skeleton_branch_metrics.md)。所有旧Dice/clDice/HD95/Precision/Recall和AV结果保留。
