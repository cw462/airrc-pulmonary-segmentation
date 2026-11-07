# 动静脉错分补充评价

本模块只读取已保存的预测标签和固定 GT。训练、推理及原有 Dice、clDice、HD95、Precision、Recall 实现均不在本模块中调用或修改。

## 固定队列与标签

AirRC 队列固定在 `configs/airrc_av_cohort.json`，来自已有 `data/splits/airrc_seed2026.json` 的 38 例验证和 38 例测试。该文件为后续 M0、A1、A2 共用的 GT 相对路径和病例列表。

已经读取 `prepare_airrc.py` 确认原始映射 `LABEL_MAP = [0, 3, 0, 1, 2]`，并核对 `evaluate_airrc.py` 和模型 `dataset.json` 的统一标签。

| 原始 AirRC 编码 | 含义 | 统一编码 |
|---:|---|---:|
| 0 | 背景 | 0 |
| 1 | 气道管腔 | 3 |
| 2 | 气道壁 | 0 |
| 3 | 肺动脉 | 1 |
| 4 | 肺静脉 | 2 |

当前固定 GT 是已经完成上述映射的 `labelsVa/` 和 `labelsTs/`，使用 `gt_label_encoding=unified`。若显式选择原始 AirRC GT，读取器使用 `airrc_original` 并直接复用 `prepare_airrc.LABEL_MAP`。不会凭原始标签数字猜测结构。

## 空间处理

`evaluation/label_geometry.py` 检查三维整数标签图的 shape、spacing、origin、direction 和由后三者组成的 LPS affine。SimpleITK 按影像物理坐标读取 NIfTI，不将其数组轴顺序当成世界坐标。

物理网格一致时直接比较；不一致时沿用 `prepare_airrc.prepare_case` 的 SimpleITK 物理参考网格流程，以 GT 影像为 reference、恒等物理变换和最近邻插值还原保存的预测标签，外部填充值为背景 0。GT 不做重采样。还原前后几何记录在 `av_geometry_checks.json`。不进行仅按数组大小的 resize。

输入必须是有正确物理元数据的标签影像。只有网络数组、没有已恢复的 nnU-Net crop/transpose/spacing 元数据时，不能声称它已经位于物理坐标系；`prediction_space=network_array` 会被拒绝。必须先使用原项目的 `convert_predicted_logits_to_segmentation_with_correct_shape` 及对应 properties 恢复和导出已有数据；本补充脚本不会调用模型生成预测。无物理交集、非整数标签或未知编码作为异常记录。

## 指标口径

令 G_A、G_V 为 GT 动脉、静脉体素集合，P_A、P_V 为预测集合。

~~~text
av_mcs = (|G_A ∩ P_V| + |G_V ∩ P_A|)
         / (|G_A| + |G_V| + |P_A| + |P_V|)

artery_to_vein_rate = |G_A ∩ P_V| / |G_A|
vein_to_artery_rate = |G_V ∩ P_A| / |G_V|

G = G_A ∪ G_V; P = P_A ∪ P_V; TP = |G ∩ P|
merged_vessel_dice = 2 * TP / (|G| + |P|)
merged_vessel_precision = TP / |P|
merged_vessel_recall = TP / |G|
~~~

MCS 按本项目指定的 HiPaS 公式集合计数展开口径实现，没有额外系数 2，分母也不替换为 GT 血管量。尚未核对 HiPaS 作者实现，不能声称与其官方代码完全一致。

完全互换 A/V 时，MCS 为 0.5、A→V 和 V→A 为 1，合并血管 Dice 为 1。全部漏检时 MCS 为 0，血管召回和 Dice 也为 0。因此错分率要与合并血管检出指标一起解释。MCS、A→V 和 V→A 越低越好。

内部与结果 CSV/JSON 保存 0–1 比率，展示时乘 100。每名患者只计算一次，验证和测试各自取 mean、sample_std（ddof=1）、median、n_valid，不合并全队列体素作为主要比率。零分母返回 NaN；CSV 写入 `nan`，JSON 中不可用的统计量为 `null`，并保存病例和原因。只有一个有效病例时 sample_std 也为 `null`。

## 使用

从仓库根目录运行，预测目录约定为 `PREDICTIONS/val/CASE_ID.nii.gz` 和 `PREDICTIONS/test/CASE_ID.nii.gz`。路径不同可显式指定 `--prediction-pattern`，不自动寻找或生成替代模型的预测。

~~~bash
python -m evaluation.av_metrics
python -m unittest discover -s tests -p test_av_evaluation.py -v

python -m evaluation.add_av_metrics \
  --cohort configs/airrc_av_cohort.json \
  --gt-root /data/Dataset926_AirRC \
  --prediction-root /results/M0/predictions \
  --per-case /results/M0/per_case.csv \
  --output /results/M0/av_metrics \
  --model-id M0
~~~

后续 A1/A2 复用同一个 cohort 文件、GT 根目录、空间处理和脚本版本，只替换对应模型的 `--prediction-root`、`--per-case`、`--output` 和 `--model-id`。不要重新抽样、修改 GT、根据结果更换病例或混入训练集。

程序输出：

- `per_case_with_av_metrics.csv` 保留原 CSV 全部字段和值，增加 case_id、六项 AV 指标和状态。原有 AirRC CSV 为每病例三行的结构级长表，因此病例级 AV 指标在这三行重复；汇总不使用重复行。
- `av_metrics_summary.json` 按验证/测试分别保存患者统计量、有效数量和不可用病例及原因。
- `av_confusion_counts.csv` 每病例一行，保留完整原始计数和推导分母；缺失预测时计数为空，不能当成零。
- `av_missing_files.csv` 列出所有缺失 GT/预测及完整预期路径。
- `av_metrics_exceptions.csv` 单独记录异常、零分母、缺失和不适用原因。
- `av_geometry_checks.json` 记录可读 GT/预测的几何及还原方式。

有缺失或异常时仍输出报告，并以退出码 2 标记不完整。原始 `per_case.csv` 保留，不被覆盖。报告中的缺失预测路径是本脚本的约定输入位置，不表示旧评估曾经创建过这些文件。

## 单独导出 M0 预测

旧 AirRC 评估只保存了分数，缺失标签图后必须先报告。本次用户随后明确允许使用同一个最佳权重补跑固定 76 例推理，并保存完整四类标签。该动作由独立命令执行，`add_av_metrics` 没有任何自动推理入口。

~~~bash
# 仅在明确允许补跑推理后执行；复用原 checkpoint、GT 和 38+38 划分。
CUDA_VISIBLE_DEVICES=0,1 bash scripts/export_airrc_predictions.sh "$RUN_ROOT" 0 1
~~~

导出复用 `evaluate_airrc.load_model`、`predict_case` 及既有 nnU-Net 空间还原。预测为背景 0、动脉 1、静脉 2、气道腔 3。逐例记录与原始评分表的 Dice 绝对差，并验证导出文件与 GT 的实际物理网格；不覆盖原分数表。完成后再运行上述 AV 补算命令。

## 适用范围

AirRC 使用完整动静脉 GT。COPD 仅对病例或队列元数据显式声明 `gt_av_complete=true` 的统一编码 GT 计算；没有该声明时排除，未知编码报错。

PARSE22/PARSE2022 只有动脉标注，不能评价完整 MCS、V→A 或合并 A/V 血管真值。本脚本将其排除于这一组六项完整 AV 评价之外，原有动脉 Dice、clDice、HD95、Precision、Recall 继续保留。当前保存的 PARSE2022 预测本身也是二值肺动脉图，无法由它恢复 A→V 计数。不会把未标注静脉当成真实背景计算动静脉分离。

## 当前 M0 补算状态

见 [M0 实际运行报告](../reports/airrc_m0/av_metrics/README.md)。该报告包含已完成的 38+38 例实际 AV 结果、患者级统计、原始计数和预测导出记录。初始缺失状态保留在 Git 历史及本机归档中。
