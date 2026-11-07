# 骨架与 GT 参考分支评价协议

本评价只读取已保存的预测和固定 GT，不调用模型、不训练、不推理。固定验证集 38 例、测试集 38 例及标签编码均沿用原 AirRC 实验。现有 Dice、clDice、HD95、Precision、Recall 文件和实现保留。

## 编码与空间

编码直接核对并导入 `evaluate_airrc.LABELS`，GT 为已经经过 `prepare_airrc.LABEL_MAP=[0,3,0,1,2]` 转换的统一标签：背景 0、肺动脉 1、肺静脉 2、气道管腔 3。原始 AirRC 标签 3/4 才是动脉/静脉，不能把原始 GT 数字直接当成统一编码。

读取器复用 `evaluation/label_geometry.py`。检查 shape、spacing、origin、direction 和 affine；只有具有正确物理元数据的预测才能以最近邻转换到不变的 GT 网格。不能使用仅按尺寸的 resize。未知标签、缺失文件及失败会记录为异常，缺失预测不会触发推理。

## 骨架指标与旧 clDice

直接导入 `evaluate_airrc.skeletonize`，固定 scikit-image **0.21.0**、`method='lee'`，结果转换为 bool。CLI 拒绝换用其他骨架库版本。GT 骨架以 GT 自身紧包围框计算，坐标还原为完整 GT 网格的 ZYX 索引，并缓存；预测骨架使用相同算法。该缓存不依赖预测。

~~~text
skeleton_recall    = |S(G) ∩ P| / |S(G)|
skeleton_precision = |S(P) ∩ G| / |S(P)|
clDice             = 2 * recall * precision / (recall + precision)
~~~

骨架指标使用体素计数，不使用物理长度权重，不膨胀掩膜，空间容差为 0。逐例输出旧 clDice、重算 clDice 和绝对差，以 1e-12 为浮点比较阈值。异常差异会调用未修改的 `score_class` 诊断，并记录算法版本、空间处理和空掩膜策略，不覆盖旧分数。

旧 clDice 的特殊策略单独保留：两掩膜都为空时 NaN；仅一方为空时 0；非空掩膜完全相同时 1；其他情况使用原实现的 `max(1, skeleton_count)` 分母及调和平均。新的 skeleton_recall 在 GT 空时为 NaN；预测骨架为空时 skeleton_precision 为 NaN。非空掩膜却产生空 Lee 骨架时也单独记录。

## 26 邻接分支图

图和参考路径只从 GT 骨架构建。全部 26 邻接候选边均保存在缓存中；各连通分量全部处理，不只取最大分量。

1. 原始图中 degree≥3 的相邻体素按 26 连通聚为 junction cluster，并收缩为一个节点。
2. 两个邻居均位于同一簇的局部 degree=2 回返体素并入该簇，避免把拐角附近的小三角形作为独立枝段。
3. 沿其余 degree=2 体素追踪端点／junction 之间的路径，用无向边访问集合保证正反方向只记录一次。真正 junction 簇内部边不作为分支，也不计长度。
4. 一个簇若只有两个外部路径，其收缩后是非分叉的转弯，而不是真正 bifurcation。将两侧路径连接为一条最大路径，内部连接取该 GT 簇的最短物理 26 邻接路径；这部分作为 degree=2 中心线连接保留。此规则固定用于消除 26 邻接在弯曲处制造的假分叉，不依赖预测或测试效果。
5. 没有端点和有效分叉的纯闭环分量单独记录并从主分支指标排除；收缩后的全 degree=2 闭环同样记录。连接在真正 junction 上的回环路径与独立纯闭环分量分别处理。
6. 孤立点、完全收缩后无有效路径的簇及零长度路径作为退化结构记录。

这些是**无根拓扑枝段**，不代表解剖学命名或分支级别。缓存包含全部骨架坐标、原始 26 邻接边、节点／簇成员、每条路径的边、物理长度和排除原因，可用于检查和后续 A1/A2 复用。

## 物理长度、覆盖率与固定阈值

~~~text
world_delta = direction @ diag(spacing_xyz) @ delta_index_xyz
edge_length_mm = norm(world_delta)
~~~

SimpleITK 数组坐标 ZYX 在应用物理矩阵前转换为 XYZ。长度不能用边数或体素数代替。

有效参考枝段最短长度固定为 **3.0 mm**。小于该长度的枝段记录数量和总长度，不进入主指标；这不是 HiPaS 官方参数，也不根据验证或测试分数调整。

每条真实骨架边的物理长度各分一半给两个端点，得到路径体素的长度权重。覆盖率为同类预测内部的加权 GT 中心线长度占比：

~~~text
coverage(b) = sum(weight(v) * 1(P(v)=1)) / sum(weight(v))
BDR80 = count(valid GT branches with coverage >= 0.80) / valid GT branch count
BDR90 = count(valid GT branches with coverage >= 0.90) / valid GT branch count
branch_coverage_mean = arithmetic mean of valid GT branch coverages
~~~

**主指标固定为 BDR80**，BDR90 只作敏感性分析。两项阈值不增加 epsilon 或空间容差。长度和覆盖权重使用稳定求和，并仅将浮点越界约束到 [0,1]。分支间等权，每患者先各自计算，再按 split/class 取患者级 mean、sample_std（ddof=1）、median、n_valid。

覆盖率不保证连通性。长分支中存在短断点时仍可能达到 BDR80，不能将其称为“完整连接率”。不能用预测分支数量除以 GT 分支数量来代替。

## 输出与复用

- `skeleton_branch_per_case.csv` 每患者每结构一行，包含六项比率、五项要求的分支计数、短分支总长度、退化统计及旧 clDice 比较。
- `skeleton_branch_summary.json` 按验证／测试和三类分别进行患者级统计。
- `skeleton_branch_exceptions.csv` 记录缺失、空间／标签错误、未定义指标和 clDice 不一致。
- `branch_details.csv` 每条有效、短、闭环或退化参考路径一行，包含长度、覆盖率、检测标记和排除原因。
- `evaluation_protocol.json` 保存精确源代码版本指纹、库版本、标签、图处理、阈值、空间／空值策略和固定病例列表。
- `missing_predictions.csv` 单独列出缺失输入；没有缺失时仅包含表头。

比率内部为 0–1，显示时乘 100；分支计数保持整数，长度为 mm。无有效 GT 分支时 BDR 和 coverage mean 为 NaN。JSON 不可用统计量写为 null，CSV 写为 nan，不填 0。

首先对两例验证病例运行三类检查并查看分支图和覆盖率图，确认图构建和旧 clDice 对应后再运行 76 例。没有用测试集调整 3 mm、80% 或 90% 阈值。

复用同一 cohort、GT_ROOT、CACHE_ROOT 和脚本版本。按模型设置 PRED_ROOT，并将该模型已有的评分表放在其 results 下；既有 clDice 比较应对该模型自己的旧结果进行，不能把 A1 的 clDice 当成应该等于 M0。

~~~bash
export COHORT=/path/to/repo/configs/airrc_av_cohort.json
export GT_ROOT=/path/to/fixed_prepared_gt
export CACHE_ROOT=/path/to/frozen_gt_branch_cache
export PRED_ROOT=/path/to/model/results/predictions

python -m evaluation.evaluate_skeleton_branches \
  --cohort "$COHORT" --gt-root "$GT_ROOT" --cache-root "$CACHE_ROOT" \
  --prediction-root "$PRED_ROOT" \
  --previous-per-case "$(dirname "$PRED_ROOT")/per_case.csv" \
  --output "$(dirname "$PRED_ROOT")/skeleton_branch" --workers 2
~~~

现有 M0 的实际汇总及预览见 [结果报告](../reports/airrc_m0/skeleton_branch/README.md)。GT 缓存仅包含由本流程生成的可信缓存文件；本机持久副本供后续 A1/A2 使用。
