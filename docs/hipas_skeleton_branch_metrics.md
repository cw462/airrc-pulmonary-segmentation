# HiPaS250例骨架与参考分支补充评价

只读取已导出的 AirRC M0 预测和固定统一GT，评价肺动脉、肺静脉；不训练、不调用推理、不改写原始指标。病例来自原HiPaS外部评价固定的公开001–250号，统一记为 `external`，不划入AirRC验证集或测试集。HiPaS没有气道GT，不计算气道骨架或分支准确率。

## 沿用的数值口径

完整算法定义见[前次AirRC协议](skeleton_branch_metrics.md)。骨架提取、分支图和旧clDice函数的源代码指纹必须与前次已交付AirRC协议一致，入口在计算前强制核对。只增加HiPaS固定病例、平铺文件路径和两类评分表的适配；不更换数值内核。

- 固定scikit-image0.21.0、`evaluate_airrc.skeletonize(method='lee')`，骨架指标按体素计数。
- Skeleton Recall = `|S(G)∩P|/|S(G)|`，Skeleton Precision = `|S(P)∩G|/|S(P)|`。
- clDice（clDSC，中心线Dice）为两者调和平均，保留原空掩膜策略；每例核对旧clDice，输出绝对差，超过1e-12时记录和诊断，不覆盖旧结果。
- 参考分支只从GT构建，26邻接、相邻分叉簇收缩、非分叉拐角合并、全部连通分量、无向边不重复计数；独立纯闭环、短枝及退化结构单独记录。
- 按XYZ物理间距和direction计算边长，最短有效GT分支3.0mm；真实junction内部长度不计。每条边长度的一半分给两端点。
- coverage为同类预测内的加权GT分支长度占比；主指标BDR80固定覆盖阈值0.80，BDR90的0.90仅作敏感性分析，不按外部结果选阈值。
- 不膨胀掩膜，空间容差0mm。患者内部各有效分支等权，然后按患者汇总mean、sample_std(ddof=1)、median、n_valid；不汇总全体患者分支作为主比率。
- 覆盖型BDR不保证连续，不称为“完整连接率”。3mm和80/90%均为本项目固定设置，不声称是HiPaS官方参数。

## 标签与空间

复用原外部评价已经保存的统一GT：0背景、1肺动脉、2肺静脉。原始两套二值NPZ的重叠按作者issue10答复归动脉，原始NPZ不改写。预测可以包含3气道，但不对气道评分。GT出现3或未知标签时报告错误。

继续使用既有读取和物理最近邻对齐流程，检查shape、spacing、origin、direction/affine，不按数组尺寸resize。公开NPZ缺少原患者origin/direction，因此现有CT、GT和预测使用相同的canonical LPS基准网格及官方逐例Resolution；这里沿用该网格，不能解释为恢复了患者原始世界坐标。物理分支长度仍使用官方间距，单位mm。

## 输出和执行

先对前两例的两类结构出分支图及覆盖图并检查，再执行250例。原有500行`per_case.csv`完整保留；新增结果每患者每结构一行，共预期500行。

- `skeleton_branch_per_case.csv`：11项新增指标、计数、额外排除统计和旧clDice逐例比较。
- `skeleton_branch_summary.json`：external下动脉、静脉的患者级统计。
- `skeleton_branch_exceptions.csv`：空间、编码、失败、未定义指标及clDice不一致。
- `branch_details.csv`：每条有效或被排除的GT参考分支、长度、coverage、detected80/90及排除原因。
- `evaluation_protocol.json`：代码及库版本、固定阈值、空值规则和全部固定病例。
- `missing_predictions.csv`：缺失输入路径，缺失不会触发训练或推理。

空GT、空预测、无有效分支的规则与前次相同；NaN不填0。GT图缓存只由固定GT生成，按病例和类别保留，后续模型复用同一GT、病例、缓存和协议，只替换预测及该模型自己的旧clDice评分表。

```bash
python -m evaluation.evaluate_hipas_skeleton_branches \
  --manifest /path/to/hipas/manifest.json \
  --airrc-protocol reports/airrc_m0/skeleton_branch/evaluation_protocol.json \
  --gt-root /path/to/hipas/labels \
  --prediction-root /path/to/hipas/predictions \
  --previous-per-case /path/to/hipas/results/per_case.csv \
  --output /path/to/hipas/skeleton_branch \
  --cache-root /path/to/fixed_hipas_gt_branch_cache --workers 4
```
