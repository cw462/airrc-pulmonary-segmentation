"""Patient-paired bootstrap comparisons for independently trained P0/P1/P2."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

from .preprocess import read_json, write_json


METRICS = ['dice', 'iou', 'cldice', 'hd95_mm', 'precision', 'recall',
           'skeleton_recall', 'skeleton_precision', 'branch_detection_rate_80', 'branch_detection_rate_90']


def patient_map(rows):
    mapping = {row['case_id']: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError('Duplicate patients cannot enter a paired comparison.')
    return mapping


def paired_bootstrap(baseline_rows, method_rows, metric, seed=2026, resamples=10000):
    baseline, method = patient_map(baseline_rows), patient_map(method_rows)
    if set(baseline) != set(method) or len(baseline) != 20:
        raise ValueError('Paired comparison requires the same20 frozen test case IDs.')
    diffs, excluded = [], []
    for case_id in sorted(baseline):
        left, right = float(baseline[case_id][metric]), float(method[case_id][metric])
        if math.isfinite(left) and math.isfinite(right):
            diffs.append(right - left)
        else:
            excluded.append(case_id)
    values = np.asarray(diffs)
    if not len(values):
        return {'n_pairs': 0, 'mean_difference': None, 'ci95_low': None, 'ci95_high': None, 'undefined_case_ids': excluded}
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(1)
    low, high = np.quantile(estimates, [.025, .975])
    return {'n_pairs': len(values), 'mean_difference': float(values.mean()),
            'ci95_low': float(low), 'ci95_high': float(high), 'undefined_case_ids': excluded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows, summaries = {}, {}
    split = read_json(root / 'parse22_seed2026.json')
    expected = set(split['test'])
    hashes = set()
    for group in ['P0', 'P1', 'P2']:
        summary = read_json(root / group / 'summary.json')
        if summary.get('state') != 'done' or summary.get('experiment_family') != 'parse22_supervised':
            raise ValueError('Only completed independent supervised experiments may enter this comparison.')
        summaries[group] = summary
        hashes.add(summary['checkpoint']['split_sha256'])
        with (root / group / 'per_case.csv').open(encoding='utf-8', newline='') as stream:
            rows[group] = list(csv.DictReader(stream))
        if set(patient_map(rows[group])) != expected or any(row['split'] != 'test' for row in rows[group]):
            raise ValueError('Test cohort differs or validation was mixed into the primary table.')
    if len(hashes) != 1:
        raise ValueError('All formal groups must share the identical split file.')
    results = []
    for baseline, method in [('P0', 'P1'), ('P0', 'P2'), ('P1', 'P2')]:
        for metric in METRICS:
            paired = paired_bootstrap(rows[baseline], rows[method], metric)
            results.append({'comparison': method + '_minus_' + baseline, 'metric': metric,
                            'unit': 'mm' if metric == 'hd95_mm' else 'fraction', **paired,
                            'bootstrap_seed': 2026, 'bootstrap_resamples': 10000,
                            'higher_is_better': metric != 'hd95_mm'})
    with (root / 'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
        fields = ['comparison', 'metric', 'unit', 'n_pairs', 'mean_difference', 'ci95_low', 'ci95_high',
                  'bootstrap_seed', 'bootstrap_resamples', 'higher_is_better', 'undefined_case_ids']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    text = ['# PARSE22独立监督实验比较', '',
            '相同64/16/20患者划分、训练种子2026、100000次实际optimizer更新；所有组从头训练。以下为固定20例测试患者的均值±样本标准差(ddof=1)，不包含AirRC→PARSE22外部结果。', '',
            '| 指标 | P0 | P1 | P2 |', '|---|---:|---:|---:|']
    for metric in METRICS:
        factor = 1 if metric == 'hd95_mm' else 100
        label = metric + ('（mm）' if metric == 'hd95_mm' else '（%）')
        cells = []
        for group in ['P0', 'P1', 'P2']:
            statistic = summaries[group]['scores']['test'][metric]
            if statistic['mean'] is None:
                cells.append('NaN')
            else:
                deviation = 'NaN' if statistic['sample_std'] is None else f"{statistic['sample_std']*factor:.2f}"
                cells.append(f"{statistic['mean']*factor:.2f} ± {deviation} (n={statistic['n_valid']})")
        text.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    text += ['', '## 患者配对差值与95%CI', '',
             '差值为后者减前者；百分比指标显示百分点，HD95显示mm且负值表示下降。固定种子2026进行10000次患者级配对percentile bootstrap。', '',
             '| 比较 | 指标 | 平均差值 | 95%CI | n_pairs |', '|---|---|---:|---:|---:|']
    for row in results:
        factor = 1 if row['metric'] == 'hd95_mm' else 100
        if row['mean_difference'] is None:
            difference, interval = 'NaN', 'NaN'
        else:
            difference = f"{row['mean_difference']*factor:.3f}"
            interval = f"[{row['ci95_low']*factor:.3f}, {row['ci95_high']*factor:.3f}]"
        text.append(f"| {row['comparison']} | {row['metric']} | {difference} | {interval} | {row['n_pairs']} |")
    text += ['', 'Precision与Recall必须同时解释，不能仅凭Recall升高认定细分支方法有效。', '',
             'P1-P0衡量Skeleton Recall Loss增量；P2-P1衡量在该损失基础上的细化增量；P2-P0衡量完整方法。三组不能单独证明模块互补、细化单独有效或CT输入本身贡献。', '',
             '这是一个训练种子的初轮实验；患者bootstrap不能代替训练随机性。确定最终方法后优先为P0和最终方法增加训练种子并报告运行间波动。', '',
             '肺血管VesMamba官方仓库未提供相同病例清单和完整训练实现，论文值仅能作为文献参考，不能据此宣称严格公平超越。', '',
             '这100例此前均已做过AirRC M0外部评价。新test20是本次监督训练留出集，不能称为研究全过程从未查看的盲测集。从划分冻结后没有用test调整算法、阈值、选模或病例。', '']
    (root / 'comparison.md').write_text('\n'.join(text), encoding='utf-8')
    write_json(root / 'comparison_metadata.json', {'state': 'done', 'split_sha256': next(iter(hashes)),
               'training_seeds': [2026], 'bootstrap_seed': 2026, 'bootstrap_resamples': 10000,
               'patient_bootstrap_is_training_seed_analysis': False, 'cross_dataset_results_included': False,
               'comparisons': results})


if __name__ == '__main__':
    main()
