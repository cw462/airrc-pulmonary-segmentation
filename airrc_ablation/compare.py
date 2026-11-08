"""Patient-paired AirRC comparisons; validation and test remain separate."""
import argparse
from pathlib import Path
import numpy as np

from .baseline import numeric, read_rows
from .common import CLASSES, read_json, write_json
from .evaluation_utils import csv_output
from .metrics import BASIC, AV

COMPARE_METRICS = BASIC + ['skeleton_recall', 'skeleton_precision',
                         'branch_detection_rate_80', 'branch_detection_rate_90', 'branch_coverage_mean']


def paired_difference(left, right, seed=2026, resamples=10000):
    if set(left) != set(right):
        raise ValueError('Pairing requires exactly the same fixed patient IDs.')
    ids = sorted(left)
    valid = [case for case in ids if numeric(left[case]) is not None and numeric(right[case]) is not None]
    delta = np.array([float(right[case])-float(left[case]) for case in valid])
    if not len(delta):
        return {'mean_difference': None, 'CI95_low': None, 'CI95_high': None, 'n_pairs': 0, 'excluded_cases': ids}
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return {'mean_difference': float(delta.mean()), 'CI95_low': float(lo), 'CI95_high': float(hi),
            'n_pairs': len(delta), 'excluded_cases': [case for case in ids if case not in valid]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args(); root = args.root
    split = read_json(root/'patient_splits.json')
    groups = ['P0_M0', 'P1', 'P2']
    for group in groups:
        summary = read_json(root/group/'summary.json')
        assert summary['state'] == 'done' and summary['patient_counts'] == {'val': 38, 'test': 38}
    tables = {group: read_rows(root/group/'per_case.csv') for group in groups}
    av_tables = {group: read_rows(root/group/'av_metrics_per_case.csv') for group in groups}
    rows = []
    for baseline, variant in [('P0_M0', 'P1'), ('P0_M0', 'P2'), ('P1', 'P2')]:
        for part in ('val', 'test'):
            for name, metrics, all_tables in [*( (n, COMPARE_METRICS, tables) for n in CLASSES.values()), ('A_V', AV, av_tables)]:
                for metric in metrics:
                    values = {}
                    for group in (baseline, variant):
                        selected = [r for r in all_tables[group] if r['split'] == part and (name == 'A_V' or r['class'] == name)]
                        values[group] = {r['case_id']: numeric(r[metric]) for r in selected}
                        assert len(selected) == 38 and set(values[group]) == set(split[part])
                    result = paired_difference(values[baseline], values[variant])
                    rows.append({'split': part, 'class': name, 'metric': metric, 'baseline': baseline, 'variant': variant,
                                 **result, 'unit': 'mm' if metric == 'hd95_mm' else 'fraction; multiply differences by100 for percentage points',
                                 'bootstrap_seed': 2026, 'bootstrap_resamples': 10000})
    csv_output(root/'comparison.csv', rows, ['split', 'class', 'metric', 'baseline', 'variant', 'mean_difference',
                'CI95_low', 'CI95_high', 'n_pairs', 'excluded_cases', 'unit', 'bootstrap_seed', 'bootstrap_resamples'])
    text = ['# AirRC M0/P1/P2 patient-paired comparisons', '',
            'Existing M0 is reused. P1/P2 train independently from scratch with the original M0 recipe.',
            'One training seed; patient bootstrap does not measure training randomness. Coverage BDR does not guarantee continuity.',
            'These three groups do not establish module complementarity or the effect of refinement alone.', '',
            '|Split|Class|Metric|Comparison|Mean difference|95% CI|Pairs|', '|---|---|---|---|---:|---|---:|']
    for row in rows:
        scale = 1 if row['metric'] == 'hd95_mm' else 100
        render = lambda value: 'NaN' if value is None else f'{value*scale:.3f}'
        text.append('|{}|{}|{}|{} − {}|{}|[{}, {}]|{}|'.format(row['split'], row['class'], row['metric'],
            row['variant'], row['baseline'], render(row['mean_difference']), render(row['CI95_low']),
            render(row['CI95_high']), row['n_pairs']))
    text.extend(['', 'HD95 differences are mm; all other differences above are percentage points. Lower HD95 and AV error rates are better.'])
    (root/'comparison.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    write_json(root/'comparison_protocol.json', {'seed': 2026, 'resamples': 10000, 'CI': 'patient paired percentile95%',
               'validation_test_separated': True, 'training_seeds': [2026], 'NaN': 'pairwise excluded and listed; never zero-filled'})


if __name__ == '__main__':
    main()
