"""Import the completed M0 numbers without retraining or re-evaluating its model."""
import argparse
import csv
import math
from pathlib import Path
import shutil

from .common import CLASSES, read_json, write_json
from .evaluation_utils import stats, csv_output
from .metrics import BASIC, SKELETON, AV


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def import_baseline(root, source):
    split = read_json(root/'patient_splits.json')
    expected = {(part, case, name) for part in ('val', 'test')
                for case in split[part] for name in CLASSES.values()}
    basic = read_rows(source/'per_case.csv')
    skeleton = read_rows(source/'skeleton_branch/skeleton_branch_per_case.csv')
    indexed = {(row['split'], row['case_id'], row['class']): row for row in skeleton}
    assert len(indexed) == len(skeleton) == 228 and set(indexed) == expected
    rows = []
    for old in basic:
        key = (old['split'], old['patient'], old['class'])
        item = indexed[key]
        row = {'split': key[0], 'case_id': key[1], 'class': key[2], 'empty': old['empty']}
        row.update({metric: numeric(old.get(metric)) for metric in BASIC})
        row.update({metric: numeric(item.get(metric)) for metric in SKELETON})
        assert abs(row['cldice'] - row['cldice_recomputed']) <= 1e-12
        rows.append(row)
    assert len(rows) == 228 and {(r['split'], r['case_id'], r['class']) for r in rows} == expected
    av_by_case = {}
    for item in read_rows(source/'av_metrics/per_case_with_av_metrics.csv'):
        key = (item['split'], item['case_id'])
        row = {'split': key[0], 'case_id': key[1], **{m: numeric(item[m]) for m in AV}}
        if key in av_by_case:
            assert av_by_case[key] == row, 'Repeated M0 AV rows disagree.'
        av_by_case[key] = row
    assert set(av_by_case) == {(p, c) for p in ('val', 'test') for c in split[p]}
    work = root/'P0_M0'; work.mkdir(exist_ok=True)
    csv_output(work/'per_case.csv', rows, ['split', 'case_id', 'class', *BASIC, *SKELETON, 'empty'])
    csv_output(work/'av_metrics_per_case.csv', list(av_by_case.values()), ['split', 'case_id', *AV])
    shutil.copy2(source/'av_metrics/av_confusion_counts.csv', work/'av_confusion_counts.csv')
    shutil.copy2(source/'skeleton_branch/evaluation_protocol.json', work/'evaluation_protocol.json')
    scores = {part: {name: {metric: stats([r[metric] for r in rows if r['split'] == part and r['class'] == name])
                           for metric in BASIC + SKELETON} for name in CLASSES.values()} for part in ('val', 'test')}
    previous = read_json(source/'summary.json')
    for part in ('val', 'test'):
        for name in CLASSES.values():
            for metric in BASIC:
                assert abs(scores[part][name][metric]['mean']-previous['scores'][part][name][metric]['mean']) < 1e-12
    summary = {'state': 'done', 'group': 'P0_M0', 'experiment_family': 'airrc_ablation',
               'patient_counts': {'val': 38, 'test': 38}, 'scores': scores,
               'av_scores': {part: {metric: stats([r[metric] for r in av_by_case.values() if r['split'] == part])
                                   for metric in AV} for part in ('val', 'test')},
               'source': 'Existing reports/airrc_m0; metrics reused unchanged, no new baseline training or inference',
               'historical_successful_optimizer_updates': None,
               'historical_AMP_skips': 'not recorded; original budget100000 training iterations',
               'original_summary': previous}
    write_json(work/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    args = parser.parse_args()
    import_baseline(args.root, args.source)


if __name__ == '__main__':
    main()
