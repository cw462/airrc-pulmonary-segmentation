"""Publish an explicit report allowlist; never copy images, weights or connection settings."""
import argparse
import json
from pathlib import Path
import shutil


def public_value(value):
    if isinstance(value, dict):
        return {key: public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if isinstance(value, str) and (value.startswith(('/home/', '/dev/shm/', 'H:', 'D:', 'C:'))):
        return value.replace('\\', '/').rsplit('/', 1)[-1]
    return value


def export(source, target):
    target.mkdir(parents=True, exist_ok=True)
    root_names = ['experiment_matrix.json', 'experiment_matrix_smoke_candidate.json', 'parse22_seed2026.json',
                  'comparison.csv', 'comparison.md', 'comparison_metadata.json', 'engineering_smoke_review.json',
                  'data_quality_summary.json', 'status.json', 'server_source_pair_verification.json',
                  'runtime_environment.json', 'source_version.json']
    group_names = ['experiment_config.json', 'split_manifest.json', 'training_history.csv', 'validation_history.csv',
                   'evaluation_config.json', 'per_case.csv', 'summary.json', 'prediction_manifest.json',
                   'validation_per_case.csv', 'branch_evaluation_protocol.json', 'evaluation_exceptions.csv',
                   'branch_details.csv',
                   'initialization.json', 'resolved_training_config.json', 'training_augmentation_snapshot.json',
                   'validation_augmentation_snapshot.json', 'training_complete.json']
    candidates = [(source / name, target / name) for name in root_names]
    candidates += [(source / 'shared' / name, target / name) for name in
                   ['training_resolution_study.json', 'preprocessing_config.json', 'shared_backbone_initialization.json']]
    for group in ['P0', 'P1', 'P2']:
        candidates += [(source / group / name, target / group / name) for name in group_names]
        candidates += [(source / group / 'smoke' / name, target / group / 'engineering_smoke' / name)
                       for name in ['smoke_report.json', 'initialization.json', 'output_shape_probe.json', 'sample_order_prefix.json']]
    for src, dst in candidates:
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == '.json':
            value = public_value(json.loads(src.read_text(encoding='utf-8')))
            dst.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        else:
            shutil.copyfile(src, dst)
    state = json.loads((source / 'status.json').read_text(encoding='utf-8')) if (source / 'status.json').exists() else {}
    complete = all((source / group / 'summary.json').exists() and
                   json.loads((source / group / 'summary.json').read_text(encoding='utf-8')).get('state') == 'done'
                   for group in ['P0', 'P1', 'P2'])
    rows = ['# PARSE22 independent supervised experiments', '',
            'This family is separate from AirRC M0 → PARSE22 cross-dataset evaluation in `reports/parse2022/`.', '',
            'Frozen patient-case split:64 train /16 validation /20 test; seed2026. Binary output:0 background,1 pulmonary artery. All formal groups start from scratch and use100000 successful optimizer updates.', '',
            '| Group | Method |', '|---|---|', '| P0 | U-Mamba Dice + CE baseline |',
            '| P1 | P0 + Skeleton Recall Loss |', '| P2 | P1 + supplied CT-guided residual refiner |', '',
            'State: **' + ('complete' if complete else str(state.get('pipeline', {}).get('state', 'engineering_checks'))) + '**. Pending summaries contain no measured scores.', '',
            '| Group | Saved optimizer updates |', '|---|---:|']
    for group in ['P0', 'P1', 'P2']:
        rows.append('| ' + group + ' | ' + str(state.get('groups', {}).get(group, {}).get('optimizer_steps', 0)) + ' |')
    rows += ['', 'Engineering smoke runs are isolated interface checks, not formal model results or formal initialization.', '',
             'Best checkpoint selection uses complete16-patient validation mean Dice every2000 updates. Test20 evaluation is deferred until all three frozen formal training trajectories finish.', '',
             'Metrics use native GT physical grids and patient macro mean/sample SD(ddof=1)/median/n_valid. clDSC(%) is the same clDice value multiplied by100. BDR80 is the primary coverage-based branch detection metric; it does not prove continuous connection.', '',
             'See [the protocol](../../docs/parse22_supervised_protocol.md) for geometry, train-only0.8mm planning, P2 architecture, prior100-case exposure, data-source correction, and interpretation limits.', '',
             'Only code, configurations and reports are public. Source CTs, label volumes, prediction volumes, weights and private connection configuration are not copied by this report exporter.', '']
    (target / 'README.md').write_text('\n'.join(rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export(args.source, args.output)


if __name__ == '__main__':
    main()
