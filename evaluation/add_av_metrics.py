"""Add patient-level AV scores using only existing physical-space label files."""

import argparse
import csv
import json
import math
from pathlib import Path
import platform

import numpy as np
import SimpleITK as sitk

from .av_metrics import compute_av_metrics
from .label_geometry import align_prediction, geometry, read_geometry, read_label


METRICS = (
    "av_mcs", "artery_to_vein_rate", "vein_to_artery_rate", "merged_vessel_dice",
    "merged_vessel_precision", "merged_vessel_recall",
)
COUNTS = (
    "gt_artery_voxels", "gt_vein_voxels", "pred_artery_voxels", "pred_vein_voxels",
    "a_to_a_voxels", "a_to_v_voxels", "a_to_other_voxels", "v_to_a_voxels",
    "v_to_v_voxels", "v_to_other_voxels", "other_to_a_voxels", "other_to_v_voxels",
    "gt_vessel_voxels", "pred_vessel_voxels", "cross_error_voxels", "vessel_tp_voxels",
    "av_mcs_denominator_voxels",
)
FORMULAS = {
    "av_mcs": "(a_to_v + v_to_a) / (gt_artery + gt_vein + pred_artery + pred_vein)",
    "artery_to_vein_rate": "a_to_v / gt_artery",
    "vein_to_artery_rate": "v_to_a / gt_vein",
    "merged_vessel_dice": "2 * vessel_tp / (gt_vessel + pred_vessel)",
    "merged_vessel_precision": "vessel_tp / pred_vessel",
    "merged_vessel_recall": "vessel_tp / gt_vessel",
}
UNDEFINED_REASONS = {
    "av_mcs": "zero_gt_plus_pred_vessel_voxels",
    "artery_to_vein_rate": "zero_gt_artery_voxels",
    "vein_to_artery_rate": "zero_gt_vein_voxels",
    "merged_vessel_dice": "zero_gt_plus_pred_vessel_voxels",
    "merged_vessel_precision": "zero_pred_vessel_voxels",
    "merged_vessel_recall": "zero_gt_vessel_voxels",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def write_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def rooted_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Cohort paths must be relative to their fixed root.")
    return root / relative


def validate_cohort(cohort):
    if cohort.get("prediction_space") != "physical_label_image":
        raise ValueError("Only physically located saved labels are supported; restore nnU-Net crop/transpose metadata first.")
    if cohort.get("prediction_label_encoding") != "unified":
        raise ValueError("Saved predictions must use the project's unified 0/1/2/3 encoding.")
    cases = cohort["cases"]
    keys = [(case["split"], case["case_id"]) for case in cases]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Expected a nonempty, unique fixed case list.")
    if any(split not in ("val", "test") for split, _ in keys):
        raise ValueError("Only separate validation and test partitions are supported.")
    if len({case_id for _, case_id in keys}) != len(keys):
        raise ValueError("A patient must not occur in both validation and test.")
    for case in cases:
        if not case["case_id"] or any(c in case["case_id"] for c in ("/", "\\", "..")):
            raise ValueError("Invalid case identifier.")
    return cases


def denominator_counts(metrics):
    ga, gv = metrics["gt_artery_voxels"], metrics["gt_vein_voxels"]
    pa, pv = metrics["pred_artery_voxels"], metrics["pred_vein_voxels"]
    return {
        "gt_vessel_voxels": ga + gv, "pred_vessel_voxels": pa + pv,
        "cross_error_voxels": metrics["a_to_v_voxels"] + metrics["v_to_a_voxels"],
        "vessel_tp_voxels": sum(metrics[name] for name in (
            "a_to_a_voxels", "a_to_v_voxels", "v_to_a_voxels", "v_to_v_voxels")),
        "av_mcs_denominator_voxels": ga + gv + pa + pv,
    }


def process_case(case, cohort, gt_root, prediction_root, prediction_pattern):
    gt_path = rooted_path(gt_root, case["gt_path"])
    prediction_path = rooted_path(prediction_root, prediction_pattern.format(**case))
    record = {
        "case_id": case["case_id"], "split": case["split"],
        "gt_path": str(gt_path), "prediction_path": str(prediction_path),
        "status": "pending", "reason": "", "metrics": dict.fromkeys(METRICS, float("nan")),
        "counts": dict.fromkeys(COUNTS), "undefined": {}, "geometry": {},
    }
    dataset = cohort.get("dataset", "").upper().replace("_", "").replace("-", "")
    if dataset in ("PARSE22", "PARSE2022"):
        record.update(status="not_applicable", reason="artery_only_gt_does_not_support_complete_av_evaluation")
        return record
    if case.get("gt_av_complete", cohort.get("gt_av_complete")) is not True:
        record.update(status="not_applicable", reason="complete_artery_and_vein_gt_not_confirmed")
        return record
    missing = []
    if not gt_path.is_file():
        missing.append("missing_gt")
    if not prediction_path.is_file():
        missing.append("missing_prediction")
    if missing:
        record.update(status="missing", reason=";".join(missing))
        if gt_path.is_file():
            try:
                record["geometry"]["gt"] = read_geometry(gt_path)
                record["geometry"]["comparison"] = "not_performed_prediction_missing"
            except Exception as error:
                record["geometry"]["gt_read_error"] = f"{type(error).__name__}: {error}"
        return record
    try:
        encoding = case.get("gt_label_encoding", cohort.get("gt_label_encoding"))
        gt = read_label(gt_path, encoding)
        prediction = read_label(prediction_path, "unified")
        record["geometry"] = {"gt": geometry(gt), "prediction": geometry(prediction)}
        prediction, alignment = align_prediction(prediction, gt)
        record["geometry"] = alignment
        gt_array, pred_array = sitk.GetArrayViewFromImage(gt), sitk.GetArrayViewFromImage(prediction)
        metrics = compute_av_metrics(gt_array, pred_array)
        record["metrics"] = {name: metrics[name] for name in METRICS}
        record["counts"] = {**{name: metrics[name] for name in COUNTS if name in metrics},
                            **denominator_counts(metrics)}
        record["undefined"] = {name: UNDEFINED_REASONS[name] for name in METRICS if math.isnan(metrics[name])}
        record.update(status="computed", reason="")
    except Exception as error:
        record.update(status="error", reason=f"{type(error).__name__}: {error}")
    return record


def summarize(records):
    summary = {}
    for split in ("val", "test"):
        cases = [row for row in records if row["split"] == split]
        scores = {}
        for metric in METRICS:
            values = [row["metrics"][metric] for row in cases if math.isfinite(row["metrics"][metric])]
            invalid = [{"case_id": row["case_id"], "reason": row["undefined"].get(metric, row["reason"])}
                       for row in cases if not math.isfinite(row["metrics"][metric])]
            scores[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "median": float(np.median(values)) if values else None,
                "n_valid": len(values), "n_unavailable": len(invalid), "invalid_cases": invalid,
            }
        summary[split] = {
            "n_expected": len(cases), "n_computed": sum(row["status"] == "computed" for row in cases),
            "n_missing": sum(row["status"] == "missing" for row in cases),
            "n_error": sum(row["status"] == "error" for row in cases),
            "n_not_applicable": sum(row["status"] == "not_applicable" for row in cases),
            "n_gt_geometry_read": sum("gt" in row["geometry"] for row in cases),
            "n_prediction_gt_geometry_checked": sum("prediction" in row["geometry"] and "gt" in row["geometry"] for row in cases),
            "n_resampled": sum(row["geometry"].get("action") == "physical_nearest_neighbor_to_gt" for row in cases),
            "scores": scores,
        }
    return summary


def augment_csv(source, destination, records):
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, original = reader.fieldnames, list(reader)
    if not fields or "split" not in fields or not ({"patient", "case_id"} & set(fields)):
        raise ValueError("per_case.csv must contain split and case_id (or legacy patient).")
    new_fields = [name for name in ("case_id", *METRICS, "av_status", "av_reason") if name not in fields]
    if set(METRICS + ("av_status", "av_reason")) & set(fields):
        raise ValueError("Input already contains AV columns; use the unchanged original per_case.csv.")
    index = {(row["split"], row["case_id"]): row for row in records}
    seen = set()
    output = []
    for row in original:
        case_id = row.get("case_id") or row.get("patient")
        if row.get("case_id") and row.get("patient") and row["case_id"] != row["patient"]:
            raise ValueError("case_id and legacy patient disagree.")
        key = (row["split"], case_id)
        if key not in index:
            raise ValueError(f"Input CSV contains a patient outside the frozen cohort: {key}.")
        seen.add(key)
        record = index[key]
        output.append({**row, "case_id": case_id, **record["metrics"],
                       "av_status": record["status"], "av_reason": record["reason"]})
    if seen != set(index):
        raise ValueError("Original CSV and frozen AV cohort must contain the same patient/split pairs.")
    write_csv(destination, fields + new_fields, output)
    return len(output)


def evaluate(cohort_path, gt_root, prediction_root, source_csv, output, model_id,
             prediction_pattern="{split}/{case_id}.nii.gz"):
    cohort = read_json(cohort_path)
    cases = validate_cohort(cohort)
    gt_root, prediction_root, source_csv, output = map(Path, (gt_root, prediction_root, source_csv, output))
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "per_case_with_av_metrics.csv"
    if destination.resolve() == source_csv.resolve():
        raise ValueError("The original per_case.csv must not be overwritten.")
    if not source_csv.is_file():
        raise FileNotFoundError(f"Missing original per-case results: {source_csv}")
    records = [process_case(case, cohort, gt_root, prediction_root, prediction_pattern) for case in cases]
    n_rows = augment_csv(source_csv, destination, records)
    counts = [{"split": row["split"], "case_id": row["case_id"], "status": row["status"],
               "reason": row["reason"], **row["counts"]} for row in records]
    write_csv(output / "av_confusion_counts.csv", ["split", "case_id", "status", "reason", *COUNTS], counts)
    missing = [{name: row[name] for name in ("split", "case_id", "reason", "gt_path", "prediction_path")}
               for row in records if row["status"] == "missing"]
    write_csv(output / "av_missing_files.csv", ["split", "case_id", "reason", "gt_path", "prediction_path"], missing)
    issues = []
    for row in records:
        if row["status"] != "computed":
            issues.append({"split": row["split"], "case_id": row["case_id"], "metric": "all_av_metrics",
                           "reason": row["reason"], "status": row["status"]})
        issues.extend({"split": row["split"], "case_id": row["case_id"], "metric": metric,
                       "reason": reason, "status": "undefined_denominator"}
                      for metric, reason in row["undefined"].items())
    write_csv(output / "av_metrics_exceptions.csv", ["split", "case_id", "metric", "reason", "status"], issues)
    write_json(output / "av_geometry_checks.json", [
        {"split": row["split"], "case_id": row["case_id"], "status": row["status"],
         "gt_path": row["gt_path"], "prediction_path": row["prediction_path"], **row["geometry"]}
        for row in records])
    incomplete = any(row["status"] in ("missing", "error") for row in records)
    summary = {
        "protocol": "av-metrics-v1", "dataset": cohort["dataset"], "model_id": model_id,
        "state": "incomplete" if incomplete else "complete", "cohort": str(cohort_path),
        "gt_root": str(gt_root), "prediction_root": str(prediction_root),
        "prediction_pattern": prediction_pattern, "original_per_case_csv": str(source_csv),
        "units": "Fractions in [0,1]; multiply by 100 for display in percent",
        "aggregation": "One metric per patient, separate val/test; sample_std uses ddof=1; no voxel pooling",
        "undefined_policy": "CSV nan, JSON null for unavailable statistics; never impute zero",
        "formulas": FORMULAS,
        "mcs_scope": "Project-specified HiPaS set-count expansion, no coefficient 2; author code has not been checked for equivalence",
        "label_encoding": {"0": "background", "1": "pulmonary_artery", "2": "pulmonary_vein", "3": "airway_lumen"},
        "geometry": "Compare shape/spacing/origin/direction and LPS affine; physical nearest-neighbor resampling of saved labels onto the unchanged GT reference grid",
        "split_results": summarize(records), "missing_files": missing,
        "original_csv_rows_preserved": n_rows,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "SimpleITK": sitk.Version_VersionString(), "platform": platform.system()},
    }
    write_json(output / "av_metrics_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prediction-pattern", default="{split}/{case_id}.nii.gz")
    args = parser.parse_args()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    summary = evaluate(args.cohort, args.gt_root, args.prediction_root, args.per_case, args.output,
                       args.model_id, args.prediction_pattern)
    for split, result in summary["split_results"].items():
        print(f"{split}: computed={result['n_computed']}/{result['n_expected']}, "
              f"missing={result['n_missing']}, errors={result['n_error']}, excluded={result['n_not_applicable']}")
    for case in summary["missing_files"]:
        for label, field in (("missing_gt", "gt_path"), ("missing_prediction", "prediction_path")):
            if label in case["reason"]:
                print(f"{case['split']} {case['case_id']} {label}: {case[field]}")
    print(f"Reports: {args.output}")
    raise SystemExit(2 if summary["state"] == "incomplete" else 0)


if __name__ == "__main__":
    main()
