"""Apply the frozen AirRC M0 to the publicly labeled PARSE2022 cases."""

import argparse
import csv
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from evaluate_airrc import METRICS, load_model, predict_case, read_json, score_class, sha256, write_json


PROTOCOL = "airrc-m0-parse2022-external-v1"


def run_case(predictor, case, output):
    started = time.monotonic()
    prediction, props = predict_case(predictor, Path(case["image_path"]))
    inference_seconds = time.monotonic() - started
    reader = predictor.plans_manager.image_reader_writer_class()
    reference, reference_props = reader.read_seg(case["label_path"])
    if prediction.shape != reference[0].shape:
        raise ValueError(f"Different CT and label shapes for {case['case_id']}.")
    for name in ("spacing", "origin", "direction"):
        if not np.allclose(props["sitk_stuff"][name], reference_props["sitk_stuff"][name], atol=1e-4, rtol=0):
            raise ValueError(f"Different CT and label {name} for {case['case_id']}.")
    reference = reference[0]
    if np.any((reference != 0) & (reference != 1)):
        raise ValueError(f"Nonbinary pulmonary artery labels for {case['case_id']}.")
    reference = reference == 1
    artery = prediction == 1
    spacing = np.asarray(reference_props["spacing"], dtype=float)
    metrics_started = time.monotonic()
    metrics = score_class(reference, artery, spacing)
    metrics_seconds = time.monotonic() - metrics_started
    target = output / "predictions" / f"{case['case_id']}.nii.gz"
    partial = target.with_name(target.name.replace(".nii.gz", ".part.nii.gz"))
    reader.write_seg(artery.astype(np.uint8), str(partial), props)
    partial.replace(target)
    return {
        "case_id": case["case_id"], "class": "pulmonary_artery", **metrics,
        "reference_voxels": int(reference.sum()), "prediction_voxels": int(artery.sum()),
        "inference_seconds": inference_seconds, "metrics_seconds": metrics_seconds,
    }


def evaluate_partition(gpu, model, cases, output):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    predictor = load_model(model, selected=400)
    for index, case in enumerate(cases, start=1):
        saved = output / "cases" / f"{case['case_id']}.json"
        prediction = output / "predictions" / f"{case['case_id']}.nii.gz"
        if saved.exists() and prediction.exists():
            row = read_json(saved)
        else:
            row = run_case(predictor, case, output)
            write_json(saved, row)
        print(f"PARSE GPU={gpu} {index}/{len(cases)} {case['case_id']} Dice={row['dice']} "
              f"inference={row['inference_seconds']:.1f}s metrics={row['metrics_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    source = read_json(args.manifest)
    cases = source["cases"]
    identifiers = [case["case_id"] for case in cases]
    if len(cases) != 100 or len(set(identifiers)) != len(cases):
        raise ValueError("Expected the 100 unique, publicly labeled PARSE2022 cases.")
    checkpoint = args.model / "fold_0" / "checkpoint_best.pth"
    digest = source["checkpoint_sha256"]
    args.output.mkdir(parents=True, exist_ok=True)
    for folder in ("cases", "predictions"):
        (args.output / folder).mkdir(exist_ok=True)
    config = {
        "protocol": PROTOCOL, "dataset": "PARSE2022", "subset": "public_labeled_training_subset",
        "evaluation_role": "external_test", "structure": "pulmonary_artery", "patients": len(cases),
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "checkpoint_step": 100000,
        "manifest_sha256": sha256(args.manifest),
        "inference": "Frozen AirRC normalization and 1 mm preprocessing, 128-cube windows, 50% overlap, Gaussian blending, no TTA",
        "prediction": "Pulmonary artery is class 1 of the original three-class argmax",
        "evaluation_grid": "Original PARSE2022 CT and label grid",
        "hd95": "Maximum of directed 95th percentiles, 6-connected boundary voxel centers, physical mm",
        "cldice": "3D Lee skeletonization and harmonic mean of centerline precision and sensitivity",
        "aggregation": "Per-patient metrics, arithmetic mean and sample standard deviation",
        "empty_policy": "Both empty excluded, one empty overlap and clDice zero with full grid diagonal for HD95",
    }
    config_path = args.output / "evaluation_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("An existing evaluation used a different model, manifest or protocol.")
    write_json(config_path, config)
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Each worker must use a separate GPU.")
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=evaluate_partition,
                        args=(gpu, args.model, cases[index::len(args.gpus)], args.output))
        for index, gpu in enumerate(args.gpus)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError("An external evaluation worker failed.")
    rows = [read_json(args.output / "cases" / f"{case['case_id']}.json") for case in cases]

    summary = {
        **config, "patients_completed": len(rows),
        "both_empty": sum(row["empty"] == "both" for row in rows),
        "one_empty": sum(row["empty"] in ("reference", "prediction") for row in rows),
        "scores": {},
    }
    for metric in METRICS:
        values = [row[metric] for row in rows if row[metric] is not None]
        summary["scores"][metric] = {
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    csv_path = args.output / "per_case.csv"
    partial = csv_path.with_suffix(".csv.part")
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(csv_path)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
