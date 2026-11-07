"""Explicitly export frozen-checkpoint predictions after inference is authorized.

This command is separate from add_av_metrics, which never starts inference.
Existing evaluation CSVs and GT images are read-only inputs.
"""

import argparse
import csv
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk

from .add_av_metrics import read_json, validate_cohort, write_json
from .label_geometry import geometry, same_grid


def export_partition(gpu, model, epoch, cases, output, records, previous):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    from evaluate_airrc import load_model, predict_case

    predictor = None
    for index, case in enumerate(cases, 1):
        case_id, split = case["case_id"], case["split"]
        destination = output / split / f"{case_id}.nii.gz"
        record_path = records / f"{split}_{case_id}.json"
        if destination.is_file() and record_path.is_file():
            print(f"EXPORT GPU={gpu} {index}/{len(cases)} {split} {case_id} already saved", flush=True)
            continue
        if predictor is None:
            predictor = load_model(model, epoch)
        started = time.monotonic()
        prediction, props = predict_case(predictor, Path(case["image_path"]))
        reader = predictor.plans_manager.image_reader_writer_class()
        reference, reference_props = reader.read_seg(case["gt_path"])
        if prediction.shape != reference[0].shape or reference_props["sitk_stuff"] != props["sitk_stuff"]:
            raise ValueError(f"Restored prediction is not on the fixed GT grid for {split}/{case_id}.")
        deltas = {}
        for label, name in ((1, "pulmonary_artery"), (2, "pulmonary_vein"), (3, "airway_lumen")):
            ga, pa = reference[0] == label, prediction == label
            denominator = int(ga.sum()) + int(pa.sum())
            dice = 2 * int(np.count_nonzero(ga & pa)) / denominator if denominator else None
            old = previous[(split, case_id, name)]
            deltas[name] = abs(dice - old) if dice is not None and old is not None else None
        finite = [value for value in deltas.values() if value is not None]
        if finite and max(finite) > 1e-3:
            raise ValueError(f"Exported Dice differs from the preserved evaluation for {case_id}: {deltas}")
        temporary = destination.with_name(destination.name.replace(".nii.gz", ".part.nii.gz"))
        reader.write_seg(prediction.astype(np.uint8), str(temporary), props)
        # Verify the actual exported header, including direction and physical origin.
        saved = sitk.ReadImage(str(temporary))
        gt_image = sitk.ReadImage(case["gt_path"])
        if not same_grid(geometry(saved), geometry(gt_image)):
            raise ValueError(f"Exported image header differs from fixed GT for {case_id}.")
        temporary.replace(destination)
        seconds = time.monotonic() - started
        write_json(record_path, {
            "split": split, "case_id": case_id, "prediction_path": str(destination),
            "gt_path": case["gt_path"], "seconds": seconds, "gpu": str(gpu),
            "dice_absolute_delta_from_preserved_results": deltas,
        })
        print(f"EXPORT GPU={gpu} {index}/{len(cases)} {split} {case_id} {seconds:.1f}s "
              f"max_Dice_delta={max(finite) if finite else None}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--source-val", type=Path, required=True)
    parser.add_argument("--source-test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--previous-per-case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    args = parser.parse_args()
    cases = validate_cohort(read_json(args.cohort))
    split = read_json(args.split_file)
    if any(sorted(row["case_id"] for row in cases if row["split"] == part) != sorted(split[part])
           for part in ("val", "test")):
        raise ValueError("Export cohort differs from the original fixed validation/test split.")
    original = read_json(args.evaluation_summary)
    epoch = original["checkpoint_epoch"]
    model = args.model.resolve()
    if model / "fold_0/checkpoint_best.pth" != Path(original["checkpoint"]).resolve():
        raise ValueError("Export must use the original selected checkpoint path.")
    previous = {}
    with args.previous_per_case.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            previous[(row["split"], row.get("case_id") or row["patient"], row["class"])] = (
                float(row["dice"]) if row["dice"] else None)
    tasks = []
    for case in cases:
        base, suffix = (args.source_val, "Va") if case["split"] == "val" else (args.source_test, "Ts")
        image = base / f"images{suffix}" / f"{case['case_id']}_0000.nii.gz"
        gt = base / f"labels{suffix}" / f"{case['case_id']}.nii.gz"
        if not image.is_file() or not gt.is_file():
            raise FileNotFoundError(f"Missing fixed input: {image}, {gt}")
        tasks.append({"split": case["split"], "case_id": case["case_id"],
                      "image_path": str(image), "gt_path": str(gt)})
    output, records = args.output.resolve(), args.records.resolve()
    records.mkdir(parents=True, exist_ok=True)
    for part in ("val", "test"):
        (output / part).mkdir(parents=True, exist_ok=True)
    config = {"checkpoint": str(model / "fold_0/checkpoint_best.pth"),
              "checkpoint_epoch": epoch, "checkpoint_step": original["checkpoint_step"],
              "inference": original["inference"], "cases": tasks}
    config_path = records / "export_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing export used a different checkpoint, case list or input configuration.")
    write_json(config_path, config)
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Each inference worker needs a separate GPU.")
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=export_partition, args=(
        gpu, model, epoch, tasks[index::len(args.gpus)], output, records, previous))
        for index, gpu in enumerate(args.gpus)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError("An authorized prediction export worker failed.")
    rows = [read_json(records / f"{row['split']}_{row['case_id']}.json") for row in tasks]
    deltas = [value for row in rows for value in row["dice_absolute_delta_from_preserved_results"].values()
              if value is not None]
    write_json(records.parent / "export_summary.json", {
        "state": "done", "checkpoint_epoch": epoch, "checkpoint_step": original["checkpoint_step"],
        "patient_counts": {part: sum(row["split"] == part for row in rows) for part in ("val", "test")},
        "maximum_dice_absolute_delta_from_preserved_results": max(deltas) if deltas else None,
        "predictions": rows,
    })


if __name__ == "__main__":
    main()
