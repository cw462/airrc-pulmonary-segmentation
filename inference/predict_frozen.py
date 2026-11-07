"""Apply the frozen AirRC model to CTs and export native-grid label images."""

import argparse
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import SimpleITK as sitk

from evaluation.add_av_metrics import read_json, write_json
from evaluation.label_geometry import read_geometry, same_grid


def predict_partition(gpu, model, epoch, cases, output):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    from evaluate_airrc import load_model, predict_case

    predictor = None
    for index, case in enumerate(cases, 1):
        case_id = case["case_id"]
        label_path = output / "predictions" / f"{case_id}_model_label.nii.gz"
        record_path = output / "results" / f"{case_id}.json"
        if label_path.is_file() and record_path.is_file():
            print(f"CT GPU={gpu} {index}/{len(cases)} {case_id} already saved", flush=True)
            continue
        if predictor is None:
            predictor = load_model(model, epoch)
        started = time.monotonic()
        prediction, properties = predict_case(predictor, Path(case["image_path"]))
        if not np.issubdtype(prediction.dtype, np.integer) or not np.isin(prediction, (0, 1, 2, 3)).all():
            raise ValueError(f"Unexpected predicted labels for {case_id}.")
        temporary = label_path.with_name(label_path.name.replace(".nii.gz", ".part.nii.gz"))
        reader = predictor.plans_manager.image_reader_writer_class()
        reader.write_seg(prediction.astype(np.uint8), str(temporary), properties)
        ct_grid, label_grid = read_geometry(case["image_path"]), read_geometry(temporary)
        if not same_grid(ct_grid, label_grid):
            raise ValueError(f"Prediction is not on the original CT physical grid for {case_id}.")
        temporary.replace(label_path)
        seconds = time.monotonic() - started
        write_json(record_path, {
            "case_id": case_id, "image_path": case["image_path"], "prediction_path": str(label_path),
            "checkpoint_epoch": epoch, "seconds": seconds, "gpu": str(gpu),
            "native_ct_geometry": ct_grid, "prediction_geometry": label_grid,
            "same_native_physical_grid": True,
            "labels": {"0": "background", "1": "pulmonary_artery", "2": "pulmonary_vein", "3": "airway_lumen"},
            "predicted_voxels": {str(label): int(np.count_nonzero(prediction == label)) for label in (0, 1, 2, 3)},
            "reference_role": "Any provided automatic segmentation is a comparison reference, not ground truth",
        })
        print(f"CT GPU={gpu} {index}/{len(cases)} {case_id} saved in {seconds:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    args = parser.parse_args()
    cases = read_json(args.manifest)["cases"]
    ids = [case["case_id"] for case in cases]
    if not ids or len(ids) != len(set(ids)) or any(not name or any(char in name for char in ("/", "\\", "..")) for name in ids):
        raise ValueError("Expected unique case identifiers without path components.")
    if any(not Path(case["image_path"]).is_file() for case in cases):
        raise FileNotFoundError("A requested original CT image is missing.")
    model, output = args.model.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("results", "predictions"):
        (output / name).mkdir(exist_ok=True)
    config = {"checkpoint": str(model / "fold_0/checkpoint_best.pth"),
              "checkpoint_epoch": args.checkpoint_epoch, "cases": cases,
              "inference": "Frozen AirRC normalization, 1 mm preprocessing, 128-cube windows, 50% overlap, Gaussian blending, no TTA; restored to native CT grid"}
    config_path = output / "inference_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing predictions used a different model or case manifest.")
    write_json(config_path, config)
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Each inference worker needs a separate GPU.")
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=predict_partition, args=(
        gpu, model, args.checkpoint_epoch, cases[index::len(args.gpus)], output))
        for index, gpu in enumerate(args.gpus)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if any(worker.exitcode != 0 for worker in workers):
        raise RuntimeError("A CT inference worker failed.")
    write_json(output / "summary.json", {
        "state": "done", "patients_completed": len(cases), **config,
        "cases": [read_json(output / "results" / f"{case_id}.json") for case_id in ids],
        "accuracy_metrics": None, "accuracy_metrics_reason": "No complete manually annotated ground truth supplied",
    })


if __name__ == "__main__":
    main()
