"""Evaluate the final AirRC checkpoint selected on validation Dice."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


DATASET = "Dataset926_AirRC"
MODEL = "AirRCTrainer__nnUNetPlans__3d_fullres"
LABELS = {1: "pulmonary_artery", 2: "pulmonary_vein", 3: "airway_lumen"}
METRICS = ("dice", "iou", "precision", "recall", "cldice", "hd95_mm")
PROTOCOL = "airrc-m0-v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def surface_points(mask, spacing):
    surface = mask & ~binary_erosion(mask, border_value=0)
    return np.column_stack(np.nonzero(surface)) * spacing


def score_class(ref, pred, spacing):
    ref_count = int(np.count_nonzero(ref))
    pred_count = int(np.count_nonzero(pred))
    if ref_count == pred_count == 0:
        return {**dict.fromkeys(METRICS), "empty": "both"}
    if ref_count == 0 or pred_count == 0:
        diagonal = float(np.linalg.norm((np.asarray(ref.shape) - 1) * spacing))
        return {**dict.fromkeys(METRICS, 0.0), "hd95_mm": diagonal,
                "empty": "reference" if ref_count == 0 else "prediction"}

    tp = int(np.count_nonzero(ref & pred))
    if tp == ref_count == pred_count:
        return {**dict.fromkeys(METRICS, 1.0), "hd95_mm": 0.0, "empty": "neither"}

    union = ref | pred
    bounds = []
    for axis in range(3):
        occupied = np.flatnonzero(union.any(axis=tuple(other for other in range(3) if other != axis)))
        bounds.append(slice(occupied[0], occupied[-1] + 1))
    ref = np.ascontiguousarray(ref[tuple(bounds)])
    pred = np.ascontiguousarray(pred[tuple(bounds)])

    ref_skel = skeletonize(ref, method="lee").astype(bool)
    pred_skel = skeletonize(pred, method="lee").astype(bool)
    tprec = np.count_nonzero(pred_skel & ref) / max(1, np.count_nonzero(pred_skel))
    tsens = np.count_nonzero(ref_skel & pred) / max(1, np.count_nonzero(ref_skel))
    cldice = 2 * tprec * tsens / (tprec + tsens) if tprec + tsens else 0.0

    ref_surface = surface_points(ref, spacing)
    pred_surface = surface_points(pred, spacing)
    ref_to_pred = cKDTree(pred_surface).query(ref_surface, workers=2)[0]
    pred_to_ref = cKDTree(ref_surface).query(pred_surface, workers=2)[0]
    hd95 = max(np.percentile(ref_to_pred, 95), np.percentile(pred_to_ref, 95))
    return {"dice": 2 * tp / (ref_count + pred_count), "iou": tp / (ref_count + pred_count - tp),
            "precision": tp / pred_count, "recall": tp / ref_count, "cldice": float(cldice),
            "hd95_mm": float(hd95), "empty": "neither"}


def load_model(model_dir, selected):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    torch.set_num_threads(4)
    dataset = read_json(model_dir / "dataset.json")
    expected = {"background": 0, **{name: label for label, name in LABELS.items()}}
    if dataset["labels"] != expected:
        raise ValueError("The trained model uses a different label mapping.")
    checkpoint = torch.load(model_dir / "fold_0" / "checkpoint_best.pth", map_location="cpu")
    if checkpoint["current_epoch"] != selected or checkpoint["trainer_name"] != "nnUNetTrainerUMambaBot":
        raise ValueError("The best checkpoint does not match the selected epoch and backbone.")
    manager = PlansManager(read_json(model_dir / "plans.json"))
    config = manager.get_configuration(checkpoint["init_args"]["configuration"])
    network = nnUNetTrainerUMambaBot.build_network_architecture(manager, dataset, config, 1, False)
    network.load_state_dict(checkpoint["network_weights"])
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
        device=torch.device("cuda", 0), verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.manual_initialization(network, manager, config, None, dataset, checkpoint["trainer_name"], None)
    return predictor


def predict_case(predictor, image, preprocessed=None):
    import torch
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

    manager, config = predictor.plans_manager, predictor.configuration_manager
    if preprocessed is None or not preprocessed.with_suffix(".pkl").is_file():
        data, _, props = DefaultPreprocessor().run_case([str(image)], None, manager, config, predictor.dataset_json)
        data = data.astype(np.float16).astype(np.float32)
    else:
        data = np.asarray(np.load(str(preprocessed) + ".npy", mmap_mode="r"), dtype=np.float32)
        with preprocessed.with_suffix(".pkl").open("rb") as stream:
            props = pickle.load(stream)

    logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(data)).cpu()
    prediction = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, manager, config, predictor.label_manager, props,
    )
    del logits
    return prediction, props


def score_case(predictor, image, label, preprocessed=None):
    prediction, props = predict_case(predictor, image, preprocessed)
    manager = predictor.plans_manager
    ref, ref_props = manager.image_reader_writer_class().read_seg(str(label))
    if ref[0].shape != prediction.shape or ref_props["sitk_stuff"] != props["sitk_stuff"]:
        raise ValueError(f"Prediction and reference geometries differ for {label.name}.")
    spacing = np.asarray(ref_props["spacing"], dtype=float)
    return {name: score_class(ref[0] == value, prediction == value, spacing) for value, name in LABELS.items()}


def summarize(rows):
    partitions = {}
    for split in ("val", "test"):
        partition = [row for row in rows if row["split"] == split]
        classes = {}
        for name in LABELS.values():
            cases = [row for row in partition if row["class"] == name]
            stats = {"patients": len(cases), "both_empty": sum(row["empty"] == "both" for row in cases),
                     "one_empty": sum(row["empty"] in {"reference", "prediction"} for row in cases)}
            for key in METRICS:
                values = [row[key] for row in cases if row[key] is not None]
                stats[key] = {"mean": float(np.mean(values)) if values else None,
                              "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
            classes[name] = stats
        partitions[split] = classes
    return partitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="AirRC run directory containing full and evaluation.")
    args = parser.parse_args()
    full = args.root / "full"
    eval_work = args.root / "evaluation"
    model = full / "results" / DATASET / MODEL
    fold = model / "fold_0"
    if (full / "state.txt").read_text().strip() != "done" or not (fold / "checkpoint_final.pth").is_file():
        raise RuntimeError("Final evaluation requires completed M0 training.")

    config = read_json(full / "run_config.json")
    interval = config["volume_validation_every_steps"] // config["steps_per_epoch"]
    candidates = [
        (epoch, read_json(fold / f"validation_epoch_{epoch:04d}.json"))
        for epoch in range(interval, config["epochs"] + 1, interval)
    ]
    selected_epoch, selected_validation = max(candidates, key=lambda item: item[1]["macro_dice"])
    checkpoint = fold / "checkpoint_best.pth"
    split = read_json(eval_work / "holdout" / "patient_splits.json")
    split_file = full / "raw" / DATASET / "splits_final.json"
    seen = read_json(split_file)[0]
    if (sha256(split_file) != config["split_sha256"] or seen != {part: split[part] for part in ("train", "val")}
            or len(split["val"]) != 38 or len(split["test"]) != 38
            or sum(len(cases) for cases in split.values()) != len(set().union(*split.values()))):
        raise ValueError("The held-out AirRC patients do not match the locked split.")
    for name, result in candidates:
        if sorted(item["case"] for item in result["cases"]) != sorted(split["val"]):
            raise ValueError(f"Validation at epoch {name} has missing or unexpected patients.")

    results = eval_work / "results"
    cases_path = results / "cases"
    cases_path.mkdir(parents=True, exist_ok=True)
    provenance = {
        "protocol": PROTOCOL, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": selected_epoch, "checkpoint_step": selected_epoch * config["steps_per_epoch"],
        "training_steps_completed": config["epochs"] * config["steps_per_epoch"],
        "split_sha256": sha256(eval_work / "holdout" / "patient_splits.json"),
        "selection_macro_dice": selected_validation["macro_dice"],
        "inference": "1 mm reference grid, 128-cube sliding windows, 50% overlap, Gaussian blending, no TTA",
        "hd95": "Maximum of directed 95th percentiles, 6-connected boundary voxel centers, physical mm",
        "cldice": "3D Lee skeletonization and harmonic mean of centerline precision and sensitivity",
        "empty_policy": "Both empty excluded, one empty overlap and clDice zero with full grid diagonal for HD95",
        "aggregation": "Per-patient metrics, arithmetic mean and sample standard deviation",
    }
    config_path = results / "evaluation_config.json"
    if config_path.is_file() and read_json(config_path) != provenance:
        raise RuntimeError("Existing evaluation used a different checkpoint or protocol.")
    write_json(config_path, provenance)
    predictor = load_model(model, selected_epoch)
    rows = []
    validation_dice = {item["case"]: item["dice"] for item in selected_validation["cases"]}
    max_delta = 0.0
    for subset, raw, suffix in (("val", full / "raw" / DATASET, "Va"), ("test", eval_work / "holdout", "Ts")):
        for index, case in enumerate(sorted(split[subset]), start=1):
            case_path = cases_path / f"{subset}_{case}.json"
            start = time.monotonic()
            if case_path.is_file():
                scores = read_json(case_path)
            else:
                prepared = None
                if subset == "val":
                    prepared = (full / "preprocessed" / DATASET /
                                predictor.configuration_manager.data_identifier / case)
                scores = score_case(predictor, raw / f"images{suffix}" / f"{case}_0000.nii.gz",
                                    raw / f"labels{suffix}" / f"{case}.nii.gz", prepared)
                write_json(case_path, scores)
            for value, name in LABELS.items():
                metrics = scores[name]
                if subset == "val" and metrics["dice"] is not None:
                    max_delta = max(max_delta, abs(metrics["dice"] - validation_dice[case][str(value)]))
                rows.append({"split": subset, "patient": case, "class": name, **metrics})
            print(f"{subset} {index}/{len(split[subset])} {case} {time.monotonic() - start:.1f}s", flush=True)
        if subset == "val" and max_delta > 1e-3:
            raise RuntimeError(f"Final validation Dice differs from selection by up to {max_delta:.6f}.")

    output = results / "per_case.csv"
    temporary = output.with_suffix(".csv.part")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    write_json(results / "summary.json", {
        **provenance, "patient_counts": {"val": 38, "test": 38}, "max_validation_dice_delta": max_delta,
        "scores": summarize(rows),
    })


if __name__ == "__main__":
    main()
