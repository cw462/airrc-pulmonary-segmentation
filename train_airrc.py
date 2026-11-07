"""Run the existing U-Mamba bottleneck trainer on the fixed AirRC split."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import shutil
import time

import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2 import paths as nnunet_paths

# This U-Mamba checkout hardcodes its data paths. Override them in this process only.
nnunet_paths.nnUNet_raw = os.environ["nnUNet_raw"]
nnunet_paths.nnUNet_preprocessed = os.environ["nnUNet_preprocessed"]
nnunet_paths.nnUNet_results = os.environ["nnUNet_results"]

from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


DATASET = "Dataset926_AirRC"
CONFIGURATION = "3d_fullres"
SEED = 2026
PATCH = [128, 128, 128]
BATCH = 2
GIB = 1024 ** 3


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, payload):
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def cache_case(task):
    out, image, label, plans, dataset = task
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    manager = PlansManager(plans)
    config = manager.get_configuration(CONFIGURATION)
    data, seg, props = DefaultPreprocessor().run_case([str(image)], str(label), manager, config, dataset)
    for suffix, array in ((".npy", data.astype(np.float16)), ("_seg.npy", seg)):
        target = out.parent / (out.name + suffix)
        partial = target.with_suffix(".npy.part")
        with partial.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        partial.replace(target)
    partial = out.with_suffix(".pkl.part")
    partial.write_bytes(pickle.dumps(props))
    partial.replace(out.with_suffix(".pkl"))


def prepare(work):
    raw = work / "raw" / DATASET
    base = work / "preprocessed" / DATASET
    base.mkdir(parents=True, exist_ok=True)
    dataset = load_json(raw / "dataset.json")
    split = load_json(raw / "splits_final.json")[0]
    train, val = split["train"], split["val"]
    if not train or not val or set(train).intersection(val):
        raise ValueError("Expected nonempty, disjoint training and validation patients.")
    if len(train) != dataset["numTraining"]:
        raise ValueError("The uploaded training split and dataset metadata differ.")

    # The framework fits intensity statistics using imagesTr and labelsTr only.
    extract_fingerprints([926], num_processes=4, clean=False, verbose=False)
    plan_experiments([926], overwrite_target_spacing=[1, 1, 1])
    plan_file = base / "nnUNetPlans.json"
    plans = load_json(plan_file)
    config = plans["configurations"][CONFIGURATION]
    if np.any(np.asarray(PATCH) % np.prod(config["pool_op_kernel_sizes"], axis=0)):
        raise ValueError("The requested patch is incompatible with the planned strides.")
    config["patch_size"] = PATCH
    config["batch_size"] = BATCH
    save_json(plan_file, plans)
    save_json(base / "splits_final.json", [split])
    save_json(base / "dataset.json", dataset)
    storage = base / config["data_identifier"]
    storage.mkdir(exist_ok=True)
    pending = []
    required_bytes = 0
    reader = sitk.ImageFileReader()
    for cases, suffix in ((train, "Tr"), (val, "Va")):
        for case in cases:
            out = storage / case
            if all((storage / (case + extension)).is_file() for extension in (".npy", "_seg.npy", ".pkl")):
                continue
            image = raw / f"images{suffix}" / f"{case}_0000.nii.gz"
            label = raw / f"labels{suffix}" / f"{case}.nii.gz"
            reader.SetFileName(str(image))
            reader.ReadImageInformation()
            required_bytes += 3 * int(np.prod(reader.GetSize())) + 512 * 1024
            pending.append((out, image, label, plans, dataset))
    available = shutil.disk_usage(storage).free
    available_memory_kb = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                               if line.startswith("MemAvailable:"))
    if available < required_bytes + 7 * GIB or available_memory_kb * 1024 < required_bytes + 16 * GIB:
        raise RuntimeError(f"Insufficient RAM-disk headroom for {required_bytes / GIB:.2f} GiB of preprocessing.")
    print(f"Preparing {len(pending)} training/validation volumes, up to {required_bytes / GIB:.2f} GiB", flush=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        for number, _ in enumerate(pool.map(cache_case, pending), start=1):
            if number % 16 == 0 or number == len(pending):
                print(f"Preprocessed {number}/{len(pending)}", flush=True)
    return plans, dataset


def dice_per_class(reference, prediction):
    result = {}
    for label in (1, 2, 3):
        expected = reference == label
        actual = prediction == label
        denom = int(expected.sum()) + int(actual.sum())
        result[str(label)] = 2 * int(np.count_nonzero(expected & actual)) / denom if denom else None
    return result


class AirRCTrainer(nnUNetTrainerUMambaBot):
    best_volume_dice = -1.0
    validate_every = 8

    def plot_network_architecture(self):
        # Avoid requiring Graphviz on the training host.
        return

    def train_step(self, batch):
        result = super().train_step(batch)
        if not np.isfinite(result["loss"]):
            raise FloatingPointError(f"Nonfinite loss at epoch {self.current_epoch}.")
        return result

    @torch.no_grad()
    def validate_volumes(self, export_first=False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        predictor = nnUNetPredictor(
            tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
            device=self.device, verbose=False, verbose_preprocessing=False, allow_tqdm=False,
        )
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager,
                                        None, self.dataset_json, "nnUNetTrainerUMambaBot", None)
        cases = self.do_split()[1]
        dataset = nnUNetDataset(self.preprocessed_dataset_folder, cases)
        reader = self.plans_manager.image_reader_writer_class()
        measurements = []
        try:
            for case in cases:
                start = time.monotonic()
                data, _, props = dataset.load_case(case)
                tensor = torch.from_numpy(np.array(data, dtype=np.float32))
                logits = predictor.predict_sliding_window_return_logits(tensor).cpu()
                prediction = convert_predicted_logits_to_segmentation_with_correct_shape(
                    logits, self.plans_manager, self.configuration_manager, self.label_manager, props,
                )
                del logits
                label = self.work / "raw" / DATASET / "labelsVa" / f"{case}.nii.gz"
                target, _ = reader.read_seg(str(label))
                if prediction.shape != target[0].shape:
                    raise ValueError(f"Prediction did not return to the reference grid for {case}.")
                if export_first and not measurements:
                    reader.write_seg(prediction, str(Path(self.output_folder) / "example_prediction.nii.gz"), props)
                seconds = time.monotonic() - start
                metrics = dice_per_class(target[0], prediction)
                measurements.append({"case": case, "seconds": seconds, "dice": metrics})
                self.print_to_log_file("Full volume", case, f"{seconds:.2f}s", metrics)
        finally:
            self.set_deep_supervision_enabled(True)
        means = {}
        for key in ("1", "2", "3"):
            scores = [item["dice"][key] for item in measurements if item["dice"][key] is not None]
            means[key] = float(np.mean(scores)) if scores else None
        macro = float(np.mean([score for score in means.values() if score is not None]))
        return {"macro_dice": macro, "mean_dice": means, "cases": measurements}

    def save_checkpoint(self, filename):
        if self.disable_checkpointing:
            return
        path = Path(filename)
        temporary = path.with_suffix(".pth.part")
        # Store the stock trainer name so the existing U-Mamba predictor can load these weights.
        state = {
            "network_weights": self.network.state_dict(), "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict(), "scheduler_state": self.lr_scheduler.state_dict(),
            "logging": self.logger.get_checkpoint(), "_best_ema": self._best_ema,
            "best_volume_dice": self.best_volume_dice, "current_epoch": self.current_epoch + 1,
            "init_args": self.my_init_kwargs, "trainer_name": "nnUNetTrainerUMambaBot",
            "inference_allowed_mirroring_axes": None,
            "rng_state": (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state()),
        }
        torch.save(state, temporary)
        temporary.replace(path)

    def load_checkpoint(self, filename):
        state = torch.load(filename, map_location="cpu")
        # The installed nnU-Net loader initializes its checkpoint only for a path.
        super().load_checkpoint(filename)
        self.lr_scheduler.load_state_dict(state["scheduler_state"])
        self.best_volume_dice = state["best_volume_dice"]
        py_rng, np_rng, torch_rng, cuda_rng = state["rng_state"]
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(torch_rng)
        torch.cuda.set_rng_state(cuda_rng)

    def on_epoch_end(self):
        self.disable_checkpointing = True
        try:
            super().on_epoch_end()
        finally:
            self.disable_checkpointing = False
        completed = self.current_epoch
        output = Path(self.output_folder)
        if completed == 1 or completed % self.validate_every == 0 or completed == self.num_epochs:
            # The base checkpoint API stores the next epoch as current_epoch + 1.
            self.current_epoch -= 1
            try:
                self.save_checkpoint(output / "checkpoint_latest.pth")
                if completed % self.validate_every == 0 or completed == self.num_epochs:
                    result = self.validate_volumes()
                    save_json(output / f"validation_epoch_{completed:04d}.json", result)
                    if result["macro_dice"] > self.best_volume_dice:
                        self.best_volume_dice = result["macro_dice"]
                        self.save_checkpoint(output / "checkpoint_best.pth")
                        self.save_checkpoint(output / "checkpoint_latest.pth")
            finally:
                self.current_epoch = completed


def run_pilot(trainer):
    trainer.on_train_start()
    trainer.on_train_epoch_start()
    batch = next(trainer.dataloader_train)
    fit_losses = [float(trainer.train_step(batch)["loss"]) for _ in range(30)]
    for _ in range(20):
        trainer.train_step(next(trainer.dataloader_train))
    start = time.monotonic()
    for index in range(200):
        trainer.train_step(next(trainer.dataloader_train))
        if (index + 1) % 50 == 0:
            print(f"Timing {index + 1}/200 steps: {time.monotonic() - start:.2f}s", flush=True)
    elapsed = time.monotonic() - start
    peak = torch.cuda.max_memory_reserved() / GIB
    result = trainer.validate_volumes(export_first=True)
    result.update({
        "seconds_per_step": elapsed / 200, "timed_steps": 200, "peak_training_reserved_gib": peak,
        "fit_loss_first": fit_losses[0], "fit_loss_last": fit_losses[-1], "patch": PATCH, "batch": BATCH,
    })
    save_json(trainer.work / "pilot_timing.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pilot", "full"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    work = Path(os.environ["nnUNet_raw"]).parent
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    plans, dataset = prepare(work)
    trainer = AirRCTrainer(plans, CONFIGURATION, 0, dataset, unpack_dataset=False, device=torch.device("cuda", 0))
    trainer.work = work
    trainer.num_epochs = 400
    trainer.num_iterations_per_epoch = 250
    split_file = work / "raw" / DATASET / "splits_final.json"
    save_json(work / "run_config.json", {
        "mode": args.mode, "seed": SEED, "epochs": trainer.num_epochs, "steps_per_epoch": 250,
        "volume_validation_every_steps": trainer.validate_every * 250, "patch": PATCH, "batch": BATCH,
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    })
    if args.mode == "pilot":
        run_pilot(trainer)
    else:
        if args.resume is not None:
            trainer.load_checkpoint(str(args.resume))
        trainer.run_training()


if __name__ == "__main__":
    main()
