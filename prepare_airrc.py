"""Prepare AirRC images in the official mask grid and keep the patient split fixed."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import random
import time

import nibabel as nib
import numpy as np
import SimpleITK as sitk


DATASET = "Dataset926_AirRC"
SPLIT_SEED = 2026
LABELS = {"background": 0, "pulmonary_artery": 1, "pulmonary_vein": 2, "airway_lumen": 3}
LABEL_MAP = np.array([0, 3, 0, 1, 2], dtype=np.uint8)
SPLIT_SUFFIX = {"train": "Tr", "val": "Va", "test": "Ts"}


def save_json(path, content):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_nifti(path, data, affine):
    temporary = path.with_name(path.name.replace(".nii.gz", ".part.nii.gz"))
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header.set_xyzt_units("mm")
    nib.save(image, temporary)
    temporary.replace(path)


def read_ct(header_path, raw_path):
    fields = dict(
        tuple(part.strip() for part in line.split("=", 1))
        for line in header_path.read_text(encoding="ascii").splitlines()
        if "=" in line
    )
    expected = {
        "NDims": "3",
        "BinaryData": "True",
        "BinaryDataByteOrderMSB": "False",
        "CompressedData": "False",
        "CenterOfRotation": "0 0 0",
        "ElementType": "MET_SHORT",
        "ElementDataFile": raw_path.name,
    }
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected MetaImage header in {header_path}")
    shape = tuple(int(n) for n in fields["DimSize"].split())
    voxels = np.fromfile(raw_path, dtype="<i2").reshape(shape[::-1])
    image = sitk.GetImageFromArray(voxels)
    image.SetOrigin(tuple(map(float, fields["Offset"].split())))
    image.SetSpacing(tuple(map(float, fields["ElementSpacing"].split())))
    # MetaImage serializes the direction matrix in column-major order.
    direction = np.asarray(fields["TransformMatrix"].split(), dtype=float).reshape(3, 3).T
    image.SetDirection(direction.ravel())
    return image


def prepare_case(source, destination, row, split):
    suffix = SPLIT_SUFFIX[split]
    case_id = row["PatientID"]
    image_out = destination / f"images{suffix}" / f"{case_id}_0000.nii.gz"
    label_out = destination / f"labels{suffix}" / f"{case_id}.nii.gz"
    if image_out.is_file() and label_out.is_file():
        return

    mask = nib.load(source / row["AirRC_mask"])
    labels = np.asanyarray(mask.dataobj)
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() >= len(LABEL_MAP):
        raise ValueError(f"Unexpected AirRC label values in {case_id}")
    lps_affine = np.diag([-1, -1, 1, 1]) @ mask.affine
    spacing = np.linalg.norm(lps_affine[:3, :3], axis=0)
    direction = lps_affine[:3, :3] / spacing
    if labels.ndim != 3 or not np.allclose(spacing, 1, atol=1e-4):
        raise ValueError(f"Expected a 1 mm 3D AirRC mask in {case_id}")

    ct = read_ct(source / row["CT_mhd"], source / row["CT_raw"])
    reference = sitk.Image(list(labels.shape), sitk.sitkFloat32)
    reference.SetOrigin(lps_affine[:3, 3])
    reference.SetSpacing(spacing)
    reference.SetDirection(direction.ravel())
    for corner in np.ndindex(2, 2, 2):
        index = tuple((n - 1) * v for n, v in zip(labels.shape, corner))
        physical = reference.TransformIndexToPhysicalPoint(index)
        source_index = np.array(ct.TransformPhysicalPointToContinuousIndex(physical))
        tolerance = 1 + 1 / np.array(ct.GetSpacing())
        if np.any(source_index < -tolerance) or np.any(source_index > np.array(ct.GetSize()) - 1 + tolerance):
            raise ValueError(f"CT and mask physical extents disagree in {case_id}")

    aligned = sitk.Resample(ct, reference, sitk.Transform(), sitk.sitkLinear, -1024, sitk.sitkFloat32)
    data = np.rint(sitk.GetArrayViewFromImage(aligned).transpose(2, 1, 0)).astype(np.int16)
    save_nifti(image_out, data, mask.affine)
    save_nifti(label_out, LABEL_MAP[labels], mask.affine)
    print(f"{split} {case_id} {image_out.stat().st_size + label_out.stat().st_size} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--test", action="store_true", help="Prepare the locked test set after model selection.")
    args = parser.parse_args()

    mappings = list(args.source.glob("*.csv"))
    if len(mappings) != 1:
        raise ValueError("Expected the case mapping CSV in the AirRC source root.")
    with mappings[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["PatientID"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Expected one CT per AirRC patient in the case mapping.")
    patients = sorted(by_id)
    random.Random(SPLIT_SEED).shuffle(patients)
    n_train = round(len(patients) * 0.70)
    n_val = round(len(patients) * 0.15)
    split = {"train": patients[:n_train], "val": patients[n_train:n_train + n_val],
             "test": patients[n_train + n_val:]}

    destination = args.cache / DATASET
    selection = ("test",) if args.test else ("train", "val")
    for name in selection:
        for kind in ("images", "labels"):
            (destination / f"{kind}{SPLIT_SUFFIX[name]}").mkdir(parents=True, exist_ok=True)
    split_file = destination / "patient_splits.json"
    if split_file.exists() and json.loads(split_file.read_text(encoding="utf-8")) != split:
        raise ValueError("The patient split already stored in this cache differs from the source CSV.")
    save_json(split_file, split)
    save_json(destination / "splits_final.json", [{"train": split["train"], "val": split["val"]}])
    save_json(destination / "dataset.json", {
        "channel_names": {"0": "CT"}, "labels": LABELS, "numTraining": len(split["train"]),
        "file_ending": ".nii.gz", "overwrite_image_reader_writer": "SimpleITKIO",
    })

    # Prepare a few actual training and validation cases first for the GPU pilot.
    queue = [(name, case_id) for name in selection for case_id in split[name][:3]]
    queue += [(name, case_id) for name in selection for case_id in split[name][3:]]
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda item: prepare_case(args.source, destination, by_id[item[1]], item[0]), queue))
    print(f"Prepared {len(queue)} cases in {time.monotonic() - started:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
