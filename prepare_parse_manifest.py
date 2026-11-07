"""Describe the 100 publicly labeled PARSE2022 cases for external inference."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="PARSE2022 train directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--airrc-summary", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    case_dirs = sorted(path for path in source.iterdir() if path.is_dir() and path.name.startswith("PA"))
    cases = []
    for directory in case_dirs:
        case = {"case_id": directory.name}
        for kind in ("image", "label"):
            files = [path for path in (directory / kind).iterdir()
                     if path.name.endswith((".nii", ".nii.gz"))]
            if len(files) != 1:
                raise ValueError(f"Expected one NIfTI in {directory.name}/{kind}.")
            case[f"{kind}_path"] = str(files[0])
        cases.append(case)
    if len(cases) != 100:
        raise ValueError(f"Expected 100 labeled PARSE2022 cases, found {len(cases)}.")
    model = json.loads(args.airrc_summary.read_text(encoding="utf-8"))
    manifest = {
        "dataset": "PARSE2022", "subset": "public_labeled_training_subset", "cases": cases,
        "checkpoint_sha256": model.get("checkpoint_sha256"),
        "source": "https://parse2022.grand-challenge.org/Dataset/",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared a manifest for {len(cases)} case pairs.")


if __name__ == "__main__":
    main()

