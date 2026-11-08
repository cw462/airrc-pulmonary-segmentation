import hashlib
import json
import os
from pathlib import Path

DATASET = 'Dataset926_AirRC'
CONFIGURATION = '3d_fullres'
CLASSES = {1: 'pulmonary_artery', 2: 'pulmonary_vein', 3: 'airway_lumen'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def configure_paths(shared, results):
    from nnunetv2 import paths
    for name, path in {'nnUNet_raw': shared / 'raw', 'nnUNet_preprocessed': shared / 'preprocessed',
                       'nnUNet_results': results}.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
        setattr(paths, name, str(path))
