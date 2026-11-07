"""Read integer labels and use the project's physical reference-grid approach.

prepare_airrc.prepare_case uses SimpleITK.Resample with a reference image,
origin, spacing and direction. Here the same physical-space operation uses
nearest-neighbor interpolation for saved label images. No array-only resize
or model inference is performed.
"""

from itertools import product
import os
import numpy as np
import SimpleITK as sitk


def _unicode_nifti(path):
    name = str(path)
    return os.name == "nt" and not name.isascii() and name.lower().endswith((".nii", ".nii.gz"))


def _nifti_fallback(path, pixels):
    # Some Windows SimpleITK builds cannot open Unicode paths. Use the same
    # NIfTI affine -> LPS geometry conversion as prepare_airrc.prepare_case.
    import nibabel as nib

    source = nib.load(str(path))
    if len(source.shape) != 3:
        raise ValueError("Expected a scalar three-dimensional NIfTI label image.")
    affine = np.diag([-1, -1, 1, 1]) @ source.affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError("Invalid NIfTI physical spacing.")
    units = source.header.get_xyzt_units()[0]
    scale = {"mm": 1.0, "unknown": 1.0, "meter": 1000.0, "micron": 0.001}.get(units)
    if scale is None:
        raise ValueError(f"Unsupported NIfTI spatial units {units!r}.")
    image = (sitk.GetImageFromArray(np.ascontiguousarray(np.asanyarray(source.dataobj).transpose(2, 1, 0)))
             if pixels else sitk.Image([1, 1, 1], sitk.sitkUInt8))
    image.SetSpacing(tuple(spacing * scale))
    image.SetOrigin(tuple(affine[:3, 3] * scale))
    image.SetDirection(tuple((affine[:3, :3] / spacing).ravel()))
    geometry(image)
    return image, list(source.shape), str(source.header.get_data_dtype())


def geometry(image):
    components = (image.GetNumberOfComponentsPerPixel() if hasattr(image, "GetNumberOfComponentsPerPixel")
                  else image.GetNumberOfComponents())
    if image.GetDimension() != 3 or components != 1:
        raise ValueError("Expected a scalar three-dimensional label image.")
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    if (not np.all(np.isfinite(spacing)) or np.any(spacing <= 0)
            or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction))
            or not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5, rtol=0)):
        raise ValueError("Invalid physical geometry; missing geometry cannot be inferred.")
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    size = list(image.GetSize())
    return {
        "size_xyz": size, "shape_zyx": size[::-1],
        "spacing_xyz_mm": spacing.tolist(), "origin_lps_mm": origin.tolist(),
        "direction": direction.ravel().tolist(), "affine_lps_mm": affine.tolist(),
    }


def read_geometry(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    try:
        reader.ReadImageInformation()
    except RuntimeError:
        if not _unicode_nifti(path):
            raise
        image, size, pixel_type = _nifti_fallback(path, pixels=False)
        result = geometry(image)
        result.update(size_xyz=size, shape_zyx=size[::-1], pixel_type=pixel_type)
        return result
    result = geometry(reader)
    result["pixel_type"] = sitk.GetPixelIDValueAsString(reader.GetPixelID())
    return result


def same_grid(first, second):
    return first["size_xyz"] == second["size_xyz"] and all(
        np.allclose(first[key], second[key], atol=1e-5, rtol=0)
        for key in ("spacing_xyz_mm", "origin_lps_mm", "direction", "affine_lps_mm")
    )


def read_label(path, encoding):
    try:
        image = sitk.ReadImage(str(path))
    except RuntimeError:
        if not _unicode_nifti(path):
            raise
        image, _, _ = _nifti_fallback(path, pixels=True)
    geometry(image)
    labels = sitk.GetArrayViewFromImage(image)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected integer label pixels in {path}.")
    unique = np.unique(labels)
    if encoding == "airrc_original":
        # Reuse the exact mapping used to create the fixed training/validation/test GT.
        from prepare_airrc import LABEL_MAP
        if unique.min() < 0 or unique.max() >= len(LABEL_MAP):
            raise ValueError(f"Unexpected original AirRC labels {unique.tolist()} in {path}.")
        converted = sitk.GetImageFromArray(LABEL_MAP[labels])
        converted.CopyInformation(image)
        image = converted
    elif encoding == "unified":
        if not np.isin(unique, (0, 1, 2, 3)).all():
            raise ValueError(f"Unexpected unified labels {unique.tolist()} in {path}.")
    else:
        raise ValueError(f"Unsupported label encoding {encoding!r}; supply an explicit project mapping.")
    return image


def _physical_bounds(image):
    # Include voxel edges, not just centers. This rejects disjoint coordinate frames.
    corners = [image.TransformContinuousIndexToPhysicalPoint(tuple(index))
               for index in product(*[(-0.5, size - 0.5) for size in image.GetSize()])]
    points = np.asarray(corners)
    return points.min(axis=0), points.max(axis=0)


def align_prediction(prediction, reference):
    before, target = geometry(prediction), geometry(reference)
    if same_grid(before, target):
        return prediction, {"action": "already_on_gt_grid", "prediction": before, "gt": target}
    p_min, p_max = _physical_bounds(prediction)
    g_min, g_max = _physical_bounds(reference)
    if np.any(np.minimum(p_max, g_max) <= np.maximum(p_min, g_min)):
        raise ValueError("Prediction and GT physical fields of view do not overlap; no array resize is allowed.")
    restored = sitk.Resample(
        prediction, reference, sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor, 0, prediction.GetPixelID(),
    )
    after = geometry(restored)
    if not same_grid(after, target):
        raise ValueError("Physical nearest-neighbor restoration did not produce the GT grid.")
    return restored, {
        "action": "physical_nearest_neighbor_to_gt", "prediction": before, "gt": target,
        "aligned_prediction": after, "outside_prediction_value": 0,
    }
