"""Binary Skeleton Recall training term, adapted to the existing nnU-Net.

Reference: MIC-DKFZ/Skeleton-Recall, Apache-2.0, commit
c7bac3c72ca7bb4f3d891235f051f0c4d07d7229. Training uses the upstream
two-voxel Manhattan tube intersected with GT, not the undilated evaluation
skeleton. Only the highest-resolution deep-supervision target is extended.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import torch
from torch import nn


class SoftSkeletonRecall(nn.Module):
    def __init__(self, batch_dice=False, smooth=1e-5):
        super().__init__()
        self.batch_dice = bool(batch_dice)
        self.smooth = float(smooth)

    def forward(self, logits, tube):
        if logits.shape[1] != 2 or tube.shape[1] != 1 or logits.shape[2:] != tube.shape[2:]:
            raise ValueError('Binary logits and same-grid single-channel tube required.')
        probability = torch.softmax(logits.float(), dim=1)[:, 1:2]
        reference = (tube > 0).to(probability.dtype)
        axes = tuple(range(2, probability.ndim))
        numerator = (probability * reference).sum(axes)
        denominator = reference.sum(axes)
        if self.batch_dice:
            numerator, denominator = numerator.sum(0), denominator.sum(0)
        recall = (numerator + self.smooth) / (denominator + self.smooth).clamp_min(1e-8)
        return -recall.mean()


class SegmentationWithSkeletonRecall(nn.Module):
    def __init__(self, base_loss, batch_dice=False, weight=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.skeleton_recall = SoftSkeletonRecall(batch_dice=batch_dice)
        self.weight = float(weight)

    def forward(self, logits, target):
        result = self.base_loss(logits, target[:, :1])
        if target.shape[1] == 2:
            result = result + self.weight * self.skeleton_recall(logits, target[:, 1:2])
        elif target.shape[1] != 1:
            raise ValueError('Expected segmentation and optional tubed-skeleton channel.')
        return result


def add_tubed_skeleton(target):
    target = np.asarray(target)
    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError('Expected B,1,D,H,W segmentation after augmentation.')
    tube = np.zeros_like(target)
    footprint = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), 2)
    for index in range(len(target)):
        mask = target[index, 0] > 0
        if mask.any():
            centerline = skeletonize(mask, method='lee').astype(bool)
            tube[index, 0] = ndi.binary_dilation(centerline, structure=footprint) & mask
    return np.concatenate([target, tube], axis=1)


class SkeletonTargetTransform:
    """Old batchgenerators callable, inserted after DS and before ToTensor."""
    def __call__(self, **data):
        targets = data['target']
        if isinstance(targets, tuple):
            data['target'] = (add_tubed_skeleton(targets[0]), *targets[1:])
        elif isinstance(targets, list):
            data['target'] = [add_tubed_skeleton(targets[0]), *targets[1:]]
        else:
            data['target'] = add_tubed_skeleton(targets)
        return data
