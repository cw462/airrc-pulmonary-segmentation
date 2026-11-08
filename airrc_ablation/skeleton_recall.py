"""Foreground-class macro Skeleton Recall Loss for the original four-label AirRC task.

Adapted mathematical term/GT tube from MIC-DKFZ/Skeleton-Recall,
Apache2, c7bac3c72ca7bb4f3d891235f051f0c4d07d7229.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import torch
from torch import nn


class SoftSkeletonRecall(nn.Module):
    def __init__(self, batch_dice=True, smooth=1e-5):
        super().__init__()
        self.batch_dice, self.smooth = bool(batch_dice), float(smooth)

    def forward(self, logits, tubes):
        if logits.shape[1] != 4 or tubes.shape[1] != 3 or logits.shape[2:] != tubes.shape[2:]:
            raise ValueError('AirRC requires four logits and three aligned foreground tubes.')
        probability = torch.softmax(logits.float(), 1)[:, 1:]
        reference = (tubes > 0).to(probability.dtype)
        axes = tuple(range(2, probability.ndim))
        numerator, denominator = (probability * reference).sum(axes), reference.sum(axes)
        if self.batch_dice:
            numerator, denominator = numerator.sum(0), denominator.sum(0)
        return -((numerator+self.smooth)/(denominator+self.smooth)).mean()


class SegmentationWithSkeletonRecall(nn.Module):
    def __init__(self, base_loss, batch_dice=True):
        super().__init__()
        self.base_loss = base_loss
        self.skeleton_recall = SoftSkeletonRecall(batch_dice)

    def forward(self, logits, target):
        value = self.base_loss(logits, target[:, :1])
        if target.shape[1] == 4:
            value = value + self.skeleton_recall(logits, target[:, 1:])
        elif target.shape[1] != 1:
            raise ValueError('Expected one segmentation channel and optionally three tubed skeletons.')
        return value


class SkeletonTargetTransform:
    def __call__(self, **data):
        targets = data['target']
        highest = targets[0] if isinstance(targets, (list, tuple)) else targets
        if highest.ndim != 5 or highest.shape[1] != 1:
            raise ValueError('Expected highest-resolution B1DHW segmentation.')
        tubes = np.zeros((len(highest), 3, *highest.shape[2:]), dtype=highest.dtype)
        footprint = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), 2)
        for item in range(len(highest)):
            for label in (1, 2, 3):
                mask = highest[item, 0] == label
                if mask.any():
                    centerline = skeletonize(mask, method='lee').astype(bool)
                    tubes[item, label-1] = ndi.binary_dilation(centerline, structure=footprint) & mask
        enriched = np.concatenate([highest, tubes], axis=1)
        if isinstance(targets, tuple):
            data['target'] = (enriched, *targets[1:])
        elif isinstance(targets, list):
            data['target'] = [enriched, *targets[1:]]
        else:
            data['target'] = enriched
        return data
