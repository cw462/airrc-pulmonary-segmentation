import io
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.ct_local_refiner import CTLocalResidualRefiner, RefinedSegmenter
from parse22_supervised.skeleton_recall import SoftSkeletonRecall, SkeletonTargetTransform


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, 1)
        self.decoder = nn.Module()
        self.decoder.deep_supervision = True

    def forward(self, x):
        logits = self.conv(x)
        return [logits, F.avg_pool3d(logits, 2)] if self.decoder.deep_supervision else logits


class Parse22ModuleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026)

    def test_skeleton_recall_matches_soft_recall_and_has_gradient(self):
        logits = torch.zeros(2, 2, 3, 3, 3, requires_grad=True)
        reference = torch.zeros(2, 1, 3, 3, 3)
        reference[:, :, 1, 1, 1] = 1
        loss = SoftSkeletonRecall(batch_dice=True)(logits, reference)
        expected = -(1 + 1e-5) / (2 + 1e-5)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum(), 0)

    def test_tube_is_inside_gt_and_only_highest_ds_target_changes(self):
        target = np.zeros((1, 1, 9, 9, 12), dtype=np.float32)
        target[0, 0, 3:6, 3:6, 1:11] = 1
        low = target[:, :, ::2, ::2, ::2].copy()
        outputs = SkeletonTargetTransform()(target=[target, low])['target']
        np.testing.assert_array_equal(outputs[0][:, :1], target)
        self.assertTrue(np.all(outputs[0][:, 1:2] <= target))
        self.assertGreater(outputs[0][:, 1:2].sum(), 0)
        self.assertIs(outputs[1], low)

    def test_refiner_zero_init_bound_and_detached_conditions(self):
        model = CTLocalResidualRefiner(2)
        ct = torch.randn(1, 1, 8, 8, 8)
        logits = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
        output, aux = model(ct, logits, return_aux=True)
        self.assertTrue(torch.equal(output, logits))
        self.assertEqual(torch.count_nonzero(aux['delta']), 0)
        with torch.no_grad():
            model.residual_head.bias.fill_(10)
        output, aux = model(ct, logits, return_aux=True)
        self.assertLessEqual(aux['delta'].abs().max(), 1)
        self.assertGreaterEqual(aux['gate'].min(), .25)
        output.sum().backward()
        torch.testing.assert_close(logits.grad, torch.ones_like(logits))

    def test_wrapper_preserves_ds_secondary_and_reloads_complete_model(self):
        backbone = DummyBackbone()
        model = RefinedSegmenter(backbone, 2)
        data = torch.randn(1, 1, 8, 8, 8)
        original = backbone(data)
        actual = model(data)
        torch.testing.assert_close(actual[0], original[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], original[1], rtol=0, atol=0)
        with torch.no_grad():
            model.refiner.residual_head.bias[1] = .5
        actual = model(data)
        self.assertFalse(torch.equal(actual[0], original[0]))
        torch.testing.assert_close(actual[1], original[1], rtol=0, atol=0)
        model.do_ds = False
        self.assertFalse(model.decoder.deep_supervision)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = RefinedSegmenter(DummyBackbone(), 2)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        restored.do_ds = False
        torch.testing.assert_close(restored(data), model(data), rtol=0, atol=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue({id(p) for p in model.refiner.parameters()}.issubset(optimizer_ids))


if __name__ == '__main__':
    unittest.main()
