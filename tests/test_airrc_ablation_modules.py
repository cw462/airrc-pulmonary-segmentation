import unittest
import numpy as np
import torch
from torch import nn

from airrc_ablation.skeleton_recall import SoftSkeletonRecall, SegmentationWithSkeletonRecall, SkeletonTargetTransform
from models.ct_local_refiner import CTLocalResidualRefiner


class AirRCVariantModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_three_class_soft_recall_is_foreground_macro(self):
        logits=torch.zeros(2,4,4,4,4,requires_grad=True)
        tube=torch.zeros(2,3,4,4,4)
        tube[0,0,:2]=1;tube[1,1,2:]=1;tube[:,2,:1]=1
        counts=tube.sum((0,2,3,4))
        expected=-((counts*.25+1e-5)/(counts+1e-5)).mean()
        loss=SoftSkeletonRecall(batch_dice=True)(logits,tube)
        self.assertTrue(torch.allclose(loss,expected))
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all());self.assertGreater(float(logits.grad.abs().sum()),0)

    def test_tubes_remain_in_own_label_and_lower_DS_unchanged(self):
        gt=np.zeros((1,1,16,16,16),dtype=np.float32)
        for label in (1,2,3):gt[0,0,2:14,label*3,label*3]=label
        lower=gt[:,:,::2,::2,::2].copy()
        output=SkeletonTargetTransform()(target=[gt,lower])['target']
        self.assertEqual(output[0].shape,(1,4,16,16,16))
        for label in (1,2,3):
            tube=output[0][0,label]>0
            self.assertTrue(np.any(tube));self.assertTrue((gt[0,0][tube]==label).all())
        self.assertIs(output[1],lower)

    def test_lower_DS_has_only_original_segmentation_loss(self):
        class Base(nn.Module):
            def forward(self,output,target):return output.sum()*0+7
        loss=SegmentationWithSkeletonRecall(Base())
        output=torch.randn(1,4,4,4,4,requires_grad=True)
        self.assertEqual(float(loss(output,torch.zeros(1,1,4,4,4))),7)

    def test_supplied_refiner_four_class_zero_init_and_feature_gradient(self):
        torch.manual_seed(2026)
        module=CTLocalResidualRefiner(4,hidden_channels=16,max_delta=1.,gate_floor=.25)
        ct=torch.randn(1,1,8,8,8);logits=torch.randn(1,4,8,8,8,requires_grad=True)
        target=torch.randint(0,4,(1,8,8,8))
        out,aux=module(ct,logits,return_aux=True)
        self.assertTrue(torch.equal(out,logits));self.assertEqual(float(aux['delta'].abs().max()),0)
        optimizer=torch.optim.SGD(module.parameters(),lr=.01)
        nn.functional.cross_entropy(out,target).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        out,aux=module(ct,logits.detach(),return_aux=True)
        nn.functional.cross_entropy(out,target).backward()
        self.assertGreater(float(module.features[0].block[0].weight.grad.abs().sum()),0)
        self.assertLessEqual(float(aux['delta'].abs().max()),1.)


if __name__=='__main__':unittest.main()
