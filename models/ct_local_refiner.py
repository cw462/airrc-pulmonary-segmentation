"""User-provided CT local refiner; only framework attribute forwarding added.

Convolutional path receptive field7x7x7; GroupNorm uses spatial statistics,
so the complete module is not a strictly finite-neighbourhood operator.
"""
import math
from contextlib import nullcontext
import torch
from torch import nn
import torch.nn.functional as F


def norm(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)


class ConvBlock3D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            norm(cout), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class CTLocalResidualRefiner(nn.Module):
    """输入为同网格归一化 CT [B,1,D,H,W] 和 logits [B,C,D,H,W]。"""
    def __init__(self, num_classes, hidden_channels=16,
                 max_delta=1.0, gate_floor=0.25):
        super().__init__()
        if num_classes < 2 or max_delta <= 0:
            raise ValueError('Invalid classes or max_delta')
        if not 0 <= gate_floor <= 1:
            raise ValueError('Invalid gate_floor')
        self.num_classes = num_classes
        self.max_delta = float(max_delta)
        self.gate_floor = float(gate_floor)
        self.features = nn.Sequential(
            ConvBlock3D(num_classes + 2, hidden_channels),
            ConvBlock3D(hidden_channels, hidden_channels))
        self.residual_head = nn.Conv3d(
            hidden_channels, num_classes, 3, padding=1)
        # 初始修正为零，初始输出等于 backbone 输出。
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, ct, logits, return_aux=False):
        if ct.ndim != 5 or logits.ndim != 5:
            raise ValueError('Expected 5D tensors')
        if ct.shape[0] != logits.shape[0] or ct.shape[1] != 1:
            raise ValueError('Invalid CT channels/batch')
        if logits.shape[1] != self.num_classes:
            raise ValueError('Invalid class count')
        if ct.shape[2:] != logits.shape[2:] or ct.device != logits.device:
            raise ValueError('CT and logits must be spatially aligned')
        # 权重保持 FP32；使用 AMP，不要对整个模型调用 .half()。
        context = (torch.autocast(device_type='cuda', enabled=False)
                   if ct.is_cuda else nullcontext())
        with context:
            probs = torch.softmax(logits.float(), dim=1)
            entropy = -(probs * probs.clamp_min(1e-7).log()).sum(1, keepdim=True)
            entropy = (entropy / math.log(self.num_classes)).clamp(0, 1)
            # 条件分支停止梯度；主 logits 加法路径仍更新 backbone。
            conditions = torch.cat(
                [ct.float(), probs.detach(), entropy.detach()], dim=1)
            raw_delta = self.residual_head(self.features(conditions))
            gate = self.gate_floor + (1-self.gate_floor)*entropy.detach()
            delta = self.max_delta * gate * torch.tanh(raw_delta)
            refined = logits.float() + delta
        if return_aux:
            return refined, dict(delta=delta, entropy=entropy.detach(), gate=gate)
        return refined


class RefinedSegmenter(nn.Module):
    """需确认 backbone 深监督最高分辨率输出位于 index=0。"""
    def __init__(self, backbone, num_classes, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.refiner = CTLocalResidualRefiner(num_classes, **kwargs)

    @property
    def decoder(self):
        # The installed U-Mamba trainer toggles decoder.deep_supervision.
        return self.backbone.decoder

    @property
    def do_ds(self):
        return getattr(self.backbone, 'do_ds', self.decoder.deep_supervision)

    @do_ds.setter
    def do_ds(self, enabled):
        if hasattr(self.backbone, 'do_ds'):
            self.backbone.do_ds = enabled
        self.decoder.deep_supervision = enabled

    def forward(self, ct, return_aux=False):
        outputs = self.backbone(ct)
        primary = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        refined, aux = self.refiner(ct, primary, return_aux=True)
        if isinstance(outputs, tuple):
            result = (refined, *outputs[1:])
        elif isinstance(outputs, list):
            result = [refined, *outputs[1:]]
        else:
            result = refined
        return (result, aux) if return_aux else result


def test_refiner():
    torch.manual_seed(2026)
    module = CTLocalResidualRefiner(2)
    ct = torch.randn(1, 1, 12, 12, 12)
    logits = torch.randn(1, 2, 12, 12, 12, requires_grad=True)
    target = torch.randint(0, 2, (1, 12, 12, 12))
    out, aux = module(ct, logits, return_aux=True)
    assert torch.equal(out, logits)
    assert torch.count_nonzero(aux['delta']) == 0
    F.cross_entropy(out, target).backward()
    assert torch.isfinite(logits.grad).all()
    grad = module.residual_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    opt = torch.optim.SGD(module.parameters(), lr=0.01)
    opt.step()
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(module(ct, logits.detach()), target).backward()
    grad = module.features[0].block[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0
    print('Refiner tests passed')


if __name__ == '__main__':
    test_refiner()
