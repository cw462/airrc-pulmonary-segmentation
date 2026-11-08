# CT 引导局部残差细化：Codex 实现说明

本模块为项目候选设计，不是论文官方实现，不保证涨点。P2=P1（基线+Skeleton Recall Loss）+本模块。PARSE22 使用2类，AirRC使用4类。

## 1. 保存为 models/ct_local_refiner.py

```python
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
```

## 2. 接入训练器

以下是接口示意，函数名必须接到仓库现有实现，不能照抄占位接口当成可运行脚本。

```python
backbone = build_existing_umamba(num_classes=2)
model = RefinedSegmenter(backbone, num_classes=2,
                        hidden_channels=16, max_delta=1.0,
                        gate_floor=0.25).to(device)
# 必须在包装后构建 optimizer，包含 refiner 参数。
optimizer = build_existing_optimizer(model.parameters())
outputs = model(ct)
loss = existing_p1_loss(outputs, target)
# 后续 backward / scaler / optimizer.step 沿用现有训练器。
```

仅细化最高分辨率输出，其他深监督输出不变。Dice/CE、Skeleton Recall Loss、深监督权重与 P1 完全一致。第一版不增加残差正则损失。

检查 nnU-Net/U-Mamba 训练器是否通过 network.do_ds、decoder.deep_supervision 等属性切换深监督；包装后必须显式转发或修改调用位置，不能以为包装器自动继承 backbone 属性。优化器、DDP、compile、checkpoint 加载和滑窗输出格式均需检查。

## 3. 推理

每个归一化 CT patch → backbone → refiner → refined logits → 原有 Gaussian 滑窗融合 → 原有空间恢复 → argmax。推理加载整个包装模型的权重，不只加载 backbone。训练及推理都启用 refiner。

三层 3×3×3 卷积的卷积路径局部感受野为7×7×7；GroupNorm 的统计跨空间位置，因此完整模块不是严格的有限邻域算子。不要在论文中声称严格局部依赖。

## 4. 固定实验要求

- P0=基线；P1=P0+Skeleton Recall Loss；P2=P1+本模块。
- 正式实验从头训练，P2不加载已训练P1权重。
- 共用划分、训练步数、优化器、增强和推理设置。
- 先构建并初始化共享 backbone，再初始化模块；核对共享初始权重哈希。模块初始化消耗随机数，需重新设置数据采样随机种子以尽可能保持数据顺序一致。
- 包装后若框架再次全局初始化网络，必须重新将 residual_head 置零，并验证初始输出相等。
- 默认 hidden_channels=16, max_delta=1.0, gate_floor=0.25。参数仅在验证集选择。
- 记录 delta 绝对值均值和分位数，确认确实学习修正。
- 记录参数量、峰值显存和推理时间。参数少不等于激活显存低；128³、FP32、16通道可能增加较多开销。
- 如果显存不足，先尝试仅对 refiner 做 gradient checkpointing；如改通道数，记录配置。不要悄悄改变其他组的 patch/batch。
- 零初始化使第一步 features 梯度可能为零，残差输出层更新后应出现梯度，这是预期现象。

## 5. 执行与验收

1. 新建模块并运行 `python -m models.ct_local_refiner`。
2. 在实际训练器执行短程 smoke test，检查 AMP、深监督输出顺序、梯度和 checkpoint 恢复。
3. 保存并重新加载 checkpoint，比较 eval 输出一致性。
4. 用实际滑窗执行一例验证，确认空间恢复和输出类别正确。
5. 确认 optimizer 包含全部 refiner 参数后启动 P2 正式训练。
6. 按与P1相同的验证 Dice 规则选 checkpoint。
7. 比较 Dice、clDice、HD95、Precision、Recall、Skeleton Recall、BDR80；不要仅依据 Recall 判断收益。

本文件提供候选模块与接入规范；示例合成测试需由 Codex 在目标环境运行，不能声称已完成真实数据验证。三组消融只能证明逐项增量，不能单独证明两模块互补或 CT 输入本身的贡献；若需要这些结论，应补对应消融。
