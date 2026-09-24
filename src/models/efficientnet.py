"""EfficientNet-B0 (ImageNet-pretrained) with a single-logit binary head,
plus helpers for the two-phase freeze / fine-tune schedule."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, 1),  # single logit -> BCEWithLogitsLoss
    )
    return model


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def freeze_backbone(model: nn.Module) -> None:
    """Phase 1: only the classification head is trainable."""
    _set_requires_grad(model.features, False)
    _set_requires_grad(model.classifier, True)


def unfreeze_top_blocks(model: nn.Module, n_blocks: int) -> None:
    """Phase 2: unfreeze the last `n_blocks` of `model.features` (9 blocks total in B0)."""
    freeze_backbone(model)
    blocks = list(model.features.children())
    n_blocks = max(0, min(n_blocks, len(blocks)))
    for block in blocks[len(blocks) - n_blocks:]:
        _set_requires_grad(block, True)


def set_frozen_bn_eval(model: nn.Module) -> None:
    """Keep BatchNorm layers inside frozen blocks in eval mode so their running
    stats are not overwritten by tiny, augmented batches. Call after model.train()."""
    for m in model.features.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
