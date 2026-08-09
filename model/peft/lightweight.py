"""Lightweight PEFT baselines: BitFit, LN-tuning, and bottleneck adapters.

All three freeze the backbone and train only a tiny subset of parameters,
reusing the same block-window semantics as LoRA (``PEFT.*.BLOCKS``).
"""

import math
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .lora import _identity_or_dropout, get_module_by_name, iter_linear_targets, set_module_by_name

# Same head keyword rule used by mark_trainable_lora_and_head, kept identical so
# the new methods match LoRA's existing "what stays trainable" behavior.
_HEAD_KEYWORDS = ("classifier", "head", "bnneck", "id_head")


def _is_head_param(name: str) -> bool:
    lname = name.lower()
    return any(k in lname for k in _HEAD_KEYWORDS)


def _block_frozen(name: str, blocks: Optional[List[int]]) -> bool:
    """True if a parameter belongs to a JPM branch or an out-of-window block."""
    if "b1." in name or "b2." in name:
        # JPM deep-copied branches — keep frozen (same as LoRA).
        return True
    if "blocks." in name:
        try:
            idx = int(name.split("blocks.")[1].split(".")[0])
            return blocks is not None and idx not in blocks
        except (IndexError, ValueError):
            return False
    # patch_embed / final norm / head are not block-scoped.
    return False


def mark_trainable_head(model: nn.Module) -> None:
    """Unfreeze classifier/head params (mirrors mark_trainable_lora_and_head)."""
    for name, module in model.named_modules():
        if _is_head_param(name):
            for p in module.parameters():
                p.requires_grad = True


def mark_trainable_bitfit(model: nn.Module, blocks: Optional[List[int]] = None, train_head: bool = True) -> None:
    """BitFit: train only bias terms (Linear biases + LayerNorm beta)."""
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if _is_head_param(name):
            continue
        if _block_frozen(name, blocks):
            continue
        if "bias" in name:
            p.requires_grad = True
    if train_head:
        mark_trainable_head(model)


def mark_trainable_lntune(
    model: nn.Module,
    blocks: Optional[List[int]] = None,
    train_head: bool = True,
    train_final_norm: bool = True,
) -> None:
    """LN-tuning: train only LayerNorm weight/bias (gamma/beta)."""
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if _is_head_param(name):
            continue
        if _block_frozen(name, blocks):
            continue
        if not train_final_norm and name.startswith("base.norm"):
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else name
        if isinstance(get_module_by_name(model, parent_name), nn.LayerNorm):
            p.requires_grad = True
    if train_head:
        mark_trainable_head(model)


class BottleneckAdapter(nn.Module):
    """Parallel bottleneck adapter: y = base(x) + scale * up(act(down(x)))."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 16,
        dropout: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.scale = float(scale)
        self.drop = _identity_or_dropout(dropout)
        self.act = nn.GELU()
        self.down = nn.Linear(self.in_features, self.r, bias=False)
        self.up = nn.Linear(self.r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        # Zero-init the up projection so the adapter is identity at init
        # (same trick as LoRA's zero-initialized lora_B).
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        update = self.up(self.act(self.drop(self.down(x))))
        return y + self.scale * update


def inject_adapters_into_vit(
    model: nn.Module,
    r: int,
    dropout: float,
    scale: float,
    targets: List[str],
    include_blocks: Optional[List[int]] = None,
) -> List[str]:
    """Wrap target nn.Linear modules with BottleneckAdapter (mirrors LoRA injection)."""
    replaced = []
    for name, lin in list(iter_linear_targets(model, targets)):
        if include_blocks is not None and len(include_blocks) > 0:
            if "blocks." in name:
                try:
                    block_idx = int(name.split("blocks.")[1].split(".")[0])
                    if block_idx not in include_blocks:
                        continue
                except (IndexError, ValueError):
                    pass

        wrapped = BottleneckAdapter(lin, r=r, dropout=dropout, scale=scale)
        set_module_by_name(model, name, wrapped)
        replaced.append(name)
    return replaced


def mark_trainable_adapters_and_head(model: nn.Module, train_head: bool = True) -> None:
    """Freeze everything except adapter down/up projections (+ head)."""
    for p in model.parameters():
        p.requires_grad = False
    for _, m in model.named_modules():
        if isinstance(m, BottleneckAdapter):
            m.down.weight.requires_grad = True
            m.up.weight.requires_grad = True
    if train_head:
        mark_trainable_head(model)
