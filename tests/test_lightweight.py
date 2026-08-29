#!/usr/bin/env python3
"""Verify BitFit, LN-tuning, and BottleneckAdapter placement in the repo."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from config import cfg, normalize_peft_config, merge_config_file
from model import make_model
from model.peft.lightweight import BottleneckAdapter


def _build_model(config_path, num_class=751, camera_num=6, view_num=1):
    merge_config_file(cfg, config_path)
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.MODEL.PRETRAIN_PATH = ""
    cfg.MODEL.DEVICE = "cpu"
    normalize_peft_config(cfg)
    cfg.freeze()
    return make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=view_num)


def _forward(model):
    dummy_input = torch.randn(2, 3, 256, 128)
    cam_label = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        output = model(dummy_input, cam_label=cam_label, view_label=cam_label)
    assert output is not None


def test_bitfit_blocks_6_11():
    model = _build_model("configs/Market/bitfit_blocks_6_11.yml")
    expected_blocks = set(cfg.PEFT.BITFIT.BLOCKS)

    trainable_bias_blocks = set()
    for name, p in model.named_parameters():
        if p.requires_grad and "bias" in name:
            if "blocks." in name:
                trainable_bias_blocks.add(int(name.split("blocks.")[1].split(".")[0]))

    # Only biases in blocks 6-11 are trainable; head stays trainable too.
    assert trainable_bias_blocks == expected_blocks, (
        f"Expected bias blocks {expected_blocks}, got {trainable_bias_blocks}"
    )
    # No non-bias backbone param is trainable.
    for name, p in model.named_parameters():
        if p.requires_grad and "bias" not in name and "classifier" not in name and "bottleneck" not in name:
            assert False, f"Unexpected trainable non-bias param: {name}"
    _forward(model)


def test_lntune_blocks_6_11():
    model = _build_model("configs/Market/lntune_blocks_6_11.yml")
    expected_blocks = set(cfg.PEFT.LNTUNE.BLOCKS)

    trainable_ln_blocks = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parent = name.rsplit(".", 1)[0]
        mod = dict(model.named_modules()).get(parent)
        if isinstance(mod, nn.LayerNorm) and "blocks." in name:
            trainable_ln_blocks.add(int(name.split("blocks.")[1].split(".")[0]))

    assert trainable_ln_blocks == expected_blocks, (
        f"Expected LN blocks {expected_blocks}, got {trainable_ln_blocks}"
    )
    # Every trainable backbone param must belong to a LayerNorm.
    for name, p in model.named_parameters():
        if p.requires_grad and "classifier" not in name and "bottleneck" not in name:
            parent = name.rsplit(".", 1)[0]
            mod = dict(model.named_modules()).get(parent)
            assert isinstance(mod, nn.LayerNorm), f"Non-LayerNorm trainable param: {name}"
    _forward(model)


def test_adapter_blocks_6_11():
    model = _build_model("configs/Market/adapter_blocks_6_11_r16.yml")
    expected_blocks = set(cfg.PEFT.ADAPTER.BLOCKS)

    adapter_modules = [n for n, m in model.named_modules() if isinstance(m, BottleneckAdapter)]
    actual_blocks = set()
    for name in adapter_modules:
        if "blocks." in name:
            actual_blocks.add(int(name.split("blocks.")[1].split(".")[0]))

    assert actual_blocks == expected_blocks, (
        f"Expected adapter blocks {expected_blocks}, got {actual_blocks}"
    )

    # Identity at init: up is zero-initialized, so output ≈ base output.
    adapter = next(iter([m for _, m in model.named_modules() if isinstance(m, BottleneckAdapter)]))
    assert torch.all(adapter.up.weight == 0), "Adapter up projection should be zero-initialized"

    x = torch.randn(2, adapter.in_features)
    with torch.no_grad():
        out = adapter(x)
        base_out = adapter.base(x)
    assert torch.allclose(out, base_out, atol=1e-5), "Adapter should be identity at init"

    _forward(model)


if __name__ == "__main__":
    test_bitfit_blocks_6_11()
    test_lntune_blocks_6_11()
    test_adapter_blocks_6_11()
    print("lightweight PEFT tests passed")
