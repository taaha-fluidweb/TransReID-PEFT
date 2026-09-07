#!/usr/bin/env python3
"""Verify LoRA block placement in the unified TransReID-PEFT repo."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from config import cfg, normalize_peft_config, merge_config_file
from model import make_model
from model.peft.lora import LoRALinear


def test_lora_block_specific():
    merge_config_file(cfg, "configs/Market/lora_blocks_6_11_r32.yml")
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.MODEL.PRETRAIN_PATH = ""
    cfg.MODEL.DEVICE = "cpu"
    normalize_peft_config(cfg)
    cfg.freeze()

    model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    model.eval()

    lora_modules = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_modules.append(name)

    expected_blocks = set(cfg.PEFT.LORA.BLOCKS)
    actual_blocks = set()
    for name in lora_modules:
        if "blocks." in name:
            actual_blocks.add(int(name.split("blocks.")[1].split(".")[0]))

    assert actual_blocks == expected_blocks, f"Expected {expected_blocks}, got {actual_blocks}"

    dummy_input = torch.randn(1, 3, 256, 128)
    cam_label = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        output = model(dummy_input, cam_label=cam_label, view_label=cam_label)
    assert output is not None


if __name__ == "__main__":
    test_lora_block_specific()
    print("LoRA block test passed")
