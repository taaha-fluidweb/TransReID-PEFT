#!/usr/bin/env python3
"""PEFT diagnostics for LoRA and SSF configurations."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import cfg, merge_config_file, normalize_peft_config, get_peft_method
from model import make_model
from model.peft.lora import LoRALinear
from model.peft.lightweight import BottleneckAdapter


def _collect_loss(output):
    if hasattr(output, "sum"):
        return output.sum()
    if isinstance(output, (list, tuple)):
        return sum(_collect_loss(x) for x in output)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Check PEFT configuration and parameter counts")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    merge_config_file(cfg, args.config_file)
    normalize_peft_config(cfg)
    cfg.defrost()
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.MODEL.PRETRAIN_PATH = ""
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()

    method = get_peft_method(cfg)
    print(f"PEFT method: {method}")

    model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    if method == "lora":
        lora_layers = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
        print(f"LoRA layers: {len(lora_layers)}")
    elif method == "ssf":
        ssf_params = [n for n, _ in model.named_parameters() if "ssf" in n]
        print(f"SSF parameters: {len(ssf_params)}")
    elif method == "bitfit":
        bias_params = [n for n, p in model.named_parameters() if p.requires_grad and "bias" in n]
        print(f"BitFit trainable bias params: {len(bias_params)}")
    elif method == "lntune":
        from torch.nn import LayerNorm
        ln_params = [n for n, p in model.named_parameters()
                     if p.requires_grad and isinstance(
                         dict(model.named_modules()).get(n.rsplit('.', 1)[0]), LayerNorm)]
        print(f"LN-tuning trainable LayerNorm params: {len(ln_params)}")
    elif method == "adapter":
        adapter_layers = [n for n, m in model.named_modules() if isinstance(m, BottleneckAdapter)]
        print(f"Adapter layers: {len(adapter_layers)}")

    if args.cpu_only:
        x = torch.randn(2, 3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
        cam = torch.zeros(2, dtype=torch.long)
        model.train()
        out = model(x, cam_label=cam, view_label=cam)
        loss = _collect_loss(out)
        loss.backward()
        print("Forward/backward smoke test passed")


if __name__ == "__main__":
    main()
