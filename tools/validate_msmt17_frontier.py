#!/usr/bin/env python3
"""
tools/validate_msmt17_frontier.py

Automated script to validate the decision-support rule table (Table 3) on MSMT17.
Runs the 5 non-dominated frontier configurations on MSMT17:
1. Full Fine-Tuning Baseline
2. LoRA 0–11 (r=8, α=16)
3. LoRA 4–11 (r=32, α=64)
4. LoRA 6–11 (r=16, α=32)
5. SSF 0–11 (Case 2)

Answers Reviewer Concern W2 cleanly without requiring a full second grid sweep.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import cfg
from config.loader import merge_config_file
from datasets.download_msmt17 import ensure_msmt17
from datasets.make_dataloader import make_dataloader
from loss.make_loss import make_loss
from model.make_model import make_model
from processor.processor import do_train
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler

MSMT17_FRONTIER_CONFIGS = [
    ("Full FT Baseline", "configs/MSMT17/vit_transreid_full_ft.yml"),
    ("LoRA 0–11 (r=8, α=16)", "configs/MSMT17/lora_blocks_0_11_r8.yml"),
    ("LoRA 4–11 (r=32, α=64)", "configs/MSMT17/lora_blocks_4_11_r32.yml"),
    ("LoRA 6–11 (r=16, α=32)", "configs/MSMT17/lora_blocks_6_11_r16.yml"),
    ("SSF 0–11 (Case 2)", "configs/MSMT17/ssf_0_11_case2.yml"),
]


def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def run_experiment(config_path: str, cpu_only: bool = False, max_epochs: int | None = None) -> dict:
    cfg.defrost()
    merge_config_file(cfg, config_path)
    if cpu_only:
        cfg.MODEL.DEVICE = "cpu"
        cfg.DATALOADER.NUM_WORKERS = 0
    if max_epochs is not None:
        cfg.SOLVER.MAX_EPOCHS = max_epochs
    cfg.freeze()

    logger = setup_logger("transreid.train")
    logger.info(f"Loaded config: {config_path}")

    # Ensure dataset is present
    try:
        ensure_msmt17(root=str(REPO_ROOT / "data"))
    except Exception as exc:
        logger.warning(f"Could not auto-download MSMT17: {exc}. Proceeding with existing data path.")

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    start_time = time.time()
    if torch.cuda.is_available() and not cpu_only:
        torch.cuda.reset_peak_memory_stats()

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query,
        local_rank=0,
    )

    elapsed_time = time.time() - start_time
    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available() and not cpu_only
        else 0.0
    )

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "param_ratio": 100.0 * trainable_params / total_params,
        "peak_vram": peak_vram,
        "time_seconds": elapsed_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Rule Table on MSMT17 Frontier Configs")
    parser.add_argument("--cpu-only", action="store_true", help="Run on CPU for testing")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count for quick testing")
    args = parser.parse_args()

    print("==========================================================================")
    print("      MSMT17 FRONTIER RULE-TABLE VALIDATION (TransReID-PEFT)")
    print("==========================================================================")

    results = []
    for name, config_rel in MSMT17_FRONTIER_CONFIGS:
        config_path = str(REPO_ROOT / config_rel)
        print(f"\n---> Running: {name} [{config_rel}]")
        res = run_experiment(config_path, cpu_only=args.cpu_only, max_epochs=args.epochs)
        res["name"] = name
        results.append(res)

    print("\n" + "=" * 75)
    print("                 MSMT17 FRONTIER VALIDATION RESULTS                      ")
    print("=" * 75)
    print(f"{'Configuration':<25} | {'Params (%)':<10} | {'VRAM (GB)':<10} | {'Time (s)':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<25} | {r['param_ratio']:>9.2f}% | {r['peak_vram']:>9.2f} | {r['time_seconds']:>9.1f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
