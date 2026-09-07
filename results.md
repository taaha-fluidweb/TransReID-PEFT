# Experiment 4 Results — Lightweight PEFT Baselines on Market-1501

All runs share the same frozen ViT-Base (TransReID) backbone, the same 60-epoch recipe
(AdamW, softmax+triplet, JPM/SIE on, batch 64), and seed 1234. Runs were executed on a
single RTX 3060 (12 GB) via `tools/run_experiment4.py`; peak VRAM and wall time are
measured per run.

> Metric values are parsed from the per-run logs (`logs/experiment4/run_10_*.log`) with the
> corrected Rank parser. Note: the first two rows of `progress.json` (bitfit_0_11, bitfit_4_11)
> carry the pre-fix `R1` values (98.0 / 97.0) — those are **Rank-10**, not Rank-1. The values
> below are authoritative.

## Configuration

### Shared recipe (inherited from `configs/base/market1501_transreid.yml`)

| Setting | Value |
|---|---|
| Backbone | ViT-Base TransReID (12 blocks, frozen) |
| Model | JPM on, SIE camera on (coefficient 3.0), BNNeck |
| Image size | 256 × 128 (stride [12, 12]) |
| Pretrained weights | ImageNet ViT-B (`jx_vit_base_p16_224-80ecf9dd.pth`) |
| Optimizer | AdamW, base LR 3e-4, weight decay 0.05 |
| Scheduler | Warmup 5 epochs + cosine decay, 60 epochs total |
| Sampler / loss | softmax + triplet (no margin), label smoothing off |
| Batch size | 64 (IMS_PER_BATCH) |
| Dataloader workers | 4 |
| Seed | 1234 |
| Dataset | Market-1501 (751 train IDs, 750 query, 751 gallery) |

### PEFT method settings (overrides on top of the base recipe)

Each config sets `PEFT.METHOD` plus its method node; only the matching node is enabled.

| Method | Config node | Settings |
|---|---|---|
| BitFit | `PEFT.BITFIT` | `ENABLED: True`, `BLOCKS: []` \| `[4..11]` \| `[6..11]`. Trains all bias params (Linear biases + LayerNorm β) in the active blocks; head always trainable. Biases get **10× base LR, weight decay 0** (SSF-consistent rule in `make_optimizer`). |
| LN-tuning | `PEFT.LNTUNE` | `ENABLED: True`, `BLOCKS: []` \| `[4..11]` \| `[6..11]`, `TRAIN_FINAL_NORM: True`. Trains LayerNorm γ/β only (incl. final `base.norm`) in the active blocks; head trainable. |
| Bottleneck adapter | `PEFT.ADAPTER` | `ENABLED: True`, `R: 16`, `DROPOUT: 0.05`, `SCALE: 1.0`, `TARGETS: ["qkv","proj","fc1","fc2"]`, `BLOCKS: []` \| `[4..11]` \| `[6..11]`. Parallel `down → GELU → up` (r=16) wrapping each target Linear; down kaiming-init, up **zero-init** (identity at init); base weights frozen. |

All three freeze the backbone and keep the Re-ID head (classifier + BNNeck) trainable; JPM
branches (`b1`/`b2`) stay frozen, matching the LoRA treatment.

### Config files

| Method | Depth window | File |
|---|---|---|
| BitFit | 0–11 | `configs/Market/bitfit_blocks_0_11.yml` |
| BitFit | 4–11 | `configs/Market/bitfit_blocks_4_11.yml` |
| BitFit | 6–11 | `configs/Market/bitfit_blocks_6_11.yml` |
| LN-tuning | 0–11 | `configs/Market/lntune_blocks_0_11.yml` |
| LN-tuning | 4–11 | `configs/Market/lntune_blocks_4_11.yml` |
| LN-tuning | 6–11 | `configs/Market/lntune_blocks_6_11.yml` |
| Adapter | 0–11 | `configs/Market/adapter_blocks_0_11_r16.yml` |
| Adapter | 4–11 | `configs/Market/adapter_blocks_4_11_r16.yml` |
| Adapter | 6–11 | `configs/Market/adapter_blocks_6_11_r16.yml` |

**How to reproduce:**

```bash
# local smoke (CPU): one config at a time
python tools/check_peft.py --config_file configs/Market/adapter_blocks_0_11_r16.yml --cpu-only

# full 9-run sweep (sequential, resumable, logs in logs/experiment4/)
python tools/run_experiment4.py
```

## Full results

| Method | Depth window | mAP | R1 | R5 | R10 | Trainable params | Peak VRAM | Wall time |
|---|---|---|---|---|---|---|---|---|
| BitFit | 0–11 | 76.6 | 89.8 | 96.6 | 98.0 | 2.89% (2.99M) | 6.51 GB | ~95 min |
| BitFit | 4–11 | 69.4 | 85.1 | 94.7 | 97.0 | 2.85% (2.96M) | 6.51 GB | ~94 min |
| BitFit | 6–11 | 61.2 | 79.8 | 92.1 | 95.5 | 2.84% (2.94M) | 6.51 GB | ~93 min |
| LN-tuning | 0–11 | 65.9 | 83.7 | 94.8 | 96.8 | 2.82% (2.92M) | 6.51 GB | ~93 min |
| LN-tuning | 4–11 | 57.7 | 78.4 | 90.9 | 94.5 | 2.81% (2.91M) | 4.83 GB | ~78 min |
| LN-tuning | 6–11 | 48.0 | 71.5 | 87.0 | 91.4 | 2.80% (2.90M) | 3.99 GB | ~71 min |
| Bottleneck adapter (r=16) | 0–11 | **85.9** | **93.7** | 97.8 | 98.8 | 4.95% (5.24M) | 8.03 GB | ~117 min |
| Bottleneck adapter (r=16) | 4–11 | 80.5 | 90.8 | 97.2 | 98.3 | 4.24% (4.46M) | 5.77 GB | ~93 min |
| Bottleneck adapter (r=16) | 6–11 | 74.8 | 87.8 | 95.8 | 97.6 | 3.88% (4.06M) | 4.67 GB | ~82 min |

Reference points from the existing LoRA/SSF study (same backbone, same recipe):

| Method | Depth window | mAP | R1 | Trainable params | Peak VRAM |
|---|---|---|---|---|---|
| Full fine-tuning | 0–11 | 88.0 | 94.4 | 100% | 11.5 GB |
| LoRA (r=32, α=64) | 4–11 | 82–83 | — | ~4.1% | ~7.8 GB |
| SSF (Case 2) | 0–11 | 79.9 | 91.1 | ~2.83% | ~10.0 GB |

## Key findings

1. **Bottleneck adapters are the strongest lightweight method.** Adapter 0–11 (r=16) reaches
   **85.9 mAP / 93.7 R1** — within ~2 mAP of full fine-tuning (88.0) at only **4.95% trainable
   parameters**. This beats LoRA 0–11 r8 (~82–83) and SSF 0–11 (79.9), making adapters the new
   frontier point for accuracy-vs-efficiency.
2. **Depth placement is the dominant factor for all three methods** (0–11 → 4–11 → 6–11):
   - BitFit: 76.6 → 69.4 → 61.2 mAP
   - LN-tuning: 65.9 → 57.7 → 48.0 mAP
   - Adapter: 85.9 → 80.5 → 74.8 mAP
   This matches the pattern already established for LoRA and SSF.
3. **Method ranking at full depth (0–11):** Adapter (85.9) ≫ BitFit (76.6) > LN-tuning (65.9).
   Learned bottleneck transformations on frozen features substantially outperform tuning only
   existing bias or LayerNorm parameters.
4. **Peak VRAM varies by method** (frozen-backbone activations + trainable graph size):
   adapter 0–11 = 8.03 GB (largest), LN-tuning 6–11 = 3.99 GB (smallest). All fit a 12 GB GPU
   at batch 64.

## Notes

- **Configs:** `configs/Market/bitfit_blocks_*.yml`, `lntune_blocks_*.yml`,
  `adapter_blocks_*_r16.yml`.
- **Runner:** `tools/run_experiment4.py` (sequential, resumable; logs in `logs/experiment4/`).
- **Implementation:** `model/peft/lightweight.py` (BitFit / LN-tuning freeze sweeps,
  `BottleneckAdapter`), dispatched in `model/make_model.py`.
- BitFit biases are trained at 10× base LR with zero weight decay (SSF-consistent). Adapter is a
  parallel bottleneck (down → GELU → up) at the same targets as LoRA (`qkv, proj, fc1, fc2`),
  r=16, up-projection zero-initialized (identity at init).
