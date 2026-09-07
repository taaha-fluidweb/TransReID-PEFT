# Efficient Person Re-Identification via LoRA and SSF: A Comparative PEFT Study on ViT Backbone in TransReID

Official repository for the paper: **"Efficient Person Re-Identification via LoRA and SSF: A Comparative PEFT Study on ViT Backbone in TransReID"** (Naseer et al., 2025).

This repository provides a unified parameter-efficient fine-tuning (PEFT) framework for Vision Transformer (ViT) backbones on TransReID, featuring **LoRA**, **SSF**, **BitFit**, **LN-Tuning**, **Bottleneck Adapters**, **Classification Control (Framing 6)**, and **MSMT17 cross-dataset validation**.

---

## 1. Abstract & Key Contributions

Vision Transformer (ViT) backbones such as TransReID achieve strong performance for person re-identification (Re-ID), but full fine-tuning is expensive in memory and compute. Parameter-efficient fine-tuning (PEFT) addresses this by freezing the backbone and learning a small set of task-specific parameters. We conduct a systematic comparison of five structurally distinct PEFT methods on TransReID:

- **LoRA** — low-rank adapters on `{qkv, proj, fc1, fc2}`
- **SSF** — per-channel scale and shift on activations
- **Bottleneck Adapters** — parallel down-projection + GELU + up-projection bottleneck ($r=16$)
- **BitFit & LN-Tuning** — lightweight tuning of bias terms and LayerNorm parameters

### Key Findings:
1. **Bottleneck Adapters reach near-baseline performance**: 0–11 adapters ($r=16$) achieve **85.9% mAP / 93.7% Rank-1** (within ~2 mAP of Full FT at 4.95% trainable parameters), outperforming LoRA (85.8 mAP) and SSF (79.9 mAP).
2. **LoRA 4–11 is the optimal memory-accuracy trade-off**: Achieves **83.2% mAP / 92.8% Rank-1** at **7.84 GB VRAM** (a ~30% memory reduction).
3. **Objective-Driven Adaptation Gap (Classification Control / Framing 6)**: Under standard Softmax Cross-Entropy classification, LoRA 0–11 ($r=8$) reaches **82.0% mAP / 92.6% Rank-1**, outperforming Full FT (**80.5% mAP / 92.1% Rank-1**) by **+1.5% mAP**, proving that the PEFT gap under Re-ID is objective-driven (triplet manifold restructuring).

---

## 2. Experimental Results Summary (Market-1501)

### Standard Re-ID Protocol (60 Epochs, AdamW)

| Method | Block Coverage | Trainable Params (%) | Peak VRAM (GB) | mAP (%) | Rank-1 (%) | Rank-5 (%) |
|---|---|---|---|---|---|---|
| **Full Fine-Tuning Baseline** | 0–11 | 100.00% | 11.5 GB | **88.0%** | **94.4%** | **98.2%** |
| **Bottleneck Adapter (r=16)** | 0–11 | 4.95% | 8.03 GB | **85.9%** 🏆 | **93.7%** 🏆 | **97.8%** |
| **LoRA (r=8, α=16)** | 0–11 | 1.99% | 11.4 GB | **85.8%** | **93.5%** | **98.0%** |
| **LoRA (r=32, α=64)** | 4–11 | 4.12% | 7.84 GB ⚡ | **83.2%** | **92.8%** | **97.8%** |
| **Bottleneck Adapter (r=16)** | 4–11 | 4.24% | 5.77 GB ⚡ | **80.5%** | **90.8%** | **97.2%** |
| **SSF (Case 2)** | 0–11 | 2.83% | 10.0 GB | **79.9%** | **91.1%** | **97.1%** |
| **BitFit** | 0–11 | 2.89% (0.12% backbone) | 6.51 GB | **76.6%** | **89.8%** | **96.6%** |
| **LN-Tuning** | 0–11 | 2.82% (0.04% backbone) | 6.51 GB | **65.9%** | **83.7%** | **94.8%** |

### Classification Control Results (Framing 6 — Softmax Cross-Entropy)

| Model | Method | Block Coverage | Trainable Params (%) | Peak VRAM | mAP (%) | Rank-1 (%) |
|---|---|---|---|---|---|---|
| **Full FT Baseline** | Full FT | 0–11 | 100.00% | 7.91 GB | 80.5% | 92.1% |
| **LoRA 0–11 ($r=8, \alpha=16$)** | LoRA | 0–11 | **1.99%** | 8.14 GB | **82.0%** (+1.5%) 🏆 | **92.6%** (+0.5%) 🏆 |
| **LoRA 4–11 ($r=32, \alpha=64$)** | LoRA | 4–11 | **4.12%** | **5.68 GB** ⚡ | **79.6%** | **90.7%** |

---

## 3. Environment & Quickstart Setup

### Installation

```bash
pip install -r requirements.txt
```

### Dataset Preparation
Place datasets under `data/`:
```
data/
└── market1501/
    ├── bounding_box_train/
    ├── bounding_box_test/
    └── query/
```
Automated dataset downloaders are available in `datasets/`:
```bash
python datasets/download_market1501.py
```

---

## 4. Running Experiments

### A. Training Single Models

```bash
# LoRA — blocks 4–11, r=32, alpha=64 (best compromise)
python train.py --config_file configs/Market/lora_blocks_4_11_r32.yml MODEL.DEVICE_ID "('0')"

# SSF — full depth (Case 2 official recipe)
python train.py --config_file configs/Market/ssf_0_11_case2.yml MODEL.DEVICE_ID "('0')"

# Bottleneck Adapters — full depth (r=16)
python train.py --config_file configs/Market/adapter_blocks_0_11_r16.yml MODEL.DEVICE_ID "('0')"

# Full Fine-Tuning Baseline
python train.py --config_file configs/Market/baselines/vit_transreid.yml MODEL.DEVICE_ID "('0')"
```

### B. Automated Benchmark Suite Runners

```bash
# Dry-run diagnostic check (CPU or GPU)
python tools/check_peft.py --config_file configs/Market/adapter_blocks_0_11_r16.yml --cpu-only

# Run full Lightweight PEFT Baselines suite (BitFit, LN-Tuning, Adapters)
python tools/run_experiment4.py

# Run Classification Control benchmark suite
python tools/run_classification_control.py

# Run MSMT17 Frontier validation suite
python tools/validate_msmt17_frontier.py
```

### C. Testing & Evaluation

```bash
python test.py --config_file configs/Market/lora_blocks_4_11_r32.yml \
  TEST.WEIGHT path/to/checkpoint.pth MODEL.DEVICE_ID "('0')"
```

### D. Unit Tests

```bash
python -m pytest tests/
```

---

## 5. Documentation & Detailed Manuscripts

- [research.md](research.md): Full academic manuscript detailing Methodology, Experiments, Configuration Maps, and Limitations.
- [results.md](results.md): Complete reviewer replication logs, hyperparameter matrices, and VRAM benchmarks.
- [docs/PEFT_IMPLEMENTATION.md](docs/PEFT_IMPLEMENTATION.md): Deep-dive into model injection architecture.
- [docs/VAST_AI.md](docs/VAST_AI.md): Setup guide for cloud GPU training (vast.ai).

---

## 6. Citation

If you use this repository or code, please cite our study and the original TransReID:

```bibtex
@article{naseer2025peft,
  title   = {Efficient Person Re-Identification via LoRA and SSF: A Comparative PEFT Study on ViT Backbone in TransReID},
  author  = {Naseer, Huzaifa and Rehman, Tameema and Ashfaq, Anas and Syed, Farrukh Hasan},
  year    = {2025}
}

@InProceedings{He_2021_ICCV,
  author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
  title     = {TransReID: Transformer-Based Object Re-Identification},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  pages     = {15013-15022}
}
```

---

## 7. Acknowledgements

Built on top of [TransReID](https://github.com/damo-cv/TransReID). LoRA formulation follows Hu et al. (ICLR 2022), SSF follows Lian et al. (NeurIPS 2022), BitFit follows Zaken et al. (ACL 2022).
