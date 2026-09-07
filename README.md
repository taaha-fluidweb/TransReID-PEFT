# PEFT-on-ReID: Parameter-Efficient Fine-Tuning for Person Re-Identification on Vision Transformers

Official repository for the paper: **"PEFT-on-ReID: Parameter-Efficient Fine-Tuning for Person Re-Identification on Vision Transformers"** (Naseer et al., 2025).

Single repository for parameter-efficient fine-tuning (PEFT) experiments on the TransReID ViT backbone, evaluating **LoRA**, **SSF**, **BitFit**, **LN-Tuning**, **Bottleneck Adapters**, **Classification Control (Framing 6)**, and **MSMT17 cross-dataset validation**.

---

## 1. Overview

Vision Transformer (ViT) backbones such as TransReID achieve strong performance for person re-identification (Re-ID), but full fine-tuning is expensive in memory and compute. Parameter-efficient fine-tuning (PEFT) addresses this by freezing the backbone and learning a small set of task-specific parameters. We conduct a systematic comparison of five structurally distinct PEFT methods on TransReID evaluated under the standard Market-1501 protocol.

### Key Contributions:
- **Comprehensive PEFT Spectrum:** We compare weight-matrix updates (LoRA), activation-affine transformations (SSF), parallel bottleneck adapters, and lightweight parameter tuning (BitFit, LN-tuning) on a frozen ViT-Base backbone.
- **Classification Control (Framing 6):** We evaluate PEFT methods under pure Softmax Cross-Entropy classification, demonstrating that the PEFT accuracy gap in Re-ID is objective-driven (triplet manifold restructuring) rather than a backbone capacity bottleneck.
- **Automated Reproducibility:** Diagnostic inspectors, unit test suites, and automated benchmark runners enable full replication of all reported metrics.

---

## 2. PEFT Paradigms & Architectures

Within each adapted transformer block:
- **LoRA:** Injects low-rank decomposition matrices $W' = W + \frac{\alpha}{r}BA$ into linear projections `{qkv, proj, fc1, fc2}`.
- **SSF:** Applies per-channel scale and shift $y = \gamma \odot x + \beta$ post-activation after Attention, MLP, LayerNorm1, and LayerNorm2. Supports zero-FLOPs inference reparameterization.
- **Bottleneck Adapters:** Parallel bottleneck blocks $y = x + W_{up} \text{GELU}(W_{down} x)$ wrapping target linear layers ($r=16$, zero-initialized up-projection).
- **BitFit & LN-Tuning:** Freezes all backbone weights while training bias terms or LayerNorm parameters ($\gamma, \beta$) respectively.

The backbone weights, patch embeddings, Side-Information Embeddings (SIE), and Jigsaw Patch Module (JPM) parameters remain frozen; PEFT modules and the Re-ID head remain trainable.

---

## 3. Configuration Space

The PEFT selection is driven by `PEFT.METHOD` in the configuration files (`configs/Market/`, `configs/classification/`, `configs/MSMT17/`):

| Axis | Values | Description |
|------|--------|-------------|
| **PEFT Method** | `lora`, `ssf`, `adapter`, `bitfit`, `lntune` | Selected via `PEFT.METHOD` |
| **Depth Placement** | `0–11` (full), `4–11` (mid+late), `6–11` (late) | Controlled via `BLOCKS` array |
| **LoRA Rank $r$ & Scale $\alpha$** | $r \in \{8, 16, 32\}$, $\alpha \in \{16, 32, 48, 64\}$ | Recipe favors $\alpha \approx 2r$ |
| **Adapter Bottleneck $r$** | $r = 16$ | Parallel bottleneck dimension |
| **SSF Optimizer Cases** | Case 1 (LoRA-matched) vs. Case 2 (SSF paper) | Controls LR and weight decay rules |

---

## 4. Key Results (Market-1501)

### 4.1. Standard Person Re-ID Protocol

All runs: single GPU, 60 epochs, AdamW optimizer, cosine learning-rate decay, seeds fixed (`1234`). Evaluated on Market-1501 under single-query protocol.

| Method | Block Coverage | Trainable Params (%) | Peak VRAM (GB) | mAP (%) | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) |
|--------|:--------------:|:--------------------:|:--------------:|:-------:|:----------:|:----------:|:-----------:|
| **Baseline (Full FT)** | 0–11 | 100.00% | 11.5 | **88.0** | **94.4** | 98.2 | 99.0 |
| **Bottleneck Adapter (r=16)** | 0–11 | 4.95% | 8.03 | **85.9** | **93.7** | 97.8 | 98.8 |
| LoRA (r=8, α=16) | 0–11 | 1.99% | 11.4 | 85.8 | 93.5 | 98.0 | 98.9 |
| **LoRA (r=32, α=64)** | 4–11 | 4.12% | **7.84** | **83.2** | **92.8** | 97.8 | 98.5 |
| Bottleneck Adapter (r=16) | 4–11 | 4.24% | 5.77 | 80.5 | 90.8 | 97.2 | 98.3 |
| SSF (Case 2) | 0–11 | 2.83% | 10.0 | 79.9 | 91.1 | 97.1 | 98.1 |
| BitFit | 0–11 | 2.89% | 6.51 | 76.6 | 89.8 | 96.6 | 98.0 |
| LN-Tuning | 0–11 | 2.82% | 6.51 | 65.9 | 83.7 | 94.8 | 96.8 |

### 4.2. Objective-Driven Adaptation Gap (Classification Control / Framing 6)

Evaluated under Softmax Cross-Entropy loss directly on global ViT features (`NECK: 'no'`), removing pairwise metric learning losses:

| Configuration | Method | Block Coverage | Trainable Params (%) | Peak VRAM (GB) | mAP (%) | Rank-1 (%) | Rank-5 (%) |
|---------------|:------:|:--------------:|:--------------------:|:--------------:|:-------:|:----------:|:----------:|
| Full FT Baseline | Full FT | 0–11 | 100.00% | 7.91 | 80.5 | 92.1 | 97.5 |
| **LoRA 0–11 (r=8, α=16)** | LoRA | 0–11 | **1.99%** | 8.14 | **82.0** | **92.6** | **98.0** |
| LoRA 4–11 (r=32, α=64) | LoRA | 4–11 | 4.12% | 5.68 | 79.6 | 90.7 | 97.0 |
| LoRA 6–11 (r=16, α=32) | LoRA | 6–11 | 1.99% | 4.39 | 72.8 | 88.0 | 95.9 |
| SSF 0–11 (Case 2) | SSF | 0–11 | 2.83% | 9.38 | 74.0 | 89.4 | 97.0 |

### Practical Design Guidelines

| Constraint | Recommended Config | Expected Outcome |
|------------|--------------------|------------------|
| Maximize accuracy | Full FT or 0–11 Adapter r=16 | Full FT: mAP ≈ 88%, 11.5 GB; Adapter: mAP ≈ 85.9%, 8.03 GB |
| ~30% VRAM savings | 4–11 LoRA (r=32, α=64) | mAP ≈ 83.2%, 7.84 GB VRAM |
| Min. trainable params | 0–11 SSF (Case 2) | mAP ≈ 79.9%, ~2.83% params |
| Tight GPU (~7 GB) | 6–11 LoRA (r=16–32, α≈2r) | mAP ≈ 75–79%, 7–7.6 GB VRAM |
| Tiny parameter budget | 0–11 BitFit | mAP ≈ 76.6%, ~0.12% backbone params |

---

## 5. Setup

### Installation

```bash
pip install -r requirements.txt
```

See `CUDA_SETUP_GUIDE.md` for GPU driver/PyTorch environment diagnostics.

### Dataset Preparation

Place datasets under `data/`:
```
data/
└── market1501/
    ├── bounding_box_train/
    ├── bounding_box_test/
    └── query/
```

Automated dataset scripts are provided:
```bash
python datasets/download_market1501.py
```

### Pretrained Backbone

Download the ImageNet-pretrained ViT-Base checkpoint (`jx_vit_base_p16_224-80ecf9dd.pth`) and specify its path via `MODEL.PRETRAIN_PATH` in the configuration file.

---

## 6. Running Experiments

### 6.1. Single Model Training

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

### 6.2. Automated Suite Runners

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

### 6.3. Evaluation

```bash
python test.py --config_file configs/Market/lora_blocks_4_11_r32.yml \
    MODEL.DEVICE_ID "('0')" TEST.WEIGHT path/to/checkpoint.pth
```

### 6.4. Unit Testing

```bash
python -m pytest tests/
```

---

## 7. Config Reference

```yaml
PEFT:
  METHOD: "lora" # "none" | "lora" | "ssf" | "bitfit" | "lntune" | "adapter"

  LORA:
    R: 32
    ALPHA: 64
    DROPOUT: 0.05
    TARGETS: ["qkv", "proj", "fc1", "fc2"]
    BLOCKS: [4, 5, 6, 7, 8, 9, 10, 11]

  ADAPTER:
    R: 16
    DROPOUT: 0.05
    TARGETS: ["qkv", "proj", "fc1", "fc2"]
    BLOCKS: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

---

## 8. Citation

If you use this repository or code, please cite our study and the original TransReID:

```bibtex
@article{naseer2025peft,
  title   = {PEFT-on-ReID: Parameter-Efficient Fine-Tuning for Person Re-Identification on Vision Transformers},
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

## 9. Acknowledgements

Built on top of [TransReID](https://github.com/damo-cv/TransReID). LoRA formulation follows Hu et al. (ICLR 2022), SSF follows Lian et al. (NeurIPS 2022), BitFit follows Zaken et al. (ACL 2022).
