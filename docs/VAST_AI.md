# Running on vast.ai (Jupyter + zip upload)

Upload **`TransReID-PEFT.zip`** and **`market1501.zip`** through the Jupyter file browser, then run one command.

## Quick start (Jupyter)

**Terminal** (from the folder where you uploaded the zips):

```bash
unzip -o TransReID-PEFT.zip
python TransReID-PEFT/run.py
```

**Notebook** (single cell):

```python
!unzip -o TransReID-PEFT.zip
!python TransReID-PEFT/run.py
```

`run.py` will automatically:

1. Unzip `market1501.zip` if present (into `data/market1501/`)
2. Install dependencies (`requirements-vast.txt` on GPU templates)
3. Download ViT-Base weights if missing
4. Run a smoke test
5. Start training (default: LoRA blocks 4–11, r=32)

## Options

```bash
# SSF instead of LoRA
python run.py --config configs/Market/ssf_0_11_case1.yml

# Setup only (no training)
python run.py --setup-only

# Already unzipped and set up — train only
python run.py --skip-unzip --skip-setup

# Save checkpoints to a persistent volume
python run.py --config configs/Market/lora_blocks_4_11_r32.yml \
  OUTPUT_DIR /workspace/logs/lora_4_11
```

## What to upload

| File | Required | Purpose |
|------|----------|---------|
| `TransReID-PEFT.zip` | Yes | This repo |
| `market1501.zip` | Yes (for training) | Dataset; extracted to `data/market1501/` |

Zip naming: `market1501.zip`, `Market-1501.zip`, or `Market1501.zip` all work.

## Instance requirements

- **Template:** PyTorch + CUDA preinstalled
- **VRAM:** ≥20 GB recommended
- **Jupyter** enabled on vast.ai

## Evaluate after training

```bash
cd TransReID-PEFT
python test.py \
  --config_file configs/Market/lora_blocks_4_11_r32.yml \
  TEST.WEIGHT ../logs/market_vit_transreid_lora_4_11/transformer_60.pth \
  MODEL.DEVICE_ID "('0')"
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Cannot train: Market-1501 not found` | Upload `market1501.zip` next to the repo zip and re-run |
| CUDA not available | Use a PyTorch+CUDA vast.ai template |
| Out of memory | `--config configs/Market/lora_blocks_6_11_r32.yml` or lower batch size |
