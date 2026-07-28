# TransReID-PEFT: Unified LoRA + SSF on TransReID

Single repository for parameter-efficient fine-tuning (PEFT) experiments on the TransReID ViT backbone, combining:

- **LoRA** — low-rank adapters on `{qkv, proj, fc1, fc2}` ([Huzaifa9559/LoRa-on-Transreid](https://github.com/Huzaifa9559/LoRa-on-Transreid))
- **SSF** — scale-and-shift on activations ([TameemaRehman/SSF_TransReID](https://github.com/TameemaRehman/SSF_TransReID))

Only one PEFT method runs at a time, selected via `PEFT.METHOD` in the config.

## Setup

```bash
pip install -r requirements.txt
```

Place Market-1501 under `data/market1501/`. Download the ViT-Base ImageNet checkpoint to `.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth`.

## Training

```bash
# LoRA — blocks 4–11, r=32, alpha=64 (paper best compromise)
python train.py --config_file configs/Market/lora_blocks_4_11_r32.yml MODEL.DEVICE_ID "('0')"

# SSF — full depth, Case 1 (LoRA-matched optimizer)
python train.py --config_file configs/Market/ssf_0_11_case1.yml MODEL.DEVICE_ID "('0')"

# SSF — full depth, Case 2 (SSF paper optimizer)
python train.py --config_file configs/Market/ssf_0_11_case2.yml MODEL.DEVICE_ID "('0')"
```

## Evaluation

```bash
python test.py --config_file configs/Market/lora_blocks_4_11_r32.yml \
  TEST.WEIGHT path/to/checkpoint.pth MODEL.DEVICE_ID "('0')"
```

LoRA checkpoints saved with `SAVE_ADAPTER_ONLY: True` use the `{"adapters": ...}` format and load automatically.

## Config layout

| File | Description |
|------|-------------|
| `configs/base/market1501_transreid.yml` | Shared 60-epoch AdamW recipe (JPM=True, SIE=True) |
| `configs/Market/lora_*.yml` | LoRA depth/rank variants |
| `configs/Market/ssf_*.yml` | SSF depth and optimizer-case variants |

Set `PEFT.METHOD` to `'none'`, `'lora'`, or `'ssf'`. Legacy configs with top-level `LORA.ENABLED: True` are still accepted.

## Diagnostics

```bash
python tools/check_peft.py --config_file configs/Market/lora_blocks_4_11_r32.yml --cpu-only
pytest tests/
```

## Project structure

```
model/peft/lora.py   — LoRA injection (post-hoc linear wrappers)
model/peft/ssf.py    — SSF modules (built into vit_pytorch Block)
config/peft_config.py — method selection, legacy compat, SSF Case 2 overrides
```

See [docs/PEFT_IMPLEMENTATION.md](docs/PEFT_IMPLEMENTATION.md) for integration details.

## Citation

If you use this code, please cite the paper and the original TransReID repository.
