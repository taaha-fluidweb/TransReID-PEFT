# PEFT Implementation Notes

## Architecture

TransReID-PEFT supports two mutually exclusive PEFT methods on the same codebase:

| Method | Injection | Config key | Checkpoint |
|--------|-----------|------------|------------|
| LoRA | Post-hoc `LoRALinear` wrappers on ViT linear layers | `PEFT.METHOD: lora` | Adapter-only optional |
| SSF | `SSF` modules in `Block.forward()` at construction | `PEFT.METHOD: ssf` | Full state dict |

## Config flow

1. `merge_config_file()` resolves `_BASE_` inheritance in YAML.
2. `normalize_peft_config()` maps legacy `LORA.ENABLED`, enforces mutual exclusion, applies SSF Case 2 optimizer overrides.
3. `make_model()` builds SSF at construction when `PEFT.METHOD == ssf`, injects LoRA after build when `PEFT.METHOD == lora`.

## Block coverage

Both methods support partial depth:

- **0–11**: all 12 blocks (LoRA: empty `BLOCKS`; SSF: empty `BLOCKS: ()`)
- **4–11**: `BLOCKS: [4,5,6,7,8,9,10,11]` or `BLOCKS: (4,5,6,7,8,9,10,11)`
- **6–11**: last six blocks only

## Optimizer rules

- **LoRA / full FT**: standard `BASE_LR` and `WEIGHT_DECAY` on all trainable params.
- **SSF**: params with `ssf` in name get `LR = 10 × BASE_LR` (or `PEFT.SSF.LR` if > 0) and `weight_decay = 0`.
- **SSF Case 2** (`OPTIMIZER_CASE: 2`): overrides to AdamW with `BASE_LR=3.5e-4`, `WEIGHT_DECAY=1e-4`, `BIAS_LR_FACTOR=2`.

## Checkpoint formats

- **LoRA adapter-only**: `{"adapters": {<qualified_name>.lora_A: Tensor, ...}}`
- **Full model**: standard PyTorch state dict (SSF and full fine-tuning)

`load_param()` in `make_model.py` detects the `adapters` key automatically.

## Source repos

This unified repo merges:

- [LoRa-on-Transreid](https://github.com/Huzaifa9559/LoRa-on-Transreid) — LoRA post-hoc injection, adapter checkpointing
- [SSF_TransReID](https://github.com/TameemaRehman/SSF_TransReID) — SSF in `vit_pytorch.py`, JPM freeze order

Original repos remain unchanged as reference implementations.
