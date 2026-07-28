"""PEFT config normalization and helpers."""

from typing import List, Optional, Set, Tuple


def _copy_legacy_lora(cfg):
    if not hasattr(cfg, "LORA") or not cfg.LORA.ENABLED:
        return
    for key in (
        "R", "ALPHA", "DROPOUT", "TARGETS", "TRAIN_HEAD",
        "MERGE_AT_EVAL", "BIAS", "SAVE_ADAPTER_ONLY", "BLOCKS",
    ):
        if hasattr(cfg.LORA, key):
            setattr(cfg.PEFT.LORA, key, getattr(cfg.LORA, key))
    if cfg.PEFT.METHOD == "none":
        cfg.PEFT.METHOD = "lora"


def normalize_peft_config(cfg):
    """Resolve legacy flags and enforce one PEFT method at a time."""
    if cfg.is_frozen():
        cfg.defrost()

    _copy_legacy_lora(cfg)

    if cfg.PEFT.SSF.ENABLED and cfg.PEFT.METHOD == "none":
        cfg.PEFT.METHOD = "ssf"

    if cfg.PEFT.METHOD == "lora" and cfg.PEFT.SSF.ENABLED:
        raise ValueError("Cannot enable both LoRA and SSF. Set PEFT.METHOD to one method only.")

    if hasattr(cfg, "LORA") and cfg.LORA.ENABLED and cfg.PEFT.METHOD == "ssf":
        raise ValueError("Cannot enable both LoRA and SSF. Set PEFT.METHOD to one method only.")

    if cfg.PEFT.METHOD == "ssf":
        cfg.PEFT.SSF.ENABLED = True
    elif cfg.PEFT.METHOD == "lora":
        cfg.PEFT.SSF.ENABLED = False
    else:
        cfg.PEFT.METHOD = "none"
        cfg.PEFT.SSF.ENABLED = False

    if cfg.PEFT.METHOD == "ssf" and cfg.PEFT.SSF.OPTIMIZER_CASE == 2:
        cfg.SOLVER.BASE_LR = 3.5e-4
        cfg.SOLVER.WEIGHT_DECAY = 1e-4
        cfg.SOLVER.WEIGHT_DECAY_BIAS = 1e-4
        cfg.SOLVER.BIAS_LR_FACTOR = 2


def get_peft_method(cfg) -> str:
    return str(cfg.PEFT.METHOD).lower()


def get_lora_blocks(cfg) -> Optional[List[int]]:
    blocks = list(cfg.PEFT.LORA.BLOCKS)
    return blocks if blocks else None


def get_ssf_blocks(cfg) -> Tuple:
    blocks = cfg.PEFT.SSF.BLOCKS
    return tuple(blocks) if blocks else ()


def get_active_blocks(cfg) -> Optional[Set[int]]:
    method = get_peft_method(cfg)
    if method == "lora":
        blocks = get_lora_blocks(cfg)
        return set(blocks) if blocks else None
    if method == "ssf":
        blocks = get_ssf_blocks(cfg)
        return set(blocks) if blocks else None
    return None
