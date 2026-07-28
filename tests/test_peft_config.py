"""Additional tests for unified PEFT config handling."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import cfg, normalize_peft_config


def test_mutual_exclusion_raises():
    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "lora"
    cfg_copy.PEFT.SSF.ENABLED = True
    with pytest.raises(ValueError, match="Cannot enable both"):
        normalize_peft_config(cfg_copy)


def test_legacy_lora_enabled_maps_to_method():
    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "none"
    cfg_copy.LORA.ENABLED = True
    cfg_copy.LORA.R = 16
    normalize_peft_config(cfg_copy)
    assert cfg_copy.PEFT.METHOD == "lora"
    assert cfg_copy.PEFT.LORA.R == 16


def test_ssf_case2_optimizer_overrides():
    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "ssf"
    cfg_copy.PEFT.SSF.OPTIMIZER_CASE = 2
    normalize_peft_config(cfg_copy)
    assert cfg_copy.SOLVER.BASE_LR == pytest.approx(3.5e-4)
    assert cfg_copy.SOLVER.WEIGHT_DECAY == pytest.approx(1e-4)
    assert cfg_copy.SOLVER.BIAS_LR_FACTOR == 2
