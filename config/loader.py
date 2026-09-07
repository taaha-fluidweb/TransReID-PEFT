import os

import yaml
from yacs.config import CfgNode as CN


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if key == "_BASE_":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_merged_yaml(config_file):
    data = _load_yaml(config_file)
    if "_BASE_" in data:
        base_path = os.path.normpath(os.path.join(os.path.dirname(config_file), data["_BASE_"]))
        data = _deep_merge(load_merged_yaml(base_path), data)
    return data


def merge_config_file(cfg, config_file):
    merged = load_merged_yaml(config_file)
    cfg.merge_from_other_cfg(CN(merged))
