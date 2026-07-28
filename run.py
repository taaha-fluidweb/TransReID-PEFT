#!/usr/bin/env python3
"""
One-shot bootstrap for vast.ai / Jupyter (zip upload workflow).

Upload TransReID-PEFT.zip (+ market1501.zip) to Jupyter, then run:

    python run.py

Or with options:

    python run.py --config configs/Market/ssf_0_11_case1.yml
    python run.py --setup-only
    python run.py --skip-unzip --skip-setup

Jupyter notebook:

    !python run.py
    # or after unzipping into the repo folder:
    %run run.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
REPO_ZIP_NAMES = ("TransReID-PEFT.zip", "transreid-peft.zip")
DATASET_ZIP_NAMES = ("market1501.zip", "Market-1501.zip", "Market1501.zip")
WEIGHT_DIR = REPO_DIR / ".cache" / "torch" / "checkpoints"
WEIGHT_FILE = WEIGHT_DIR / "jx_vit_base_p16_224-80ecf9dd.pth"
VIT_URL = (
    "https://github.com/rwightman/pytorch-image-models/releases/download/"
    "v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
)
MARKET_DIR = REPO_DIR / "data" / "market1501"
DEFAULT_CONFIG = "configs/Market/lora_blocks_4_11_r32.yml"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd or REPO_DIR)


def _find_zip(search_dirs: list[Path], names: tuple[str, ...]) -> Path | None:
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def unzip_file(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def resolve_repo_dir() -> Path:
    """Return repo root — handles running from parent dir after zip extract."""
    if (REPO_DIR / "train.py").is_file():
        return REPO_DIR
    nested = REPO_DIR / "TransReID-PEFT"
    if (nested / "train.py").is_file():
        return nested
    return REPO_DIR


def unzip_archives(skip_unzip: bool) -> Path:
    global REPO_DIR, WEIGHT_DIR, WEIGHT_FILE, MARKET_DIR

    if skip_unzip:
        REPO_DIR = resolve_repo_dir()
        WEIGHT_DIR = REPO_DIR / ".cache" / "torch" / "checkpoints"
        WEIGHT_FILE = WEIGHT_DIR / "jx_vit_base_p16_224-80ecf9dd.pth"
        MARKET_DIR = REPO_DIR / "data" / "market1501"
        return REPO_DIR

    cwd = Path.cwd()
    search_dirs = [cwd, REPO_DIR, REPO_DIR.parent, cwd.parent]

    repo_zip = _find_zip(search_dirs, REPO_ZIP_NAMES)
    if repo_zip and not (REPO_DIR / "train.py").is_file():
        # Extract next to the zip so TransReID-PEFT/ appears
        unzip_file(repo_zip, repo_zip.parent)

    REPO_DIR = resolve_repo_dir()
    WEIGHT_DIR = REPO_DIR / ".cache" / "torch" / "checkpoints"
    WEIGHT_FILE = WEIGHT_DIR / "jx_vit_base_p16_224-80ecf9dd.pth"
    MARKET_DIR = REPO_DIR / "data" / "market1501"

    if not (MARKET_DIR / "bounding_box_train").is_dir():
        dataset_zip = _find_zip(search_dirs, DATASET_ZIP_NAMES)
        if dataset_zip:
            data_dir = REPO_DIR / "data"
            unzip_file(dataset_zip, data_dir)
            # Normalize layout: zip may contain market1501/ or flat folders
            if not (MARKET_DIR / "bounding_box_train").is_dir():
                for child in data_dir.iterdir():
                    if child.is_dir() and (child / "bounding_box_train").is_dir():
                        if child.name != "market1501":
                            target = data_dir / "market1501"
                            if target.exists():
                                break
                            child.rename(target)
                        break

    return REPO_DIR


def setup(skip_smoke: bool = False) -> None:
    print(f"\n==> TransReID-PEFT setup")
    print(f"    Repo: {REPO_DIR}")
    print(f"    Python: {sys.version.split()[0]}")

    import torch

    print(f"    PyTorch: {torch.__version__}")
    req_file = "requirements-vast.txt" if torch.cuda.is_available() else "requirements.txt"
    if torch.cuda.is_available():
        print(f"    CUDA OK: {torch.cuda.get_device_name(0)}")
    else:
        print("    WARNING: CUDA not available — using full requirements.txt")

    run([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])

    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    if WEIGHT_FILE.exists():
        print(f"    ViT weights present: {WEIGHT_FILE.name}")
    else:
        print("    Downloading ViT-Base pretrained weights...")
        urllib.request.urlretrieve(VIT_URL, WEIGHT_FILE)
        print(f"    Saved to {WEIGHT_FILE}")

    if (MARKET_DIR / "bounding_box_train").is_dir():
        print("    Market-1501: OK")
    else:
        print(
            "    WARNING: data/market1501/ not found.\n"
            "    Upload market1501.zip alongside the repo zip, or extract manually."
        )

    if not skip_smoke:
        print("    Running smoke test...")
        run([
            sys.executable,
            "tools/check_peft.py",
            "--config_file",
            DEFAULT_CONFIG,
            "--cpu-only",
        ])


def train(config: str, extra_args: list[str]) -> None:
    if not (MARKET_DIR / "bounding_box_train").is_dir():
        raise SystemExit(
            "Cannot train: Market-1501 not found at data/market1501/.\n"
            "Upload market1501.zip and re-run: python run.py"
        )

    cmd = [
        sys.executable,
        "train.py",
        "--config_file",
        config,
        "MODEL.DEVICE_ID",
        "('0')",
        *extra_args,
    ]
    print(f"\n==> Starting training")
    print(f"    Config: {config}")
    run(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unzip, setup, and train TransReID-PEFT")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Training config YAML (default: LoRA 4-11 r32)",
    )
    parser.add_argument("--skip-unzip", action="store_true", help="Skip zip extraction")
    parser.add_argument("--skip-setup", action="store_true", help="Skip dependency install / weights")
    parser.add_argument("--setup-only", action="store_true", help="Setup only, do not train")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke test during setup")
    parser.add_argument("extra", nargs="*", help="Extra args passed to train.py")
    args = parser.parse_args()

    print("==> TransReID-PEFT run.py")
    repo = unzip_archives(skip_unzip=args.skip_unzip)
    os.chdir(repo)

    if not args.skip_setup:
        setup(skip_smoke=args.skip_smoke)

    if args.setup_only:
        print("\nSetup complete (--setup-only).")
        return 0

    train(args.config, args.extra)
    print("\nTraining finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
