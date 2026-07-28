# Running on vast.ai (Jupyter + zip upload)

Upload **`TransReID-PEFT.zip`** to Jupyter, then run one command. **Market-1501 downloads automatically** (~153 MB from Google Drive).

## Quick start (Jupyter)

**Terminal** (from the folder where you uploaded the zip):

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

1. Unzip the repo (if needed)
2. **Download Market-1501** (Google Drive, with HTTP mirror fallback)
3. Install dependencies (`requirements-vast.txt` on GPU templates)
4. Download ViT-Base weights if missing
5. Run a smoke test
6. Start training (default: LoRA blocks 4–11, r=32)

## Optional: upload dataset zip to skip download

If you already have `market1501.zip`, upload it next to the repo zip. Setup will use it instead of downloading.

Supported names: `market1501.zip`, `Market-1501.zip`, `Market1501.zip`, `Market-1501-v15.09.15.zip`

## Options

```bash
python run.py --config configs/Market/ssf_0_11_case1.yml
python run.py --setup-only
python run.py --setup-only --download-all-datasets   # Duke + Occ-Duke too
python run.py --skip-unzip --skip-setup
```

## Datasets

| Dataset | Auto-download | Notes |
|---------|---------------|-------|
| **Market-1501** | Yes (default) | ~153 MB, Google Drive |
| **DukeMTMC-reID** | `--download-all-datasets` | ~2.2 GB zip from Google Drive |
| **Occluded-Duke** | `--download-all-datasets` | Built from Duke + [official ICCV19 split lists](https://github.com/lightas/ICCV19_Pose_Guided_Occluded_Person_ReID/tree/master/dataset) |

Dataset loaders also download on first use when you train with `duke` or `occ_duke` configs.

Optional local zips (upload to skip download):

- `market1501.zip`, `Market-1501.zip`, …
- `DukeMTMC-reID.zip`, `dukemtmcreid.zip`, `duke.zip`

Final layouts:

```
data/market1501/
data/dukemtmcreid/
data/Occluded_Duke/
├── bounding_box_train/
├── bounding_box_test/
└── query/
```

## Dataset source (Market-1501)

Market-1501 is downloaded from the [official Zheng Lab Google Drive](https://zheng-lab-anu.github.io/Project/project_reid.html) (same dataset used in the paper).

Final layout:

```
data/market1501/
├── bounding_box_train/
├── bounding_box_test/
└── query/
```

## Instance requirements

- **Template:** PyTorch + CUDA preinstalled
- **VRAM:** ≥20 GB recommended
- **Disk:** ~500 MB free for dataset + weights

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Google Drive download fails | Upload `market1501.zip` manually; setup will use it |
| CUDA not available | Use a PyTorch+CUDA vast.ai template |
| Out of memory | `--config configs/Market/lora_blocks_6_11_r32.yml` |
