# Running on vast.ai

**Recommended:** `git clone` your branch directly on the instance. Zip upload still works as a fallback.

## Quick start (git clone — recommended)

```bash
git clone -b taaha https://github.com/taaha-fluidweb/TransReID-PEFT.git
cd TransReID-PEFT
python run.py --skip-unzip
```

After you push changes from your local machine:

```bash
cd TransReID-PEFT
git pull
python run.py --skip-unzip --config configs/Market/lora_blocks_4_11_r32.yml
```

Use `--skip-unzip` whenever you cloned or pulled — there is no zip to extract.

`run.py` will automatically:

1. **Download Market-1501** (Google Drive, with HTTP mirror fallback)
2. Install dependencies (`requirements-vast.txt` on GPU templates)
3. Download ViT-Base weights if missing
4. Run a smoke test
5. Start training (default: LoRA blocks 4–11, r=32)

### Update workflow (local → vast)

| Step | Where | Command |
|------|-------|---------|
| Edit code / configs | Local (branch `taaha`) | — |
| Commit & push | Local | `git add . && git commit -m "..." && git push` |
| Pull latest | vast.ai | `git pull` |
| Re-run setup or train | vast.ai | `python run.py --skip-unzip ...` |

**Note:** `git pull` only updates code. Datasets (`data/`) and ViT weights (`.cache/`) stay on disk — you don't re-download them each pull.

---

## Fallback: zip upload

Upload **`TransReID-PEFT.zip`** to Jupyter if git is unavailable:

```bash
unzip -o TransReID-PEFT.zip
python TransReID-PEFT/run.py
```

**Notebook:**

```python
!git clone -b taaha https://github.com/taaha-fluidweb/TransReID-PEFT.git
!python TransReID-PEFT/run.py --skip-unzip
```

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
